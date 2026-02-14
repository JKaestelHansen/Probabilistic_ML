# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from generate_blob_images import generate_blob_images

%load_ext autoreload
%autoreload 2

# %%
"""
Instance segmentation with uncertainty quantification on synthetic blob images.

Approach:
    - U-Net (encoder-decoder with skip connections) for semantic segmentation
    - MC Dropout for epistemic uncertainty (Gal & Ghahramani 2016)
    - Heteroscedastic BCE loss for learned aleatoric uncertainty (Kendall & Gal 2017)
    - Connected-component instance extraction + Hungarian matching
    - Evaluation using the uncertainty_quantification/ package

Follows the same uncertainty decomposition as DL_MCDropout_predictions.py:
    total_var = aleatoric_var + epistemic_var

References:
    - Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning..."
    - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
    - Scalia et al. (2020) "Evaluating Scalable Uncertainty Estimation Methods..."
    - Kæstel-Hansen et al. (2024) PLOS Computational Biology
"""

# %%
# ---- Data generation ----
# Generate separate train/val/test sets with different seeds
# Resized to 128x128 for practical CPU training

IMG_SIZE = 128

train_images, train_masks = generate_blob_images(
    200, save_dir="generated_data/seg_train", seed=0)
val_images, val_masks = generate_blob_images(
    50, save_dir="generated_data/seg_val", seed=1)
test_images, test_masks = generate_blob_images(
    50, save_dir="generated_data/seg_test", seed=2)

print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")
print(f"Test:  {len(test_images)} images")
print(f"Example image shape: {train_images[0].shape}")
print(f"Example mask shape:  {train_masks[0].shape} (H, W, n_blobs)")


# %%
# ---- Dataset ----

class BlobSegmentationDataset(Dataset):
    def __init__(self, images, masks, img_size=128):
        self.img_size = img_size
        self.images = []
        self.binary_masks = []
        self.instance_masks = masks  # keep original full-res masks for evaluation

        for img, mask in zip(images, masks):
            # Normalize image per-sample
            img_norm = (img - img.mean()) / (img.std() + 1e-6)

            # Resize image: (1, 1, H, W) for interpolate
            img_t = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            img_t = F.interpolate(img_t, size=(img_size, img_size),
                                  mode='bilinear', align_corners=False)
            self.images.append(img_t.squeeze(0))  # (1, H, W)

            # Binary mask: any blob channel = 1
            binary = mask.any(axis=2).astype(np.float32)
            mask_t = torch.tensor(binary).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(img_size, img_size), mode='nearest')
            self.binary_masks.append(mask_t.squeeze(0))  # (1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.binary_masks[idx]


# %%
# ---- U-Net architecture with MC Dropout ----
# Similar structure to CNN_MLP in DL_MCDropout_predictions.py but encoder-decoder
# Dropout uses training=True at inference (same pattern as enable_dropout)

class DoubleConv(nn.Module):
    """Two conv layers with ReLU. No batch norm (avoids BN/MC Dropout conflict)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2,
                 features=[32, 64, 128], dropout_rate=0.2):
        """
        U-Net for segmentation with MC Dropout.

        out_channels=2: channel 0 = logit, channel 1 = log_variance (aleatoric)
        Dropout with training=True is applied in bottleneck and decoder
        for MC Dropout inference.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = F.dropout(x, p=self.dropout_rate, training=True)
            x = dec(x)

        return self.final_conv(x)


# %%
# ---- Loss function ----
# Heteroscedastic BCE following Kendall & Gal (2017)
# Same principle as aleatoric_uncertainty_loss in DL_MCDropout_predictions.py
# but adapted for classification: sample noisy logits then compute BCE

def heteroscedastic_bce_loss(logits, log_var, target, n_samples=10):
    """
    Heteroscedastic binary cross-entropy loss (Kendall & Gal 2017).

    Models learned aleatoric uncertainty by sampling noisy logits
    from N(logit, sigma^2) and computing mean BCE across samples.

    Parameters:
        logits: predicted logits, shape (B, 1, H, W)
        log_var: predicted log-variance, shape (B, 1, H, W)
        target: ground truth binary mask, shape (B, 1, H, W)
        n_samples: number of Monte Carlo samples for the integral
    """
    sigma = torch.sqrt(F.softplus(log_var) + 1e-6)
    eps = torch.randn(n_samples, *logits.shape, device=logits.device)
    noisy_logits = logits.unsqueeze(0) + sigma.unsqueeze(0) * eps
    target_expanded = target.unsqueeze(0).expand_as(noisy_logits)
    loss = F.binary_cross_entropy_with_logits(
        noisy_logits, target_expanded, reduction='none')
    return loss.mean()


def enable_dropout(model):
    """Function to enable the dropout layers during test-time
    (same as DL_MCDropout_predictions.py)"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# %%
# ---- Setup ----

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset = BlobSegmentationDataset(train_images, train_masks, img_size=IMG_SIZE)
val_dataset = BlobSegmentationDataset(val_images, val_masks, img_size=IMG_SIZE)
test_dataset = BlobSegmentationDataset(test_images, test_masks, img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = UNet(in_channels=1, out_channels=2,
             features=[32, 64, 128], dropout_rate=0.2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')


# %%
# ---- Training ----
# Same structure as DL_MCDropout_predictions.py training loop

num_epochs = 50
best_val_loss = float('inf')

training_losses = []
validation_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        output = model(images)
        logits = output[:, 0:1, :, :]
        log_var = output[:, 1:2, :, :]

        loss = heteroscedastic_bce_loss(logits, log_var, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / len(images)

    training_losses.append(running_loss)

    # Validation
    model.eval()
    enable_dropout(model)
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            logits = output[:, 0:1, :, :]
            log_var = output[:, 1:2, :, :]
            loss = heteroscedastic_bce_loss(logits, log_var, masks)
            val_loss += loss.item() / len(images)

    validation_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Loss {running_loss:.4f}, Val Loss {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_segmentation_model.pth')
        print(f'\tBest val loss: {val_loss:.4f}')

plt.figure(figsize=(5, 4))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
# ---- MC Dropout inference ----
# Same pattern as DL_MCDropout_predictions.py lines 376-398
# T stochastic forward passes with dropout enabled at test time

forward_passes = 20

model.load_state_dict(torch.load('best_segmentation_model.pth',
                                  map_location=device, weights_only=True))
model.eval()
enable_dropout(model)

all_logits = []   # will be (T, N, 1, H, W)
all_log_vars = []

for t in tqdm(range(forward_passes), desc='MC forward passes'):
    batch_logits = []
    batch_log_vars = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            output = model(images)
            batch_logits.append(output[:, 0:1, :, :].cpu().numpy())
            batch_log_vars.append(output[:, 1:2, :, :].cpu().numpy())
    all_logits.append(np.concatenate(batch_logits, axis=0))
    all_log_vars.append(np.concatenate(batch_log_vars, axis=0))

all_logits = np.stack(all_logits)      # (T, N, 1, H, W)
all_log_vars = np.stack(all_log_vars)  # (T, N, 1, H, W)

print(f'MC predictions shape: {all_logits.shape}')


# %%
# ---- Uncertainty decomposition ----
# Same decomposition as DL_MCDropout_predictions.py lines 406-417:
#   epistemic = var of predictions across MC passes
#   aleatoric = mean of learned variances
#   total = aleatoric + epistemic

# Logits -> probabilities
all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

# Mean prediction
mean_prob = np.mean(all_probs, axis=0)  # (N, 1, H, W)

# Epistemic uncertainty: variance of sigmoid outputs across forward passes
epi_var = np.var(all_probs, axis=0)  # (N, 1, H, W)

# Aleatoric uncertainty: mean of learned variances
# softplus(log_var) = variance in logit space
# Transform to probability space via delta method: var_p ~ sigmoid'(mu)^2 * var_logit
alea_var_logit = np.mean(np.log1p(np.exp(all_log_vars)), axis=0)  # mean softplus
mean_logit = np.mean(all_logits, axis=0)
sigmoid_deriv = mean_prob * (1 - mean_prob)
alea_var = (sigmoid_deriv ** 2) * alea_var_logit  # (N, 1, H, W)

# Total uncertainty
total_var = alea_var + epi_var  # (N, 1, H, W)

print(f'Epistemic  var range: [{epi_var.min():.6f}, {epi_var.max():.6f}]')
print(f'Aleatoric  var range: [{alea_var.min():.6f}, {alea_var.max():.6f}]')
print(f'Total      var range: [{total_var.min():.6f}, {total_var.max():.6f}]')


# %%
# ---- Visualization: uncertainty maps ----

n_examples = 4
fig, axes = plt.subplots(n_examples, 6, figsize=(20, 3.5 * n_examples))
col_titles = ['Input', 'Ground Truth', 'Prediction',
              'Total Unc.', 'Epistemic Unc.', 'Aleatoric Unc.']

for i in range(n_examples):
    img = test_dataset.images[i].numpy().squeeze()
    gt = test_dataset.binary_masks[i].numpy().squeeze()
    pred = mean_prob[i].squeeze()

    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 1].imshow(gt, cmap='gray')
    axes[i, 2].imshow(pred, cmap='gray')
    im3 = axes[i, 3].imshow(np.sqrt(total_var[i].squeeze()), cmap='hot')
    im4 = axes[i, 4].imshow(np.sqrt(epi_var[i].squeeze()), cmap='hot')
    im5 = axes[i, 5].imshow(np.sqrt(alea_var[i].squeeze()), cmap='hot')

    for j in range(6):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == 0:
            axes[i, j].set_title(col_titles[j], fontsize=12)

plt.tight_layout()
plt.savefig('segmentation_uncertainty_maps.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# ---- Pixel-level UQ evaluation ----
# Using the uncertainty_quantification package (same as DL_MCDropout_predictions.py)

from uncertainty_quantification.calibration import (
    prep_reliability_diagram,
    error_based_calibration,
    expected_normalized_calibration_error,
    max_calibration_error,
)
from uncertainty_quantification.confidence import (
    quantile_and_oracle_errors,
    ranking_confidence_curve,
    area_confidence_oracle_error,
    error_drop,
    decreasing_ratio,
)
from uncertainty_quantification.chi_squared import chi_squared_anees

# Flatten all pixel predictions
pixel_preds = mean_prob.flatten()
pixel_targets = np.array(
    [test_dataset.binary_masks[i].numpy() for i in range(len(test_dataset))]
).flatten()
pixel_total_unc = np.sqrt(total_var.flatten())
pixel_errors = np.abs(pixel_preds - pixel_targets)
pixel_var = total_var.flatten()

# Subsample for computational efficiency (millions of pixels)
n_pixel_eval = 50000
rng = np.random.default_rng(42)
idx = rng.choice(len(pixel_preds),
                 size=min(n_pixel_eval, len(pixel_preds)), replace=False)

pixel_preds_s = pixel_preds[idx]
pixel_targets_s = pixel_targets[idx]
pixel_unc_s = pixel_total_unc[idx]
pixel_errors_s = pixel_errors[idx]
pixel_var_s = pixel_var[idx]

n_quantiles = 50

# Reliability diagram
count, perc, ECE_pixel, Sharpness_pixel = prep_reliability_diagram(
    pixel_targets_s, pixel_preds_s, pixel_unc_s, n_quantiles)

# Error-based calibration
avg_empirical_error, avg_predicted_uncertainty = error_based_calibration(
    pixel_targets_s, pixel_preds_s, pixel_var_s, num_bins=n_quantiles)

# Confidence curves
quantile_errs_px, oracle_errs_px = quantile_and_oracle_errors(
    pixel_unc_s, pixel_errors_s, n_quantiles)

auco_pixel = area_confidence_oracle_error(
    quantile_errs_px, oracle_errs_px, quantiles=n_quantiles)
err_drop_pixel = error_drop(quantile_errs_px[::-1])
decr_ratio_pixel = decreasing_ratio(quantile_errs_px[::-1])

ence_pixel = expected_normalized_calibration_error(
    pixel_errors_s, pixel_unc_s, n_quantiles=n_quantiles)
MCE_pixel = max_calibration_error(count, perc)


# %%
# ---- Pixel-level UQ plots ----

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Reliability diagram
axes[0].plot(perc, count, lw=1)
axes[0].scatter(perc, count, s=10)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[0].set_xlabel('Expected Confidence')
axes[0].set_ylabel('Empirical Coverage')
axes[0].set_title('Pixel Reliability Diagram')

# Error-based calibration
axes[1].plot(avg_predicted_uncertainty, avg_empirical_error,
             'o-', lw=0.5, markersize=3)
maxval = max(max(avg_predicted_uncertainty), max(avg_empirical_error))
axes[1].plot([0, maxval], [0, maxval], linestyle='--', color='gray', lw=0.5)
axes[1].set_xlabel('Avg. Pred. Uncertainty')
axes[1].set_ylabel('Avg. Empirical Error')
axes[1].set_title('Pixel Error-based Calibration')

# Confidence curve
axes[2].plot(quantile_errs_px[::-1], 'o-',
             label='Quantile Error', markersize=1, lw=0.5)
axes[2].plot(oracle_errs_px[::-1], 'o-',
             label='Oracle Error', markersize=1, lw=0.5)
xticks = np.linspace(0, n_quantiles, 5)
axes[2].set_xticks(ticks=xticks,
                   labels=np.round(1 - xticks / n_quantiles, 2))
axes[2].set_xlabel('Fraction of data retained')
axes[2].set_ylabel('Norm. Error')
axes[2].set_title('Pixel Confidence Curve')
axes[2].legend()

plt.tight_layout()
plt.savefig('pixel_level_uq.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# ---- Instance extraction and matching ----

def extract_instances(pred_prob, threshold=0.5):
    """Extract instances via connected components on thresholded prediction."""
    binary = (pred_prob > threshold).astype(np.int32)
    labeled, n_instances = ndimage.label(binary)
    instances = []
    for i in range(1, n_instances + 1):
        instances.append(labeled == i)
    return instances


def compute_iou(mask1, mask2):
    """Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def match_instances(pred_instances, gt_instance_masks, pred_shape,
                    iou_threshold=0.1):
    """
    Match predicted instances to GT instances via Hungarian algorithm on IoU.

    Parameters:
        pred_instances: list of boolean masks at pred_shape resolution
        gt_instance_masks: (H_orig, W_orig, n_blobs) original resolution
        pred_shape: (H, W) of prediction
        iou_threshold: minimum IoU to count as a match

    Returns:
        matches: list of (pred_idx, gt_idx, iou) tuples
        iou_matrix: full IoU matrix
        unmatched_pred: indices of false positive predictions
        unmatched_gt: indices of missed GT blobs
    """
    n_gt = gt_instance_masks.shape[2]
    if len(pred_instances) == 0 or n_gt == 0:
        return ([], np.zeros((0, 0)),
                list(range(len(pred_instances))), list(range(n_gt)))

    # Resize GT masks to prediction resolution
    gt_resized = []
    for b in range(n_gt):
        gt_b = gt_instance_masks[:, :, b].astype(np.float32)
        gt_t = torch.tensor(gt_b).unsqueeze(0).unsqueeze(0)
        gt_t = F.interpolate(gt_t, size=pred_shape, mode='nearest')
        gt_resized.append(gt_t.squeeze().numpy() > 0.5)

    # IoU cost matrix
    n_pred = len(pred_instances)
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_iou(pred_instances[i], gt_resized[j])

    # Hungarian matching (maximize IoU = minimize negative IoU)
    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_pred = list(range(n_pred))
    unmatched_gt = list(range(n_gt))

    for r, c in zip(row_idx, col_idx):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c, iou_matrix[r, c]))
            if r in unmatched_pred:
                unmatched_pred.remove(r)
            if c in unmatched_gt:
                unmatched_gt.remove(c)

    return matches, iou_matrix, unmatched_pred, unmatched_gt


# %%
# ---- Object-level UQ evaluation ----
# For each matched blob: IoU error (1 - IoU) vs aggregated uncertainty

blob_ious = []
blob_epi_uncs = []
blob_alea_uncs = []
blob_total_uncs = []
total_pred_instances = 0
total_gt_instances = 0
total_false_positives = 0
total_missed = 0

for i in range(len(test_dataset)):
    pred = mean_prob[i].squeeze()
    pred_instances = extract_instances(pred, threshold=0.5)
    gt_masks = test_dataset.instance_masks[i]

    total_pred_instances += len(pred_instances)
    total_gt_instances += gt_masks.shape[2]

    matches, iou_mat, unmatched_p, unmatched_g = match_instances(
        pred_instances, gt_masks,
        pred_shape=(IMG_SIZE, IMG_SIZE))

    total_false_positives += len(unmatched_p)
    total_missed += len(unmatched_g)

    for pred_idx, gt_idx, iou in matches:
        mask = pred_instances[pred_idx]
        blob_ious.append(iou)
        blob_epi_uncs.append(np.mean(np.sqrt(epi_var[i].squeeze()[mask])))
        blob_alea_uncs.append(np.mean(np.sqrt(alea_var[i].squeeze()[mask])))
        blob_total_uncs.append(np.mean(np.sqrt(total_var[i].squeeze()[mask])))

blob_ious = np.array(blob_ious)
blob_errors = 1 - blob_ious
blob_total_uncs = np.array(blob_total_uncs)
blob_epi_uncs = np.array(blob_epi_uncs)
blob_alea_uncs = np.array(blob_alea_uncs)

print(f'Matched blobs:    {len(blob_ious)}')
print(f'False positives:  {total_false_positives}')
print(f'Missed GT blobs:  {total_missed}')
print(f'Mean IoU:         {np.mean(blob_ious):.4f}')


# %%
# ---- Object-level UQ plots ----

n_q_obj = min(20, max(3, len(blob_ious) // 3))

if len(blob_ious) >= 10:
    obj_quantile_errs, obj_oracle_errs = quantile_and_oracle_errors(
        blob_total_uncs, blob_errors, n_q_obj)
    obj_auco = area_confidence_oracle_error(
        obj_quantile_errs, obj_oracle_errs, quantiles=n_q_obj)
    obj_err_drop = error_drop(obj_quantile_errs[::-1])
    obj_decr_ratio = decreasing_ratio(obj_quantile_errs[::-1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Uncertainty vs IoU error
    axes[0].scatter(blob_total_uncs, blob_errors, s=15, alpha=0.6)
    axes[0].set_xlabel('Mean Blob Uncertainty')
    axes[0].set_ylabel('1 - IoU (Error)')
    axes[0].set_title('Object: Uncertainty vs Error')

    # Epistemic vs Aleatoric per blob
    axes[1].scatter(blob_epi_uncs, blob_alea_uncs,
                    c=blob_errors, cmap='viridis', s=15, alpha=0.6)
    axes[1].set_xlabel('Epistemic Uncertainty')
    axes[1].set_ylabel('Aleatoric Uncertainty')
    axes[1].set_title('Object: Epistemic vs Aleatoric (color = error)')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='1 - IoU')

    # Object confidence curve
    axes[2].plot(obj_quantile_errs[::-1], 'o-',
                 label='Quantile Error', markersize=3, lw=0.5)
    axes[2].plot(obj_oracle_errs[::-1], 'o-',
                 label='Oracle Error', markersize=3, lw=0.5)
    xticks_obj = np.linspace(0, n_q_obj, 5)
    axes[2].set_xticks(ticks=xticks_obj,
                       labels=np.round(1 - xticks_obj / n_q_obj, 2))
    axes[2].set_xlabel('Fraction of blobs retained')
    axes[2].set_ylabel('Norm. Error')
    axes[2].set_title('Object Confidence Curve')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('object_level_uq.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print(f'Only {len(blob_ious)} matched blobs — skipping object-level plots')
    obj_auco = obj_err_drop = obj_decr_ratio = float('nan')


# %%
# ---- Summary metrics ----

# Pixel accuracy
pixel_preds_binary = (mean_prob.flatten() > 0.5).astype(float)
pixel_acc = np.mean(pixel_preds_binary == pixel_targets)
pixel_iou_fg = compute_iou(pixel_preds_binary, pixel_targets)

print()
print('=' * 60)
print('PIXEL-LEVEL Uncertainty Metrics:')
print(f'  ECE  (Expected Calibration Error):             {ECE_pixel:.4f}')
print(f'  MCE  (Max Calibration Error):                  {MCE_pixel:.4f}')
print(f'  ENCE (Expected Normalized Calibration Error):  {ence_pixel:.4f}')
print(f'  AUCO (Area Under Confidence Oracle Error):     {auco_pixel:.4f}')
print(f'  Error Drop:                                    {err_drop_pixel:.4f}')
print(f'  Decreasing Ratio:                              {decr_ratio_pixel:.4f}')
print(f'  Sharpness:                                     {Sharpness_pixel:.4f}')

if len(blob_ious) >= 10:
    print()
    print('OBJECT-LEVEL Uncertainty Metrics:')
    print(f'  Mean IoU:         {np.mean(blob_ious):.4f}')
    print(f'  AUCO:             {obj_auco:.4f}')
    print(f'  Error Drop:       {obj_err_drop:.4f}')
    print(f'  Decreasing Ratio: {obj_decr_ratio:.4f}')
    print(f'  Matched blobs:    {len(blob_ious)}')
    print(f'  False positives:  {total_false_positives}')
    print(f'  Missed GT:        {total_missed}')

print()
print('Accuracy Metrics:')
print(f'  Pixel Accuracy:    {pixel_acc:.4f}')
print(f'  Pixel IoU (fg):    {pixel_iou_fg:.4f}')
print(f'  Mean blob IoU:     {np.mean(blob_ious):.4f}')
print('=' * 60)


# %%
# ---- Prediction overlays: GT vs predicted masks on input images ----

def resize_gt_masks(gt_instance_masks, target_shape):
    """Resize each GT instance mask to target_shape via nearest interpolation."""
    n_gt = gt_instance_masks.shape[2]
    resized = []
    for b in range(n_gt):
        gt_b = gt_instance_masks[:, :, b].astype(np.float32)
        gt_t = torch.tensor(gt_b).unsqueeze(0).unsqueeze(0)
        gt_t = F.interpolate(gt_t, size=target_shape, mode='nearest')
        resized.append(gt_t.squeeze().numpy() > 0.5)
    return resized


n_show = min(6, len(test_dataset))
fig, axes = plt.subplots(n_show, 4, figsize=(18, 4 * n_show))
col_titles = ['Input Image', 'GT Mask Overlay', 'Predicted Mask Overlay',
              'Error Map (TP/FP/FN)']

for i in range(n_show):
    img = test_dataset.images[i].numpy().squeeze()
    gt = test_dataset.binary_masks[i].numpy().squeeze()
    pred = mean_prob[i].squeeze()
    pred_binary = (pred > 0.5).astype(float)

    # TP / FP / FN map
    tp = np.logical_and(pred_binary, gt)
    fp = np.logical_and(pred_binary, ~gt.astype(bool))
    fn = np.logical_and(~pred_binary.astype(bool), gt.astype(bool))
    error_rgb = np.zeros((*gt.shape, 3))
    error_rgb[tp, 1] = 1.0   # green = TP
    error_rgb[fp, 0] = 1.0   # red   = FP
    error_rgb[fn, 2] = 1.0   # blue  = FN

    # Input
    axes[i, 0].imshow(img, cmap='gray')

    # GT overlay (green)
    axes[i, 1].imshow(img, cmap='gray')
    gt_overlay = np.zeros((*gt.shape, 4))
    gt_overlay[gt > 0.5] = [0, 1, 0, 0.45]
    axes[i, 1].imshow(gt_overlay)

    # Prediction overlay (blue) + GT contours (red dashed)
    axes[i, 2].imshow(img, cmap='gray')
    pred_overlay = np.zeros((*pred.shape, 4))
    pred_overlay[pred_binary > 0.5] = [0.2, 0.4, 1.0, 0.45]
    axes[i, 2].imshow(pred_overlay)
    axes[i, 2].contour(gt, levels=[0.5], colors='red',
                        linewidths=0.8, linestyles='--')

    # Error map
    axes[i, 3].imshow(img, cmap='gray', alpha=0.3)
    axes[i, 3].imshow(error_rgb, alpha=0.7)

    for j in range(4):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == 0:
            axes[i, j].set_title(col_titles[j], fontsize=12)

# Legend for error map
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='TP'),
                   Patch(facecolor='red', label='FP'),
                   Patch(facecolor='blue', label='FN')]
axes[0, 3].legend(handles=legend_elements, loc='upper right',
                   fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig('prediction_overlays.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# ---- Accuracy metrics plots ----

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) IoU histogram for matched blobs
axes[0, 0].hist(blob_ious, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(np.mean(blob_ious), color='red', linestyle='--',
                     label=f'Mean IoU = {np.mean(blob_ious):.3f}')
axes[0, 0].axvline(np.median(blob_ious), color='orange', linestyle='--',
                     label=f'Median IoU = {np.median(blob_ious):.3f}')
axes[0, 0].set_xlabel('IoU')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('IoU Distribution (Matched Blobs)')
axes[0, 0].legend(fontsize=9)

# (b) Per-image detection counts: predicted vs GT
n_preds_per_img = []
n_gt_per_img = []
for i in range(len(test_dataset)):
    pred = mean_prob[i].squeeze()
    n_preds_per_img.append(len(extract_instances(pred, threshold=0.5)))
    n_gt_per_img.append(test_dataset.instance_masks[i].shape[2])

x_pos = np.arange(len(test_dataset))
bar_width = 0.35
axes[0, 1].bar(x_pos - bar_width / 2, n_gt_per_img, bar_width,
                label='GT blobs', color='forestgreen', alpha=0.7)
axes[0, 1].bar(x_pos + bar_width / 2, n_preds_per_img, bar_width,
                label='Predicted', color='steelblue', alpha=0.7)
axes[0, 1].set_xlabel('Image Index')
axes[0, 1].set_ylabel('Blob Count')
axes[0, 1].set_title('Detection Counts per Image')
axes[0, 1].legend(fontsize=9)
# thin out x-ticks for readability
tick_step = max(1, len(test_dataset) // 10)
axes[0, 1].set_xticks(x_pos[::tick_step])

# (c) Pixel precision-recall at varying thresholds
thresholds = np.linspace(0.05, 0.95, 40)
precisions = []
recalls = []
f1_scores = []
for thr in thresholds:
    pred_bin = (mean_prob.flatten() > thr).astype(float)
    tp_sum = np.sum(np.logical_and(pred_bin, pixel_targets))
    fp_sum = np.sum(np.logical_and(pred_bin, 1 - pixel_targets))
    fn_sum = np.sum(np.logical_and(1 - pred_bin, pixel_targets))
    prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

axes[1, 0].plot(recalls, precisions, 'o-', markersize=2, lw=1)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Pixel Precision-Recall Curve')
axes[1, 0].set_xlim(0, 1.05)
axes[1, 0].set_ylim(0, 1.05)

# (d) Pixel F1 / Precision / Recall vs threshold
axes[1, 1].plot(thresholds, precisions, label='Precision', lw=1)
axes[1, 1].plot(thresholds, recalls, label='Recall', lw=1)
axes[1, 1].plot(thresholds, f1_scores, label='F1', lw=1.5, linestyle='--')
best_f1_idx = np.argmax(f1_scores)
axes[1, 1].axvline(thresholds[best_f1_idx], color='gray', linestyle=':',
                     label=f'Best F1 thr = {thresholds[best_f1_idx]:.2f}')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Precision / Recall / F1 vs Threshold')
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_metrics.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# ---- Instance overlay colored by uncertainty ----
# Each predicted blob is colored by its mean uncertainty on a coolwarm colormap
# Low uncertainty = cool (blue), high uncertainty = warm (red)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

n_show = min(6, len(test_dataset))
fig, axes = plt.subplots(n_show, 3, figsize=(16, 4 * n_show))
col_titles = ['Instances by Total Unc.',
              'Instances by Epistemic Unc.',
              'Instances by Aleatoric Unc.']

# Gather global uncertainty ranges for consistent colorbars
all_inst_total = []
all_inst_epi = []
all_inst_alea = []
per_image_data = []

for i in range(n_show):
    pred = mean_prob[i].squeeze()
    pred_instances = extract_instances(pred, threshold=0.5)

    inst_total_uncs = []
    inst_epi_uncs = []
    inst_alea_uncs = []
    for inst_mask in pred_instances:
        inst_total_uncs.append(np.mean(np.sqrt(total_var[i].squeeze()[inst_mask])))
        inst_epi_uncs.append(np.mean(np.sqrt(epi_var[i].squeeze()[inst_mask])))
        inst_alea_uncs.append(np.mean(np.sqrt(alea_var[i].squeeze()[inst_mask])))

    all_inst_total.extend(inst_total_uncs)
    all_inst_epi.extend(inst_epi_uncs)
    all_inst_alea.extend(inst_alea_uncs)
    per_image_data.append((pred_instances, inst_total_uncs,
                           inst_epi_uncs, inst_alea_uncs))

# Normalizers for consistent color scaling across images
norm_total = Normalize(vmin=min(all_inst_total) if all_inst_total else 0,
                       vmax=max(all_inst_total) if all_inst_total else 1)
norm_epi = Normalize(vmin=min(all_inst_epi) if all_inst_epi else 0,
                     vmax=max(all_inst_epi) if all_inst_epi else 1)
norm_alea = Normalize(vmin=min(all_inst_alea) if all_inst_alea else 0,
                      vmax=max(all_inst_alea) if all_inst_alea else 1)
cmap = plt.cm.coolwarm

for i in range(n_show):
    img = test_dataset.images[i].numpy().squeeze()
    gt = test_dataset.binary_masks[i].numpy().squeeze()
    pred_instances, inst_total, inst_epi, inst_alea = per_image_data[i]

    norms = [norm_total, norm_epi, norm_alea]
    unc_lists = [inst_total, inst_epi, inst_alea]

    for col in range(3):
        axes[i, col].imshow(img, cmap='gray')

        # Build RGBA overlay: each instance colored by its uncertainty
        overlay = np.zeros((*img.shape, 4))
        for k, inst_mask in enumerate(pred_instances):
            if k < len(unc_lists[col]):
                rgba = cmap(norms[col](unc_lists[col][k]))
                overlay[inst_mask, 0] = rgba[0]
                overlay[inst_mask, 1] = rgba[1]
                overlay[inst_mask, 2] = rgba[2]
                overlay[inst_mask, 3] = 0.6

        axes[i, col].imshow(overlay)
        # Draw GT contours for reference
        axes[i, col].contour(gt, levels=[0.5], colors='lime',
                              linewidths=0.6, linestyles='--')
        axes[i, col].set_xticks([])
        axes[i, col].set_yticks([])
        if i == 0:
            axes[i, col].set_title(col_titles[col], fontsize=12)

# Add colorbars
for col, (norm, label) in enumerate(
        zip(norms, ['Total Unc.', 'Epistemic Unc.', 'Aleatoric Unc.'])):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[:, col].tolist(), fraction=0.02,
                 pad=0.01, label=label)

plt.tight_layout()
plt.savefig('instance_uncertainty_overlay.png', dpi=150, bbox_inches='tight')
plt.show()


# %%
# ---- Per-image FP/FN maps with uncertainty ----
# Shows where the model is uncertain AND wrong

n_show_err = min(4, len(test_dataset))
fig, axes = plt.subplots(n_show_err, 3, figsize=(14, 4 * n_show_err))
col_titles = ['FP regions (unc. heatmap)', 'FN regions (unc. heatmap)',
              'Boundary uncertainty']

for i in range(n_show_err):
    img = test_dataset.images[i].numpy().squeeze()
    gt = test_dataset.binary_masks[i].numpy().squeeze()
    pred = mean_prob[i].squeeze()
    pred_binary = (pred > 0.5).astype(float)
    unc = np.sqrt(total_var[i].squeeze())

    fp_mask = np.logical_and(pred_binary, ~gt.astype(bool))
    fn_mask = np.logical_and(~pred_binary.astype(bool), gt.astype(bool))

    # FP regions colored by uncertainty
    axes[i, 0].imshow(img, cmap='gray')
    fp_unc = np.where(fp_mask, unc, np.nan)
    axes[i, 0].imshow(fp_unc, cmap='Reds', alpha=0.7,
                       vmin=0, vmax=np.nanmax(unc))

    # FN regions colored by uncertainty
    axes[i, 1].imshow(img, cmap='gray')
    fn_unc = np.where(fn_mask, unc, np.nan)
    axes[i, 1].imshow(fn_unc, cmap='Blues', alpha=0.7,
                       vmin=0, vmax=np.nanmax(unc))

    # Boundary uncertainty: uncertainty at edges of predicted blobs
    from scipy.ndimage import binary_dilation, binary_erosion
    boundary = binary_dilation(pred_binary.astype(bool)) ^ \
               binary_erosion(pred_binary.astype(bool))
    axes[i, 2].imshow(img, cmap='gray')
    boundary_unc = np.where(boundary, unc, np.nan)
    axes[i, 2].imshow(boundary_unc, cmap='hot', alpha=0.8,
                       vmin=0, vmax=np.nanmax(unc))
    axes[i, 2].contour(gt, levels=[0.5], colors='lime',
                        linewidths=0.6, linestyles='--')

    for j in range(3):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == 0:
            axes[i, j].set_title(col_titles[j], fontsize=12)

plt.tight_layout()
plt.savefig('error_uncertainty_maps.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
