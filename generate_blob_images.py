# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
def generate_blob_images(N, save_dir="generated_data", seed=None):
    """
    Generate N 512x512 images with Gaussian background, random blobs, and Poisson noise.

    Parameters
    ----------
    N : int
        Number of images to generate.
    save_dir : str
        Directory to save outputs.
    seed : int, optional
        Random seed for reproducibility.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    os.makedirs(save_dir, exist_ok=True)

    H, W = 512, 512
    all_images = []
    all_masks = []

    for i in range(N):
        # Uniform background sampled from N(100, 10)
        background_value = rng.normal(100, 10)
        image = np.full((H, W), background_value, dtype=np.float64)

        # Random number of blobs: 3-10
        n_blobs = rng.integers(3, 11)
        masks = np.zeros((H, W, n_blobs), dtype=np.uint8)

        # Precompute coordinate grids
        yy, xx = np.mgrid[:H, :W]

        for b in range(n_blobs):
            diameter = rng.integers(2, 21)
            radius = diameter / 2.0
            cy = rng.integers(0, H)
            cx = rng.integers(0, W)

            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            blob_mask = dist <= radius

            masks[:, :, b] = blob_mask.astype(np.uint8)
            image[blob_mask] += 20

        # Poisson noise: clamp to >= 0 since Poisson requires non-negative lambda
        image = np.clip(image, 0, None)
        image = rng.poisson(lam=image).astype(np.float64)

        all_images.append(image)
        all_masks.append(masks)

    # Save
    np.save(os.path.join(save_dir, "images.npy"), np.stack(all_images))
    for i, m in enumerate(all_masks):
        np.save(os.path.join(save_dir, f"masks_{i:04d}.npy"), m)

    print(f"Saved {N} images to {save_dir}/images.npy")
    print(f"Saved {N} mask arrays to {save_dir}/masks_XXXX.npy")

    return all_images, all_masks


# %%
def plot_example(image, masks, save_path="example_blob_image.png"):
    """Plot an example image: raw, combined mask, and overlay."""
    n_blobs = masks.shape[2]
    combined_mask = masks.any(axis=2).astype(np.float64)

    # Color each blob differently
    colored_masks = np.zeros((512, 512, 3), dtype=np.float64)
    cmap = plt.cm.tab10
    for b in range(n_blobs):
        color = cmap(b % 10)[:3]
        for c in range(3):
            colored_masks[:, :, c] += masks[:, :, b] * color[c]
    colored_masks = np.clip(colored_masks, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: Raw image
    im0 = axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Generated image", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Pixel value")

    # Panel 2: Binary mask overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(combined_mask, cmap="Reds", alpha=0.4)
    axes[1].set_title("Combined blob mask overlay", fontsize=13)

    # Panel 3: Individual blob masks (colored)
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(colored_masks, alpha=0.45)
    axes[2].set_title(f"Individual blob masks ({n_blobs} blobs)", fontsize=13)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {save_path}")
    return save_path


# %%
if __name__ == "__main__":
    N = 10
    images, masks = generate_blob_images(N, save_dir="generated_data", seed=42)
    plot_example(images[0], masks[0], save_path="example_blob_image.png")
