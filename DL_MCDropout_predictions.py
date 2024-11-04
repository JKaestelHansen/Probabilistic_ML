# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

%load_ext autoreload
%autoreload 2


# %%
"""
confirm with:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012061#sec020
https://pubs.acs.org/doi/epdf/10.1021/acs.jcim.9b00975?ref=article_openPDF

for evaluating and comparing uncertainty estimates and their 
confindence/calibration metrics (UC / UQ)

"""

def aleatoric_uncertainty_loss(y_true, y_pred, variance_pred):
    """
    Computes aleatoric uncertainty loss 

    Scalia et al. 
    "Evaluating Scalable Uncertainty Estimation Methods 
     for Deep Learning-Based Molecular Property Prediction"

    Parameters:
    y_true: Tensor of true values, shape [batch_size]
    y_pred: Tensor of predicted means from the model, shape [batch_size]
    sigma_pred: Tensor of predicted standard deviations (uncertainty), shape [batch_size]

    Returns:
    Tensor: Loss value (mean of batch losses)
    """
    # Compute the loss for each sample
    squared_diff = (y_true - y_pred) ** 2
    loss = (0.5 * squared_diff / variance_pred + 0.5 * torch.log(variance_pred)).mean()
    
    return loss

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    

class CNN_MLP(nn.Module):
    def __init__(self, input_size, in_channels,
                 cnn_channels, pooling, kernel_size,
                 hidden_sizes, output_size, dropout_rate, device='cpu'):
        super(CNN_MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.cnn_layers = len(cnn_channels)
        self.cnn = nn.Sequential()
        self.in_channels = in_channels
        for i in range(self.cnn_layers):
            self.cnn.add_module(f'conv_{i}', nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_channels[i], kernel_size=kernel_size))
            self.cnn.add_module(f'relu_{i}', nn.ReLU())
            self.cnn.add_module(f'maxpool_{i}', nn.MaxPool1d(kernel_size=pooling))
            self.in_channels = cnn_channels[i]

        cnn_output_size = input_size
        for i in range(self.cnn_layers):
            cnn_output_size = (cnn_output_size - kernel_size) / pooling 
            cnn_output_size = int(np.ceil(cnn_output_size))

        cnn_output_size *= cnn_channels[-1]
        
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(cnn_output_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.fc.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.fc.append(nn.Linear(hidden_sizes[-1], output_size))

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        for i, layer in enumerate(self.fc):
            x = layer(x)
            if i < len(self.fc) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=True)
        return x
    
class Tracks_to_Dataset(Dataset):
    def __init__(self, X, Y, X_padtoken=0, device='cpu'):
        maxlens = np.max([len(x) for x in X])
        X = [torch.tensor(x.astype(np.float32), device=device).to(torch.float32) for x in X]
        Y = [torch.tensor(y.astype(np.float32), device=device).to(torch.float32) for y in Y]
        # seems like pre-padding is smartest https://arxiv.org/abs/1903.07288, 
        # but lstm should only see variable len input
        self.X = [nn.ConstantPad1d((maxlens-len(x), 0), X_padtoken)(x.permute(*torch.arange(x.ndim - 1, -1, -1))).float() for x in X]
        self.y = Y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# import data
X = pickle.load(open("data/directed_tracks/tracks.pkl", "rb"))
y = pickle.load(open("data/directed_tracks/speeds.pkl", "rb"))

lengths = [len(x) for x in X]
torch.set_default_device('mps')
print('torch.mps.is_available()', torch.mps.is_available())
print(torch.Generator(device='mps').device)

# make a pytorch MLP model with monte carlo dropout that takes a list of hidden layer sizes

    
# split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

# create a pytorch dataset
train_dataset = Tracks_to_Dataset(X_train, y_train, device= 'mps' if torch.mps.is_available() else 'cpu')
test_dataset = Tracks_to_Dataset(X_test, y_test, device= 'mps' if torch.mps.is_available() else 'cpu')
val_dataset = Tracks_to_Dataset(X_val, y_val, device= 'mps' if torch.mps.is_available() else 'cpu')


# create a pytorch dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator(device='mps'))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, generator=torch.Generator(device='mps'))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, generator=torch.Generator(device='mps'))

# create a pytorch model
input_size = len(X[0])  # Assuming all inputs have the same length
print(input_size)
hidden_size = [100, 10]
output_size = 2  # Mean and variance
dropout_rate = 0.2
in_channels = 3
cnn_channels = [8, 16, 32, 64]
pooling = 2
kernel_size = 5

model = CNN_MLP(input_size, in_channels,
                cnn_channels, pooling, kernel_size,
                hidden_size, output_size, dropout_rate,
                device='mps' if torch.mps.is_available() else 'cpu')
print(model)

# create a loss function and optimizer
criterion = nn.GaussianNLLLoss()

print('NOTE: try ADAM with weigt decay')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight decays
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 100
best_val_loss = float('inf')

training_losses = []
validation_losses = []


forward_passes = 10
WA = 0.3  # Weight ratio for the WA * gaussianNLL (1-WA) * aleatoric uncertainty loss

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        # do 10 monte carlo passes with dropout
        # could mess with training and the point of having dropout
        # seems like it is okay to do this in the training phase

        mean_mc = []
        var_mc = []
        for j in range(forward_passes):  # Monte Carlo Dropout
            outputs = model(inputs)
            mean, var = outputs[:, 0], outputs[:, 1]
            var = nn.functional.softplus(var)  # Ensure variance is positive
            mean_mc.append(mean)
            var_mc.append(var)
        
        mean = torch.mean(torch.stack(mean_mc), dim=0)
        alea_var = torch.mean(torch.stack(var_mc), dim=0)
        epi_var = torch.var(torch.stack(mean_mc), dim=0)
        # var = alea_var + epi_var
        var = alea_var  # Uncomment this line if you want to use only aleatoric variance

        # # predict without monte carlo dropout loops
        # outputs = model(inputs)
        # mean, var = outputs[:, 0], outputs[:, 1]
        # var = nn.functional.softplus(var)  # Ensure variance is positive

        #loss = criterion(outputs, labels.view(-1, 1))
        # loss = aleatoric_uncertainty_loss(labels.view(-1, 1), mean, var)
        #loss = criterion(mean, labels, var) #+ 0.5 * torch.mean(torch.log(var))  # Add a regularization term
        # loss = criterion(mean, labels, var)
        loss = WA * criterion(mean, labels, var) + (1-WA) * aleatoric_uncertainty_loss(labels.view(-1, 1), mean, var)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / len(labels)
    
    training_losses.append(running_loss)

    # evaluate the model on val set
    n_samples = len(val_loader.dataset)  # Number of samples in the validation set
    with torch.no_grad():
        dropout_predictions = np.empty((0, n_samples, 1))
        dropout_predictions_var = np.empty((0, n_samples, 1))
        for i in range(forward_passes):
            predictions = np.empty((0,1))
            predicted_var = np.empty((0,1))
            model.eval()
            enable_dropout(model)
            for j, (image, label) in enumerate(val_loader):
                image = image.to(torch.device('mps')) if torch.mps.is_available() else image
                output = model(image)
                mean, var = output[:, 0], output[:, 1]
                var = nn.functional.softplus(var)
                predictions = np.vstack((predictions, mean.detach().cpu().numpy().reshape(-1,1)))
                predicted_var = np.vstack((predicted_var, var.detach().cpu().numpy().reshape(-1,1)))
            
            dropout_predictions = np.vstack((dropout_predictions,
                                     predictions[np.newaxis, :, :]))
            dropout_predictions_var = np.vstack((dropout_predictions_var,
                                        predicted_var[np.newaxis, :, :]))
        
        mean = torch.tensor(np.mean(dropout_predictions, axis=0).astype(np.float32))
        mean_var_alea = torch.tensor(np.mean(dropout_predictions_var, axis=0).astype(np.float32))
        mean_var_epi = torch.tensor(np.var(dropout_predictions, axis=0).astype(np.float32))
        # mean_var = mean_var_alea + mean_var_epi
        mean_var = mean_var_alea  # Uncomment this line if you want to use only aleatoric variance
        labels = torch.tensor(np.array([label.detach().cpu().numpy() for label in val_loader.dataset.y]).reshape(-1, 1))
        
        val_loss = criterion(mean, labels, mean_var) 
        validation_losses.append(val_loss.item())

        print(f'Epoch {epoch+1}, Loss {running_loss}', f'Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), 'best_model.pth')
            print(f'\t Best validation Loss: {val_loss}')

plt.figure(figsize=(5, 4))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('should I add the epistermic uncertainty to the training/validation loss?')
print('any literaure on this?')
print('does not look like I should add epistermic unc before calculating loss')
print('probably beacuse the loss cant deliineate between aleatoric and epistermic uncertainty in this setup??')
print('loss function, weight decay, dropout, etc. all greatly affect the uncertainty estimates')

# %%
forward_passes = 10
n_samples = len(test_loader.dataset)

dropout_predictions = np.empty((0, n_samples, 1))
dropout_predictions_var = np.empty((0, n_samples, 1))
for i in tqdm(range(forward_passes)):
    predictions = np.empty((0,1))
    predicted_var = np.empty((0,1))
    best_model.eval()
    enable_dropout(best_model)
    for j, (image, label) in enumerate(test_loader):
        image = image.to(torch.device('mps')) if torch.mps.is_available() else image
        with torch.no_grad():
            output = best_model(image)
            mean, var = output[:, 0], output[:, 1]
            var = nn.functional.softplus(var)  # Ensure variance is positive
        predictions = np.vstack((predictions, mean.detach().cpu().numpy().reshape(-1,1)))
        predicted_var = np.vstack((predicted_var, var.detach().cpu().numpy().reshape(-1,1)))

    dropout_predictions = np.vstack((dropout_predictions,
                                     predictions[np.newaxis, :, :]))
    dropout_predictions_var = np.vstack((dropout_predictions_var,
                                        predicted_var[np.newaxis, :, :]))
    # dropout predictions - shape (forward_passes, n_samples, n_classes)


print(dropout_predictions[0].shape)
print(dropout_predictions_var[0].shape)

# Calculating mean across multiple MCD forward passes 
mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
alea_var = np.mean(dropout_predictions_var, axis=0)  # shape (n_samples, n_classes)

print(mean.shape)

# Calculating variance across multiple MCD forward passes 
epi_var = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

# Total variance
variance = epi_var 
variance = alea_var 
variance = alea_var + epi_var

# %%

# functions for uncertainty quantification from Michael & Kæstel-Hansen PLOS 
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012061#sec020

from uncertainty_quantification.confidence import (
    quantile_and_oracle_errors,
    ranking_confidence_curve,
    area_confidence_oracle_error,
    error_drop,
    decreasing_ratio
)

from uncertainty_quantification.calibration import (
    prep_reliability_diagram,
    confidence_based_calibration,
    expected_normalized_calibration_error,
    max_calibration_error,
    error_based_calibration
)
from uncertainty_quantification.chi_squared import (
    chi_squared_anees
)

# jupyter notebook magic line to auto reload the imported modules
%load_ext autoreload
%autoreload 2




# Example usage:
# y_true = torch.tensor([2.0, 3.0, 4.0])           # True values
# y_pred = torch.tensor([2.5, 2.9, 4.1])           # Predicted means
# sigma_pred = torch.tensor([0.5, 0.3, 0.4])       # Predicted uncertainties (standard deviation)

# calibration_results = error_based_calibration(y_true, y_pred, sigma_pred, num_bins=3)
# print("Error-based calibration results:", calibration_results)


n_quantiles = 20
quantiles = np.arange(0, 1.1, 1 / n_quantiles).astype(np.float32)
errors = ((mean.squeeze() - y_test)**2)**0.5 # Root Mean Squared Error
uncertainties = np.sqrt(variance.squeeze())
y_pred = mean.squeeze()
target = y_test
var_pred = variance.squeeze()

quantile_errs, oracle_errs = quantile_and_oracle_errors(
    uncertainties, errors, n_quantiles)

count, perc, ECE, Sharpness = prep_reliability_diagram(target, y_pred, uncertainties, n_quantiles)

average_loss_in_quantile, average_loss_in_oracle_quantile = ranking_confidence_curve(
    errors, uncertainties, quantiles=n_quantiles)

auco = area_confidence_oracle_error(quantile_errs, oracle_errs, quantiles=n_quantiles)
err_drop = error_drop(quantile_errs[::-1])
decr_ratio = decreasing_ratio(quantile_errs[::-1])

ence = expected_normalized_calibration_error(errors, uncertainties, n_quantiles=n_quantiles)

MCE = max_calibration_error(count, perc)

csa = chi_squared_anees(
    target, y_pred, var_pred, eps=0.0001
)

avg_empirical_error, avg_predicted_uncertainty = error_based_calibration(target, y_pred, var_pred, num_bins=n_quantiles)


# plot predictions, target, and uncertainties to assess
plt.figure(figsize=(10, 4))
plt.errorbar(np.arange(n_samples)[::10], mean.squeeze()[::10], 
            yerr=np.sqrt(variance.squeeze())[::10], fmt='o', 
            label='Predictive Mean',
            ecolor='k', capthick=2, markersize=5)
plt.scatter(np.arange(n_samples)[::10], y_test[::10], label='True Value', color='r', s=10, zorder=5)
plt.xlabel('Sample Index')
plt.ylabel('Prediction')
plt.legend()

# true vs predicted to assess accuracy
plt.figure(figsize=(4, 3))
plt.scatter(mean.squeeze(), y_test, label='True Value', s=10, zorder=5)
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()  # Show all plots

# histogram of uncertainties to assess spread
plt.figure(figsize=(4, 3))
plt.hist(uncertainties, bins=100)
plt.xlabel('Uncertainties')
plt.ylabel('Count')
plt.title('Histogram of Uncertainties')
plt.show()  # Show all plots

# scatter plot of uncertainties vs errors to assess calibration
# higher uncertainties pair with higher errors
plt.figure(figsize=(4, 3))
plt.scatter(uncertainties, errors, s=1)
plt.xlabel('Uncertainties')
plt.ylabel('Errors')
plt.title('Uncertainties vs Errors')
plt.show()  # Show all plots

# scatter plot of uncertainties vs target to assess patterns or bias
plt.figure(figsize=(4, 3))
plt.scatter(uncertainties, target, s=1)
plt.xlabel('Uncertainties')
plt.ylabel('Target')
plt.title('Uncertainties vs True vals')
plt.show()  # Show all plots

# quantile_and_oracle_errors and ranking_confidence_curve are the same
# but quantile_and_oracle_errors normalizes to case of all errors included (quantile=1)
# Quantile Error is the average error in each quantile of uncertainty (error in bins of uncertainty)
# Oracle Error is the average error in each quantile of oracle errors (error in bins of error) corresponding 
# if quantile errors follow the oracle errors, the model is well calibrated
plt.figure(figsize=(4, 3))
plt.plot(quantile_errs[::-1], 'o-', label='Quantile Error')
plt.plot(oracle_errs[::-1], 'o-', label='Oracle Error')
plt.xticks(np.arange(n_quantiles+2), np.round(quantiles[::-1], 2), rotation=45)
plt.xlabel('Quantile')
plt.ylabel('Max norm. Error')
plt.title('Rank-based confidence curve')
plt.legend()
plt.show()  # Show all plots

# error_based_calibration is a plot of the average error versus the average uncertainty
# when binning by uncertainty
# being on the diagonal is good calibration
plt.figure(figsize=(4, 3))
plt.plot(avg_predicted_uncertainty, avg_empirical_error, 'o-')
maxval = max(max(avg_predicted_uncertainty), max(avg_empirical_error))
plt.plot([0, maxval], [0, maxval], linestyle='--', color='gray')
plt.xlabel('Avg. Pred. Uncertainty')
plt.ylabel('Avg. Empirical Error')
plt.title('Error-based Calibration')
plt.show()


# Reliability Diagram
# Reliability Diagram is a plot of the predicted confidence against the true confidence
# binning the predictions by confidence and plotting against accuracy in each bin
# if predicting probability of 10%, we want accuracy to be 10% etc.
# The diagonal is the line of perfect calibration
# The closer the plot is to the diagonal, the better the calibration
plt.figure(figsize=(4, 3))
plt.plot(perc, count)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Percentile')
plt.ylabel('Pred. Conf.')
plt.title('Reliability Diagram')
plt.show()  # Show all plots

# plotting number of standard deviations away predictions are from the true value
std_away = np.abs(mean.squeeze() - y_test)/np.hstack(np.sqrt(variance))
print( std_away.shape)
plt.figure(figsize=(4, 3))
plt.hist(std_away, bins=100)
plt.xlabel('Standard Deviations Away from True Value')
plt.ylabel('Count')
plt.title('Histogram of Standard Deviations Away from True Value')
plt.show()  # Show all plots

print()
print('Uncertainty calibration and confidence metrics:')
# measures differences in quantile error curves and oracle error curves
print(f'\tArea Under Confidence Oracle Error: {auco}')

# Difference between first uncertainty quantile and last uncertainty quantile
print(f'\tError Drop: {err_drop}')

# fractions of uncertainties larger than the next quantiles uncertainties, 
# to cover monotonicity
print(f'\tDecreasing Ratio: {decr_ratio}')

# reduced chi squared statistic
# A method would thus be over-confident if the empirical error 
# is larger than the uncertainties it predicts.
print(f'\tReduced Chi Squared Statistic: {csa}')

# average error between bins of the reliability diagram
# showing the average deviation from the true value in each bin
print(f'\tExpected Calibration Error: {ECE}')

# max difference in reliability diagram
# showing worst case deviation from the true value in each bin
print(f'\tMax Calibration Error: {MCE}')

# expected normalized calibration error
# measures the mean of differences between the predicted root mean variance and the RMSE 
# per bin normalized by root mean variance
# of the error-based calibration diagram
print(f'\tExpected Normalized Calibration Error: {ence}')

# np.std(uncertainties, ddof=1) / np.mean(uncertainties)
# measures diversity in the uncertainty estimates
# because outputting constant uncertainty is not useful
print(f'\tSharpness: {Sharpness}')  # Show all plots

print()
print('Accuracy metrics:')
print(f'\tRoot Mean Squared Error: {np.mean(errors)}')
print(f'\tMean Squared Error: {np.mean((mean.squeeze() - y_test)**2)}')
print(f'\tRoot Mean Squared Error vs mean value: {np.mean(np.abs(mean.squeeze() - y_test))/np.mean(y_test)}')
print(f'\tMean Squared Error vs mean value: {np.mean((mean.squeeze() - y_test)**2)/np.mean(y_test)}')
print(f'\tR2 Score: {1 - np.sum((mean.squeeze() - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)}')

# %%