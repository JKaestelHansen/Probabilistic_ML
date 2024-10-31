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
criterion = nn.MSELoss()
criterion = nn.GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 30
best_val_loss = float('inf')

training_losses = []
validation_losses = []


forward_passes = 10

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
        loss = criterion(mean, labels, var)
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



# %%

# Evaluate the model on the test set
test_predictions = []
test_variances = []
for sample in samples:
    model.load_state_dict(sample)
    model.eval()
    predictions = []
    variances = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            output = model(inputs)
            mean, var = output[:, 0], output[:, 1]
            var = nn.functional.softplus(var)
            predictions.append(mean)
            variances.append(var)
    test_predictions.append(torch.cat(predictions))
    test_variances.append(torch.cat(variances))

# Calculate the mean and variance of the predictions
test_predictions = torch.mean(torch.stack(test_predictions), dim=0)
test_alea_variances = torch.mean(torch.stack(test_variances), dim=0)
test_epi_variances = torch.var(torch.stack(test_predictions), dim=0)
test_variances = test_alea_variances + test_epi_variances



plt.figure(figsize=(10, 4))
plt.errorbar(np.arange(n_samples)[::10], mean.squeeze()[::10], 
            yerr=np.sqrt(test_variances.squeeze())[::10], fmt='o', 
            label='Predictive Mean',
            ecolor='k', capthick=2, markersize=5)
plt.scatter(np.arange(n_samples)[::10], y_test[::10], label='True Value', color='r', s=10, zorder=5)
plt.xlabel('Sample Index')
plt.ylabel('Prediction')
plt.legend()

plt.figure(figsize=(6, 5))
plt.scatter(mean.squeeze(), y_test, label='True Value', s=10, zorder=5)
plt.xlabel('Predicted Value')
plt.ylabel('True Value')

plt.figure(figsize=(6, 5))
plt.scatter(test_variances, y_test, label='True Value', s=10, zorder=5)
plt.xlabel('Variance')
plt.ylabel('True Value')

rse = ((mean.squeeze() - y_test)**2)**0.5
plt.figure(figsize=(6, 5))
plt.scatter(rse, test_variances, label='True Value', s=5, zorder=5)
plt.xlabel('Root Squared Error')
plt.ylabel('Variance')


std_away = np.abs(mean.squeeze() - y_test)/np.hstack(np.sqrt(test_variances))
print( std_away.shape)

plt.figure(figsize=(6, 5))
plt.hist(std_away, bins=100)
plt.xlabel('Standard Deviations Away from True Value')
plt.ylabel('Count')
plt.title('Histogram of Standard Deviations Away from True Value')
plt.show()  # Show all plots
