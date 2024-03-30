import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.nn.utils import weight_norm


class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Baseline model
    
# Base LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output size is 1 for binary classification

    def forward(self, X):
        # _, (h, _) = self.lstm(X)
        # out = self.fc(h[-1])  # Use the last hidden state
        # return out
        # Get all hidden states
        output, _ = self.lstm(X)
        # Calculate the average across the sequence dimension
        avg_hidden_state = output.mean(dim=1)
        # Pass the average hidden state through the fully connected layer
        out = self.fc(avg_hidden_state)
        return out

# Base Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_dim, output_size):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)  # Pooling
        x = self.fc(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Base Temporal Convolutional Network model
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert (batch_size, seq_length, num_inputs) to (batch_size, num_inputs, seq_length)
        y = self.network(x)
        y = y.transpose(1, 2)  # Convert back to (batch_size, seq_length, num_channels)
        y = y[:, -1, :]  # Take the last time step
        return self.fc(y)


# Improved model

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert (batch_size, seq_length, num_inputs) to (batch_size, num_inputs, seq_length)
        y = self.network(x)
        y = y.transpose(1, 2)  # Convert back to (batch_size, seq_length, num_channels)
        return y

class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_dim, output_size, tcn_channels, kernel_size, dropout, post_hidden_dims):
        super(TransformerModel, self).__init__()
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size, dropout)
        
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=tcn_channels[-1],  # Output size of TCN
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        # Post-Layers
        post_layers = []
        final_dim = tcn_channels[-1]  # Initialize with the output size of the TCN
        if post_hidden_dims is not None:
            if isinstance(post_hidden_dims, int):
                post_hidden_dims = [post_hidden_dims]  # Convert single integer to list
            for dim in post_hidden_dims:
                post_layers.append(nn.Linear(final_dim, dim))
                post_layers.append(nn.ReLU())
                post_layers.append(nn.Dropout(0.1))
                final_dim = dim  # Update for next layer input
        self.post_fc = nn.Sequential(*post_layers)
        
        # Final Output Layer
        self.fc = nn.Linear(tcn_channels[-1], output_size)

    def forward(self, x):
        x = self.tcn(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)  # Pooling
        x = self.fc(x)
        return x


# loading the data
dp = pd.read_csv('/content/drive/MyDrive/csc413/dataset_32_prices128.csv')
dp.head()

# Load headlines
headlines = dp['headlines'].to_list()
headline_features = []
for headline in headlines:
    headline = headline.strip('[]').split(' ')
    headline_features.append([float(item.strip('\n')) for item in headline if item != ''])

# Load prices
prices = dp['prices'].to_list()
price_features = []
for price in prices:
    price = price.strip('[]').split(' ')
    price_features.append([float(item.strip('\n').replace(',', '')) for item in price if item != ''])

# Concatenate headlines and prices
concatenated_features = [headline + price for headline, price in zip(headline_features, price_features)]

# Convert to NumPy array
features = np.array(concatenated_features)
print(features.shape)

# Load targets
targets = dp['y'].to_numpy()


# Split the dataset into training and temporary sets (80% training, 20% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.2, random_state=42)

# Split the temporary set into validation and test sets (50% validation, 50% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Initialize the model
input_size = 32 * 25 + 128
num_layers = 2
num_heads = 32
hidden_dim = 32
output_size = 1
num_classes = 16
num_channels = [32, 64, 128]  # Example channel sizes
kernel_size = 3
dropout = 0.5
post_hidden_dims = 32

# model = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
# model = TransformerModel(input_size, num_layers, num_heads, hidden_dim, output_size)
# model = LSTM(input_size, hidden_dim)
model = ImprovedTransformerModel(input_size, num_layers, num_heads, hidden_dim, output_size, num_channels, kernel_size, dropout, post_hidden_dims)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-03)

# Training loop
counter = 0
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_features.unsqueeze(1))
        loss = criterion(outputs.squeeze(1), batch_targets)
        loss.backward()
        optimizer.step()

        counter += 1
        if counter % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {counter}, Loss: {loss.item()}')

    # Validation
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        val_outputs = model(X_val.unsqueeze(1)).squeeze()
        val_loss = criterion(val_outputs, y_val)
        val_preds = torch.round(torch.sigmoid(val_outputs))
        val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy}')

model.to('cpu')
# Test evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(1)).squeeze()
    test_preds = torch.round(torch.sigmoid(test_outputs))
    test_accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
    test_precision = precision_score(y_test.numpy(), test_preds.numpy())
    test_recall = recall_score(y_test.numpy(), test_preds.numpy())
    test_f1 = f1_score(y_test.numpy(), test_preds.numpy())

print(f'Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')

