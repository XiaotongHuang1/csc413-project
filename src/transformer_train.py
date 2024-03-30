import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Define the dataset
class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Define the model
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


# Load your dataset
features = np.load("features.npy")
targets = np.load("targets.npy")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

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
input_size = 25 * 256 + 30
num_layers = 2
num_heads = 8
hidden_dim = 512
output_size = 1

model = TransformerModel(input_size, num_layers, num_heads, hidden_dim, output_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features.unsqueeze(1))
        loss = criterion(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.unsqueeze(1)).squeeze()
        val_loss = criterion(val_outputs, y_val)
        val_preds = torch.round(torch.sigmoid(val_outputs))
        val_accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy}')

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
