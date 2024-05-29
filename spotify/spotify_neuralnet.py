import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path = 'spotify_cleaned_labeled.csv'
data = pd.read_csv(file_path)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
data['stream_class'] = label_encoder.fit_transform(data['stream_class'])

# Select numerical columns and target
numerical_columns = ['artist_count','in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                     'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 
                     'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 
                     'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
X = data[numerical_columns].values
y = data['stream_class'].values

# Preprocessing
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define neural network model
class SongClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SongClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Increased dropout to prevent overfitting
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1]
output_dim = len(np.unique(y))
model = SongClassifier(input_dim, output_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 150  # Increased number of epochs
batch_size = 32   # Changed batch size
train_losses = []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0.0
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    average_loss = epoch_loss / (X_train.size()[0] // batch_size)
    train_losses.append(average_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}')

# Plotting training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss') 
plt.legend()
plt.show()

# Testing the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Accuracy: {accuracy*100:.2f}%')
