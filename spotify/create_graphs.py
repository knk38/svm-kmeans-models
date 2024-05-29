import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from scipy.stats import mode

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

# Helper function to plot learning curves
def plot_learning_curves(sizes, train_scores, test_scores, title):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_scores, label='Training Accuracy')
    plt.plot(sizes, test_scores, label='Testing Accuracy')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()

# Neural Network Model
class SongClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SongClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to train and evaluate a neural network
def evaluate_nn(X_train, y_train, X_test, y_test, sizes):
    train_accuracies = []
    test_accuracies = []
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    
    for size in sizes:
        subset_X_train = X_train[:size]
        subset_y_train = y_train[:size]
        
        # Convert to PyTorch tensors
        subset_X_train = torch.tensor(subset_X_train, dtype=torch.float32)
        subset_y_train = torch.tensor(subset_y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        model = SongClassifier(input_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training the model
        num_epochs = 50
        batch_size = 32
        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(subset_X_train.size()[0])
            for i in range(0, subset_X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = subset_X_train[indices], subset_y_train[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_outputs = model(subset_X_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_accuracy = accuracy_score(subset_y_train, train_predicted)
            
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test_tensor, test_predicted)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    plot_learning_curves(sizes, train_accuracies, test_accuracies, 'Neural Network Learning Curve')

# SVM Model
def evaluate_svm(X_train, y_train, X_test, y_test, sizes):
    train_accuracies = []
    test_accuracies = []
    
    for size in sizes:
        subset_X_train = X_train[:size]
        subset_y_train = y_train[:size]
        
        svm_model = SVC(kernel='rbf')
        svm_model.fit(subset_X_train, subset_y_train)
        
        train_accuracy = accuracy_score(subset_y_train, svm_model.predict(subset_X_train))
        test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    plot_learning_curves(sizes, train_accuracies, test_accuracies, 'SVM Learning Curve')

# K-means Model
def evaluate_kmeans(X_train, y_train, X_test, y_test, sizes):
    train_accuracies = []
    test_accuracies = []
    
    for size in sizes:
        subset_X_train = X_train[:size]
        subset_y_train = y_train[:size]
        
        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
        kmeans.fit(subset_X_train)
        
        train_clusters = kmeans.predict(subset_X_train)
        cluster_labels = np.zeros_like(train_clusters)
        for i in range(len(np.unique(y))):
            mask = (train_clusters == i)
            if np.sum(mask) > 0:
                cluster_labels[mask] = mode(subset_y_train[mask])[0]
        
        test_clusters = kmeans.predict(X_test)
        y_pred = np.zeros_like(test_clusters)
        for i in range(len(np.unique(y))):
            y_pred[test_clusters == i] = cluster_labels[train_clusters == i][0]
        
        train_accuracy = accuracy_score(subset_y_train, cluster_labels)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    plot_learning_curves(sizes, train_accuracies, test_accuracies, 'K-means Learning Curve')

# Define different sizes of the training set
training_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

# Evaluate each model
evaluate_nn(X_train, y_train, X_test, y_test, training_sizes)
evaluate_svm(X_train, y_train, X_test, y_test, training_sizes)
evaluate_kmeans(X_train, y_train, X_test, y_test, training_sizes)
