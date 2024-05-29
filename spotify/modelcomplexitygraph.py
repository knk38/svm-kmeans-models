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

# Helper function to plot validation curves
def plot_validation_curves(param_range, train_scores, test_scores, param_name, title):
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, label='Training Accuracy')
    plt.plot(param_range, test_scores, label='Testing Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()

# Neural Network Model
class SongClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SongClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to evaluate a neural network with varying hyperparameters
def evaluate_nn_hyperparameters(X_train, y_train, X_test, y_test, hidden_dims, learning_rates):
    for lr in learning_rates:
        train_accuracies = []
        test_accuracies = []
        for hidden_dim in hidden_dims:
            print(f"Evaluating NN with hidden_dim={hidden_dim} and lr={lr}")
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train))
            
            subset_X_train = torch.tensor(X_train, dtype=torch.float32)
            subset_y_train = torch.tensor(y_train, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            
            model = SongClassifier(input_dim, hidden_dim, output_dim)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
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
        
        plot_validation_curves(hidden_dims, train_accuracies, test_accuracies, 'Hidden Layer Dimension', f'NN Learning Rate: {lr}')

# SVM Model
def evaluate_svm_hyperparameters(X_train, y_train, X_test, y_test, C_values, kernels):
    for kernel in kernels:
        train_accuracies = []
        test_accuracies = []
        for C in C_values:
            print(f"Evaluating SVM with C={C} and kernel={kernel}")
            svm_model = SVC(C=C, kernel=kernel)
            svm_model.fit(X_train, y_train)
            
            train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
            test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        
        plot_validation_curves(C_values, train_accuracies, test_accuracies, 'C value', f'SVM Kernel: {kernel}')

# K-means Model
def evaluate_kmeans_hyperparameters(X_train, y_train, X_test, y_test, cluster_numbers, inits):
    for init in inits:
        train_accuracies = []
        test_accuracies = []
        for clusters in cluster_numbers:
            print(f"Evaluating K-means with clusters={clusters} and init={init}")
            kmeans = KMeans(n_clusters=clusters, init=init, random_state=42)
            kmeans.fit(X_train)
            
            train_clusters = kmeans.predict(X_train)
            cluster_labels = np.zeros_like(train_clusters)
            for i in range(clusters):
                mask = (train_clusters == i)
                if np.sum(mask) > 0:
                    cluster_labels[mask] = mode(y_train[mask])[0]
            
            test_clusters = kmeans.predict(X_test)
            y_pred = np.zeros_like(test_clusters)
            for i in range(clusters):
                y_pred[test_clusters == i] = cluster_labels[train_clusters == i][0]
            
            train_accuracy = accuracy_score(y_train, cluster_labels)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        
        plot_validation_curves(cluster_numbers, train_accuracies, test_accuracies, 'Number of Clusters', f'K-means Init: {init}')

# Define hyperparameter ranges
hidden_dims = [32, 64, 128, 256]
learning_rates = [0.001, 0.01, 0.1]
C_values = [0.1, 1, 10, 100]
kernels = ['linear', 'rbf']
cluster_numbers = [5, 10, 15, 20]
inits = ['k-means++', 'random']

# Evaluate each model with hyperparameter ranges
evaluate_nn_hyperparameters(X_train, y_train, X_test, y_test, hidden_dims, learning_rates)
evaluate_svm_hyperparameters(X_train, y_train, X_test, y_test, C_values, kernels)
evaluate_kmeans_hyperparameters(X_train, y_train, X_test, y_test, cluster_numbers, inits)
