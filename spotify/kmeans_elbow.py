import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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

# Determine the optimal number of clusters using the Elbow Method
wcss = []
max_clusters = 200
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
optimal_clusters = 721  # Replace with the number of clusters determined from the elbow point

# Train K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_train)

# Predicting clusters for the training set
train_clusters = kmeans.predict(X_train)

# Map each cluster to the most frequent class label in that cluster
cluster_labels = np.zeros_like(train_clusters)
for i in range(optimal_clusters):
    mask = (train_clusters == i)
    if np.sum(mask) > 0:
        cluster_labels[mask] = mode(y_train[mask])[0]

# Predict clusters for the test set
test_clusters = kmeans.predict(X_test)

# Map the test clusters to the most frequent class labels in the training set clusters
y_pred = np.zeros_like(test_clusters)
for i in range(optimal_clusters):
    y_pred[test_clusters == i] = cluster_labels[train_clusters == i][0]

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
