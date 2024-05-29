import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

# Define and train SVM model
svm_model = SVC(kernel='rbf')  
svm_model.fit(X_train, y_train)

# Testing the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
