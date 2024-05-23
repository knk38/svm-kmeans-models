import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'spotify.csv'
df = pd.read_csv(file_path)

# Select features and target
features = df.drop(columns=['streams'])
target = df['streams']

# Encode categorical features
categorical_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
for col in categorical_columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# Normalize numerical features
numerical_columns = ['artist_count', 'released_year', 'released_month', 'released_day',
                     'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                     'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 
                     'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 
                     'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

scaler = StandardScaler()
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {mae}')

# Save the model
model.save('music_streams_prediction_model.h5')
