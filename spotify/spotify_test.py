import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('music_streams_prediction_model.h5')

# Load new data
new_data = pd.read_csv('path_to_new_data.csv')

# Preprocess new data
categorical_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
for col in categorical_columns:
    le = LabelEncoder()
    # You need to fit LabelEncoder on the same categories as used during training
    le.fit(df[col])  # 'df' is the original DataFrame used during training
    new_data[col] = le.transform(new_data[col])

numerical_columns = ['artist_count', 'released_year', 'released_month', 'released_day',
                     'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                     'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 
                     'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 
                     'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

scaler = StandardScaler()
# You need to fit StandardScaler on the same distribution as used during training
scaler.fit(df[numerical_columns])  # 'df' is the original DataFrame used during training
new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

# Drop any columns that are not part of the input features
new_data = new_data.drop(columns=['streams'], errors='ignore')

# Ensure the new data has the same columns as the training data
# Assuming the columns are in the same order
new_data = new_data[features.columns]

# Make predictions
predictions = model.predict(new_data)

# Output predictions
new_data['predicted_streams'] = predictions
print(new_data[['track_name', 'artist(s)_name', 'predicted_streams']])
