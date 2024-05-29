import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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

# Preprocessing
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Add the scaled features back to the dataframe
data[numerical_columns] = X_scaled

# Separate the data by class
high_class = data[data['stream_class'] == label_encoder.transform(['High'])[0]]
average_class = data[data['stream_class'] == label_encoder.transform(['Average'])[0]]
low_class = data[data['stream_class'] == label_encoder.transform(['Low'])[0]]

# Plot the second feature (released_year)
plt.figure(figsize=(10, 6))
plt.plot(high_class.index, high_class['released_year'], label='High', color='r')
plt.plot(average_class.index, average_class['released_year'], label='Average', color='g')
plt.plot(low_class.index, low_class['released_year'], label='Low', color='b')
plt.xlabel('Song Index')
plt.ylabel('Normalized Released Year')
plt.title('Normalized Released Year by Stream Class')
plt.legend()
plt.show()
