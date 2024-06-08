import pandas as pd
import numpy as np

def preprocess_csv(input_file_path, output_file_path):
    # Load the dataset with potential encoding issues
    try:
        df = pd.read_csv(input_file_path, encoding='latin1')
    except UnicodeDecodeError:
        # If 'latin1' encoding also fails, try 'iso-8859-1'
        df = pd.read_csv(input_file_path, encoding='iso-8859-1')
    
    # Clean numerical columns to remove commas and convert to float
    numerical_columns = ['artist_count', 'released_year', 'released_month', 'released_day',
                         'in_spotify_playlists', 'in_spotify_charts', 'streams',
                         'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 
                         'in_deezer_charts', 'in_shazam_charts', 'bpm', 'danceability_%', 
                         'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 
                         'liveness_%', 'speechiness_%']
    
    for col in numerical_columns:
        df[col] = df[col].astype(str).str.replace(',', '').replace('', '0').astype(float)
    
    # Ensure all string columns are encoded in Unicode
    string_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
    for col in string_columns:
        df[col] = df[col].astype(str)
    
    # Check for and handle NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Calculate mean of streams
    mean_streams = df['streams'].mean()
    
    # Create class labels based on the mean
    df['stream_class'] = pd.cut(df['streams'], bins=[-np.inf, mean_streams/2, mean_streams*1.5, np.inf], labels=['Low', 'Average', 'High'])
    
    # Save the cleaned and labeled DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, encoding='utf-8')

# Define file paths
input_file_path = 'spotify.csv'  # Input CSV file path
output_file_path = 'spotify_cleaned_labeled.csv'  # Output CSV file path

# Preprocess the CSV file
preprocess_csv(input_file_path, output_file_path)


