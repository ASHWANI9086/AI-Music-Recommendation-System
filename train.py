import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load dataset (FIXED PATH)
df = pd.read_csv("data/songs.csv")

# Select only useful numerical features (YOUR DATASET BASED)
features = ['acousticness', 'danceability', 'energy',
            'instrumentalness', 'liveness', 'loudness',
            'speechiness', 'tempo', 'valence']

df = df[features + ['name']]

# Fill missing values
df.fillna(0, inplace=True)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Similarity matrix
similarity = cosine_similarity(scaled_data)

# Save files
pickle.dump(df, open('songs.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Model + similarity saved successfully!")