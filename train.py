import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# LOAD DATASET
# =========================================================

print("🚀 Loading Dataset...")

df = pd.read_csv("data/songs.csv")

print("✅ Dataset Loaded Successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =========================================================
# CLEAN DATA
# =========================================================

print("\n🧹 Cleaning Dataset...")

df.columns = df.columns.str.lower()

required_columns = [
    "name",
    "album",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "popularity"
]

for col in required_columns:
    if col not in df.columns:
        raise Exception(f"❌ Missing Column: {col}")

df = df[required_columns]

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

print("✅ Missing values removed")

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\n⚙️ Feature Engineering...")

features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "popularity"
]

X = df[features]

# =========================================================
# SCALING
# =========================================================

print("\n📏 Scaling Features...")

scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

print("✅ Data Scaled")

# =========================================================
# KMEANS CLUSTERING
# =========================================================

print("\n🎯 Applying KMeans Clustering...")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df["cluster"] = clusters

print("✅ Clustering Completed")

# =========================================================
# PCA DIMENSIONALITY REDUCTION
# =========================================================

print("\n📉 Applying PCA...")

pca = PCA(n_components=2)

pca_result = pca.fit_transform(scaled_data)

df["pca_x"] = pca_result[:, 0]
df["pca_y"] = pca_result[:, 1]

print("✅ PCA Completed")

# =========================================================
# t-SNE VISUALIZATION
# =========================================================

print("\n🎨 Applying t-SNE...")

sample_size = min(500, len(df))

sample_data = scaled_data[:sample_size]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',
    random_state=42
)

tsne_result = tsne.fit_transform(sample_data)

tsne_df = pd.DataFrame({
    "tsne_x": tsne_result[:, 0],
    "tsne_y": tsne_result[:, 1]
})

tsne_df.to_csv("data/tsne_data.csv", index=False)

print("✅ t-SNE Completed")

# =========================================================
# COSINE SIMILARITY
# =========================================================

print("\n🎵 Calculating Cosine Similarity...")

similarity = cosine_similarity(scaled_data)

print("✅ Similarity Matrix Created")

# =========================================================
# MOOD LABELING
# =========================================================

print("\n😊 Generating Mood Labels...")

moods = []

for _, row in df.iterrows():

    if row["energy"] > 0.7 and row["danceability"] > 0.7:
        moods.append("Party")

    elif row["valence"] > 0.6:
        moods.append("Happy")

    elif row["acousticness"] > 0.7:
        moods.append("Chill")

    elif row["energy"] < 0.4:
        moods.append("Sad")

    else:
        moods.append("Mixed")

df["mood"] = moods

print("✅ Mood Labels Added")

# =========================================================
# SAVE FILES
# =========================================================

print("\n💾 Saving Models...")

df.to_csv("data/processed_songs.csv", index=False)

joblib.dump(similarity, "model/similarity.pkl")
joblib.dump(kmeans, "model/kmeans.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca, "model/pca.pkl")

print("✅ All Files Saved Successfully")

# =========================================================
# SUMMARY
# =========================================================

print("\n===================================")
print("🎧 AI MUSIC RECOMMENDATION SYSTEM")
print("===================================")

print("Total Songs:", len(df))
print("Total Features:", len(features))
print("Clusters:", len(df['cluster'].unique()))
print("Moods:", df['mood'].unique())

print("\n🔥 Technologies Used:")
print("✔ KMeans Clustering")
print("✔ PCA")
print("✔ t-SNE")
print("✔ Cosine Similarity")
print("✔ StandardScaler")

print("\n✅ TRAINING COMPLETED SUCCESSFULLY")