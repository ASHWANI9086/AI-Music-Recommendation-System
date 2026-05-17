import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Music Recommendation",
    page_icon="🎧",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.big-title {
    font-size: 55px;
    font-weight: bold;
    color: #1DB954;
}

.song-card {
    background-color: #181818;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    transition: 0.3s;
}

.song-card:hover {
    transform: scale(1.02);
    background-color: #242424;
}

.song-name {
    font-size: 24px;
    font-weight: bold;
    color: white;
}

.song-info {
    color: #B3B3B3;
    font-size: 16px;
}

.metric-box {
    background: #181818;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_songs.csv")

@st.cache_resource
def load_similarity():
    return joblib.load("model/similarity.pkl")

df = load_data()
similarity = load_similarity()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='big-title'>🎧 AI Music Recommendation System</div>",
    unsafe_allow_html=True
)

st.markdown("""
### 🚀 Advanced ML Based Recommendation Engine

This project uses:

- ✅ KMeans Clustering
- ✅ Cosine Similarity
- ✅ PCA
- ✅ t-SNE Visualization
- ✅ Mood Classification
- ✅ Streamlit Premium UI
""")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎵 Filters")

search_song = st.sidebar.text_input("🔍 Search Song")

mood_filter = st.sidebar.selectbox(
    "🎭 Mood Filter",
    ["All", "Happy", "Sad", "Party", "Chill", "Mixed"]
)

cluster_filter = st.sidebar.selectbox(
    "🎯 Cluster",
    ["All"] + list(df["cluster"].unique())
)

# =========================================================
# FILTERING
# =========================================================

filtered_df = df.copy()

if search_song:
    filtered_df = filtered_df[
        filtered_df["name"].str.contains(search_song, case=False)
    ]

if mood_filter != "All":
    filtered_df = filtered_df[
        filtered_df["mood"] == mood_filter
    ]

if cluster_filter != "All":
    filtered_df = filtered_df[
        filtered_df["cluster"] == cluster_filter
    ]

# =========================================================
# SONG SELECTION
# =========================================================

selected_song = st.selectbox(
    "🎵 Select Your Favorite Song",
    filtered_df["name"].values
)

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend(song_name):

    idx = df[df["name"] == song_name].index[0]

    distances = similarity[idx]

    song_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    recommendations = []

    for i in song_list:

        temp_df = df.iloc[i[0]]

        recommendations.append({
            "name": temp_df["name"],
            "album": temp_df["album"],
            "popularity": temp_df["popularity"],
            "mood": temp_df["mood"],
            "cluster": temp_df["cluster"],
            "similarity": round(i[1] * 100, 2)
        })

    return recommendations

# =========================================================
# BUTTON
# =========================================================

if st.button("🚀 Generate Recommendations"):

    recommendations = recommend(selected_song)

    st.success(f"Showing songs similar to: {selected_song}")

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎵 Total Songs", len(df))

    with col2:
        st.metric("🎯 Clusters", len(df["cluster"].unique()))

    with col3:
        st.metric("😊 Moods", len(df["mood"].unique()))

    st.divider()

    # =====================================================
    # SONG CARDS
    # =====================================================

    for song in recommendations:

        st.markdown(f"""
        <div class="song-card">

        <div class="song-name">
        🎵 {song['name']}
        </div>

        <br>

        <div class="song-info">
        💿 Album: {song['album']}
        </div>

        <div class="song-info">
        😊 Mood: {song['mood']}
        </div>

        <div class="song-info">
        🎯 Cluster: {song['cluster']}
        </div>

        <div class="song-info">
        ⭐ Popularity: {song['popularity']}
        </div>

        <div class="song-info">
        🔥 Similarity Score: {song['similarity']}%
        </div>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# VISUALIZATION SECTION
# =========================================================

st.divider()

st.subheader("📊 PCA Visualization")

fig = px.scatter(
    df,
    x="pca_x",
    y="pca_y",
    color=df["cluster"].astype(str),
    hover_data=["name", "album", "mood"],
    title="Song Clusters Visualization using PCA"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.divider()

st.subheader("📁 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
## 👨‍💻 Project Team

### Team Leader
- Ashwani Kumar  
- Registration No: 12319687  
- Roll No: 35  

### Team Member
- Karan Yadav  
- Registration No: 12319301  
- Roll No: 13  

---

### 🧠 Machine Learning Algorithms Used

✔ KMeans Clustering  
✔ Cosine Similarity  
✔ PCA  
✔ t-SNE  
✔ StandardScaler  

---

### 🎯 Project Objective

To build an intelligent AI-powered music recommendation system that recommends songs based on similarity, mood, clustering, and audio features using unsupervised machine learning algorithms.
""")