import streamlit as st
import pickle
import requests
import base64

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Music Recommender", layout="wide")

# -----------------------------
# 🔐 SPOTIFY API KEYS (PASTE YOURS)
# -----------------------------
CLIENT_ID = "fee410581d5643049784cd6c164b8766"
CLIENT_SECRET = "5b8fe2a9f72f438485fd6a909d0a621b"

# Initialize token globally
token = None

# -----------------------------
# 🔑 GET SPOTIFY TOKEN
# -----------------------------
def get_spotify_token():
    global token
    try:
        auth_url = "https://accounts.spotify.com/api/token"
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(auth_url, data=auth_data)
        if response.status_code == 200:
            token = response.json()['access_token']
            return token
        else:
            print("❌ Failed to get Spotify token")
            return None
    except Exception as e:
        print(f"❌ Token Error: {e}")
        return None

# Get token on startup
token = get_spotify_token()

# -----------------------------
# 🔍 SEARCH SONG ON SPOTIFY
def get_song_details(song_name):
    try:
        # ✅ Use global token (faster)
        global token

        if not token:
            return None, "No Token", None

        # ✅ Better query (remove strict track:)
        query = song_name.replace(" ", "%20")
        url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"

        headers = {
            "Authorization": f"Bearer {token}"
        }

        response = requests.get(url, headers=headers)

        # ✅ Handle API failure
        if response.status_code != 200:
            print("❌ Spotify API Error:", response.text)
            return None, "API Error", None

        data = response.json()

        # ✅ No results found
        if not data.get('tracks') or len(data['tracks']['items']) == 0:
            return None, "Not Found", None

        track = data['tracks']['items'][0]

        # ✅ Safe extraction
        image = track['album']['images'][0]['url'] if track['album']['images'] else None
        artist = track['artists'][0]['name'] if track['artists'] else "Unknown"
        preview = track['preview_url']  # can be None

        return image, artist, preview

    except Exception as e:
        print("❌ Search Error:", e)
        return None, "Error", None
# 📦 LOAD MODEL FILES
# 📦 LOAD MODEL FILES
# -----------------------------
songs = pickle.load(open("songs.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# -----------------------------
# 🎯 RECOMMEND FUNCTION
# -----------------------------
def recommend(song):
    index = songs[songs['name'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended = []
    for i in distances[1:10]:
        recommended.append(songs.iloc[i[0]]['name'])

    return recommended

# -----------------------------
# 🎧 HEADER UI
# -----------------------------
st.markdown("""
    <h1 style='text-align: center; color: #1DB954;'>🎧 AI Music Recommendation System</h1>
    <p style='text-align: center;'>ML + Spotify Powered 🎵</p>
""", unsafe_allow_html=True)

# -----------------------------
# 🎵 SONG SELECT
# -----------------------------
selected_song = st.selectbox("🎵 Choose a song", songs['name'].values)

# -----------------------------
# 🚀 BUTTON
# -----------------------------
if st.button("🚀 Recommend"):

    recommendations = recommend(selected_song)

    st.markdown("## 🎶 Recommended Songs")

    # Scrollable container
    scroll_container = st.container()

    for song in recommendations:
        image, artist, preview = get_song_details(song)

        with scroll_container:
            st.markdown("""
                <div style="
                    background-color:#181818;
                    padding:15px;
                    border-radius:12px;
                    margin-bottom:15px;
                    display:flex;
                    align-items:center;
                ">
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1,3])

            with col1:
                if image:
                    st.image(image, width=100)
                else:
                    st.image("https://via.placeholder.com/100", width=100)

            with col2:
                st.markdown(f"### 🎵 {song}")
                st.markdown(f"👤 {artist}")

                if preview:
                    st.audio(preview)
                else:
                    st.write("❌ No Preview Available")

            st.markdown("</div>", unsafe_allow_html=True)