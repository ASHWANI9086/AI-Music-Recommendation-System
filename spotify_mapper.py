import pandas as pd
import requests
import base64
import time
import pickle

# 🔐 YOUR KEYS
CLIENT_ID = "fee410581d5643049784cd6c164b8766"
CLIENT_SECRET = "5b8fe2a9f72f438485fd6a909d0a621b"

# 🔑 GET TOKEN
def get_token():
    auth_string = CLIENT_ID + ":" + CLIENT_SECRET
    auth_base64 = base64.b64encode(auth_string.encode()).decode()

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    result = requests.post(url, headers=headers, data=data)
    return result.json()["access_token"]

# 🔍 SEARCH SONG
def search_song(token, song_name):
    try:
        url = f"https://api.spotify.com/v1/search?q=track:{song_name}&type=track&limit=1"
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return None, None, None

        data = response.json()

        if len(data['tracks']['items']) == 0:
            return None, None, None

        track = data['tracks']['items'][0]

        return (
            track['album']['images'][0]['url'] if track['album']['images'] else None,
            track['artists'][0]['name'],
            track['preview_url']
        )

    except Exception as e:
        print(f"❌ Skipped: {song_name}")
        return None, None, None

# 📦 LOAD DATA
df = pd.read_csv("data/songs.csv")

token = get_token()

images = []
artists = []
previews = []

print("🚀 Mapping songs with Spotify...")

for song in df['name']:
    img, artist, preview = search_song(token, song)

    images.append(img)
    artists.append(artist if artist else "Unknown")
    previews.append(preview)

    print(f"✔ {song}")

    time.sleep(0.2)  # avoid rate limit

# 💾 SAVE ENRICHED DATA
df['image'] = images
df['artist'] = artists
df['preview'] = previews

pickle.dump(df, open("songs_spotify.pkl", "wb"))

print("✅ DONE! Spotify data mapped & saved")