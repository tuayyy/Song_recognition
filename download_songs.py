import os
import yt_dlp
import requests

# ✅ Base Folder for Sorting Songs by Genre
GENRE_FOLDER = "data/genres"
SONG_LIST_FOLDER = "songs"  # Folder containing genre-based song lists
os.makedirs(GENRE_FOLDER, exist_ok=True)

# ✅ YouTube API Key (Replace with your actual API key)
YOUTUBE_API_KEY = ""
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# ✅ Function to Read Songs from Genre-Specific Files
def load_songs_by_genre():
    """Load songs from different genre-specific text files."""
    genre_songs = {}
    
    if not os.path.exists(SONG_LIST_FOLDER):
        print(f"⚠️ Folder '{SONG_LIST_FOLDER}' not found! Create it and add song lists.")
        return {}

    for filename in os.listdir(SONG_LIST_FOLDER):
        if filename.endswith(".txt"):  # Ensure it's a song list file
            genre = os.path.splitext(filename)[0]  # Extract genre from filename
            genre_songs[genre] = []

            with open(os.path.join(SONG_LIST_FOLDER, filename), "r", encoding="utf-8") as file:
                genre_songs[genre] = [line.strip() for line in file if line.strip()]
    
    return genre_songs

# ✅ Check if a song already exists in any genre folder
def song_exists(song_name):
    """Check if a song already exists in any genre folder."""
    song_filename = song_name.replace(" ", "_") + ".wav"
    
    for genre in os.listdir(GENRE_FOLDER):
        genre_folder = os.path.join(GENRE_FOLDER, genre)
        if os.path.isdir(genre_folder) and song_filename in os.listdir(genre_folder):
            print(f"⏭️ Skipping '{song_name}' (Already exists in '{genre}').")
            return True  # Song already exists in some genre
    
    return False  # Song does not exist

# ✅ Search for a song on YouTube
def search_youtube(song_name):
    """Search for a song on YouTube and return the video ID of the best match."""
    params = {
        "part": "snippet",
        "q": song_name + " official audio",
        "key": YOUTUBE_API_KEY,
        "maxResults": 1
    }
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    result = response.json()
    
    if "items" in result and len(result["items"]) > 0:
        return f"https://www.youtube.com/watch?v={result['items'][0]['id']['videoId']}"
    else:
        print(f"⚠️ No YouTube results found for: {song_name}")
        return None

# ✅ Download YouTube audio and convert to WAV format
def download_audio(video_url, output_path):
    """Download the YouTube audio and convert to WAV format."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': output_path + '.%(ext)s',
        'cookiefile': 'cookies.txt'  # ✅ Uses YouTube cookies for age-restricted videos
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"✅ Download complete: {output_path}.wav")
    except yt_dlp.utils.DownloadError as e:
        print(f"⚠️ Download error: {e}")

# ✅ Download song and categorize by genre (Avoids Duplicates)
def download_song(song_name, genre):
    """Search and download a song from YouTube and save it in the correct genre folder."""
    if song_exists(song_name):  # Check if the song already exists anywhere
        return  # Skip if found

    genre_folder = os.path.join(GENRE_FOLDER, genre)
    os.makedirs(genre_folder, exist_ok=True)

    output_path = os.path.join(genre_folder, song_name.replace(" ", "_"))

    print(f"🎵 Searching for: {song_name}")
    video_url = search_youtube(song_name)

    if video_url:
        print(f"📥 Downloading: {song_name} from {video_url} into '{genre}' category")
        download_audio(video_url, output_path)
    else:
        print(f"⚠️ Could not find {song_name} on YouTube.")

# ✅ Count number of songs in each genre
def count_songs_per_genre():
    """Count and display the number of songs in each genre."""
    print("\n📊 **Song Count per Genre:**")
    for genre in os.listdir(GENRE_FOLDER):
        genre_path = os.path.join(GENRE_FOLDER, genre)
        if os.path.isdir(genre_path):
            num_songs = len([file for file in os.listdir(genre_path) if file.endswith(".wav")])
            print(f"🎶 {genre}: {num_songs} songs")

# ✅ Read songs from multiple genre-specific files and download them
if __name__ == "__main__":
    genre_songs = load_songs_by_genre()
    
    if not genre_songs:
        print("⚠️ No songs found in genre-specific files. Please add some song lists.")
    else:
        for genre, songs in genre_songs.items():
            for song in songs:
                download_song(song, genre)
    
    # ✅ Show the total number of songs per genre
    count_songs_per_genre()
