import librosa
import librosa.display
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import multiprocessing

# Paths
GENRE_FOLDER = "data/genres"
FINGERPRINT_30S_FOLDER = "data/fingerprint_30s"
FINGERPRINT_FOLDER = "data/fingerprints"


# Ensure directories exist
os.makedirs(FINGERPRINT_30S_FOLDER, exist_ok=True)

# Parameters
SEGMENT_DURATION = 30  # 30s segments
OVERLAP = 15  # Overlap by 15 seconds
PEAK_NEIGHBORHOOD_SIZE = 40
AMPLITUDE_THRESHOLD = -30
DPI = 100


def save_fingerprint(S, sr, save_path):
    """Helper function to generate and save a fingerprint from a spectrogram."""
    peaks = scipy.ndimage.maximum_filter(S, size=PEAK_NEIGHBORHOOD_SIZE) == S
    peak_freqs, peak_times = np.where(peaks & (S > AMPLITUDE_THRESHOLD))

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.scatter(peak_times, peak_freqs, marker="o", color="red", s=10)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close()
    print(f"✅ Saved: {save_path}")


def process_song(song_path, save_folder):
    """Process a song to generate overlapping 30s fingerprint segments."""
    y, sr = librosa.load(song_path, sr=22050)
    total_duration = librosa.get_duration(y=y, sr=sr)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    existing_files = set(os.listdir(save_folder))  # List existing fingerprints
    segment_start = 0
    segment_index = 0

    while segment_start + SEGMENT_DURATION <= total_duration:
        segment_filename = f"{os.path.basename(song_path).split('.')[0]}_seg{segment_index}_fingerprint.png"
        segment_path = os.path.join(save_folder, segment_filename)
        
        if segment_filename not in existing_files:  # Check in-memory list instead of repeated disk access
            start_sample = int(segment_start * sr)
            end_sample = int((segment_start + SEGMENT_DURATION) * sr)
            segment_S = S[:, start_sample // 512 : end_sample // 512]  # Use precomputed spectrogram
            save_fingerprint(segment_S, sr, segment_path)
        else:
            print(f"⏭️ Skipping: {segment_filename} (Already Exists)")

        segment_start += OVERLAP
        segment_index += 1

def process_genre(genre):
    """Process all songs in a genre folder."""
    genre_path = os.path.join(GENRE_FOLDER, genre)
    genre_fingerprint_path = os.path.join(FINGERPRINT_30S_FOLDER, genre)
    os.makedirs(genre_fingerprint_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            song_path = os.path.join(genre_path, file)
            process_song(song_path, genre_fingerprint_path)

def generate_fingerprint_plot(file_path, save_path):
    """Generate and save a fingerprint constellation map."""
    if os.path.exists(save_path):  # Skip if file already exists
        print(f"⏭️ Skipping fingerprint (already exists): {save_path}")
        return

    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    peaks = scipy.ndimage.maximum_filter(S, size=PEAK_NEIGHBORHOOD_SIZE) == S
    peak_freqs, peak_times = np.where(peaks & (S > AMPLITUDE_THRESHOLD))
    
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.scatter(peak_times, peak_freqs, marker="o", color="red", s=10)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()
    print(f"✅ Fingerprint saved: {save_path}")

# Process each genre
for genre in os.listdir(GENRE_FOLDER):
    genre_path = os.path.join(GENRE_FOLDER, genre)
    genre_fingerprint_path = os.path.join(FINGERPRINT_FOLDER, genre)

    os.makedirs(genre_fingerprint_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            song_name = os.path.splitext(file)[0]
            audio_path = os.path.join(genre_path, file)
            fingerprint_path = os.path.join(genre_fingerprint_path, song_name + "_fingerprint.png")
            
            generate_fingerprint_plot(audio_path, fingerprint_path)

print("🎉 All fingerprints processed!")


def main():
    available_genres = [genre for genre in os.listdir(GENRE_FOLDER) if os.path.isdir(os.path.join(GENRE_FOLDER, genre))]
    print("Available genres:", ", ".join(available_genres))
    selected_genres = input("Enter genres to process (comma-separated): ").split(',')
    selected_genres = [g.strip() for g in selected_genres if g.strip() in available_genres]
    
    if not selected_genres:
        print("No valid genres selected. Exiting.")
        return
    
    with multiprocessing.Pool(processes=min(len(selected_genres), os.cpu_count())) as pool:
        pool.map(process_genre, selected_genres)  # Process selected genres in parallel

    print("🎉 All overlapping 30s fingerprints processed! (Skipped existing)")



if __name__ == "__main__":
    main()