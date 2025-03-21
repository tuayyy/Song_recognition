import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.ndimage

# ✅ Load genre labels
with open("model/labels.json", "r") as f:
    GENRE_LABELS = json.load(f)

# ✅ Load trained model
model = tf.keras.models.load_model("model/merged_audio_model.keras")

FINGERPRINT_FOLDER = "filtered_fingerprint_merged"
PEAK_NEIGHBORHOOD_SIZE = 40
AMPLITUDE_THRESHOLD = -30  # dB threshold


def record_audio(filename="test.wav", duration=30, sr=22050):
    """Record audio and save it as a .wav file."""
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    wav.write(filename, sr, (audio * 32767).astype("int16"))
    print("✅ Recording finished.")


def generate_fingerprint_segments(audio_path, save_folder, segment_duration=30, overlap=15):
    """Generate fingerprints for overlapping 30-second and full-song segments."""
    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    segment_start = 0
    segment_index = 0
    fingerprint_paths = []

    while segment_start + segment_duration <= total_duration:
        segment_filename = f"test_seg{segment_index}_fingerprint.png"
        segment_path = os.path.join(save_folder, segment_filename)

        start_sample = int(segment_start * sr)
        end_sample = int((segment_start + segment_duration) * sr)
        segment = y[start_sample:end_sample]

        save_fingerprint(segment, sr, segment_path)
        fingerprint_paths.append(segment_path)
        segment_start += overlap
        segment_index += 1
    
    # ✅ Full song fingerprint
    full_song_path = os.path.join(save_folder, "full_song_fingerprint.png")
    save_fingerprint(y, sr, full_song_path)
    fingerprint_paths.append(full_song_path)

    return fingerprint_paths


def save_fingerprint(y, sr, save_path):
    """Generate and save a fingerprint spectrogram."""
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    peaks = scipy.ndimage.maximum_filter(S, size=PEAK_NEIGHBORHOOD_SIZE) == S
    peak_freqs, peak_times = np.where(peaks & (S > AMPLITUDE_THRESHOLD))
    
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.scatter(peak_times, peak_freqs, marker="o", color="red", s=8)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()
    print(f"✅ Fingerprint saved: {save_path}")


def preprocess_fingerprint(fingerprint_path):
    """Load and preprocess a fingerprint image for model input."""
    img = image.load_img(fingerprint_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def recognize_song():
    """Record, process, and classify song using AI-based fingerprint matching."""
    choice = input("Choose input method: [1] Record with mic [2] Use existing file: ")
    
    if choice == "1":
        record_audio()
        audio_path = "test.wav"
    elif choice == "2":
        audio_files = [f for f in os.listdir() if f.endswith(".wav")]
        if audio_files:
            audio_path = audio_files[0]
            print(f"✅ Found file: {audio_path}")
        else:
            audio_path = input("Enter the path of the audio file: ")
    else:
        print("❌ Invalid choice. Defaulting to microphone recording.")
        record_audio()
        audio_path = "test.wav"

    # ✅ Clear temp fingerprint folder
    fingerprint_folder = "temp_fingerprint_merged"
    if os.path.exists(fingerprint_folder):
        shutil.rmtree(fingerprint_folder)
    os.makedirs(fingerprint_folder, exist_ok=True)

    # ✅ Generate 30-second segment and full-song fingerprints
    segment_fingerprints = generate_fingerprint_segments(audio_path, fingerprint_folder)

    # ✅ Predict genre for each segment
    predictions = []
    for fingerprint_path in segment_fingerprints:
        img_array = preprocess_fingerprint(fingerprint_path)
        prediction = model.predict(img_array)[0]
        predictions.append(prediction)

    # ✅ Aggregate predictions: Average over all segments and full song
    avg_prediction = np.mean(predictions, axis=0)
    predicted_label_idx = np.argmax(avg_prediction)
    predicted_genre = GENRE_LABELS[predicted_label_idx]
    confidence = avg_prediction[predicted_label_idx] * 100

    print(f"🎵 Predicted Genre: {predicted_genre} ({confidence:.2f}% confidence)")
    print(f"✅ file: {audio_path}")

recognize_song()