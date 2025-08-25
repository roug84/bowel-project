import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

DATA_DIR = "bowel_dataset/data"

# List all files in the data folder
all_files = sorted(os.listdir(DATA_DIR))
print(f"Total files in {DATA_DIR}: {len(all_files)}")
print(f"First 20 files:\n{all_files[:20]}")

# Separate CSV and WAV files
csv_files = sorted([f for f in all_files if f.endswith(".csv")])
wav_files = sorted([f for f in all_files if f.endswith(".wav")])

print(f"\nTotal CSV files: {len(csv_files)}")
print(f"Total WAV files: {len(wav_files)}")


# Function to print info about CSV file content
def explore_csv(file_path):
    print(f"\nExploring CSV file: {file_path}")
    df = pd.read_csv(file_path)
    print(df.head())
    print(f"Columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")


def plot_audio_with_annotations(wav_path, csv_path):
    y, sr = librosa.load(wav_path, sr=None)

    annotations = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    fig, (ax_wave, ax_spec) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Waveform plot
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_ylabel("Amplitude")

    # Mel spectrogram plot
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='mel', ax=ax_spec)
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Mel frequency")
    ax_spec.set_title("Mel Spectrogram")

    bowel_intervals = []
    if annotations is not None and not annotations.empty:
        valid_anns = annotations.dropna(subset=["start", "end"])
        for _, row in valid_anns.iterrows():
            start, end = row["start"], row["end"]
            bowel_intervals.append((start, end))
            ax_wave.axvspan(start, end, color="orange", alpha=0.4)
            ax_spec.axvline(x=start, color="red", linestyle="--", linewidth=1)
            ax_spec.axvline(x=end, color="red", linestyle="--", linewidth=1)

    n_intervals = len(bowel_intervals)
    ax_wave.set_title(f"Waveform with {n_intervals} bowel sound intervals annotated")

    plt.tight_layout()
    plt.show()


# Explore one CSV file as before
print("\n--- Sample CSV file exploration ---")
explore_csv(os.path.join(DATA_DIR, csv_files[0]))

# Ask user how many examples to visualize
num_examples = input(f"\nEnter the number of examples to plot (max {len(wav_files)}): ")
try:
    num_examples = int(num_examples)
    if num_examples < 1 or num_examples > len(wav_files):
        raise ValueError()
except ValueError:
    print("Invalid input, using 3 examples by default.")
    num_examples = 3

# Plot requested number of examples
for i in range(num_examples):
    wav_file = wav_files[i]
    csv_file = wav_file.replace(".wav", ".csv")
    print(f"\nPlotting example {i+1}: {wav_file} with annotations from {csv_file}")
    plot_audio_with_annotations(os.path.join(DATA_DIR, wav_file), os.path.join(DATA_DIR, csv_file))


# Find low amplitude files (possible 'no sound') and plot one with annotations too
def find_low_amplitude_audio(threshold=0.01):
    low_amp_files = []
    for f in wav_files:
        y, sr = librosa.load(os.path.join(DATA_DIR, f), sr=None)
        mean_amp = np.mean(np.abs(y))
        if mean_amp < threshold:
            low_amp_files.append((f, mean_amp))
    return sorted(low_amp_files, key=lambda x: x[1])


low_amp_candidates = find_low_amplitude_audio()
print(f"\nFound {len(low_amp_candidates)} low amplitude audio files (possible no sound):")
for f, amp in low_amp_candidates[:5]:
    print(f"{f} - mean amplitude: {amp}")

if low_amp_candidates:
    print("\nPlotting one low amplitude audio file with annotations:")
    wav_file = low_amp_candidates[0][0]
    csv_file = wav_file.replace(".wav", ".csv")
    plot_audio_with_annotations(os.path.join(DATA_DIR, wav_file), os.path.join(DATA_DIR, csv_file))
else:
    print("\nNo low amplitude audio files found with threshold =", 0.01)
