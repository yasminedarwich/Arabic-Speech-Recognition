#count of wav files
import os

# Specify the directory path where your WAV files are located
folder_path = "D:\AllArabicAudioTextLocal\AllArabicWavFilesLocal"

# Initialize a counter to keep track of the number of WAV files
wav_file_count = 0

# Iterate over the files in the directory
for filename in os.listdir(folder_path):
    # Check if the file has a .wav extension (case-insensitive)
    if filename.lower().endswith(".wav"):
        wav_file_count += 1

print(f"Number of WAV files in the folder: {wav_file_count}")
