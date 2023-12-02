#count of txt files
import os

# Specify the directory path where your txt files are located
folder_path = "D:\AllArabicAudioTextLocal\AllArabicTextFilesLocal"

# Initialize a counter to keep track of the number of WAV files
txt_file_count = 0

# Iterate over the files in the directory
for filename in os.listdir(folder_path):
    # Check if the file has a .txt extension (case-insensitive)
    if filename.lower().endswith(".txt"):
        txt_file_count += 1

print(f"Number of text files in the folder: {txt_file_count}")