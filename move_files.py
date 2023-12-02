import shutil
import os

def move_files(source_folder, destination_folder):
    try:
        # Ensure the source folder exists
        if not os.path.exists(source_folder):
            print(f"Source folder '{source_folder}' does not exist.")
            return

        # Ensure the destination folder exists; create it if it doesn't
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # List all files in the source folder
        files = os.listdir(source_folder)

        # Move each file to the destination folder
        for file in files:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.move(source_path, destination_path)
            print(f"Moved '{file}' to '{destination_folder}'")

        print("All files have been moved successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Replace these paths with your source and destination folder paths
source_folder_path = "D:/AllArabicAudioTextLocal/AllArabicWavFiles"
destination_folder_path = "D:/AllArabicAudioTextLocal/AllArabicWavFilesLocal"

move_files(source_folder_path, destination_folder_path)
