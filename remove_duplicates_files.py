import os
import hashlib

def get_file_hash(file_path):
    """Calculate and return the hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in 64k chunks
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

def find_duplicate_files(folder_path):
    """Find and return a dictionary of duplicate files in a folder."""
    file_hashes = {}
    duplicate_files = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = get_file_hash(file_path)

            if file_hash in file_hashes:
                duplicate_files[file_path] = file_hashes[file_hash]
            else:
                file_hashes[file_hash] = file_path

    return duplicate_files

def remove_duplicate_files(folder_path):
    """Remove duplicate files from a folder."""
    duplicate_files = find_duplicate_files(folder_path)
    
    for duplicate, original in duplicate_files.items():
        print(f"Removing duplicate: {duplicate}")
        os.remove(duplicate)

if __name__ == "__main__":
    folder_path = "D:\AllArabicAudioTextLocal\AllArabicWavFilesLocal"
    remove_duplicate_files(folder_path)

