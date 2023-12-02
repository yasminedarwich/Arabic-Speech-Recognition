import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import librosa  # For audio feature extraction

# Define the path to your data folders
audio_data_folder = "D:/AllArabicWavFilesLocal"
transcription_data_folder = "D:/AllArabicTextFilesLocal"

# Load audio data and transcriptions
audio_files = os.listdir(audio_data_folder)
transcription_files = os.listdir(transcription_data_folder)

# Initialize empty lists to store audio features and transcriptions
X = []  # Audio features
y = []  # Transcriptions

# Loop through audio files and load features and transcriptions
for audio_file in audio_files:
    if audio_file.endswith(".wav"):
        # Load audio features (MFCCs) as audio features
        audio_path = os.path.join(audio_data_folder, audio_file)
        audio, sr = librosa.load(audio_path, sr=None)  # Load the audio file
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCCs
        audio_features = np.mean(mfcc, axis=1)  # Use the mean as features

        # Load transcription
        transcription_file = os.path.join(transcription_data_folder, audio_file.replace(".wav", ".txt"))
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription = f.read().strip()

        X.append(audio_features)
        y.append(transcription)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a CountVectorizer to convert text data to numerical format
vectorizer = CountVectorizer()
X_train_text = vectorizer.fit_transform(y_train)
X_test_text = vectorizer.transform(y_test)

# Combine audio features and text features
X_train_combined = np.concatenate((X_train, X_train_text.toarray()), axis=1)
X_test_combined = np.concatenate((X_test, X_test_text.toarray()), axis=1)

# Initialize a Support Vector Machine (SVM) classifier
clf = SVC()

# Train the classifier
clf.fit(X_train_combined, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_combined)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
