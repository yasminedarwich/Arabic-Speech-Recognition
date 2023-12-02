import os
import Levenshtein
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the path to your data folder
data_folder = 'D:/AllArabicAudioTextLocal'

# Initialize lists to store file paths and corresponding transcriptions
audio_paths = []
transcriptions = []

# Iterate through the 'AllArabicWavFiles' folder to collect audio file paths
audio_folder = os.path.join(data_folder, 'AllArabicWavFilesLocal')
for filename in os.listdir(audio_folder):
    if filename.endswith('.wav'):
        audio_path = os.path.join(audio_folder, filename)
        audio_paths.append(audio_path)

# Iterate through the 'AllArabicTranscription' folder to collect transcription text
transcription_folder = os.path.join(data_folder, 'AllArabicTextFilesLocal')
for filename in os.listdir(transcription_folder):
    if filename.endswith('.txt'):
        transcription_path = os.path.join(transcription_folder, filename)
        with open(transcription_path, 'r', encoding='utf-8') as transcript_file:
            transcription = transcript_file.read().strip()
        transcriptions.append(transcription)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_paths, transcriptions, test_size=0.2, random_state=42)

# Initialize a function to calculate Word Error Rate (WER)
def calculate_wer(true_transcriptions, predicted_transcriptions):
    wer = sum(Levenshtein.distance(true_transcriptions[i], predicted_transcriptions[i]) for i in range(len(true_transcriptions))/len(true_transcriptions))
    return wer

# Define the number of MFCC coefficients based on your data
num_mfcc_coefficients = 13  # Update with your actual number of MFCC coefficients

# Determine the number of unique characters in your transcription data
transcription_chars = set(''.join(transcriptions))
output_vocab_size = len(transcription_chars)

# Implement your data loading and preprocessing for audio and transcriptions here
# Preprocess audio data and encode transcriptions

# Set a maximum sequence length (you can adjust this based on your data)
max_sequence_length = 500  # Adjust this as needed

def preprocess_audio(audio_paths, num_mfcc_coefficients, max_sequence_length):
    mfcc_features = []
    for audio_path in audio_paths:
        # Load audio file using librosa
        audio, sr = librosa.load(audio_path, sr=None)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc_coefficients)

        # Ensure all sequences have the same length by padding or truncating
        if mfcc.shape[1] < max_sequence_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_sequence_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_sequence_length]

        mfcc_features.append(mfcc)

    return np.array(mfcc_features)

# Preprocess audio data
X_train_processed = preprocess_audio(X_train, num_mfcc_coefficients, max_sequence_length)
X_test_processed = preprocess_audio(X_test, num_mfcc_coefficients, max_sequence_length)

# Transpose the input data to have the correct shape for the LSTM layer
X_train_processed = np.transpose(X_train_processed, (0, 2, 1))
X_test_processed = np.transpose(X_test_processed, (0, 2, 1))

# Define a mapping from characters to indices
char_to_index = {char: index for index, char in enumerate(transcription_chars)}

# Encode transcriptions
def encode_transcriptions(transcriptions, output_vocab_size, max_sequence_length):
    encoded_transcriptions = []

    for transcription in transcriptions:
        # Initialize an array to hold the integer encoding for this transcription
        integer_encoding = []

        for char in transcription:
            if char in char_to_index:
                # Append the index of the character
                integer_encoding.append(char_to_index[char])

        # Ensure all sequences have the same length by padding or truncating
        if len(integer_encoding) < max_sequence_length:
            integer_encoding = integer_encoding + [0] * (max_sequence_length - len(integer_encoding))
        else:
            integer_encoding = integer_encoding[:max_sequence_length]

        encoded_transcriptions.append(integer_encoding)

    return np.array(encoded_transcriptions)

# Encode transcriptions
y_train_encoded = encode_transcriptions(y_train, output_vocab_size, max_sequence_length)
y_test_encoded = encode_transcriptions(y_test, output_vocab_size, max_sequence_length)

# Define the model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(max_sequence_length, num_mfcc_coefficients)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(output_vocab_size, activation='softmax')))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Train the model
model.fit(X_train_processed, y_train_encoded, validation_data=(X_test_processed, y_test_encoded), epochs=10, batch_size=32)

# Perform inference on the test data
predicted_transcriptions_encoded = model.predict(X_test_processed)

# Define a mapping from indices to characters for decoding
index_to_char = {index: char for char, index in char_to_index.items()}

# Implement your decoding logic here
# This function takes the model's predicted transcriptions
# and converts them into a human-readable format (e.g., plain text).

# Placeholder code for decoding predicted transcriptions
def decode_predictions(predicted_transcriptions_encoded):
    decoded_transcriptions = []

    for transcription_encoded in predicted_transcriptions_encoded:
        # Map indices back to characters
        decoded_text = ''.join([index_to_char[index] for index in transcription_encoded])

        decoded_transcriptions.append(decoded_text)

    return decoded_transcriptions

# Decode the predicted transcriptions
decoded_predictions = decode_predictions(predicted_transcriptions_encoded)

# Calculate the Word Error Rate (WER)
wer = calculate_wer(y_test, decoded_predictions)

print(f'Word Error Rate (WER): {wer}')


