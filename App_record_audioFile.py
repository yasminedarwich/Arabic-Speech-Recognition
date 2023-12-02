import streamlit as st
import os
import speech_recognition as sr
from tempfile import NamedTemporaryFile

# Define the ArabicTranscriptionModel class
class ArabicTranscriptionModel:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def audio_to_text_arabic(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data, language="ar-AR")
            return text
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service; {e}")
            return None

# Create a Streamlit app
st.title("Audio Transcription App")

# Create a SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize the ArabicTranscriptionModel
model = ArabicTranscriptionModel()

# Add a button to trigger audio recording
if st.button("Record Audio"):
    # Start recording audio from the microphone
    with sr.Microphone() as source:
        st.write("Recording... Speak something!")
        audio = recognizer.listen(source)

    # Transcribe the recorded audio using the model
    try:
        # Use the audio-to-text method from your model
        text = model.audio_to_text_arabic(audio)
        st.write(f"Transcription: {text}")
    except sr.UnknownValueError:
        st.write("Could not understand audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")

# Add an option to upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    with NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())

    # Transcribe the uploaded audio file
    audio_file = sr.AudioFile(temp_audio.name)
    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = model.audio_to_text_arabic(audio_data)
        st.write(f"Transcription from uploaded audio: {text}")
    except sr.UnknownValueError:
        st.write("Could not understand audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
















####################
##################
##you should run ""streamlit run App_record_audioFile.py