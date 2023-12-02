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
            st.error("Speech recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service; {e}")
            return None

# Create a Streamlit app with a custom background color
st.markdown(
    """
    <style>
    .reportview-container {
        background: #c1f0c1; /* Change the background color here (light green) */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for styling
st.write(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ff7f50; /* Coral */
        color: #ffffff;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff6b36; /* Darker coral on hover */
    }
    div.stFileUploader > div > div > div > div {
        background-color: #ffcc00; /* Yellow */
        color: #000000;
    }
    .stTextInput {
        background-color: #f9ed69; /* Light yellow for text input */
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a Streamlit app
st.title("Audio Transcription App")

# Section for Recording Audio
st.header("Record Audio🎤")

# Create a SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize the ArabicTranscriptionModel
model = ArabicTranscriptionModel()

# Add a button to trigger audio recording
if st.button("Generate Transcripts✨"):
    # Start recording audio from the microphone
    with sr.Microphone() as source:
        st.info("Recording... Speak something!")
        audio = recognizer.listen(source)

    # Transcribe the recorded audio using the model
    try:
        # Use the audio-to-text method from your model
        text = model.audio_to_text_arabic(audio)
        st.success(f"Transcription: {text}")
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

# Section for Uploading Audio Files
st.header("Upload Audio File📤")

# Add an option to upload an audio file
uploaded_file = st.file_uploader("Choose an audio file to upload", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    with NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())

    # Transcribe the uploaded audio file
    audio_file = sr.AudioFile(temp_audio.name)
    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = model.audio_to_text_arabic(audio_data)
        st.success(f"Transcription from uploaded audio: {text}")
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

# Section to save transcripts
st.header("Save Transcripts📥")
st.write("Copy and paste the transcription you want to download here:")
transcript = st.text_area("Transcript Text")

# Format options
format_option = st.radio("Choose Format", ["Text File", "SRT File", "VTT File"])

if st.button("Click To Download⬇️"):
    if format_option == "Text File":
        st.download_button(
            label="Download Transcript as Text",
            data=transcript,
            key="text_file",
            file_name="transcript.txt",
        )
    elif format_option == "SRT File":
        st.download_button(
            label="Download Transcript as SRT",
            data=transcript,
            key="srt_file",
            file_name="transcript.srt",
        )
    elif format_option == "VTT File":
        st.download_button(
            label="Download Transcript as VTT",
            data=transcript,
            key="vtt_file",
            file_name="transcript.vtt",
        )












####################
##################
##you should run "streamlit run StreamlitAppCuteButtons.py"