import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

# Define a function to reshape Arabic text
def reshape_arabic(text):
    return get_display(arabic_reshaper.reshape(text))


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

# Set the page width
st.set_page_config(layout="wide")

# Custom styling with Ocean Breeze Palette
st.markdown(
    """
    <style>
    body {
        background-color: #6FC0D3;  /* Ocean Breeze background */
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #1A4557;  /* Dark blue sidebar background */
        color: #FFFFFF;
    }
    .sidebar .stRadio > div > label {
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #3A90A2;  /* Blue button color */
        color: #FFFFFF;
    }
    .stButton > button:hover {
        background-color: #2A6881;  /* Darker blue on hover */
    }
    .stTextInput {
        background-color: #FFFFFF;  /* White text input background */
        color: #000000;
    }
    .stSelectbox select {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3 {
        color: #3A90A2;  /* Blue headers */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a Streamlit app
col1, col2 = st.columns(2)  # Two columns with equal width
with col1:
    st.image("podeo-logo-english-white.png", width=300)

# Add title to second column
with col2:
    st.title("Audio-Transcription App")

# Create a sidebar for navigation
st.sidebar.title("Hey there! 👋🏻")


# This is the transcription page

st.header("Record Audio 🎤")

# Create a SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize the ArabicTranscriptionModel
model = ArabicTranscriptionModel()

# Add a button to trigger audio recording
micAudio = audiorecorder("Click to record", "Click to stop recording") # Start recording audio from the microphone

if len(micAudio) > 0:
    # To play audio in frontend:
    st.audio(micAudio.export().read())  
    
    recordedFile = micAudio.export("audio.wav", format="wav")
    if recordedFile is not None:
        with NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(recordedFile.read())

        # Transcribe the uploaded audio file
        audio_file = sr.AudioFile(temp_audio.name)
        with audio_file as source:
            audio_data = recognizer.record(source)

        try:
            text = model.audio_to_text_arabic(audio_data)
            st.success(f"Transcription from uploaded audio: {text}")
            transcript = st.text_area("Transcript Text", value = text)
             # Section to save transcripts
            st.header("Save Transcripts 📥")
            st.write("Copy and paste the transcription you want to download here:")
            ###transcript = st.text_area("Transcript Text")
        
            # Format options
            format_option = st.radio("Choose Format", ["Text File", "SRT File", "VTT File"])
        
            if st.button("Click To Download ⬇️"):
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

        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

##   st.button("Record an Audio✨")

# Section for Calculating Transcription Cost
st.header("Transcription Cost 💰")
# Mention the transcription cost per minute
st.write("The transcription cost is $2.65 per minute.")



# Section for Uploading Audio Files
st.header("Upload an Audio File 📤")

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
        transcript = st.text_area("Transcript Text", value = text)
         # Section to save transcripts
        st.header("Save Transcripts 📥")
        st.write("Copy and paste the transcription you want to download here:")
        ###transcript = st.text_area("Transcript Text")
    
        # Format options
        format_option = st.radio("Choose Format", ["Text File", "SRT File", "VTT File"])
    
        if st.button("Click To Download ⬇️"):
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
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
