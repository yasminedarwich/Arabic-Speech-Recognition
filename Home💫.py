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
with col1:
    st.image("podeo-logo-english-white.png", width=300)

# Add title to second column
with col2:
    st.title("Audio-Transcription App")

# Create a sidebar for navigation
st.sidebar.title("Hey there! 👋🏻")
st.sidebar.write('''
    I am Yasmine Darwich, an MSBA student at AUB 👱‍♀️🏫! 

    I enjoy working with datasets and analyzing information 🎯. 
    
    Let's connect:

    [LinkedIn](https://www.linkedin.com/in/yasmine-darwich/)

    ''')
page = st.sidebar.radio("Select a Page", ["Home💫", "Transcription🎤", "Analytics📊"])


st.markdown(
        """
        Podeo is the Arab World's largest podcasting platform, dedicated to managing, distributing, and producing audio podcasts that reach millions of listeners.

        With this app, you can easily transcribe your audio recordings. You have two options:
        1. Record Audio 🎤: Click the "Generate Transcripts✨" button, speak into your microphone, and let the app transcribe your speech.
        2. Upload Audio File 📤: Upload an audio file in formats like WAV, MP3, or OGG, and get it transcribed instantly.

        After transcription, you can save the text in various formats like Text File, SRT, or VTT. Click "Click To Download⬇️" to save your transcript.

        Enjoy seamless transcription with the Podeo Audio-Transcription App!
        """
    )

    # Add images, GIFs, or any content you want to display on the introduction page
    col3, col4, col5 = st.columns([0.2, 0.6, 0.2])
    with col4:
        st.image("Podcastcaptions.png")
