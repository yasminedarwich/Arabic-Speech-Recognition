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

###########


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper

# Read the CSV file
df2 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/episodesWithMostLikes.csv?token=GHSAT0AAAAAACKRHIALIWMTB3F6KSEGETVEZLWM6MA")

# Sort the DataFrame by likes_count in descending order and select the top 10 rows
df2 = df2.sort_values(by="likes_count", ascending=False).head(10)

# Reshape the Arabic words to show correctly
df2['name'] = [get_display(arabic_reshaper.reshape(item)) for item in df2.name.values]

# Create a DataFrame with the data
data = pd.DataFrame({'episodes': df2['name'], 'likes': df2['likes_count']})

# Sort the DataFrame by likes_count in descending order
data = data.sort_values(by="likes", ascending=True)

# Create a horizontal bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(data['episodes'], data['likes'], color='skyblue')

# Annotate each bar with the number of likes
for bar, like in zip(bars, data['likes']):
    ax.text(like, bar.get_y() + bar.get_height() / 2, str(like), va='center', color='blue')

# Customize the plot
ax.set_xlabel("Number of Likes")
ax.set_ylabel("Episode")
ax.set_title("Top 10 Most Popular Podcasts")
ax.invert_yaxis()  # To display the episodes in descending order

# Display the Matplotlib plot in Streamlit
st.pyplot(fig)
