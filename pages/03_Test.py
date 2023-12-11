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
import plotly.express as px
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

# Create an interactive horizontal bar chart using Plotly Express
fig = px.bar(data, x='likes', y='episodes', orientation='h', text='likes', title="Top 10 Most Popular Podcasts")
fig.update_layout(xaxis_title="Number of Likes", yaxis_title="Episode")
fig.update_traces(marker_color='skyblue')

# Display the Plotly Express plot in Streamlit
st.plotly_chart(fig)



##############

import streamlit as st
import pandas as pd
from bidi.algorithm import get_display
import arabic_reshaper

# Read the CSV file
df3 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv?token=GHSAT0AAAAAACKRHIALJ3AVIXXFC3KN66ZYZLWM6WA')

# Reshape the Arabic words to show correctly
if 'author' in df3.columns:
    df3['reshaped_author'] = [get_display(arabic_reshaper.reshape(item)) for item in df3['author'].astype(str)]
else:
    # Adjust the column name based on the actual column name in your DataFrame
    st.error("Column 'author' not found in the DataFrame. Please adjust the column name.")
    st.stop()

st.title("Least Popular Authors")

# Display authors in two rows with 5 columns on each row
row1, row2 = st.columns(2)

for index, row in df3.iterrows():
    if index < 5:
        row1.write(
            f'<div style="margin: 10px; padding: 15px; border: 1px solid #ddd; text-align: center;">'
            f'{row["reshaped_author"]}'
            f'</div>', unsafe_allow_html=True)
    else:
        row2.write(
            f'<div style="margin: 10px; padding: 15px; border: 1px solid #ddd; text-align: center;">'
            f'{row["reshaped_author"]}'
            f'</div>', unsafe_allow_html=True)

