import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import arabic_reshaper
from arabic_reshaper import reshape
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

# Path to the Arabic font file in OTF format (adjust the path accordingly)
font_path = "https://github.com/yasminedarwich/Arabic-Speech-Recognition/blob/main/ArabicFont.otf"

# Cache the font loading operation for better performance
@st.cache
def load_font():
    pass  # Placeholder function; no Streamlit functions should be called inside this function

# Load the font
load_font()

# Set the Arabic font using st.markdown() outside the cached function
st.markdown(f'<style>div {{ font-family: "ArabicFont", sans-serif; }}</style>', unsafe_allow_html=True)

###########

import streamlit as st
import pandas as pd
import plotly.express as px
from bidi.algorithm import get_display
import arabic_reshaper

# Read the CSV file
df2 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/episodesWithMostLikes.csv?token=GHSAT0AAAAAACKRHIALGP25WZU7YTHCQAEYZLWPBTQ")

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


############


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


#################


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df5 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/categoriesCount.csv')

st.title("Most Popular Catgories")

# Create a hierarchical structure for sunburst chart
sunburst_data = {'labels': df5['name'], 'parents': [''] * len(df5), 'values': df5['category_count']}
sunburst_df = pd.DataFrame(sunburst_data)

# Create the sunburst chart using Plotly
fig = go.Figure(go.Sunburst(labels=sunburst_df['labels'], parents=sunburst_df['parents'], values=sunburst_df['values']))

# Set layout properties for better aesthetics
fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    sunburstcolorway=["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#FFA07A", "#FFD700", "#20B2AA", "#FF4500", "#7FFF00"],
)

# Display the Plotly sunburst chart in Streamlit
st.plotly_chart(fig)



###########


# Load the DataFrame from the provided URL
df8 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/TopCountries.csv")

# Streamlit app code
st.title('Visualizing Listens Count by Country')

# Bar chart
bar_chart = px.bar(df8, x='country', y='listens_count', title='Listens Count by Country',
                   labels={'listens_count': 'Listens Count','country': 'Country'},
                   color='listens_count', color_continuous_scale='Viridis')
st.plotly_chart(bar_chart)


#########


from streamlit_card import card

# Define categories, icons, and background images
categories_info = {
    "Business": {"icon": "👜", "background": "business_background_url"},
    "Science & Technology": {"icon": "🔬", "background": "science_background_url"},
    "Society & Culture": {"icon": "🌐", "background": "society_background_url"},
    "Business & Finance": {"icon": "💰", "background": "business_finance_background_url"},
    "Religion & Spirituality": {"icon": "⛪", "background": "religion_background_url"},
    "Arts": {"icon": "🎨", "background": "arts_background_url"},
    "TV & Film": {"icon": "📺", "background": "tv_film_background_url"},
    "Sports": {"icon": "⚽", "background": "sports_background_url"},
    "economy": {"icon": "💼", "background": "economy_background_url"},
    "Gaming & Hobbies": {"icon": "🎮", "background": "gaming_background_url"},
    "finance": {"icon": "💸", "background": "finance_background_url"},
    "Education": {"icon": "🎓", "background": "education_background_url"},
    "Health & Medicine": {"icon": "⚕️", "background": "health_background_url"},
    "money": {"icon": "💰", "background": "money_background_url"},
    "News & Politics": {"icon": "📰", "background": "news_politics_background_url"},
    "Comedy": {"icon": "😄", "background": "comedy_background_url"},
    "news": {"icon": "📰", "background": "news_background_url"},
    "History": {"icon": "📜", "background": "history_background_url"},
    "Government & Organizations": {"icon": "🏛️", "background": "government_background_url"},
    "Kids & Family": {"icon": "👨‍👩‍👧‍👦", "background": "kids_family_background_url"},
}


# Function to display category cards
def display_category_card(category, info):
    """
    Display a category card with an icon and background image using streamlit_card.
    """
    with st.beta_container():
        hasClicked = card(
            title=f"{info['icon']} {category}",
            text="Some description",
            image=info["background"],
        )

# Display category cards
for category, info in categories_info.items():
    display_category_card(category, info)

