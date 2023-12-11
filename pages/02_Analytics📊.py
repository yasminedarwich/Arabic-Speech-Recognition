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



###############


df2 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/episodesWithMostLikes.csv?token=GHSAT0AAAAAACKRHIALIWMTB3F6KSEGETVEZLWM6MA")

# Sort the DataFrame by likes_count in descending order and select the top 10 rows
df2 = df2.sort_values(by="likes_count", ascending=False).head(10)

# Reshape the Arabic words to show correctly
x = [get_display(arabic_reshaper.reshape(item)) for item in df2.name.values]

# Get the number of likes for each episode
likes = df2["likes_count"]
episodes = x

# Create a horizontal bar chart of the number of likes for the top 10 episodes using Matplotlib
fig2, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.barh(episodes, likes, color='skyblue')

# Annotate each bar with the number of likes
for bar, like in zip(bars, likes):
    ax.text(like, bar.get_y() + bar.get_height() / 2, str(like), va='center', color='blue')

# Customize the plot
ax.set_xlabel("Number of Likes")
ax.set_ylabel("Episode")
ax.set_title("Top 10 Most Popular Podcasts")
ax.invert_yaxis()  # To display the episodes in descending order

# Display the Matplotlib plot in Streamlit
st.pyplot(fig2)


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


###################


df1 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/date_posted.csv?token=GHSAT0AAAAAACKRHIALGNTFVQKWYDXLOCMGZLWM5WQ")

# Convert 'date_posted' to datetime if it's not already
df1['date_posted'] = pd.to_datetime(df1['date_posted'])

# Filter for episodes posted after 2010
df1_filtered = df1[df1['date_posted'] >= pd.to_datetime("2010-01-01")]

# Group episodes by year and count the number of episodes for each year
episode_counts_by_year = df1_filtered.resample('Y', on='date_posted').size().reset_index()
episode_counts_by_year.columns = ["Year", "Number of Episodes"]

# Create an interactive Plotly line plot
fig1 = px.line(episode_counts_by_year, x="Year", y="Number of Episodes", title='Trend of Episodes by Year (After 2010)')

# Customize the plot layout
fig1.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Episodes',
)

# Display the Plotly plot in Streamlit
st.plotly_chart(fig1)




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

st.title('Visualizing Listens Count by Country')

# Bar chart
bar_chart = px.bar(df8, x='country', y='listens_count', title='Listens Count by Country',
                   labels={'listens_count': 'Listens Count','country': 'Country'},
                   color='listens_count', color_continuous_scale='Viridis')
st.plotly_chart(bar_chart)


#########


from streamlit_card import card

st.title('Top Meta-Tags')

# Define categories, icons, and background images
categories_info = {
    "Science & Technology": {"icon": "🔬", "background": "science_background_url"},
    "Society & Culture": {"icon": "🌐", "background": "society_background_url"},
    "Business & Finance": {"icon": "💰", "background": "business_finance_background_url"},
    "Religion & Spirituality": {"icon": "⛪", "background": "religion_background_url"},
    "Arts": {"icon": "🎨", "background": "arts_background_url"},
    "Sports": {"icon": "⚽", "background": "sports_background_url"},
    "Economy": {"icon": "💼", "background": "economy_background_url"},
    "Gaming & Hobbies": {"icon": "🎮", "background": "gaming_background_url"},
    "Education": {"icon": "🎓", "background": "education_background_url"},
    "Health & Medicine": {"icon": "⚕️", "background": "health_background_url"},
    "Money": {"icon": "💰", "background": "money_background_url"},
    "News & Politics": {"icon": "📰", "background": "news_politics_background_url"},
    "Comedy": {"icon": "😄", "background": "comedy_background_url"},
    "History": {"icon": "📜", "background": "history_background_url"},
    "Government & Organizations": {"icon": "🏛️", "background": "government_background_url"},
    "Kids & Family": {"icon": "👨‍👩‍👧‍👦", "background": "kids_family_background_url"},
}


# Function to display category cards
def display_category_card(category, info):
    """
    Display a category card with an icon and background image using streamlit_card.
    """
    with st.container():
        hasClicked = card(
            title=f"{info['icon']} {category}",
            text="", 
            image=info["background"],
            styles={
                "card": {
                    "width": "150px",
                    "height": "150px",
                    "border-radius": "20px"
                },
                "title": {
                    "font-size": "14px"  # Adjust the font size as needed
                }
            }
        )

# Create 4 columns
columns = st.columns(4)

# Display category cards in a 4 x 4 grid
for i, (category, info) in enumerate(categories_info.items()):
    # Determine the column to place the card in
    col_index = i % 4
    # Display the card in the appropriate column
    with columns[col_index]:
        display_category_card(category, info)



#########


import pandas as pd
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display

# Create a DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10MostPopularAuthors.csv")

# Reshape the Arabic words to show correctly on matplotlib
df['author_display'] = df['author'].apply(lambda item: get_display(arabic_reshaper.reshape(item)))

df = df.sort_values(by='likes_count', ascending=False)

# Select the top 10 rows (authors with the most likes)
top_10_authors = df.head(10)

# Streamlit App
st.title('Top 10 Most Popular Authors')

# Bar Chart using Altair (a popular visualization library compatible with Streamlit)
import altair as alt


# Bar Chart using Altair
chart = alt.Chart(top_10_authors).mark_bar().encode(
    x=alt.X('author_display:N', title='Author', sort='-y'),  # Sort by likes_count descending
    y=alt.Y('likes_count:Q', title='Likes Count'),
    tooltip=['author_display:N', 'likes_count:Q']
).properties(
    width=600,
    height=400
).configure_axis(
    labelAngle=-45,
    labelAlign='right'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)


##############



# Assuming you have a DataFrame named df4 with columns 'browser' and 'listens_count'
df4 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/Top_browsers.csv")

# Replace '0' and 'nan' with 'Podeo' in the 'browser' column
df4['browser'] = df4['browser'].replace({'0': 'Podeo', pd.NA: 'Podeo'})

# Group by the 'browser' column and sum the 'listens_count'
grouped_df4 = df4.groupby('browser', as_index=False)['listens_count'].sum()

# Sort the DataFrame by listens_count in descending order
sorted_df4 = grouped_df4.sort_values(by='listens_count', ascending=False)

# Convert the values in the 'browser' column to strings
sorted_df4['browser'] = sorted_df4['browser'].astype(str)

# Streamlit App
st.title('Listens Count by Browser')

# Bar Chart using Altair
chart = alt.Chart(sorted_df4).mark_bar().encode(
    x=alt.X('browser:N', title='Browser', sort='-y'),  # Sort by listens_count descending
    y=alt.Y('listens_count:Q', title='Listens Count'),
    tooltip=['browser:N', 'listens_count:Q']
).properties(
    width=600,
    height=400
).configure_axis(
    labelAngle=-45,
    labelAlign='right'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)



############



df6= pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10authors.csv")
sorted_df6 = df6.sort_values(by='episode_count', ascending=False)

# Select the top 3 authors
top_authors = sorted_df6.head(10)

# Streamlit App
st.title('Top 3 Authors Producing the Most Episodes')

# Bar Chart using Altair
chart = alt.Chart(top_authors).mark_bar().encode(
    x=alt.X('author:N', title='Author', sort='-y'),  # Sort by episode_count descending
    y=alt.Y('episode_count:Q', title='Episode Count'),
    tooltip=['author:N', 'episode_count:Q']
).properties(
    width=600,
    height=400
).configure_axis(
    labelAngle=-45,
    labelAlign='right'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)
