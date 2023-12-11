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


# This is the Analytics page
st.title("Podeo Podcasts Analysis")

# Load your data, assuming you have the data loaded into the 'df1' DataFrame
#C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/date_posted.csv
 #Load your data, assuming you have the data loaded into the 'df1' DataFrame
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

###########C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/episodesWithMostLikes.csv
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

#C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/top10LeastPopularPodcasts.csv
df3 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv?token=GHSAT0AAAAAACKRHIALJ3AVIXXFC3KN66ZYZLWM6WA')

# Sort the DataFrame by 'likes_count' in ascending order
df3 = df3.sort_values(by='likes_count', ascending=True)

# Select the top 10 rows (episodes with the least likes)
bottom_10_episodes = df3.head(10)

# Reshape the Arabic words to show correctly
x = [get_display(arabic_reshaper.reshape(item)) for item in bottom_10_episodes['name']]

# Create a bar chart to visualize the top 10 episodes with the least likes
fig3, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(x, bottom_10_episodes['likes_count'], color='skyblue')

# Annotate each bar with the number of likes
for bar, like in zip(bars, bottom_10_episodes['likes_count']):
    ax.text(bar.get_x() + bar.get_width() / 2, like, str(like), ha='center', va='bottom', color='blue')

# Customize the plot
ax.set_xlabel('Episode Name')
ax.set_ylabel('Likes Count')
ax.set_title('Top 10 Least Popular Podcasts')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

# Display the Matplotlib plot in Streamlit
st.pyplot(fig3)

#####C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/top10MostPopularAuthors.csv
# 
df3 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv?token=GHSAT0AAAAAACKRHIALJ3AVIXXFC3KN66ZYZLWM6WA")

# Reshape the Arabic words to show correctly
x = df3['author'].apply(lambda item: get_display(arabic_reshaper.reshape(item)))

df3 = df3.sort_values(by='likes_count', ascending=False)

# Select the top 10 rows (authors with the most likes)
top_10_authors = df3.head(10)

# Create a bar chart to visualize the top 10 authors using Matplotlib
fig3, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(x, top_10_authors['likes_count'], color='skyblue')

# Add percentage labels on top of each bar
total_likes = top_10_authors['likes_count'].sum()
for bar in bars:
    percentage = (bar.get_height() / total_likes) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.2f}%', ha='center')

# Customize the plot
ax.set_xlabel('Author')
ax.set_ylabel('Likes Count')
ax.set_title('Top 10 Most Popular Authors')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability

# Display the Matplotlib plot in Streamlit
st.pyplot(fig3)

######C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/top10LeastPopularAuthors.csv
df4 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv?token=GHSAT0AAAAAACKRHIALJ3AVIXXFC3KN66ZYZLWM6WA')

# Sort the DataFrame by 'likes_count' in ascending order
df4 = df4.sort_values(by='likes_count', ascending=True)

# Select the top 10 rows (least popular authors)
bottom_10_authors = df4.head(10)

# Create an interactive bar chart using Plotly to visualize the top 10 least popular authors
fig4 = px.bar(
    bottom_10_authors,
    x='author',
    y='likes_count',
    labels={'likes_count': 'Likes Count'},
    title='Top 10 Least Popular Authors'
)

# Customize the layout for better interaction
fig4.update_xaxes(categoryorder='total ascending', title='Author', tickangle=45)
fig4.update_yaxes(title='Likes Count')

# Display the interactive Plotly chart in Streamlit
st.plotly_chart(fig4)

#C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/categoriesCount.csv

df5 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/categoriesCount.csv?token=GHSAT0AAAAAACKRHIALMFXA76ZBY6N4VNXUZLWM7KQ')

# Calculate the total count of categories
total_count = df5['category_count'].sum()

# Calculate the category percentages
df5['Category_Percentage'] = (df5['category_count'] / total_count) * 100

# Create the pie chart
fig5 = st.pie_chart(
    data=df5,
    labels=df5['namecategory'],
    values=df5['Category_Percentage'],
    label_format="%1.1f%%",
)

# Customize the layout for better interaction
#fig5.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])  # Adjust the appearance of pie slices

# Display the interactive Plotly chart in Streamlit
#st.plotly_chart(fig5)
#
#C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/Newest podcasts.csv
df6 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/Newest%20podcasts.csv?token=GHSAT0AAAAAACKRHIAL6JTCXRZVDR7HYT4SZLWM74Q")

# Reshape Arabic words to display correctly
df6['name'] = df6['name'].apply(lambda item: get_display(arabic_reshaper.reshape(item)))
# Sort the DataFrame by the date posted in descending order
df6 = df6.sort_values(by='date_posted', ascending=False)

# Add a rank column
df6['Rank'] = range(1, 11)

# Create a table using Matplotlib
fig6, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df6.values, colLabels=df6.columns, cellLoc='center', loc='center')

# Add the table to your Streamlit app
st.title("Newest 10 Podcasts")
st.pyplot(fig6)

#
#C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/oldestPodcasts.csv
df7 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/oldestPodcasts.csv?token=GHSAT0AAAAAACKRHIAKI3WNMJE5UVLAPIKUZLWNAEQ")

# Reshape Arabic words to display correctly
df7['name'] = df7['name'].apply(lambda item: get_display(arabic_reshaper.reshape(item)))
# Sort the DataFrame by the date posted in descending order
df7 = df7.sort_values(by='date_posted', ascending=True)

# Add a rank column
df7['Rank'] = range(1, 11)

# Create a table using Matplotlib
fig7, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df6.values, colLabels=df6.columns, cellLoc='center', loc='center')

# Add the table to your Streamlit app
st.title("Oldest 10 Podcasts")
st.pyplot(fig7)



############
########
##you should run "streamlit run NewStreamlit.py
