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
from transformers import pipeline
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess(text):
  # Lowercase text
  text = text.lower()
  # Remove punctuation
  text = ''.join([c for c in text if not c.isalnum() and c != " "])
  # Tokenize text
  words = nltk.word_tokenize(text)
  # Remove stop words
  words = [word for word in words if word.lower() not in stopwords.words('arabic')]
  return ' '.join(words)

# Function to perform sentiment analysis
def analyze_sentiment(text):
  # Preprocess the text
  preprocessed_text = preprocess(text)
  # Create TextBlob object
  blob = TextBlob(preprocessed_text)
  # Analyze sentiment
  sentiment = blob.sentiment
  return sentiment.polarity, sentiment.subjectivity



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
st.image("podeoCapture.PNG", caption="Image1")
st.title("Podeo Audio-Transcription App")

# Create a sidebar for navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Select a Page", ["Home💫", "Transcription🎤", "Analytics📊"])

if page == "Home💫":
    # This is the introduction page
    st.title("Welcome to Podeo Transcription App")
    st.markdown(
        """
        Podeo is the Arab World's largest podcasting platform, dedicated to managing, distributing, and producing audio podcasts that reach millions of listeners.

        With this app, you can easily transcribe your audio recordings. You have two options:
        1. Record Audio🎤: Click the "Generate Transcripts✨" button, speak into your microphone, and let the app transcribe your speech.
        2. Upload Audio File📤: Upload an audio file in formats like WAV, MP3, or OGG, and get it transcribed instantly.

        After transcription, you can save the text in various formats like Text File, SRT, or VTT. Click "Click To Download⬇️" to save your transcript.

        Enjoy seamless transcription with Podeo Audio-Transcription App!
        """
    )

    # Add images, GIFs, or any content you want to display on the introduction page
    st.image("Capture34.PNG", caption="Image2")
   

elif page == "Transcription🎤":
    # This is the transcription page
    # Section for Recording Audio
    st.header("Record Audio🎤")

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
                st.header("Save Transcripts📥")
                st.write("Copy and paste the transcription you want to download here:")
                ###transcript = st.text_area("Transcript Text")
            
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

            except sr.UnknownValueError:
                st.warning("Could not understand audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")

 ##   st.button("Record an Audio✨")

       


    # Section for Calculating Transcription Cost
    st.header("Transcription Cost💰")
    # Mention the transcription cost per minute
    st.write("The transcription cost is $2.65 per minute.")



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
            transcript = st.text_area("Transcript Text", value = text)
             # Section to save transcripts
            st.header("Save Transcripts📥")
            st.write("Copy and paste the transcription you want to download here:")
            ###transcript = st.text_area("Transcript Text")
        
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
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

def transcribe_and_analyze(audio_file):
  # Transcribe the audio
  transcript = transcribe_audio(audio_file)

  # Perform sentiment analysis
  polarity, subjectivity = analyze_sentiment(transcript)

  # Display results
  st.write(f"Transcript: {transcript}")
  st.write(f"Sentiment Polarity: {polarity}")
  st.write(f"Sentiment Subjectivity: {subjectivity}")

if st.button("Transcribe and Analyze"):
  uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
  if uploaded_file is not None:
    transcribe_and_analyze(uploaded_file)




elif page == "Analytics📊":
    # This is the Analytics page
    st.title("Podeo Podcasts Analysis")

    # Load your data, assuming you have the data loaded into the 'df1' DataFrame
    #C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/date_posted.csv
     #Load your data, assuming you have the data loaded into the 'df1' DataFrame
    df1 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/date_posted.csv")

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
    df2 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/episodesWithMostLikes.csv")

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
    df3 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv')

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
    df3 = pd.read_csv("https://github.com/yasminedarwich/Arabic-Speech-Recognition/tree/main/EDA_Local/top10MostPopularAuthors.csv")

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
    df4 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/top10LeastPopularAuthors.csv')

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

    df5 = pd.read_csv('https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/categoriesCount.csv')

    # Calculate the total count of categories
    total_count = df5['category_count'].sum()

    # Calculate the category percentages
    df5['Category_Percentage'] = (df5['category_count'] / total_count) * 100

    # Create an interactive pie chart using Plotly for the category breakdown
    fig5 = px.pie(
        df5,
        values='Category_Percentage',
        names='name',
        title='Top Categories Listened To',
    )

    # Customize the layout for better interaction
    fig5.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])  # Adjust the appearance of pie slices

    # Display the interactive Plotly chart in Streamlit
    st.plotly_chart(fig5)
    #
    #C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/Newest podcasts.csv
    df6 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/Newest%20podcasts.csv")

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
    df7 = pd.read_csv("https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/EDA_Local/oldestPodcasts.csv")

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

    # Load your dataset into a DataFrame (replace 'your_dataset.csv' with the actual filename)
    ##C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/episodes_creators.csv
    #df8 = pd.read_csv('C:/Users/User/Desktop/DonaLeb/PodeoCodesLocal/EDA_Local/episodes_creators.csv')
    

    

# Run the app



####################
##################
##you should run "streamlit run NewStreamlit.py
