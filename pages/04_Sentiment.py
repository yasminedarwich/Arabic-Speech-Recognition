import os
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
import streamlit as st

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load Arabic stop words
arabic_stopwords = set(stopwords.words('arabic'))

# Read and preprocess your Arabic text files
def preprocess(text):
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word.lower() not in arabic_stopwords]
    return ' '.join(words)

# Function to analyze sentiment
def analyze_sentiment(text):
    preprocessed_text = preprocess(text)
    blob = TextBlob(preprocessed_text)
    sentiment_score = blob.sentiment.polarity

    # Classify sentiment into positive, negative, or neutral
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit App
st.title('Arabic Sentiment Analysis')

# Input text box for user input
user_input = st.text_area('Enter your Arabic text here:', '')

# Analyze sentiment on button click
if st.button('Analyze Sentiment'):
    if user_input:
        sentiment_result = analyze_sentiment(user_input)
        st.write(f'Sentiment: {sentiment_result}')
    else:
        st.warning('Please enter some text for analysis.')

# Print overall sentiment statistics (you can adjust this part based on your specific needs)
st.header('Overall Sentiment Statistics')

# You can replace this with your actual sentiment statistics calculation
# For now, I'm using placeholder values based on your existing code
total_files = 100  # replace with your actual total_files count
positive_percentage = 30.0  # replace with your actual positive_percentage
negative_percentage = 20.0  # replace with your actual negative_percentage
neutral_percentage = 50.0  # replace with your actual neutral_percentage

st.write(f'Total Files: {total_files}')
st.write(f'Positive Percentage: {positive_percentage}%')
st.write(f'Negative Percentage: {negative_percentage}%')
st.write(f'Neutral Percentage: {neutral_percentage}%')
