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
