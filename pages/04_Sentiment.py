import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load AraBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2-sentiment")

# Define sentiment labels
labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using AraBERT.
    """
    encoded_text = tokenizer(text, return_tensors="pt")
    output = model(**encoded_text)
    logits = output.logits.squeeze(0)
    predicted_label_id = logits.argmax().item()
    predicted_label = labels[predicted_label_id]
    return predicted_label

# Streamlit app layout
st.title("AraBERT Sentiment Analysis")
st.write("Enter your text below to get sentiment analysis.")

text = st.text_area("Input Text")

if st.button("Analyze Sentiment"):
    if text:
        predicted_sentiment = predict_sentiment(text)
        st.write(f"Sentiment: **{predicted_sentiment}**")
    else:
        st.write("Please enter some text to analyze.")

st.markdown(
    """
    This app uses AraBERT, a pre-trained Transformer model for Arabic language understanding,
    to perform sentiment analysis on the input text.
    """
)
