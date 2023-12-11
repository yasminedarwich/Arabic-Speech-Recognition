import streamlit as st
import ktrain
from ktrain import text

# Load the saved predictor
MODEL_NAME = 'aubmindlab/bert-base-arabertv01'
p = ktrain.load_predictor('/tmp/arabic_predictor')

# Streamlit app
def main():
    st.title("Arabic Sentiment Analysis with BERT")

    # Input text
    input_text = st.text_area("Enter Arabic text:", "الغرفة كانت نظيفة ، الطعام ممتاز ، وأنا أحب المنظر من غرفتي.")

    # Analyze sentiment on button click
    if st.button("Analyze Sentiment"):
        result = p.predict(input_text)
        st.write(f"Sentiment: {result}")

if __name__ == "__main__":
    main()
