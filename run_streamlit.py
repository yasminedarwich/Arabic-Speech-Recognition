from pyngrok import ngrok
import streamlit as st

# Function to run the Streamlit app
def run():
    st.write("Running your Streamlit app...")
    !streamlit run /content/drive/MyDrive/ColabNotebooks/app.py  # Specify the full path to your app.py script

# Create a public URL to access the app in Colab
public_url = ngrok.connect(port='8501')
st.write('Streamlit app is available at', public_url)

if __name__ == '__main__':
    run()

