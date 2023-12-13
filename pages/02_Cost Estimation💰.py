import streamlit as st

# Set the page config
st.set_page_config(
    page_title="Arabic Speech Recognition",
    page_icon="https://raw.githubusercontent.com/yasminedarwich/Arabic-Speech-Recognition/main/static/voice.ico",  # Path to your favicon in the static folder
    layout="wide"
)

# Set the theme
st.markdown(
    """
    <style>
        :root {
            --primaryColor: #6FC0D3;
            --secondaryBackgroundColor: #6FC0D3;
            --textColor: #6FC0D3;
        }
        body {
            color: var(--textColor);
            background-color: var(--secondaryBackgroundColor);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--primaryColor);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def calculate_transcription_cost(duration_minutes, cost_per_minute, audio_quality, num_speakers, urgency):
    try:
        duration_minutes = float(duration_minutes)
        cost_per_minute = float(cost_per_minute)

        # Adjust transcription cost based on factors
        audio_quality_factor = 1 + (audio_quality - 5) / 10  # Adjust based on a scale of 1-10
        num_speakers_factor = 1 + (num_speakers - 1) * 0.2  # Adjust based on the number of speakers
        urgency_factor = 1 + (urgency - 1) * 0.1  # Adjust based on urgency/turnaround time

        transcription_cost = duration_minutes * cost_per_minute * audio_quality_factor * num_speakers_factor * urgency_factor
        return transcription_cost
    except ValueError:
        return None

def main():

    col1, col2, col3 = st.columns(3)
    col2.image("hero-img.png", use_column_width=True)

    st.markdown("<h1 style='text-align: center;'>Arabic Transcription Cost Estimation</h1>", unsafe_allow_html=True)

    # Input
    duration_minutes = st.text_input("Enter Duration (minutes):", "0")

    # Sliders for additional factors
    audio_quality = st.slider("Audio Quality (1-10)", min_value=1, max_value=10, value=5)
    num_speakers = st.slider("Number of Speakers", min_value=1, max_value=5, value=1)
    urgency = st.slider("Urgency/Turnaround Time (1-5)", min_value=1, max_value=5, value=3)

    # Calculate
    cost_per_minute = 2.65  # Cost of Arabic transcription per minute
    transcription_cost = calculate_transcription_cost(duration_minutes, cost_per_minute, audio_quality, num_speakers, urgency)

    # Display Result
    if transcription_cost is not None:
        st.subheader("Estimated Transcription Cost:")
        st.write(f"${transcription_cost:.2f}")
    else:
        st.warning("Please enter a valid numerical value for duration.")

    # Reset Button
    if st.button("Reset"):
        st.text_input("Enter Duration (minutes):", "0")

if __name__ == "__main__":
    main()

