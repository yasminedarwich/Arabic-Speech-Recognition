import streamlit as st

def calculate_transcription_cost(duration_minutes, cost_per_minute):
    try:
        duration_minutes = float(duration_minutes)
        cost_per_minute = float(cost_per_minute)
        transcription_cost = duration_minutes * cost_per_minute
        return transcription_cost
    except ValueError:
        return None

def main():
    st.title("Arabic Transcription Cost Estimation")

    # Input
    duration_minutes = st.text_input("Enter Duration (minutes):", "0")

    # Calculate
    cost_per_minute = 2.5  # Cost of Arabic transcription per minute
    transcription_cost = calculate_transcription_cost(duration_minutes, cost_per_minute)

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
