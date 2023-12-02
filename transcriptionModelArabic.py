import os
import speech_recognition as sr

def audio_to_text_arabic(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="ar-AR")
        return text
    except sr.UnknownValueError:
        print(f"Speech recognition could not understand the audio: {audio_file}")
    except sr.RequestError as e:
        print(f"Error with speech recognition service; {e}")

def main():
    audio_directory = 'D:\AllArabicAudioTextLocal\AllArabicWavFilesLocal'
    text_directory = 'D:\AllArabicAudioTextLocal\AllArabicTextFilesLocal'

    os.makedirs(text_directory, exist_ok=True)  # Ensure the text_directory exists

    for root, _, files in os.walk(audio_directory):
        for filename in files:
            audio_file = os.path.join(root, filename)
            print(f"Processing audio file: {audio_file}")

            # Generate a transcription file path with the same name but .txt extension
            transcription_file_path = os.path.join(text_directory, os.path.splitext(filename)[0] + '.txt')

            # Check if a transcription file already exists for this audio file
            if os.path.exists(transcription_file_path):
                print(f"Transcription already exists for: {audio_file}")
                continue  # Skip this audio file if transcription already exists

            try:
                transcription = audio_to_text_arabic(audio_file)
                print("Transcription:", transcription)

                # Save the transcription in a text file
                with open(transcription_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(transcription)
            except Exception as e:
                print(f"Error processing audio file: {audio_file}")
                print(f"Error message: {e}")

if __name__ == "__main__":
    main()
