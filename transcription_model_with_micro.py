# @title Run audio script
## script created by Eric Lam
AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);

function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""
import io
import wave
from base64 import b64decode

import ffmpeg
import numpy as np
from google.colab.output import eval_js
from IPython.display import HTML
from scipy.io.wavfile import read as wav_read

import os
import speech_recognition as sr

from google.colab import drive
drive.mount('/content/drive')

def write_wav(f, sr, x, normalized=False):
    f = wave.open(f, "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)

    wave_data = x.astype(np.short)
    f.writeframes(wave_data.tobytes())
    f.close()

def get_audio():
    # call microphone
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(",")[1])

    process = (
        ffmpeg.input("pipe:0")
        .output("pipe:1", format="wav")
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    output, _ = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    riff = output[:4] + bytes(b) + output[8:]
    sr, audio = wav_read(io.BytesIO(riff))
    # save to Google Drive
    human_sound_file = "/content/drive/MyDrive/saved_records/demo.wav"  # Change the path as needed
    write_wav(human_sound_file, sr, audio)

get_audio()

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
    directory_path = '/content/drive/MyDrive/saved_records'  # Replace with the actual directory path containing audio files.

    for root, _, files in os.walk(directory_path):
        for filename in files:
            audio_file = os.path.join(root, filename)
            print(f"Processing audio file: {audio_file}")

            try:
                transcription = audio_to_text_arabic(audio_file)
                print("Transcription:", transcription)

                # Create a folder for each audio file to store its transcription in a text file
                folder_name = os.path.splitext(filename)[0]
                folder_path = os.path.join('/content/drive/MyDrive/tanscriptions_records', folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Save the transcription in a text file inside the folder
                text_file_path = os.path.join(folder_path, 'transcription.txt')
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(transcription)
            except Exception as e:
                print(f"Error processing audio file: {audio_file}")
                print(f"Error message: {e}")

if __name__ == "__main__":
    main()
