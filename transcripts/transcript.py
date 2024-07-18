import whisper
import yt_dlp
from pydub import AudioSegment
import os
import numpy as np
import torch

def download_audio_from_youtube(url, output_path="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'nocheckcertificate': True  # Bypass SSL verification
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Correcting the file name if double extension occurs
    if not os.path.exists(output_path) and os.path.exists(output_path + ".mp3"):
        os.rename(output_path + ".mp3", output_path)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def convert_audio_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def save_transcription(transcription, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(transcription)

def main(youtube_url, output_txt_path):
    mp3_path = "audio.mp3"
    wav_path = "audio.wav"

    try:
        download_audio_from_youtube(youtube_url, mp3_path)
        convert_audio_to_wav(mp3_path, wav_path)
        transcription = transcribe_audio(wav_path)
        save_transcription(transcription, output_txt_path)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return output_txt_path

if __name__ == "__main__":
    youtube_url = input("Enter the YouTube video URL: ")
    output_txt_path = input("Enter the name for the transcription file (with .txt extension): ")
    txt_file_path = main(youtube_url, output_txt_path)
    print(f"Transcription saved to {txt_file_path}")
