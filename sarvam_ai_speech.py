import requests
import base64
from dotenv import load_dotenv
import os

load_dotenv()

def speech_to_text(base64_audio,language_code="ta-IN"):
    """
    Sends a base64-encoded audio to the speech-to-text API and returns the transcribed text in the same language as the audio.
    
    Link: https://sarvam.ai/docs/speech-to-text

    Parameters:
    - base64_audio (str): Base64-encoded audio content.
    - language_code (str): Language code for transcription (default is "ta-IN").
        - Avalailable language codes: hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, gu-IN
    - with_timestamps (bool): Whether to include timestamps in the transcription (default is False).

    Returns:
    - str: Transcribed text or error message from the API.
    """
    
    url = "https://api.sarvam.ai/speech-to-text"

    data = {
        "model": "saarika:v1",
        "language_code": language_code,
        "with_timestamps": "true"
    }

    files = {
        "file": ("input.wav", base64.b64decode(base64_audio), "audio/wav")
    }

    headers = {
    'api-subscription-key': os.getenv('SARVAM_API_KEY')
    }

    try:
        response = requests.post(url, data=data, files=files,headers=headers)
        return response.json()['transcript']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def speech_to_text_translate(base64_audio):
    """
    Sends a base64-encoded audio file to the speech-to-text API and returns the transcribed text in english.
    
    Link: https://sarvam.ai/docs/speech-to-text-translate

    Parameters:
    - base64_audio (str): Base64-encoded audio content.

    Returns:
    - str: Transcribed text or error message from the API.
    """
    
    url = "https://api.sarvam.ai/speech-to-text-translate"

    data = {
        "model": "saaras:v1",
        "prompt":"The audio file will always be in Tamil language.",
    }

    files = {
        "file": ("input.wav", base64.b64decode(base64_audio), "audio/wav")
    }

    headers = {
    'api-subscription-key': os.getenv('SARVAM_API_KEY')
    }

    try:
        response = requests.post(url, data=data, files=files,headers=headers)
        return response.json()['transcript']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Sample usage
file_path = "sample_input2.wav"  # 'Am i in the right place?'

base64_audio = ""
with open(file_path, "rb") as audio_file:
    base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

print(speech_to_text(base64_audio))
print(speech_to_text_translate(base64_audio))