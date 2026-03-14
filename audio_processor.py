import base64
from gtts import gTTS
import os

class AudioProcessor:
    def __init__(self):
        self.temp_audio_file = "temp_response.mp3"

    def text_to_speech_html(self, text: str, lang: str = 'en') -> str:
        """
        Converts text to speech using Google TTS, saves it as an MP3, 
        and returns the HTML markup to autoplay the audio in the Streamlit app.
        """
        # Create TTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(self.temp_audio_file)

        # Read the audio file
        with open(self.temp_audio_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
        # Create an invisible HTML audio tag that plays automatically
        md = f"""
            <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        
        # Cleanup temp file
        if os.path.exists(self.temp_audio_file):
            os.remove(self.temp_audio_file)
            
        return md
