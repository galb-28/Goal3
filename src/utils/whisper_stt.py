"""Whisper speech-to-text integration."""

import os
import tempfile
from pathlib import Path
from typing import Optional
import whisper
from dotenv import load_dotenv

load_dotenv()


class WhisperSTT:
    """Whisper speech-to-text transcription."""
    
    def __init__(self, model_size: Optional[str] = None):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       Defaults to value from .env or 'base'
        """
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "base")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            print(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'en' for English)
            
        Returns:
            Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False  # Use FP32 for better compatibility
            )
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, language: str = "en") -> str:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio file content as bytes
            language: Language code (default: 'en' for English)
            
        Returns:
            Transcribed text
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            return self.transcribe_audio(tmp_path, language)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Global instance to avoid reloading model
_whisper_instance = None


def get_whisper_instance() -> WhisperSTT:
    """Get or create global Whisper instance."""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperSTT()
    return _whisper_instance


def transcribe(audio_input, language: str = "en") -> str:
    """
    Convenience function to transcribe audio.
    
    Args:
        audio_input: Path to audio file or bytes
        language: Language code
        
    Returns:
        Transcribed text
    """
    stt = get_whisper_instance()
    
    if isinstance(audio_input, bytes):
        return stt.transcribe_audio_bytes(audio_input, language)
    else:
        return stt.transcribe_audio(audio_input, language)
