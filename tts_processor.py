# [file name]: tts_processor.py
import os
import io
import tempfile
import logging
import asyncio
from typing import Optional, Dict, List
from pydub import AudioSegment
import numpy as np
import re

logger = logging.getLogger(__name__)

class OptimizedTTSProcessor:
    def __init__(self, use_neural=True, voice_quality="high"):
        self.use_neural = use_neural
        self.voice_quality = voice_quality
        self.available_engines = []
        self.primary_engine = None
        self.fallback_engines = []
        
        self.voice_config = {
            "id": {
                "neural_voice": "id-ID-GadisNeural",
                "standard_voice": "id-ID-ArdiNeural",
                "speed": 1.0,
                "pitch": 1.0
            },
            "en": {
                "neural_voice": "en-US-AriaNeural", 
                "standard_voice": "en-US-DavisNeural",
                "speed": 1.0,
                "pitch": 1.0
            }
        }
        
        self._init_engines()
    
    def _init_engines(self):
        try:
            import edge_tts
            self.available_engines.append("edge_tts")
            if not self.primary_engine:
                self.primary_engine = "edge_tts"
            logger.info("Edge-TTS initialized successfully")
        except ImportError:
            logger.warning("Edge-TTS not available")
        
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self._configure_pyttsx3()
            self.available_engines.append("pyttsx3")
            if not self.primary_engine:
                self.primary_engine = "pyttsx3"
            logger.info("pyttsx3 initialized successfully")
        except Exception as e:
            logger.warning(f"pyttsx3 failed to initialize: {e}")
        
        self.available_engines.append("gtts")
        if not self.primary_engine:
            self.primary_engine = "gtts"
        
        logger.info(f"Available TTS engines: {self.available_engines}")
        logger.info(f"Primary engine: {self.primary_engine}")
    
    def _configure_pyttsx3(self):
        try:
            self.pyttsx3_engine.setProperty('rate', 170)  # Slightly faster for more natural speech
            self.pyttsx3_engine.setProperty('volume', 0.95)
            
            voices = self.pyttsx3_engine.getProperty('voices')
            
            # Prefer natural sounding voices
            preferred_voices = []
            for voice in voices:
                voice_name = voice.name.lower()
                if any(keyword in voice_name for keyword in ['natural', 'premium', 'zira', 'david', 'gadis']):
                    preferred_voices.append(voice)
            
            if preferred_voices:
                self.pyttsx3_engine.setProperty('voice', preferred_voices[0].id)
                logger.info(f"Selected high-quality voice: {preferred_voices[0].name}")
                    
        except Exception as e:
            logger.warning(f"Error configuring pyttsx3: {e}")
    
    def _add_prosody(self, text: str) -> str:
        """Add SSML prosody tags to make speech more natural"""
        # Add pauses for punctuation
        text = re.sub(r'([.!?])', r'\1<break time="300ms"/>', text)
        text = re.sub(r'([,;:])', r'\1<break time="150ms"/>', text)
        
        # Emphasize important words (questions, exclamations)
        if '?' in text:
            text = f'<prosody rate="95%">{text}</prosody>'
        elif '!' in text:
            text = f'<prosody rate="105%" volume="loud">{text}</prosody>'
        else:
            # Vary pitch and rate slightly for more natural sound
            text = f'<prosody rate="98%" pitch="+2%">{text}</prosody>'
            
        return text
    
    def synthesize_speech(self, text: str, language: str = "id", voice_type: str = "neural") -> Optional[bytes]:
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        # Add prosody for more natural speech
        text_with_prosody = self._add_prosody(text)
        
        for engine in [self.primary_engine] + [e for e in self.available_engines if e != self.primary_engine]:
            try:
                if engine == "edge_tts":
                    return asyncio.run(self._synthesize_edge_tts(text_with_prosody, language, voice_type))
                elif engine == "pyttsx3":
                    return self._synthesize_pyttsx3(text, language)
                elif engine == "gtts":
                    return self._synthesize_gtts(text, language)
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
                continue
        
        logger.error("All TTS engines failed")
        return None
    
    async def _synthesize_edge_tts(self, text: str, language: str, voice_type: str) -> bytes:
        import edge_tts
        
        voice_key = "neural_voice" if voice_type == "neural" else "standard_voice"
        voice = self.voice_config.get(language, self.voice_config["id"])[voice_key]
        
        # Wrap text in SSML for better control
        ssml_text = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
            <voice name="{voice}">
                {text}
            </voice>
        </speak>
        """
        
        communicate = edge_tts.Communicate(ssml_text, voice)
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio = audio.set_frame_rate(22050).set_channels(1)
        
        audio = self._enhance_audio(audio)
        
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        
        logger.info("Speech synthesized with Edge-TTS")
        return buffer.getvalue()
    
    def _synthesize_pyttsx3(self, text: str, language: str) -> bytes:
        tmp_name = None
        try:
            # Create a temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            
            # Set voice properties
            if hasattr(self.pyttsx3_engine, 'setProperty'):
                voices = self.pyttsx3_engine.getProperty('voices')
                for voice in voices:
                    if language == "id" and "indonesia" in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
                    elif language == "en" and "english" in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
            
            # Save to file and wait
            self.pyttsx3_engine.save_to_file(text, tmp_name)
            self.pyttsx3_engine.runAndWait()
            
            # Read and process the audio
            audio = AudioSegment.from_wav(tmp_name)
            audio = audio.set_frame_rate(22050).set_channels(1)
            
            audio = self._enhance_audio(audio)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            
            logger.info("Speech synthesized with pyttsx3")
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            raise
        finally:
            # Clean up temporary file if it exists
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
    
    def _synthesize_gtts(self, text: str, language: str) -> bytes:
        tmp_name = None
        try:
            from gtts import gTTS
            
            lang_code = "id" if language == "id" else "en"
            tld = "co.id" if language == "id" else "com"
            
            # Create a temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_name = tmp.name
            
            tts = gTTS(
                text=text,
                lang=lang_code,
                tld=tld,
                slow=False,
                lang_check=False
            )
            
            tts.save(tmp_name)
            
            # Read and process the audio
            audio = AudioSegment.from_mp3(tmp_name)
            audio = audio.set_frame_rate(22050).set_channels(1)
            
            audio = self._enhance_audio(audio)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            
            logger.info("Speech synthesized with gTTS")
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise
        finally:
            # Clean up temporary file if it exists
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
    
    def _enhance_audio(self, audio: AudioSegment) -> AudioSegment:
        try:
            # Normalize audio
            audio = audio.normalize()
            
            # Apply EQ for better clarity
            audio = audio.high_pass_filter(100)  # Remove low-frequency noise
            audio = audio.low_pass_filter(8000)  # Smooth high frequencies
            
            # Add slight reverb for natural sound
            if self.voice_quality == "high":
                # Simple reverb effect
                echo = audio - 8  # Quieter echo
                combined = audio.overlay(echo, position=50)  # 50ms delay
                audio = combined
            
            return audio
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio
    
    def get_available_voices(self, language: str = "id") -> List[Dict]:
        voices = []
        
        if "edge_tts" in self.available_engines:
            if language == "id":
                voices.extend([
                    {"engine": "edge_tts", "voice": "id-ID-ArdiNeural", "gender": "male", "quality": "neural"},
                    {"engine": "edge_tts", "voice": "id-ID-GadisNeural", "gender": "female", "quality": "neural"}
                ])
            else:
                voices.extend([
                    {"engine": "edge_tts", "voice": "en-US-AriaNeural", "gender": "female", "quality": "neural"},
                    {"engine": "edge_tts", "voice": "en-US-DavisNeural", "gender": "male", "quality": "neural"},
                    {"engine": "edge_tts", "voice": "en-US-AnaNeural", "gender": "female", "quality": "neural"}
                ])
        
        return voices
    
    def get_engine_status(self) -> Dict:
        return {
            "primary_engine": self.primary_engine,
            "available_engines": self.available_engines,
            "voice_quality": self.voice_quality,
            "neural_voices": self.use_neural
        }