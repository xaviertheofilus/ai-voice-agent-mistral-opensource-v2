# [file name]: stt_processor.py
import os
import io
import tempfile
import logging
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class OptimizedSTTProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        os.makedirs("models", exist_ok=True)
        
        try:
            logger.info(f"Loading faster-whisper model: {model_size} on {device}")
            
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root="./models",
                cpu_threads=4
            )
            
            logger.info("Faster-whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {str(e)}")
            raise e
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        try:
            if self.model is None:
                raise Exception("STT model not initialized")
            
            # Convert to mono 16kHz with minimal processing
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
                
            try:
                audio.export(tmp_name, format="wav")
                
                # Use faster settings for real-time transcription
                segments, info = self.model.transcribe(
                    tmp_name,
                    language="auto",
                    task="transcribe",
                    beam_size=3,           # Better accuracy while maintaining speed
                    best_of=2,             # Better accuracy
                    temperature=0.2,       # Slight randomness for more natural results
                    condition_on_previous_text=True,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.5,  # More sensitive to speech
                    initial_prompt=None,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,  # More sensitive
                        speech_pad_ms=200
                    )
                )
                
                transcription = ""
                for segment in segments:
                    transcription += segment.text
                
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                
            transcription = transcription.strip()
            logger.info(f"Transcription completed: '{transcription[:50]}...'")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return "Maaf, tidak dapat memproses audio. Silakan coba lagi."
    
    def transcribe_realtime(self, audio_chunks):
        try:
            accumulated_audio = AudioSegment.empty()
            
            for chunk in audio_chunks:
                chunk_audio = AudioSegment.from_file(io.BytesIO(chunk))
                accumulated_audio += chunk_audio
                
                if len(accumulated_audio) >= 2000:  # Process every 2 seconds
                    partial_transcript = self.transcribe_audio(
                        accumulated_audio.export(format="wav").read()
                    )
                    
                    if partial_transcript and partial_transcript.strip():
                        yield partial_transcript
                    
                    accumulated_audio = accumulated_audio[-500:]  # Keep last 500ms for context
                    
        except Exception as e:
            logger.error(f"Error in real-time transcription: {str(e)}")
            yield "Error in transcription"
    
    def is_speech(self, audio_data: bytes) -> bool:
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            samples = np.array(audio.get_array_of_samples())
            
            if len(samples) > 0:
                samples = samples.astype(np.float32) / np.max(np.abs(samples))
                rms = np.sqrt(np.mean(samples**2))
                speech_threshold = 0.02  # More sensitive threshold
                return rms > speech_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error in speech detection: {str(e)}")
            return True
    
    def get_model_info(self) -> dict:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": self.model is not None
        }