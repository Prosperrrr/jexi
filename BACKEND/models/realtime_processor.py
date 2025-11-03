import numpy as np
import torch
from df.enhance import enhance, init_df
import base64
import io
import soundfile as sf

class RealtimeProcessor:
    """
    Real-time audio processing for live streams
    """
    
    def __init__(self):
        print("Initializing real-time processor...")
        
        # Load DeepFilterNet for real-time noise reduction
        print("  Loading DeepFilterNet for real-time noise reduction...")
        self.df_model, self.df_state, _ = init_df()
        
        self.sample_rate = 16000  # Standard for real-time
        
        print("Real-time processor ready!")
    
    def process_audio_chunk(self, audio_data):
        """
        Process a single audio chunk with DeepFilterNet
        
        Args:
            audio_data: Base64 encoded audio or numpy array
            
        Returns:
            Base64 encoded clean audio
        """
        try:
            # Decode if base64
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            else:
                audio_array = audio_data
                sr = self.sample_rate
            
            # Ensure correct sample rate
            if sr != self.df_state.sr():
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.df_state.sr())
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Enhance with DeepFilterNet
            with torch.no_grad():
                enhanced = enhance(self.df_model, self.df_state, audio_tensor)
            
            # Convert back to numpy
            clean_audio = enhanced.squeeze(0).numpy()
            
            # Encode to base64 for transmission
            buffer = io.BytesIO()
            sf.write(buffer, clean_audio, self.df_state.sr(), format='WAV')
            buffer.seek(0)
            clean_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return clean_audio_b64
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_audio_stream(self, audio_stream):
        """
        Process continuous audio stream
        Generator function for streaming
        
        Args:
            audio_stream: Generator yielding audio chunks
            
        Yields:
            Clean audio chunks
        """
        for chunk in audio_stream:
            clean_chunk = self.process_audio_chunk(chunk)
            if clean_chunk:
                yield clean_chunk