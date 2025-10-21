import os
import json
import librosa
import soundfile as sf
import numpy as np
import whisper
import torch
from df.enhance import enhance, init_df, load_audio, save_audio as df_save_audio
from pathlib import Path
from datetime import datetime

class SpeechProcessor:
    """
    Processes speech audio: noise reduction, transcription, enhancement
    """
    
    def __init__(self):
        print("Loading DeepFilterNet model for noise reduction...")
        self.df_model, self.df_state, _ = init_df()
        
        print("Loading Whisper model for speech...")
        self.whisper_model = whisper.load_model("base")  # Use base for speed
        
        self.sample_rate = 16000  # Standard for speech
        self.processed_dir = "processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Progress tracking
        self.current_progress = {}
        
        print("✅ Speech processor initialized!")
    
    def process(self, audio_path, job_id):
        """
        Main processing function - reduce noise, transcribe, enhance
        
        Args:
            audio_path (str): Path to uploaded audio file
            job_id (str): Unique job identifier
            
        Returns:
            dict: Processing results with metadata
        """
        try:
            print(f"\n{'='*50}")
            print(f"PROCESSING SPEECH JOB: {job_id}")
            print(f"File: {audio_path}")
            print(f"{'='*50}\n")
            
            # Initialize progress
            self._update_progress(job_id, 0, "Starting speech processing...")
            
            # Create job directory
            job_dir = os.path.join(self.processed_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            
            # Step 1: Load audio
            print("Step 1/4: Loading audio...")
            self._update_progress(job_id, 10, "Loading audio file...")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=audio, sr=sr)
            print(f"✅ Audio loaded: {duration:.1f}s at {sr}Hz")
            
            # Step 2: Noise reduction
            print("\nStep 2/4: Reducing background noise...")
            self._update_progress(job_id, 25, "Cleaning audio with AI noise reduction...")
            clean_audio = self.reduce_noise(audio, sr)
            print("✅ Noise reduced!")
            
            # Step 3: Transcription
            print("\nStep 3/4: Transcribing speech with Whisper...")
            self._update_progress(job_id, 50, "Transcribing speech to text...")
            transcript = self.transcribe_speech(clean_audio, sr)
            self._update_progress(job_id, 85, "Transcription complete!")
            print("✅ Transcription complete!")
            
            # Step 4: Save enhanced audio
            print("\nStep 4/4: Saving enhanced audio...")
            self._update_progress(job_id, 90, "Saving processed audio...")
            clean_audio_path = os.path.join(job_dir, "clean_audio.wav")
            self.save_audio(clean_audio, sr, clean_audio_path)
            print("✅ Audio saved!")
            
            # Compile metadata
            metadata = {
                "job_id": job_id,
                "filename": os.path.basename(audio_path),
                "status": "completed",
                "content_type": "speech",
                "duration": f"{int(duration // 60)}:{int(duration % 60):02d}",
                "sample_rate": sr,
                "transcript": transcript,
                "clean_audio_path": clean_audio_path,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save metadata
            self.save_metadata(job_id, metadata)
            
            # Clear progress tracking
            self._update_progress(job_id, 100, "Processing complete!")
            self._clear_progress(job_id)
            
            print(f"\n{'='*50}")
            print(f"✅ SPEECH JOB COMPLETED: {job_id}")
            print(f"{'='*50}\n")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error processing speech job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Update progress with error
            self._update_progress(job_id, -1, f"Error: {str(e)}")
            
            # Save error metadata
            error_metadata = {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_metadata(job_id, error_metadata)
            
            return error_metadata
    
    def reduce_noise(self, audio, sr):
        """
        Reduce background noise using DeepFilterNet
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Clean audio array
        """
        try:
            # DeepFilterNet expects specific sample rate
            if sr != self.df_state.sr():
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.df_state.sr())
            else:
                audio_resampled = audio
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_resampled).unsqueeze(0)
            
            # Enhance with DeepFilterNet
            enhanced = enhance(self.df_model, self.df_state, audio_tensor)
            
            # Convert back to numpy
            clean_audio = enhanced.squeeze(0).numpy()
            
            # Resample back to original sample rate if needed
            if sr != self.df_state.sr():
                clean_audio = librosa.resample(clean_audio, orig_sr=self.df_state.sr(), target_sr=sr)
            
            print(f"  DeepFilterNet noise reduction applied")
            return clean_audio
            
        except Exception as e:
            print(f"  Warning: DeepFilterNet failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Falling back to original audio")
            return audio  # Return original if noise reduction fails
    
    def transcribe_speech(self, audio, sr):
        """
        Transcribe speech to text using Whisper
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            dict: Transcript with plain text and timestamped segments
        """
        try:
            # Whisper expects float32
            audio = audio.astype(np.float32)
            
            # Transcribe
            result = self.whisper_model.transcribe(audio)
            
            # Extract plain text
            plain_text = result.get('text', '').strip()
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    "start": round(segment['start'], 2),
                    "end": round(segment['end'], 2),
                    "text": segment['text'].strip()
                })
            
            word_count = len(plain_text.split())
            
            print(f"  Transcribed {word_count} words")
            
            if not plain_text:
                return {
                    "plain": "No speech detected",
                    "segments": [],
                    "word_count": 0
                }
            
            return {
                "plain": plain_text,
                "segments": segments,
                "word_count": word_count
            }
            
        except Exception as e:
            print(f"  Warning: Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "plain": "Transcription failed",
                "segments": [],
                "word_count": 0
            }
    
    def save_audio(self, audio, sr, output_path):
        """Save processed audio to file"""
        # Normalize audio to prevent clipping
        audio_normalized = audio / np.max(np.abs(audio))
        
        # Save as WAV
        sf.write(output_path, audio_normalized, sr)
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path} ({file_size_mb:.2f} MB)")
    
    def save_metadata(self, job_id, metadata):
        """Save metadata to JSON file"""
        job_dir = os.path.join(self.processed_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_metadata(self, job_id):
        """Load metadata from JSON file"""
        metadata_path = os.path.join(self.processed_dir, job_id, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_status(self, job_id):
        """Get processing status for a job"""
        metadata = self.get_metadata(job_id)
        
        if not metadata:
            return {"status": "not_found", "message": "Job ID not found"}
        
        if metadata['status'] == 'completed':
            return {
                "status": "completed",
                "job_id": job_id,
                "message": "Processing complete"
            }
        elif metadata['status'] == 'failed':
            return {
                "status": "failed",
                "job_id": job_id,
                "error": metadata.get('error', 'Unknown error')
            }
        else:
            return {
                "status": "processing",
                "job_id": job_id,
                "message": "Still processing..."
            }
    
    def _update_progress(self, job_id, percent, message):
        """Update processing progress for a job"""
        self.current_progress[job_id] = {
            "percent": percent,
            "message": message,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _clear_progress(self, job_id):
        """Clear progress tracking after job completes"""
        if job_id in self.current_progress:
            del self.current_progress[job_id]
    
    def get_progress(self, job_id):
        """Get current progress for a job"""
        return self.current_progress.get(job_id, None)
    
    def export_transcript_txt(self, job_id):
        """Export transcript as plain text file"""
        metadata = self.get_metadata(job_id)
        if not metadata:
            return None
        
        transcript = metadata.get('transcript', {})
        plain_text = transcript.get('plain', '')
        
        # Create txt file
        job_dir = os.path.join(self.processed_dir, job_id)
        txt_path = os.path.join(job_dir, "transcript.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(plain_text)
        
        return txt_path
    
    def export_transcript_json(self, job_id):
        """Export transcript as JSON with timestamps"""
        metadata = self.get_metadata(job_id)
        if not metadata:
            return None
        
        transcript = metadata.get('transcript', {})
        
        # Create JSON file
        job_dir = os.path.join(self.processed_dir, job_id)
        json_path = os.path.join(job_dir, "transcript.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def export_transcript_srt(self, job_id):
        """Export transcript as SRT subtitle file"""
        metadata = self.get_metadata(job_id)
        if not metadata:
            return None
        
        transcript = metadata.get('transcript', {})
        segments = transcript.get('segments', [])
        
        # Create SRT file
        job_dir = os.path.join(self.processed_dir, job_id)
        srt_path = os.path.join(job_dir, "transcript.srt")
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start = self._format_timestamp_srt(segment['start'])
                end = self._format_timestamp_srt(segment['end'])
                text = segment['text']
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        return srt_path
    
    def _format_timestamp_srt(self, seconds):
        """Format timestamp for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"