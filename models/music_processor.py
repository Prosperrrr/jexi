import os
import json
import librosa
import torch
import whisper
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
import torchaudio
from pathlib import Path
import uuid
from datetime import datetime

class MusicProcessor:
    """
    Processes music files: separates stems, transcribes lyrics, analyzes audio
    """
    
    def __init__(self):
        print("Loading Demucs model (this may take a minute)...")
        self.demucs_model = get_model('htdemucs_6s')  # 6 stems model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.demucs_model.to(self.device)
        
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")  # Use base for speed
        
        self.sample_rate = 44100
        self.processed_dir = "processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Progress tracking
        self.current_progress = {}  # Store progress for each job_id
        
    def process(self, audio_path, job_id):
        """
        Main processing function - separates stems, transcribes, analyzes
        
        Args:
            audio_path (str): Path to uploaded audio file
            job_id (str): Unique job identifier
            
        Returns:
            dict: Processing results with metadata
        """
        try:
            print(f"\n{'='*50}")
            print(f"PROCESSING JOB: {job_id}")
            print(f"File: {audio_path}")
            print(f"{'='*50}\n")
            
            # Initialize progress
            self._update_progress(job_id, 0, "Starting processing...")
            
            # Create job directory
            job_dir = os.path.join(self.processed_dir, job_id)
            stems_dir = os.path.join(job_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            
            # Step 1: Separate stems (longest step - 70% of time)
            print("Step 1/3: Separating stems with Demucs...")
            self._update_progress(job_id, 10, "Separating audio stems...")
            stem_paths = self.separate_stems(audio_path, stems_dir, job_id)
            self._update_progress(job_id, 70, "Stems separated successfully!")
            print(" Stems separated successfully!")
            
            # Step 2: Transcribe lyrics from vocals (20% of time)
            print("\nStep 2/3: Transcribing lyrics with Whisper...")
            self._update_progress(job_id, 75, "Transcribing lyrics...")
            vocals_path = stem_paths['vocals']['path'] if isinstance(stem_paths['vocals'], dict) else stem_paths['vocals']
            lyrics = self.transcribe_lyrics(vocals_path)
            self._update_progress(job_id, 90, "Lyrics transcribed!")
            print(" Lyrics transcribed!")
            
            # Step 3: Analyze audio (10% of time)
            print("\nStep 3/3: Analyzing audio...")
            self._update_progress(job_id, 95, "Analyzing audio properties...")
            analysis = self.analyze_audio(audio_path)
            self._update_progress(job_id, 100, "Processing complete!")
            print("Analysis complete!")
            
            # Compile metadata
            metadata = {
                "job_id": job_id,
                "filename": os.path.basename(audio_path),
                "status": "completed",
                "content_type": "music",
                "key": analysis['key'],
                "bpm": analysis['bpm'],
                "duration": analysis['duration'],
                "sample_rate": analysis['sample_rate'],
                "lyrics": lyrics,
                "stems": {
                    name: {
                        "path": info["path"],
                        "active": info["active"]
                    } for name, info in stem_paths.items()
                },
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save metadata
            self.save_metadata(job_id, metadata)
            
            # Clear progress tracking
            self._clear_progress(job_id)
            
            print(f"\n{'='*50}")
            print(f"JOB COMPLETED: {job_id}")
            print(f"{'='*50}\n")
            
            return metadata
            
        except Exception as e:
            print(f" Error processing job {job_id}: {e}")
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
    
    def separate_stems(self, audio_path, output_dir, job_id=None):
        """
        Separate audio into 6 stems using Demucs
        
        Returns:
            dict: Paths to each stem file with activity status
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # FIX: Convert mono to stereo if needed
        if wav.shape[0] == 1:
            # Duplicate mono channel to create stereo
            wav = wav.repeat(2, 1)
            print("  ℹnfo: Converted mono audio to stereo")
        
        # Resample if needed (Demucs expects 44.1kHz)
        if sr != self.demucs_model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, self.demucs_model.samplerate)
        
        # Convert to correct format for Demucs (add batch dimension)
        wav = wav.to(self.device)
        
        # Apply Demucs model
        print("  Running Demucs separation (this takes 3-5 minutes)...")
        if job_id:
            self._update_progress(job_id, 20, "Processing audio with Demucs AI...")
        
        with torch.no_grad():
            sources = apply_model(self.demucs_model, wav[None], device=self.device)[0]
        
        if job_id:
            self._update_progress(job_id, 60, "Saving separated stems...")
        
        # Save each stem and check if it's active
        stem_names = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        stem_paths = {}
        
        for i, name in enumerate(stem_names):
            stem_path = os.path.join(output_dir, f"{name}.wav")
            
            # Save audio
            save_audio(sources[i], stem_path, self.demucs_model.samplerate)
            
            # Check if stem is active (has actual content vs silence)
            is_active = self._check_stem_activity(sources[i])
            
            stem_paths[name] = {
                "path": stem_path,
                "active": is_active
            }
            
            status = "✓ Active" if is_active else "○ Silent/Minimal"
            print(f"  {status}: {name}.wav")
        
        return stem_paths
    
    def _check_stem_activity(self, stem_tensor):
        """
        Check if a stem has meaningful audio content
        
        Args:
            stem_tensor: Audio tensor [channels, samples]
            
        Returns:
            bool: True if stem is active, False if mostly silent
        """
        # Calculate RMS energy
        rms = torch.sqrt(torch.mean(stem_tensor ** 2))
        
        # Threshold: if RMS > 0.01, consider it active
        # (This is a reasonable threshold for music)
        threshold = 0.01
        
        return rms.item() > threshold
    
    def transcribe_lyrics(self, vocals_path):
        """
        Transcribe lyrics from vocals stem using Whisper
        
        Returns:
            dict: Transcribed lyrics with timestamps
        """
        try:
            # Get full transcription with word-level timestamps
            result = self.whisper_model.transcribe(vocals_path)
            
            # Extract plain text
            plain_lyrics = result.get('text', '').strip()
            
            # Extract segments with timestamps
            lyrics_with_time = []
            for segment in result.get('segments', []):
                lyrics_with_time.append({
                    "start": round(segment['start'], 2),  # Start time in seconds
                    "end": round(segment['end'], 2),      # End time in seconds
                    "text": segment['text'].strip()
                })
            
            if not plain_lyrics:
                return {
                    "plain": "No lyrics detected (instrumental or unclear vocals)",
                    "timestamped": []
                }
            
            return {
                "plain": plain_lyrics,
                "timestamped": lyrics_with_time
            }
            
        except Exception as e:
            print(f"  Warning: Could not transcribe lyrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                "plain": "Transcription failed",
                "timestamped": []
            }
    
    def analyze_audio(self, audio_path):
        """
        Analyze audio for key, BPM, duration
        
        Returns:
            dict: Audio analysis data
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Get duration
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{minutes}:{seconds:02d}"
        
        # Detect tempo/BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo) if not isinstance(tempo, (list, tuple)) else int(tempo[0])
        
        # Detect key (simplified - uses chroma features)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_index = chroma.mean(axis=1).argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        
        # Determine if major or minor (basic heuristic)
        # This is simplified - proper key detection needs more analysis
        key_full = f"{key} major"  # Placeholder
        
        return {
            "key": key_full,
            "bpm": bpm,
            "duration": duration_str,
            "sample_rate": sr
        }
    
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
        """
        Get processing status for a job
        
        Returns:
            dict: Status information
        """
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