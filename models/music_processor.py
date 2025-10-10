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
        print("Loading Demucs model ...")
        self.demucs_model = get_model('htdemucs_6s')  # 6 stems model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.demucs_model.to(self.device)
        
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")  # Use base for speed
        
        self.sample_rate = 44100
        self.processed_dir = "processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
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
            
            # Create job directory
            job_dir = os.path.join(self.processed_dir, job_id)
            stems_dir = os.path.join(job_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            
            # Step 1: Separate stems (longest step)
            print("Step 1/3: Separating stems with Demucs...")
            stem_paths = self.separate_stems(audio_path, stems_dir)
            print("✅ Stems separated successfully!")
            
            # Step 2: Transcribe lyrics from vocals
            print("\nStep 2/3: Transcribing lyrics with Whisper...")
            lyrics = self.transcribe_lyrics(stem_paths['vocals'])
            print("✅ Lyrics transcribed!")
            
            # Step 3: Analyze audio
            print("\nStep 3/3: Analyzing audio...")
            analysis = self.analyze_audio(audio_path)
            print("✅ Analysis complete!")
            
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
                "stems": stem_paths,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save metadata
            self.save_metadata(job_id, metadata)
            
            print(f"\n{'='*50}")
            print(f"✅ JOB COMPLETED: {job_id}")
            print(f"{'='*50}\n")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error processing job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error metadata
            error_metadata = {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_metadata(job_id, error_metadata)
            
            return error_metadata
    
    def separate_stems(self, audio_path, output_dir):
        """
        Separate audio into 6 stems using Demucs
        
        Returns:
            dict: Paths to each stem file
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Resample if needed (Demucs expects 44.1kHz)
        if sr != self.demucs_model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, self.demucs_model.samplerate)
        
        # Convert to correct format for Demucs (add batch dimension)
        wav = wav.to(self.device)
        
        # Apply Demucs model
        print("  Running Demucs separation (this takes 3-5 minutes)...")
        with torch.no_grad():
            sources = apply_model(self.demucs_model, wav[None], device=self.device)[0]
        
        # Save each stem
        stem_names = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        stem_paths = {}
        
        for i, name in enumerate(stem_names):
            stem_path = os.path.join(output_dir, f"{name}.wav")
            # Save audio (Demucs outputs shape: [sources, channels, samples])
            save_audio(sources[i], stem_path, self.demucs_model.samplerate)
            stem_paths[name] = stem_path
            print(f"  ✓ Saved {name}.wav")
        
        return stem_paths
    
    def transcribe_lyrics(self, vocals_path):
        """
        Transcribe lyrics from vocals stem using Whisper
        
        Returns:
            str: Transcribed lyrics
        """
        try:
            result = self.whisper_model.transcribe(vocals_path)
            lyrics = result['text'].strip()
            
            if not lyrics:
                return "No lyrics detected (instrumental or unclear vocals)"
            
            return lyrics
            
        except Exception as e:
            print(f"  Warning: Could not transcribe lyrics: {e}")
            return "Transcription failed"
    
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