import librosa
import numpy as np
from pydub import AudioSegment
import os
import tempfile
import time

class AudioClassifier:
    """
    Classifies audio content as either 'music' or 'speech'
    Uses spectral features and rhythm analysis
    """
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate for analysis
    
    def classify(self, audio_path):
        """
        Classify audio file as 'music' or 'speech' using three-point sampling
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: 'music' or 'speech'
        """
        try:
            # Convert non-WAV formats to temporary WAV file
            if not audio_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(audio_path)
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_wav.close()  
                audio.export(temp_wav.name, format='wav')
                audio_path_to_load = temp_wav.name
                cleanup_temp = True
            else:
                audio_path_to_load = audio_path
                cleanup_temp = False
            
            # Get total duration
            duration = librosa.get_duration(path=audio_path_to_load)
            
            # Determine sampling points
            clip_length = 30
            if duration < clip_length:
                # Short audio - analyze whole thing
                points_to_check = [0]
            elif duration < 90:
                # Medium audio - check beginning and middle
                points_to_check = [0, duration / 2 - (clip_length / 2)]
            else:
                # Long audio - check beginning, middle, end (3-point)
                points_to_check = [
                    0,  # Beginning
                    duration / 2 - (clip_length / 2),  # Middle
                    max(0, duration - clip_length)  # End
                ]
            
            decisions = []
            print("\n=== AUDIO CLASSIFICATION DEBUG ===")
            print(f"File: {audio_path}")
            print(f"Duration: {duration:.1f}s")
            print(f"Checking {len(points_to_check)} point(s)...\n")
            
            # Analyze each point
            for i, start_time in enumerate(points_to_check):
                print(f"--- Sample {i+1} (at {int(start_time)}s) ---")
                
                # Load clip
                y, sr = librosa.load(audio_path_to_load, sr=self.sample_rate, 
                                    offset=start_time, duration=clip_length)
                
                # Extract features
                features = self._extract_features(y, sr)
                
                # Show key features
                print(f"  beat_strength: {features['beat_strength']:.3f}")
                print(f"  tempo: {features['tempo']:.1f}")
                print(f"  zcr_mean: {features['zcr_mean']:.3f}")
                
                # Make decision
                decision = self._make_decision(features)
                decisions.append(decision)
                print(f"  → {decision}\n")
            
            # Cleanup temp file
            if cleanup_temp:
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass
            
            # Vote on final classification
            from collections import Counter
            vote_count = Counter(decisions)
            final_decision = vote_count.most_common(1)[0][0]
            
            print(f"Votes: {dict(vote_count)}")
            print(f"🎯 Final Classification: {final_decision.upper()}")
            print("===================================\n")
            
            return final_decision
            
        except Exception as e:
            print(f"Error classifying audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_features(self, y, sr):
        """Extract audio features for classification"""
        features = {}
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if isinstance(tempo, np.ndarray) else tempo
        beat_strength = len(beats) / (len(y) / sr) if len(y) > 0 else 0
        features['beat_strength'] = beat_strength
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        rms = librosa.feature.rms(y=y)[0]
        features['rms_std'] = np.std(rms)
        
        return features
    
    def _make_decision(self, features):
        """
        Decision logic based on extracted features (IMPROVED THRESHOLDS)
        
        Music typically has:
        - Very strong, regular beat patterns
        - Higher spectral rolloff (fuller frequency spectrum)
        - Lower ZCR (smoother, more tonal)
        - More consistent energy
        
        Speech typically has:
        - Higher zero crossing rate (more abrupt transitions)
        - Lower spectral rolloff (concentrated in speech frequencies)
        - Irregular or weak beats
        - More energy variation (pauses between words)
        """
        music_score = 0
        speech_score = 0
        
        # Zero Crossing Rate - STRONG speech indicator
        if features['zcr_mean'] > 0.15:
            speech_score += 3
        elif features['zcr_mean'] < 0.08:
            music_score += 3  # Increased from 2
        
        # Beat strength - STRICTER for music (main improvement!)
        if features['beat_strength'] > 2.0:  # Raised from 1.5
            music_score += 4  # Increased weight
        elif features['beat_strength'] < 0.6:  # Stricter threshold
            speech_score += 3
        elif features['beat_strength'] < 1.0:  # Middle ground
            speech_score += 1
        
        # Spectral Rolloff - Music uses fuller frequency spectrum
        if features['spectral_rolloff_mean'] > 6000:  # Raised from 5000
            music_score += 2
        elif features['spectral_rolloff_mean'] < 3000:  # Stricter
            speech_score += 2
        
        # Spectral Centroid - Speech typically has lower centroid
        if features['spectral_centroid_mean'] < 1800:  # Stricter
            speech_score += 2
        elif features['spectral_centroid_mean'] > 3500:  # Higher threshold
            music_score += 2
        
        # Spectral Centroid Variance
        if features['spectral_centroid_std'] > 1200:
            if features['zcr_mean'] > 0.15:
                speech_score += 1
            else:
                music_score += 1
        
        # Spectral Bandwidth
        if features['spectral_bandwidth_mean'] > 2000:  # Raised
            music_score += 1
        elif features['spectral_bandwidth_mean'] < 1200:  # Stricter
            speech_score += 1
        
        # RMS Energy variation - Speech has more pauses
        if features['rms_std'] < 0.03:  # Stricter
            music_score += 2
        elif features['rms_std'] > 0.07:  # Raised
            speech_score += 2
        
        # MFCC Std
        if features['mfcc_std'] > 100:
            speech_score += 2
        elif features['mfcc_std'] > 40:
            music_score += 1
        
        # Tempo check - Very extreme tempos unlikely for music
        if features['tempo'] < 60 or features['tempo'] > 200:
            speech_score += 1
        elif 80 <= features['tempo'] <= 180:  # Typical music range
            music_score += 1
        
        print(f"  Music: {music_score} | Speech: {speech_score}")
        
        if music_score > speech_score:
            return "music"
        else:
            return "speech"
    
    def get_confidence(self, audio_path):
        """
        Get classification with confidence score
        
        Returns:
            tuple: (classification, confidence_percentage)
        """
        classification = self.classify(audio_path)
        # TODO: Implement proper confidence scoring based on score margins
        return classification, 75.0