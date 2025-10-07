import librosa
import numpy as np
from pydub import AudioSegment
import os
import tempfile

class AudioClassifier:
    """
    Classifies audio content as either 'music' or 'speech'
    Uses spectral features and rhythm analysis
    A heuristic-based rough set features optimization algorithm for compressed audio
    """
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate for analysis
    
    def classify(self, audio_path):
        """
        Classify audio file as 'music' or 'speech'
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: 'music' or 'speech'
        """
        try:
            # Convert M4A/other formats to WAV for librosa compatibility
            if audio_path.lower().endswith('.m4a'):
                audio = AudioSegment.from_file(audio_path, format='m4a')
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_wav.close()  # Close the file handle before export
                audio.export(temp_wav.name, format='wav')
                audio_path_to_load = temp_wav.name
                cleanup_temp = True
            else:
                audio_path_to_load = audio_path
                cleanup_temp = False
            
            # Load audio file
            y, sr = librosa.load(audio_path_to_load, sr=self.sample_rate, duration=30)  # Analyze first 30 seconds
            
            # Clean up temp file if created
            if cleanup_temp:
                try:
                    os.unlink(temp_wav.name)
                except PermissionError:
                    # If still locked, try again after a brief moment
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass  # If still can't delete, OS will clean it up later
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # DEBUG: Print features to see what's happening
            print("\n=== AUDIO CLASSIFICATION DEBUG ===")
            print(f"File: {audio_path}")
            print(f"Features extracted:")
            for key, value in features.items():
                # Fix: Handle both scalar and array values
                if isinstance(value, (np.ndarray, list)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value:.4f}")
            
            # Classify based on features
            classification = self._make_decision(features)
            print(f"Classification: {classification}")
            print("===================================\n")
            
            return classification
            
        except Exception as e:
            print(f"Error classifying audio: {e}")
            import traceback
            traceback.print_exc()
            return None  # Return None instead of defaulting to speech
    
    def _extract_features(self, y, sr):
        """Extract audio features for classification"""
        features = {}
        
        # 1. Spectral Centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # 2. Zero Crossing Rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 3. Spectral Rolloff (frequency distribution)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # 4. MFCCs (timbre characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        # 5. Tempo and Beat Strength (rhythm detection)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if isinstance(tempo, np.ndarray) else tempo
        features['beat_strength'] = len(beats) / (len(y) / sr)  # Beats per second
        
        # 6. Spectral Bandwidth (frequency range)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # 7. RMS Energy (loudness variation)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_std'] = np.std(rms)
        
        return features
    
    def _make_decision(self, features):
        """
        Decision logic based on extracted features
        
        Music typically has:
        - Very strong, regular beat patterns (beat_strength > 1.5)
        - Higher spectral rolloff (fuller frequency spectrum)
        - Lower ZCR (smoother, more tonal)
        - More consistent energy (less silence/pauses)
        
        Speech typically has:
        - Higher zero crossing rate (more abrupt transitions)
        - Lower spectral rolloff (concentrated in speech frequencies)
        - Irregular or weak beats
        - More energy variation (pauses between words/sentences)
        - Lower spectral centroid (less "bright")
        """
        
        music_score = 0
        speech_score = 0
        
        # Zero Crossing Rate - STRONG speech indicator
        # Speech has more abrupt transitions between sounds
        if features['zcr_mean'] > 0.15:  # High ZCR = likely speech
            speech_score += 3
        elif features['zcr_mean'] < 0.08:  # Low ZCR = likely music (smoother)
            music_score += 2
        
        # Beat strength analysis - Music has VERY strong, regular beats
        if features['beat_strength'] > 1.5:  # Strong regular beats = music
            music_score += 3
        elif features['beat_strength'] < 0.8:  # Weak/irregular = speech
            speech_score += 2
        else:
            # Middle ground - use other features
            pass
        
        # Spectral Rolloff - Music uses fuller frequency spectrum
        if features['spectral_rolloff_mean'] > 5000:  # High rolloff = music
            music_score += 2
        elif features['spectral_rolloff_mean'] < 3500:  # Low rolloff = speech
            speech_score += 2
        
        # Spectral Centroid - Speech typically has lower centroid
        if features['spectral_centroid_mean'] < 2000:  # Lower = speech
            speech_score += 2
        elif features['spectral_centroid_mean'] > 3000:  # Higher = music
            music_score += 2
        
        # Spectral Centroid Variance, But high variance can mean speech too!
        # Speech can be very dynamic (shouting, whispering, etc.)
        if features['spectral_centroid_std'] > 1200:  # Very high variance
            # Could be either - check with other features
            if features['zcr_mean'] > 0.15:  # If also high ZCR, likely speech
                speech_score += 1
            else:
                music_score += 1
        
        # Spectral Bandwidth
        if features['spectral_bandwidth_mean'] > 1800:
            music_score += 1
        elif features['spectral_bandwidth_mean'] < 1400:
            speech_score += 1
        
        # RMS Energy variation - Speech has more pauses (higher variation)
        if features['rms_std'] < 0.04:  # Low variation = consistent = music
            music_score += 2
        elif features['rms_std'] > 0.06:  # High variation = pauses = speech
            speech_score += 1
        
        # MFCC Std - Extremely high values can indicate speech dynamics
        if features['mfcc_std'] > 100:  # Very high = likely speech
            speech_score += 2
        elif features['mfcc_std'] > 40:  # High = likely music
            music_score += 1
        
        # Tempo check - Very extreme tempos unlikely for music
        if features['tempo'] < 60 or features['tempo'] > 200:
            speech_score += 1
        
        # Final decision
        print(f"  Music score: {music_score}")
        print(f"  Speech score: {speech_score}")
        
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
        # TODO: Implement proper confidence scoring
        # For now, return a basic confidence
        return classification, 75.0  # Placeholder confidence