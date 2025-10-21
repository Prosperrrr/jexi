import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import csv
from pydub import AudioSegment
import tempfile
import os

class YAMNetClassifier:
    """
    Audio classifier using Google's YAMNet pre-trained model
    Maps audio to 'music' or 'speech' categories
    """
    
    def __init__(self):
        print("Loading YAMNet model (this may take a minute)...")
        # Load YAMNet model from TensorFlow Hub
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Define which YAMNet classes map to our categories
        self.music_classes = {
            'Music', 'Musical instrument', 'Plucked string instrument', 
            'Guitar', 'Electric guitar', 'Bass guitar', 'Acoustic guitar',
            'Piano', 'Keyboard (musical)', 'Drum', 'Drum kit', 'Snare drum',
            'Bass drum', 'Timpani', 'Tabla', 'Synthesizer', 'Brass instrument',
            'Trumpet', 'Trombone', 'French horn', 'Saxophone', 'Clarinet',
            'Flute', 'Violin', 'Fiddle', 'Cello', 'Double bass',
            'Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra',
            'Electronic music', 'Techno', 'House music', 'Dubstep',
            'Drum and bass', 'Hip hop music', 'Rock music', 'Heavy metal',
            'Punk rock', 'Progressive rock', 'Rock and roll', 'Psychedelic rock',
            'Country', 'Jazz', 'Blues', 'Reggae', 'Soul music', 'Funk',
            'Pop music', 'Latin music', 'Salsa music', 'Carnatic music'
        }
        
        self.speech_classes = {
            'Speech', 'Narration, monologue', 'Speech synthesizer',
            'Conversation', 'Male speech, man speaking', 'Female speech, woman speaking',
            'Child speech, kid speaking', 'Babbling', 'Baby laughter',
            'Whispering', 'Laughter', 'Shouting, screaming', 'Crying, sobbing',
            'Cheering', 'Throat clearing', 'Cough', 'Sneeze', 'Sniff',
            'Sermon', 'Podcast', 'Audiobook'
        }
        
        print("✅ YAMNet model loaded successfully!")
    
    def _load_class_names(self):
        """Load YAMNet class names"""
        class_map_path = self.model.class_map_path().numpy()
        class_names = []
        
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        
        return class_names
    
    def classify(self, audio_path):
        """
        Classify audio file as 'music' or 'speech'
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: {
                'classification': 'music' or 'speech',
                'confidence': float (0-100),
                'top_predictions': list of top 5 predictions
            }
        """
        try:
            print(f"\n{'='*60}")
            print(f"YAMNET CLASSIFICATION")
            print(f"File: {audio_path}")
            print(f"{'='*60}\n")
            
            # Convert to WAV if needed
            wav_path = self._convert_to_wav(audio_path)
            
            # Load audio with librosa (YAMNet expects 16kHz mono)
            waveform, sr = librosa.load(wav_path, sr=16000, mono=True)
            
            # Cleanup temp file if created
            if wav_path != audio_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
            
            # Run YAMNet inference
            scores, embeddings, spectrogram = self.model(waveform)
            
            # Get predictions (average scores across all time frames)
            prediction_scores = tf.reduce_mean(scores, axis=0)
            
            # Get top 5 predictions
            top_5_indices = tf.argsort(prediction_scores, direction='DESCENDING')[:5]
            
            top_predictions = []
            for i, idx in enumerate(top_5_indices):
                class_name = self.class_names[idx]
                score = float(prediction_scores[idx]) * 100
                top_predictions.append({
                    'class': class_name,
                    'confidence': round(score, 2)
                })
                print(f"{i+1}. {class_name}: {score:.2f}%")
            
            # Determine if music or speech
            result = self._categorize(prediction_scores)
            
            print(f"\n🎯 Final Classification: {result['classification'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.2f}%")
            print(f"{'='*60}\n")
            
            result['top_predictions'] = top_predictions
            
            return result
            
        except Exception as e:
            print(f"❌ Error classifying audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_to_wav(self, audio_path):
        """Convert audio to WAV format if needed"""
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        try:
            audio = AudioSegment.from_file(audio_path)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav.close()
            audio.export(temp_wav.name, format='wav')
            return temp_wav.name
        except Exception as e:
            print(f"Warning: Could not convert audio format: {e}")
            return audio_path
    
    def _categorize(self, prediction_scores):
        """
        Categorize YAMNet predictions into 'music' or 'speech'
        
        Args:
            prediction_scores: TensorFlow tensor of class scores
            
        Returns:
            dict with classification and confidence
        """
        music_score = 0.0
        speech_score = 0.0
        
        # Sum up scores for music and speech categories
        for i, score in enumerate(prediction_scores):
            class_name = self.class_names[i]
            score_value = float(score)
            
            if class_name in self.music_classes:
                music_score += score_value
            elif class_name in self.speech_classes:
                speech_score += score_value
        
        # Determine winner
        total_score = music_score + speech_score
        
        if total_score == 0:
            # Fallback to top prediction
            top_class_idx = tf.argmax(prediction_scores)
            top_class = self.class_names[top_class_idx]
            
            if any(music_word in top_class.lower() for music_word in ['music', 'singing', 'instrument']):
                return {
                    'classification': 'music',
                    'confidence': float(prediction_scores[top_class_idx]) * 100
                }
            else:
                return {
                    'classification': 'speech',
                    'confidence': float(prediction_scores[top_class_idx]) * 100
                }
        
        # Calculate confidence
        if music_score > speech_score:
            classification = 'music'
            confidence = (music_score / total_score) * 100
        else:
            classification = 'speech'
            confidence = (speech_score / total_score) * 100
        
        print(f"\nScore Breakdown:")
        print(f"  Music: {music_score:.4f}")
        print(f"  Speech: {speech_score:.4f}")
        print(f"  Total: {total_score:.4f}")
        
        return {
            'classification': classification,
            'confidence': round(confidence, 2)
        }