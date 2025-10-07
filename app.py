from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============= ROUTES =============

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to Jexi API",
        "version": "1.0.0",
        "status": "running"
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

# Offline Processing Routes
@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """
    Upload audio file for offline processing
    Auto-detects Music vs Speech and routes to correct pipeline
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # TODO: Implement content classification (Music vs Speech)
        # TODO: Route to appropriate pipeline (Spleeter or Whisper+RNNoise)
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "status": "processing"
        }), 200
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/process/music', methods=['POST'])
def process_music():
    """
    Process music file with Demucs for stem separation
    """
    # TODO: Implement Demucs processing
    return jsonify({"message": "Music processing endpoint - Coming soon"}), 200

@app.route('/api/process/speech', methods=['POST'])
def process_speech():
    """
    Process speech file with noisereduce + Whisper
    """
    # TODO: Implement noisereduce noise reduction
    # TODO: Implement Whisper transcription
    return jsonify({"message": "Speech processing endpoint - Coming soon"}), 200

# Real-time Processing Routes
@app.route('/api/realtime/noise-reduction', methods=['POST'])
def realtime_noise_reduction():
    """
    Real-time noise reduction using noisereduce
    WebSocket or streaming endpoint for live audio
    """
    # TODO: Implement WebSocket for live noisereduce processing
    return jsonify({"message": "Real-time noise reduction - Coming soon"}), 200

@app.route('/api/realtime/transcription', methods=['POST'])
def realtime_transcription():
    """
    Real-time transcription using Deepgram API
    """
    # TODO: Implement Deepgram streaming integration
    return jsonify({"message": "Real-time transcription - Coming soon"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)