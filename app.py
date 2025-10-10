from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import threading
import uuid
from models.classifier import AudioClassifier
from models.music_processor import MusicProcessor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize models
classifier = AudioClassifier()
music_processor = MusicProcessor()

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Store for tracking background jobs
processing_jobs = {}

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

# ============= UPLOAD & AUTO-CLASSIFY =============

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """
    Upload audio file - auto-classifies and starts processing
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
        
        # Classify audio content
        try:
            content_type = classifier.classify(filepath)
            
            if content_type is None:
                return jsonify({
                    "error": "Failed to classify audio",
                    "filename": filename
                }), 500
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())[:8]
            
            # Start processing based on content type
            if content_type == "music":
                # Start music processing in background
                processing_jobs[job_id] = {"status": "processing", "type": "music"}
                thread = threading.Thread(
                    target=process_music_background,
                    args=(filepath, job_id)
                )
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    "job_id": job_id,
                    "filename": filename,
                    "content_type": "music",
                    "status": "processing",
                    "message": "Music processing started",
                    "check_status_url": f"/api/process/music/{job_id}/status",
                    "estimated_time": "3-5 minutes"
                }), 200
                
            elif content_type == "speech":
                # TODO: Implement speech processing
                return jsonify({
                    "message": "Speech processing not yet implemented",
                    "filename": filename,
                    "content_type": "speech"
                }), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Processing failed: {str(e)}",
                "filename": filename
            }), 500
    
    return jsonify({"error": "Invalid file type"}), 400

def process_music_background(filepath, job_id):
    """Background function to process music"""
    try:
        music_processor.process(filepath, job_id)
        processing_jobs[job_id] = {"status": "completed", "type": "music"}
    except Exception as e:
        processing_jobs[job_id] = {"status": "failed", "type": "music", "error": str(e)}

# ============= MUSIC PROCESSING ENDPOINTS =============

@app.route('/api/process/music/<job_id>/status', methods=['GET'])
def get_music_status(job_id):
    """Check status of music processing job"""
    # First check if job is currently processing (in memory)
    if job_id in processing_jobs:
        job_info = processing_jobs[job_id]
        if job_info['status'] == 'processing':
            return jsonify({
                "status": "processing",
                "job_id": job_id,
                "message": "Still processing... This may take 3-5 minutes"
            }), 200
        elif job_info['status'] == 'failed':
            return jsonify({
                "status": "failed",
                "job_id": job_id,
                "error": job_info.get('error', 'Unknown error')
            }), 200
    
    # If not in memory, check metadata file (completed jobs)
    status_info = music_processor.get_status(job_id)
    return jsonify(status_info), 200

@app.route('/api/process/music/<job_id>', methods=['GET'])
def get_music_results(job_id):
    """Get results of completed music processing"""
    metadata = music_processor.get_metadata(job_id)
    
    if not metadata:
        return jsonify({"error": "Job not found"}), 404
    
    if metadata['status'] != 'completed':
        return jsonify({
            "error": "Job not completed yet",
            "status": metadata['status']
        }), 400
    
    # Build download URLs for stems (only active ones)
    stems_urls = {}
    active_stems = []
    
    for stem_name in ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']:
        stem_info = metadata['stems'].get(stem_name, {})
        is_active = stem_info.get('active', True)  # Default to True for backwards compatibility
        
        stem_data = {
            "url": f"/api/download/{job_id}/{stem_name}.wav",
            "active": is_active
        }
        stems_urls[stem_name] = stem_data
        
        if is_active:
            active_stems.append(stem_name)
    
    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "metadata": {
            "filename": metadata['filename'],
            "key": metadata['key'],
            "bpm": metadata['bpm'],
            "duration": metadata['duration'],
            "sample_rate": metadata['sample_rate'],
            "lyrics": metadata['lyrics'],
            "processed_at": metadata['processed_at']
        },
        "stems": stems_urls
    }), 200

@app.route('/api/download/<job_id>/<stem_file>', methods=['GET'])
def download_stem(job_id, stem_file):
    """Download individual stem file"""
    stem_path = os.path.join('processed', job_id, 'stems', stem_file)
    
    if not os.path.exists(stem_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(stem_path, as_attachment=True)

# ============= LEGACY/PLACEHOLDER ENDPOINTS =============

@app.route('/api/process/music', methods=['POST'])
def process_music():
    """
    Legacy endpoint - use /api/upload instead
    """
    return jsonify({"message": "Use /api/upload endpoint instead"}), 400

@app.route('/api/process/speech', methods=['POST'])
def process_speech():
    """
    Process speech file with noisereduce + Whisper
    """
    # TODO: Implement noisereduce noise reduction
    # TODO: Implement Whisper transcription
    return jsonify({"message": "Speech processing endpoint - Coming soon"}), 200

# ============= REAL-TIME PROCESSING ENDPOINTS =============

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