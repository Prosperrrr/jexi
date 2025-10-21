import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import uuid
from models.classifier import AudioClassifier
from models.yamnet_classifier import YAMNetClassifier
from models.music_processor import MusicProcessor
from models.speech_processor import SpeechProcessor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize models
# classifier = AudioClassifier()  # Old classifier (this is included for documentation/backup)
yamnet_classifier = YAMNetClassifier()  
music_processor = MusicProcessor()
speech_processor = SpeechProcessor()  

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Store for tracking background jobs and file metadata
processing_jobs = {}
uploaded_files = {}  # NEW: Store file info before processing

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
    STAGE 1: Upload file and classify with YAMNet
    Returns classification for user confirmation
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
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())[:8]
        
        try:
            # Classify with YAMNet (takes ~5 seconds)
            result = yamnet_classifier.classify(filepath)
            
            if result is None:
                return jsonify({
                    "error": "Failed to classify audio",
                    "filename": filename,
                    "suggestion": "Please specify if this is music or speech"
                }), 500
            
            # Store file info for later processing
            uploaded_files[file_id] = {
                "filename": filename,
                "filepath": filepath,
                "detected_type": result['classification'],
                "confidence": result['confidence'],
                "top_predictions": result['top_predictions']
            }
            
            return jsonify({
                "file_id": file_id,
                "filename": filename,
                "detected_type": result['classification'],
                "confidence": result['confidence'],
                "top_predictions": result['top_predictions'],
                "status": "awaiting_confirmation",
                "message": f"AI detected this as {result['classification'].upper()} with {result['confidence']:.0f}% confidence"
            }), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Classification failed: {str(e)}",
                "filename": filename
            }), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/process/<file_id>', methods=['POST'])
def confirm_and_process(file_id):
    """
    STAGE 2: User confirms content type and starts processing
    """
    if file_id not in uploaded_files:
        return jsonify({"error": "File ID not found"}), 404
    
    file_info = uploaded_files[file_id]
    
    # Get user's confirmed content type (or use detected)
    data = request.get_json() or {}
    content_type = data.get('content_type', file_info['detected_type'])
    
    if content_type not in ['music', 'speech']:
        return jsonify({"error": "Invalid content type"}), 400
    
    # Generate processing job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Start processing
    if content_type == "music":
        processing_jobs[job_id] = {"status": "processing", "type": "music"}
        thread = threading.Thread(
            target=process_music_background,
            args=(file_info['filepath'], job_id)
        )
        #uses threading instead of background processing like celery,e.t.c
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "file_id": file_id,
            "filename": file_info['filename'],
            "content_type": "music",
            "status": "processing",
            "message": "Music processing started",
            "check_status_url": f"/api/process/music/{job_id}/status",
            "estimated_time": "3-5 minutes"
        }), 200
        
    elif content_type == "speech":
        processing_jobs[job_id] = {"status": "processing", "type": "speech"}
        thread = threading.Thread(
            target=process_speech_background,
            args=(file_info['filepath'], job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "file_id": file_id,
            "filename": file_info['filename'],
            "content_type": "speech",
            "status": "processing",
            "message": "Speech processing started",
            "check_status_url": f"/api/process/speech/{job_id}/status",
            "estimated_time": "2-3 minutes"
        }), 200

def process_music_background(filepath, job_id):
    """Background function to process music"""
    try:
        music_processor.process(filepath, job_id)
        processing_jobs[job_id] = {"status": "completed", "type": "music"}
    except Exception as e:
        processing_jobs[job_id] = {"status": "failed", "type": "music", "error": str(e)}

def process_speech_background(filepath, job_id):
    """Background function to process speech"""
    try:
        speech_processor.process(filepath, job_id)
        processing_jobs[job_id] = {"status": "completed", "type": "speech"}
    except Exception as e:
        processing_jobs[job_id] = {"status": "failed", "type": "speech", "error": str(e)}

# ============= MUSIC PROCESSING ENDPOINTS =============

@app.route('/api/process/music/<job_id>/status', methods=['GET'])
def get_music_status(job_id):
    """Check status of music processing job"""
    # First check if job is currently processing (in memory)
    if job_id in processing_jobs:
        job_info = processing_jobs[job_id]
        
        # Get detailed progress if available
        progress_info = music_processor.get_progress(job_id)
        
        if job_info['status'] == 'processing':
            response = {
                "status": "processing",
                "job_id": job_id,
                "message": "Processing audio..."
            }
            
            # Add progress details if available
            if progress_info:
                response["progress"] = progress_info["percent"]
                response["current_step"] = progress_info["message"]
                response["updated_at"] = progress_info["updated_at"]
            
            return jsonify(response), 200
            
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
        "stems": stems_urls,
        "active_stems": active_stems  # List of stems that have content
    }), 200

@app.route('/api/download/<job_id>/<stem_file>', methods=['GET'])
def download_stem(job_id, stem_file):
    """Download individual stem file"""
    stem_path = os.path.join('processed', job_id, 'stems', stem_file)
    
    if not os.path.exists(stem_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(stem_path, as_attachment=True)

# ============= SPEECH PROCESSING ENDPOINTS =============

@app.route('/api/process/speech/<job_id>/status', methods=['GET'])
def get_speech_status(job_id):
    """Check status of speech processing job"""
    # First check if job is currently processing (in memory)
    if job_id in processing_jobs:
        job_info = processing_jobs[job_id]
        
        # Get detailed progress if available
        progress_info = speech_processor.get_progress(job_id)
        
        if job_info['status'] == 'processing':
            response = {
                "status": "processing",
                "job_id": job_id,
                "message": "Processing speech..."
            }
            
            # Add progress details if available
            if progress_info:
                response["progress"] = progress_info["percent"]
                response["current_step"] = progress_info["message"]
                response["updated_at"] = progress_info["updated_at"]
            
            return jsonify(response), 200
            
        elif job_info['status'] == 'failed':
            return jsonify({
                "status": "failed",
                "job_id": job_id,
                "error": job_info.get('error', 'Unknown error')
            }), 200
    
    # If not in memory, check metadata file (completed jobs)
    status_info = speech_processor.get_status(job_id)
    return jsonify(status_info), 200

@app.route('/api/process/speech/<job_id>', methods=['GET'])
def get_speech_results(job_id):
    """Get results of completed speech processing"""
    metadata = speech_processor.get_metadata(job_id)
    
    if not metadata:
        return jsonify({"error": "Job not found"}), 404
    
    if metadata['status'] != 'completed':
        return jsonify({
            "error": "Job not completed yet",
            "status": metadata['status']
        }), 400
    
    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "metadata": {
            "filename": metadata['filename'],
            "duration": metadata['duration'],
            "sample_rate": metadata['sample_rate'],
            "transcript": metadata['transcript'],
            "processed_at": metadata['processed_at']
        },
        "downloads": {
            "clean_audio": f"/api/download/speech/{job_id}/clean_audio.wav",
            "transcript_txt": f"/api/download/transcript/{job_id}/txt",
            "transcript_json": f"/api/download/transcript/{job_id}/json",
            "transcript_srt": f"/api/download/transcript/{job_id}/srt"
        }
    }), 200

@app.route('/api/download/speech/<job_id>/<filename>', methods=['GET'])
def download_speech_audio(job_id, filename):
    """Download processed speech audio"""
    file_path = os.path.join('processed', job_id, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/download/transcript/<job_id>/<format>', methods=['GET'])
def download_transcript(job_id, format):
    """
    Download transcript in specified format
    Formats: txt, json, srt
    """
    if format not in ['txt', 'json', 'srt']:
        return jsonify({"error": "Invalid format. Use: txt, json, or srt"}), 400
    
    try:
        if format == 'txt':
            file_path = speech_processor.export_transcript_txt(job_id)
        elif format == 'json':
            file_path = speech_processor.export_transcript_json(job_id)
        elif format == 'srt':
            file_path = speech_processor.export_transcript_srt(job_id)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "Transcript not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    Legacy endpoint - use /api/upload + /api/process/{file_id} instead
    """
    return jsonify({"message": "Use /api/upload endpoint"}), 400

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
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)