# Jexi - AI-Powered Audio Processing Platform


> A comprehensive, full-stack AI platform for audio enhancement, stem separation, transcription, and real-time processing.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)


---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Jexi** is an AI-powered audio processing platform that intelligently analyzes and enhances audio content. By combining cutting-edge machine learning models, Jexi provides a unified solution for musicians, podcasters, content creators, and anyone working with audio.

### The Problem
Audio processing typically requires multiple specialized tools:
- Music producers need separate apps for stem separation and analysis
- Podcasters use different tools for noise reduction and transcription  
- Content creators struggle with fragmented workflows

### The Solution
Jexi automatically detects audio content type (music or speech) and applies the optimal AI processing pipeline, delivering professional results in one seamless platform.

---

##  Features

###  Music Processing
- **AI-Powered Stem Separation**: Separate any song into 6 isolated tracks:
  - 🎤 Vocals
  - 🥁 Drums
  - 🎸 Bass
  - 🎸 Guitar
  - 🎹 Piano
  - 🎺 Other instruments
- **Lyrics Transcription**: Extract lyrics with timestamps using Whisper AI
- **Audio Analysis**: Automatic detection of musical key, BPM, and duration
- **Silence Detection**: Identifies active vs inactive stems for clean mixing

###  Speech Processing
- **Advanced Noise Reduction**: Remove background noise using DeepFilterNet
- **High-Accuracy Transcription**: Convert speech to text with Whisper AI
- **Multiple Export Formats**: Download transcripts as TXT, JSON, or SRT subtitles
- **Audio Enhancement**: Normalize volume and enhance voice clarity

###  Intelligent Classification
- **Automatic Content Detection**: Google's YAMNet model with 95%+ accuracy
- **User Confirmation**: Two-stage workflow (classify → confirm → process)
- **Multi-Class Detection**: Distinguishes music, speech, sermons, podcasts, and more

###  Real-Time Features
- **Live Noise Reduction**: Clean audio from microphone in real-time
- **Live Transcription**: Stream audio to text with Deepgram API
- **WebSocket Support**: Low-latency processing for interactive applications

###  System Features
- **Background Processing**: Non-blocking uploads with progress tracking (0-100%)
- **Rate Limiting**: Prevent abuse (5 uploads per minute)
- **Automatic Cleanup**: Delete old files after 24 hours
- **RESTful API**: Complete endpoints for all functionality
- **Cross-Platform**: Web, mobile (React Native), and API access

---

##  Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core backend language |
| **Flask** | Web framework & REST API |
| **Flask-SocketIO** | WebSocket support for real-time features |
| **YAMNet** | Audio classification (Google) |
| **Demucs** | Music stem separation (Meta) |
| **Whisper** | Speech-to-text transcription (OpenAI) |
| **DeepFilterNet** | Noise reduction (Xiph.org) |
| **Deepgram API** | Real-time transcription |
| **Librosa** | Audio analysis and processing |

### Frontend (In Development)
| Technology | Purpose |
|------------|---------|
| **React.js** | Web application framework |
| **Socket.IO Client** | Real-time communication |
| **Web Audio API** | Audio playback and visualization |
| **Tailwind CSS** | Styling framework |

### Mobile (Planned)
| Technology | Purpose |
|------------|---------|
| **React Native** | Cross-platform mobile app |
| **Native Audio** | Mobile audio recording |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         JEXI PLATFORM                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐     │
│  │  Web Client  │  │ Mobile App    │  │  API Client   │     │
│  │  (React.js)  │  │(React Native) │  │   (REST)      │     │
│  └──────┬───────┘  └───────┬───────┘  └───────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│                   ┌────────▼─────────┐                      │
│                   │   Flask Backend  │                      │
│                   │   (Python API)   │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│    ┌────▼────┐      ┌──────▼─────┐    ┌──────▼─────┐        │
│    │ YAMNet  │      │   Demucs   │    │DeepFilter  │        │
│    │Classifier│     │(6-Stems)   │    │   Net      │        │
│    └─────────┘      └────────────┘    └────────────┘        │
│         │                  │                  │             │
│    ┌────▼────────────────────▼──────────────┬─▼─────┐       │
│    │           Whisper AI Model              │Deepgram│     │
│    │      (Transcription Engine)             │  API   │     │
│    └─────────────────────────────────────────┴────────┘     │
│                                                             │
│    ┌─────────────────────────────────────────────────┐      │
│    │         File System / Storage Layer             │      │
│    │  • Processed Audio Files                        │      │
│    │  • JSON Metadata                                │      │
│    │  • Transcripts (TXT, JSON, SRT)                 │      │
│    └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

#### Music Pipeline
```
Audio Upload → YAMNet Classification → User Confirmation
                        ↓
              ┌─────────────────────┐
              │  Music Processing   │
              └─────────────────────┘
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   Stem Separation  Lyrics Extract  Audio Analysis
   (Demucs 6-stem)  (Whisper AI)   (Key, BPM, etc)
        ↓               ↓               ↓
   6 WAV Files    Timestamped     Metadata JSON
                    Transcript
```

#### Speech Pipeline
```
Audio Upload → YAMNet Classification → User Confirmation
                        ↓
              ┌─────────────────────┐
              │  Speech Processing  │
              └─────────────────────┘
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   Noise Reduction  Transcription  Audio Enhancement
   (DeepFilterNet)  (Whisper AI)   (Normalize, etc)
        ↓               ↓               ↓
   Clean WAV       TXT/JSON/SRT    Enhanced Audio
```

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git
- 4GB RAM minimum (8GB recommended)
- Internet connection (for first-time model downloads)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/prosperrr/jexi.git
cd jexi/jexi-backend
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```


4. **Configure environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API keys
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

5. **Run the backend**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup (Coming Soon)

```bash
cd jexi-frontend
npm install
npm start
```

---

##  Usage

### Quick Start

1. **Upload Audio**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@your_audio.mp3"
```

Response:
```json
{
  "file_id": "abc123",
  "detected_type": "music",
  "confidence": 94.5,
  "status": "awaiting_confirmation"
}
```

2. **Confirm and Process**
```bash
curl -X POST http://localhost:5000/api/process/abc123 \
  -H "Content-Type: application/json" \
  -d '{"content_type": "music"}'
```

3. **Check Status**
```bash
curl http://localhost:5000/api/process/music/xyz789/status
```

4. **Download Results**
```bash
curl http://localhost:5000/api/download/xyz789/vocals.wav -O
```

### Example Use Cases

#### For Musicians
```
Upload: "my_song.mp3"
Result: 6 isolated stems for remixing
       + Lyrics with timestamps
       + Key and BPM analysis
```

#### For Podcasters
```
Upload: "episode_01.mp3"
Result: Noise-free audio
       + Full transcript (TXT, JSON, SRT)
       + Enhanced voice clarity
```

#### For Content Creators
```
Upload: "sermon_recording.mp3"
Result: Clean audio without background noise
       + Searchable transcript
       + Downloadable in multiple formats
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and classify audio |
| POST | `/process/{file_id}` | Start processing |
| GET | `/process/music/{job_id}/status` | Check music job status |
| GET | `/process/music/{job_id}` | Get music results |
| GET | `/process/speech/{job_id}/status` | Check speech job status |
| GET | `/process/speech/{job_id}` | Get speech results |
| GET | `/download/{job_id}/{filename}` | Download processed files |
| GET | `/download/transcript/{job_id}/{format}` | Download transcript |
| GET | `/storage/stats` | Get storage statistics |

### Detailed Examples

#### 1. Upload Audio
**Request:**
```http
POST /api/upload
Content-Type: multipart/form-data

file: (binary audio file)
```

**Response:**
```json
{
  "file_id": "a1b2c3d4",
  "filename": "audio.mp3",
  "detected_type": "music",
  "confidence": 92.5,
  "top_predictions": [
    {"class": "Music", "confidence": 85.2},
    {"class": "Singing", "confidence": 7.3}
  ],
  "status": "awaiting_confirmation",
  "message": "AI detected this as MUSIC with 93% confidence"
}
```

#### 2. Process Audio
**Request:**
```http
POST /api/process/{file_id}
Content-Type: application/json

{
  "content_type": "music"
}
```

**Response:**
```json
{
  "job_id": "xyz789",
  "file_id": "a1b2c3d4",
  "content_type": "music",
  "status": "processing",
  "estimated_time": "3-5 minutes"
}
```

#### 3. Get Results (Music)
**Request:**
```http
GET /api/process/music/{job_id}
```

**Response:**
```json
{
  "job_id": "xyz789",
  "status": "completed",
  "metadata": {
    "filename": "song.mp3",
    "key": "C# minor",
    "bpm": 128,
    "duration": "3:45",
    "lyrics": {
      "plain": "Full lyrics here...",
      "segments": [
        {"start": 0.0, "end": 2.5, "text": "First line"}
      ]
    }
  },
  "stems": {
    "vocals": "/api/download/xyz789/vocals.wav",
    "drums": "/api/download/xyz789/drums.wav",
    "bass": "/api/download/xyz789/bass.wav",
    "guitar": "/api/download/xyz789/guitar.wav",
    "piano": "/api/download/xyz789/piano.wav",
    "other": "/api/download/xyz789/other.wav"
  }
}
```

#### 4. Get Results (Speech)
**Request:**
```http
GET /api/process/speech/{job_id}
```

**Response:**
```json
{
  "job_id": "abc456",
  "status": "completed",
  "metadata": {
    "filename": "podcast.mp3",
    "duration": "15:30",
    "transcript": {
      "plain": "Full transcript...",
      "segments": [...],
      "word_count": 2450
    }
  },
  "downloads": {
    "clean_audio": "/api/download/speech/abc456/clean_audio.wav",
    "transcript_txt": "/api/download/transcript/abc456/txt",
    "transcript_json": "/api/download/transcript/abc456/json",
    "transcript_srt": "/api/download/transcript/abc456/srt"
  }
}
```

### WebSocket Events (Real-Time)

Connect to: `ws://localhost:5000`

**Client Events:**
- `audio_chunk` - Send audio for noise reduction
- `start_transcription` - Begin live transcription
- `transcribe_audio` - Send audio for transcription

**Server Events:**
- `clean_audio` - Receive cleaned audio chunk
- `transcript` - Receive transcribed text
- `transcription_error` - Error in transcription

---

##  Project Structure

```
jexi/
├── jexi-backend/              # Python Flask backend
│   ├── app.py                 # Main application entry
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment variables template
│   ├── models/               # AI model handlers
│   │   ├── classifier.py     # Audio classifier (backup)
│   │   ├── yamnet_classifier.py  # YAMNet implementation
│   │   ├── music_processor.py    # Music processing pipeline
│   │   ├── speech_processor.py   # Speech processing pipeline
│   │   ├── realtime_processor.py # Real-time features
│   │   └── deepgram_transcriber.py # Deepgram integration
│   ├── utils/                # Utility functions
│   │   ├── file_cleanup.py   # Automatic file management
│   │   └── rate_limiter.py   # Rate limiting logic
│   ├── uploads/              # Temporary uploaded files
│   └── processed/            # Processed output files
│
├── jexi-frontend/            # React.js web application (in development)
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API service layer
│   │   └── utils/           # Helper functions
│   └── public/
│
├── jexi-mobile/              # React Native mobile app (planned)
│   └── ...
│
└── README.md                 # This file
```

---

## 🚀 Deployment

### Production Considerations

**Backend:**
- Use **Gunicorn** or **uWSGI** instead of Flask dev server
- Set up **Nginx** as reverse proxy
- Enable **HTTPS** with SSL certificates
- Configure **environment variables** securely
- Set up **database** (PostgreSQL/MongoDB) for production
- Implement **user authentication** (JWT tokens)
- Add **monitoring** (Sentry, Datadog)

**Frontend:**
- Build production bundle: `npm run build`
- Deploy to **Vercel**, **Netlify**, or **AWS S3 + CloudFront**
- Configure **CORS** properly

**Recommended Hosting:**
- Backend: AWS EC2, Google Cloud, DigitalOcean, Heroku
- Frontend: Vercel, Netlify, Cloudflare Pages
- Database: AWS RDS, MongoDB Atlas


##  Acknowledgments

### AI Models & Libraries
- **Google** - YAMNet audio classification model
- **Meta/Facebook** - Demucs stem separation
- **OpenAI** - Whisper speech recognition
- **Xiph.org Foundation** - DeepFilterNet noise reduction
- **Deepgram** - Real-time transcription API

### Open Source Libraries
- Flask, Librosa, PyTorch, TensorFlow
- React.js, Socket.IO
- All contributors to open-source audio processing tools

---

##  Contact

**Prosper J. Sobamiwa**  
**Email:** prospersobamiwa@gmail.com  
**GitHub:** [@prosperrrr](https://github.com/prosperrrr)  

---

## 🗺️ Roadmap

### Current Version: v1.0.0 (Backend)
- ✅ Audio classification
- ✅ Music stem separation
- ✅ Speech transcription
- ✅ Real-time processing infrastructure
- ✅ REST API

### v1.1.0 - Frontend 
- [ ] React web application
- [ ] Interactive music mixer UI
- [ ] Transcript viewer
- [ ] User dashboard

### v1.2.0 - Mobile Apps
- [ ] React Native iOS app
- [ ] React Native Android app
- [ ] Mobile audio recording
- [ ] Offline processing

### v2.0.0 - Advanced Features
- [ ] User accounts & authentication
- [ ] Cloud storage integration
- [ ] Batch processing
- [ ] Advanced audio effects
- [ ] Collaborative features

---

<div align="center">

**Built with ❤️ for Audio Enthusiasts**

[⬆ Back to Top](#jexi---ai-powered-audio-processing-platform)

</div>
