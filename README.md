# For CTRL+HACK+DEL 36hr Hackathon ([devpost](https://devpost.com/software/neighbor-chat-bot))
# Neighbor: AI Companion with a Clinical Heart
Neighbor is a proactive health-monitoring companion designed to support the elderly aging in place. It uses deep learning and signal processing to conversationally engage users while passively monitoring for cognitive decline and respiratory issues.

## 🚀 Overview

Neighbor turns everyday technology into an early-awareness tool. Using only a standard smartphone/computer microphone, it:
- **Engages** users in natural conversation to reduce isolation and block fraudulent interactions.
- **Analyzes** 768-dimensional acoustic embeddings to detect early-stage dementia.
- **Extracts** respiratory digital biomarkers (BPM, variability) to monitor for labored breathing.
- **Visualizes** clinical risk assessments in a real-time, glassmorphic dashboard.

---

## 🛠️ System Architecture

- **Backend**: Python 3.10+ (Flask, PyTorch, Librosa)
- **Frontend**: React 19 (Vite, Tailwind CSS, Recharts)
- **AI Stack**:
  - **STT**: OpenAI Whisper
  - **TTS**: ElevenLabs API
  - **Cognitive Model**: Wav2Vec2-XL (DementiaBank Fine-tuned)
  - **Inference Engine**: Ollama (Luna-4b locally)

---

## 📋 Requirements

### Hardware
- **OS**: Windows (Required for `winsound` and FFmpeg paths)
- **GPU**: NVIDIA GPU with CUDA support recommended (8GB+ VRAM for local models)
- **Audio**: Standard Microphone and Speakers

### Software
- **Python**: 3.10 or 3.11
- **Node.js**: 18.x or 20.x
- **Ollama**: [Download Here](https://ollama.com/)
- **FFmpeg**: Included in `bin/` or install via `choco install ffmpeg`

---

## 🔧 Installation & Setup

### 1. Backend Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd Dementia-CHD

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install core dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flask flask-cors openai-whisper librosa numpy requests pyaudio pydub elevenlabs transformers python-dotenv
```

### 2. Frontend Setup
```bash
cd dashboard
npm install
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
TTS=your_elevenlabs_api_key_here
```

### 4. Local Model Setup (Ollama)
Ensure Ollama is running and pull the required model:
```bash
ollama pull pheem49/Luna:4b
```

---

## 🛰️ Running the Project

### Start the AI Agent (Backend)
The backend runs the Flask API and the voice interaction loop.
```bash
# From the root directory
python autoanswer.py
```

### Start the Dashboard (Frontend)
The dashboard polls the backend every 3 seconds for health updates.
```bash
# From the /dashboard directory
npm run dev
```

---

## ⚠️ Implementation Notes & Troubleshooting

- **PyTorch Versions**: During development, we encountered compatibility issues between PyTorch 2.6 and certain safetensors. The current implementation is stable on **PyTorch 2.2+ with CUDA 12.1**.
- **PyAudio on Windows**: If `pip install pyaudio` fails, download the pre-compiled `.whl` file from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).
- **FFmpeg Path**: `autoanswer.py` expects FFmpeg in `bin\ffmpeg-8.0.1-essentials_build\bin\`. Adjust the `ffmpeg_path` variable in `autoanswer.py` if your installation differs.
- **Model Loading**: The first run will download the `shields/wav2vec2-xl-960h-dementiabank` model (~1.2GB). Ensure a stable connection.

---

## 🤝 Contribution
Neighbor is a "Early Awareness" tool, not a diagnostic medical device. Contributions focused on improving signal-to-noise ratios in respiratory detection are welcome.

**Neighbor: Turning silence into early awareness.**
