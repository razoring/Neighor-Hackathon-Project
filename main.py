import os
import time
import shutil
import io
import json
import torch
import librosa
import numpy as np
from dotenv import load_dotenv

# --- SET FFMPEG PATH FIRST (before importing pydub) ---
project_root = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
ffprobe_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe")

# Set environment before importing pydub
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

# API & Framework
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment, silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Set pydub paths after import
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API INITIALIZATION ---
# Using the modern google-genai SDK
genai_client = genai.Client(api_key=os.getenv("GEMINI"))
eleven_client = ElevenLabs(api_key=os.getenv("TTS"))

GEMINI_MODEL = "gemini-2.0-flash"
DEMENTIA_MODEL_ID = "shields/wav2vec2-xl-960h-dementiabank"

# Storage
SESSION_STORAGE = "temp_sessions"
os.makedirs(SESSION_STORAGE, exist_ok=True)

# --- LOAD AI MODELS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Wav2Vec2 DementiaBank Model...")
try:
    processor = Wav2Vec2Processor.from_pretrained(DEMENTIA_MODEL_ID)
    wav_model = Wav2Vec2Model.from_pretrained(DEMENTIA_MODEL_ID).to(device)
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading specialized model: {e}. Falling back to base.")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# --- HELPERS ---

def text_to_speech(text: str):
    """ElevenLabs TTS conversion."""
    try:
        audio_stream = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_turbo_v2_5"
        )
        return b"".join(chunk for chunk in audio_stream)
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        return None

def trim_silence_pydub(audio_path: str):
    """Removes silence from audio using pydub."""
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = silence.split_on_silence(
            audio, min_silence_len=600, silence_thresh=audio.dBFS - 16
        )
        output = AudioSegment.empty()
        for chunk in chunks:
            output += chunk
        
        trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
        output.export(trimmed_path, format="wav")
        return trimmed_path
    except Exception as e:
        print(f"Silence trimming failed: {e}")
        return audio_path

def extract_acoustic_features(audio_path: str):
    """Extracts high-level embeddings from the DementiaBank model."""
    try:
        # Load audio (downsample to 16kHz for Wav2Vec2)
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = wav_model(**inputs)
        
        # Get mean and std of the last hidden states
        embeddings = outputs.last_hidden_state.cpu()
        mean_feats = torch.mean(embeddings, dim=1).numpy().tolist()[0]
        std_feats = torch.std(embeddings, dim=1).numpy().tolist()[0]
        
        # Return a subset (first 15 dimensions) to provide as context to Gemini
        return mean_feats[:15], std_feats[:15]
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None, None

# --- ENDPOINTS ---

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...), session_id: str = Form(...), history: str = Form(...)):
    """Handles the casual chat loop."""
    session_dir = os.path.join(SESSION_STORAGE, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Save user audio
    user_filename = f"user_{int(time.time())}.wav"
    user_audio_path = os.path.join(session_dir, user_filename)
    with open(user_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Friendly Chat Logic with Gemini
    # Note: For a hackathon, we assume the user is speaking naturally. 
    # You could use Gemini's multimodal power to 'listen' to the audio here.
    prompt = f"""
    You are a friendly, caring phone companion named Rachel. 
    You are having a casual chat with an elderly person. 
    Keep your response very warm, short (1-2 sentences), and ask a simple follow-up question.
    History: {history}
    """
    
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        response_text = response.text
    except Exception as e:
        print(f"Gemini Chat Error with {GEMINI_MODEL}: {e}")
        try:
            # Fallback to gemini-pro
            response = genai_client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            response_text = response.text
        except:
            response_text = "It's so good to hear from you. How has your morning been?"

    return JSONResponse(content={"response_text": response_text})

@app.get("/audio_response")
async def get_audio_stream(text: str):
    """Streams the ElevenLabs audio back to the frontend."""
    audio_data = text_to_speech(text)
    if audio_data:
        return StreamingResponse(iter([audio_data]), media_type="audio/mpeg")
    return JSONResponse(status_code=500, content={"error": "TTS Generation Failed"})

@app.post("/analyze")
async def analyze_full_session(session_id: str = Form(...)):
    """Trims silence, extracts features, and runs Gemini Diagnosis."""
    session_dir = os.path.join(SESSION_STORAGE, session_id)
    
    if not os.path.exists(session_dir):
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    # 1. Stitch User Clips
    user_files = sorted([f for f in os.listdir(session_dir) if f.startswith("user_")])
    if not user_files:
        return JSONResponse(status_code=400, content={"error": "No audio data collected"})

    try:
        combined_audio = AudioSegment.empty()
        for f in user_files:
            combined_audio += AudioSegment.from_file(os.path.join(session_dir, f))
        
        full_path = os.path.join(session_dir, "full_conversation.wav")
        combined_audio.export(full_path, format="wav")

        # 2. Trim Silence
        trimmed_path = trim_silence_pydub(full_path)

        # 3. Extract Features from Wav2Vec2-DementiaBank
        means, stds = extract_acoustic_features(trimmed_path)
        
        if means is None or stds is None:
            return JSONResponse(status_code=500, content={"error": "Feature extraction failed"})
        
        # Return raw features for testing (without Gemini analysis)
        return JSONResponse(content={
            "label": "Feature Extraction Success",
            "means": means,
            "stds": stds,
            "message": "Acoustic features extracted successfully"
        })

    except Exception as e:
        print(f"Analysis Crash: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)