import os
import time
import shutil
import io
import json
import torch
import librosa
import numpy as np
from scipy.io import wavfile
import requests
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
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment, silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

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
# Using local Ollama with gemma3:4b model
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "pheem49/Luna:4b"
eleven_client = ElevenLabs(api_key=os.getenv("TTS"))
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

# Load a CTC head for speech-to-text (fallback to facebook/wav2vec2-base-960h)
try:
    asr_model = Wav2Vec2ForCTC.from_pretrained(DEMENTIA_MODEL_ID).to(device)
except Exception:
    try:
        asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        asr_model = None

# --- HELPERS ---

def query_ollama(prompt: str) -> str:
    """Query local Ollama model."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Ollama Error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Ollama Connection Error: {e}")
        return ""

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


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Wav2Vec2 CTC model."""
    if asr_model is None:
        return ""
    try:
        # load waveform
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = asr_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription.strip()
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

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
    
    # Save user audio (keep original format, pydub will handle conversion)
    user_filename = f"user_{int(time.time())}.webm"
    user_audio_path = os.path.join(session_dir, user_filename)
    
    with open(user_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Reject empty uploads early
    try:
        if os.path.getsize(user_audio_path) == 0:
            os.remove(user_audio_path)
            return JSONResponse(status_code=400, content={"error": "Uploaded audio is empty"})
    except Exception:
        pass

    # Transcribe the received audio and include the transcript in the prompt
    transcript = transcribe_audio(user_audio_path)
    print(f"Transcript: {transcript}")

    # Friendly Chat Logic with Ollama
    base_prompt = open("prompt.txt","r").read()
    # Build prompt with transcript and optional history
    if transcript and transcript.strip():
        prompt = f"{base_prompt}\n\nUser transcript: \"{transcript}\"\n"
    else:
        prompt = base_prompt

    if history and history.strip():
        prompt = f"{prompt}\nConversation History: {history}\n"

    response_text = query_ollama(prompt)
    if not response_text or len(response_text.strip()) == 0:
        response_text = "It's so good to hear from you. How has your morning been?"

    return JSONResponse(content={"response_text": response_text, "transcript": transcript})

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
            audio_file_path = os.path.join(session_dir, f)
            # AudioSegment will handle webm, wav, or other formats via ffmpeg
            try:
                combined_audio += AudioSegment.from_file(audio_file_path)
            except Exception as load_err:
                print(f"Warning: Could not load {f}: {load_err}")
                continue
        
        if len(combined_audio) == 0:
            return JSONResponse(status_code=400, content={"error": "No valid audio data found"})
        
        full_path = os.path.join(session_dir, "full_conversation.wav")
        combined_audio.export(full_path, format="wav")

        # 2. Trim Silence
        trimmed_path = trim_silence_pydub(full_path)

        # 3. Extract Features from Wav2Vec2-DementiaBank
        means, stds = extract_acoustic_features(trimmed_path)
        
        if means is None or stds is None:
            return JSONResponse(status_code=500, content={"error": "Feature extraction failed"})
        
        # 4. Expert Interpretation using Ollama
        diagnosis_prompt = f"""Act as a Neuro-Speech Pathologist. You have analyzed a patient's speech using the 
Shields Wav2Vec2 DementiaBank model. 

Acoustic Feature Fingerprint (Mean): {means}
Acoustic Feature Fingerprint (Std): {stds}

Based on these high-dimensional embeddings, determine the likelihood of cognitive impairment. 
Respond in JSON format:
{{
    "label": "Dementia" or "Healthy",
    "score": 0-100,
    "confidence": "High" | "Medium" | "Low",
    "explanation": "Explain how the acoustic features led to this result."
}}

Respond ONLY with valid JSON, no other text."""
        
        diagnosis_response = query_ollama(diagnosis_prompt)
        
        # Try to parse JSON from Ollama's response. The frontend expects
        # either a structured model explanation (gemini_explanation) or
        # a local_inference object. We'll return both keys so the UI can
        # choose the best available data.
        gemini_explanation = {}
        local_inference = {}

        try:
            parsed = json.loads(diagnosis_response)
            if isinstance(parsed, dict) and ("label" in parsed or "score" in parsed):
                gemini_explanation = parsed
            else:
                # Parsed JSON but doesn't contain expected fields: place into explanation
                gemini_explanation = {"explanation": parsed}
        except Exception:
            # Ollama didn't return strict JSON — fall back to a local heuristic
            local_inference = {
                "label": "Unknown",
                "score": 0,
                "confidence": "Low",
                "explanation": diagnosis_response if diagnosis_response else "Analysis completed",
                "means": means,
                "stds": stds
            }

        # If Ollama produced a structured explanation, still provide the raw features
        if not local_inference:
            local_inference = {
                "label": gemini_explanation.get("label", "Unknown"),
                "score": gemini_explanation.get("score", 0),
                "confidence": gemini_explanation.get("confidence", "Low"),
                "explanation": gemini_explanation.get("explanation", ""),
                "means": means,
                "stds": stds,
            }

        return JSONResponse(content={
            "local_inference": local_inference,
            "gemini_explanation": gemini_explanation,
            "raw_ollama": diagnosis_response
        })

    except Exception as e:
        print(f"Analysis Crash: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)