import os
import time
import shutil
import io
import torch
import librosa
import numpy as np
import google.generativeai as genai
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment, silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import speech_recognition as sr

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API INITIALIZATION ---
# Fixed: Explicitly using the current stable model string
genai.configure(api_key=os.getenv("GEMINI"))
eleven = ElevenLabs(api_key=os.getenv("TTS"))

MODEL_ID = "shields/wav2vec2-xl-960h-dementiabank"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading Model on {device}...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device)
    print("Model Loaded.")
except Exception as e:
    print(f"Model Load Error: {e}. Falling back to base.")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

SESSION_STORAGE = "temp_sessions"
os.makedirs(SESSION_STORAGE, exist_ok=True)

# --- UTILITIES ---

def text_to_speech(text: str):
    try:
        # ElevenLabs generates audio
        audio_stream = eleven.generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1"
        )
        return b"".join(chunk for chunk in audio_stream)
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def trim_silence(audio_path):
    # This requires ffmpeg installed on your Windows PATH
    try:
        audio = AudioSegment.from_file(audio_path)
        nonsilent_chunks = silence.split_on_silence(
            audio, min_silence_len=500, silence_thresh=audio.dbFS - 16
        )
        output = AudioSegment.empty()
        for chunk in nonsilent_chunks:
            output += chunk
        
        trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
        output.export(trimmed_path, format="wav")
        return trimmed_path
    except Exception as e:
        print(f"Trim failed (Likely missing ffmpeg): {e}")
        return audio_path # Return original if trim fails

def extract_features(audio_path):
    # librosa also needs ffmpeg for certain formats
    audio_input, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.last_hidden_state.cpu()
    mean_embedding = torch.mean(hidden_states, dim=1).numpy().tolist()[0]
    std_embedding = torch.std(hidden_states, dim=1).numpy().tolist()[0]
    return mean_embedding[:10], std_embedding[:10]

# --- ENDPOINTS ---

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...), session_id: str = Form(...), history: str = Form(...)):
    session_dir = os.path.join(SESSION_STORAGE, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    user_audio_path = os.path.join(session_dir, f"user_{int(time.time())}.wav")
    with open(user_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # FIX: Use 'gemini-pro' if 'gemini-1.5-flash' returns 404 in your current lib version
    # Or ensure your library is updated: pip install -U google-generativeai
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Friendly chat history: {history}. Reply naturally and briefly to the user."
        chat_response = model_gemini.generate_content(prompt)
        response_text = chat_response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        response_text = "I'm sorry, I'm having a bit of trouble connecting. How are you feeling today?"

    return JSONResponse(content={"response_text": response_text})

@app.get("/audio_response")
async def get_audio(text: str):
    audio = text_to_speech(text)
    if audio:
        return FileResponse(io.BytesIO(audio), media_type="audio/mpeg")
    return JSONResponse(status_code=500, content={"error": "TTS failed"})

@app.post("/analyze")
async def analyze_session(session_id: str = Form(...)):
    session_dir = os.path.join(SESSION_STORAGE, session_id)
    
    # Combined user audio logic
    files = sorted([f for f in os.listdir(session_dir) if f.startswith("user_")])
    if not files:
        return JSONResponse(status_code=400, content={"error": "No audio found"})

    try:
        combined = AudioSegment.empty()
        for f in files:
            combined += AudioSegment.from_file(os.path.join(session_dir, f))
        
        full_path = os.path.join(session_dir, "full.wav")
        combined.export(full_path, format="wav")
        trimmed_path = trim_silence(full_path)
        
        m, s = extract_features(trimmed_path)
        
        # Expert Diagnosis with Gemini
        diag_model = genai.GenerativeModel('gemini-1.5-flash')
        diag_prompt = f"""
        Analyze these speech features from the DementiaBank Wav2Vec2 model:
        Mean: {m}
        Std: {s}
        Provide a JSON response with: label, score (0-100), confidence (High/Low), and explanation.
        """
        res = diag_model.generate_content(diag_prompt)
        # Handle possible markdown in response
        clean_json = res.text.replace("```json", "").replace("```", "").strip()
        return JSONResponse(content=json.loads(clean_json))
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)