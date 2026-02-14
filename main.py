import os
import time
import shutil
import io
import torch
import librosa
import numpy as np
import google.generativeai as genai
import json
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment, silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import speech_recognition as sr

load_dotenv()

# --- CONFIGURATION ---
app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize APIs
genai.configure(api_key=os.getenv("GEMINI"))
eleven = ElevenLabs(api_key=os.getenv("TTS"))

# Load the specific DementiaBank Model
# Note: This model is an ASR model fine-tuned on DementiaBank. 
# We will use it as a Feature Extractor to get embeddings specific to this domain.
MODEL_ID = "shields/wav2vec2-xl-960h-dementiabank"
print("Loading Model...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    print("Model Loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to standard wav2vec2 if specific one fails or is gated
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Storage for session audio
SESSION_STORAGE = "temp_sessions"
os.makedirs(SESSION_STORAGE, exist_ok=True)

# --- HELPER FUNCTIONS ---

def text_to_speech(text: str):
    """Converts text to speech using ElevenLabs."""
    try:
        audio_stream = eleven.generate(
            text=text,
            voice="Rachel", # Friendly voice
            model="eleven_monolingual_v1"
        )
        # Convert generator to bytes
        audio_bytes = b"".join(chunk for chunk in audio_stream)
        return audio_bytes
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def trim_silence(audio_path):
    """Trims silence from the audio file using pydub."""
    audio = AudioSegment.from_file(audio_path)
    # Detect non-silent chunks
    nonsilent_chunks = silence.split_on_silence(
        audio, min_silence_len=500, silence_thresh=audio.dbFS - 16
    )
    # Recombine
    output = AudioSegment.empty()
    for chunk in nonsilent_chunks:
        output += chunk
    
    trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
    output.export(trimmed_path, format="wav")
    return trimmed_path

def extract_features(audio_path):
    """
    Runs the audio through the Wav2Vec2 model to extract embeddings.
    These embeddings represent the 'acoustic signature' as learned by the DementiaBank model.
    """
    audio_input, _ = librosa.load(audio_path, sr=16000)
    
    # Process inputs
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    # move tensors to device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state (features)
    # Shape: [1, Sequence_Length, Hidden_Size]
    hidden_states = outputs.last_hidden_state.cpu()
    
    # Calculate simple statistics on the embeddings to pass to Gemini
    # (Mean and Variance of the acoustic features)
    mean_embedding = torch.mean(hidden_states, dim=1).numpy().tolist()[0]
    std_embedding = torch.std(hidden_states, dim=1).numpy().tolist()[0]
    
    return mean_embedding[:10], std_embedding[:10] # Return first 10 dims for brevity in prompt


def run_local_inference(trimmed_audio_path: str):
    """
    Run inference locally on the available device (GPU if present).
    Extract features via the loaded wav2vec2 model, transcribe audio, and score locally.
    Returns a dict with keys: label, score, confidence, features_mean, features_std, transcript
    """
    try:
        features_mean, features_std = extract_features(trimmed_audio_path)
        transcript = transcribe_audio(trimmed_audio_path)
        score_obj = local_score_from_features(features_mean, features_std)
        return {
            "label": score_obj.get("label"),
            "score": score_obj.get("score"),
            "confidence": score_obj.get("confidence"),
            "features_mean": features_mean,
            "features_std": features_std,
            "transcript": transcript,
        }
    except Exception as e:
        print(f"Local inference failed: {e}")
        return None


def transcribe_audio(trimmed_audio_path: str):
    """Try to transcribe using `speech_recognition` if available; return string or None."""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(trimmed_audio_path) as source:
            audio = r.record(source)
        # try Google Web Speech (requires internet but no API key for small requests)
        try:
            text = r.recognize_google(audio)
            return text
        except Exception:
            try:
                text = r.recognize_sphinx(audio)
                return text
            except Exception:
                return None
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None


def local_score_from_features(mean_feats, std_feats):
    """Simple heuristic scorer converting features to label/score/confidence."""
    try:
        vals = [abs(x) for x in mean_feats]
        score = int(min(100, max(0, sum(vals) / (len(vals) or 1) * 10)))
        # Higher variance indicates more irregular speech -> increase score slightly
        vare = sum([abs(x) for x in std_feats]) / (len(std_feats) or 1)
        score = min(100, int(score + vare * 5))
        label = "Dementia" if score > 55 else "Healthy"
        confidence = "High" if score > 75 or score < 25 else ("Medium" if score > 45 and score < 65 else "Low")
        return {"label": label, "score": score, "confidence": confidence}
    except Exception as e:
        print(f"Scoring failed: {e}")
        return {"label": "Unknown", "score": 0, "confidence": "Low"}

# --- ENDPOINTS ---

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...), session_id: str = Form(...), history: str = Form(...)):
    """
    Receives user audio, transcribes, gets Gemini response, returns TTS audio.
    """
    # 1. Save User Audio
    session_dir = os.path.join(SESSION_STORAGE, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    user_audio_path = os.path.join(session_dir, f"user_{int(time.time())}.wav")
    with open(user_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Transcribe User Audio (Using Gemini Multimodal or Whisper)
    # For simplicity/speed in this example, we use Gemini Pro Vision/1.5 if available, 
    # or just assume the frontend sends text. Let's use Gemini to transcribe/listen.
    
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    
    # Upload audio to Gemini for transcription/understanding
    # (In a real hackathon, standard STT like Whisper is safer, but let's use Gemini capabilities)
    # Reverting to simple STT pattern:
    
    # Let's prompt Gemini to be a friendly chat companion
    prompt = f"""
    You are a friendly, empathetic phone companion having a casual chat. 
    Keep your responses short (1-2 sentences) and conversational.
    History: {history}
    User just said something (audio provided). Reply naturally.
    """
    
    # Note: Sending audio bytes directly to Gemini 1.5 is possible via the API 
    # but requires File API upload. For hackathon speed, we'll assume the prompt works
    # or you can use `speech_recognition` library here. 
    # *Mocking Transcript for the code block stability if no STT key provided*
    # Real implementation: Use Whisper here.
    
    # Generating Response
    chat_response = model_gemini.generate_content([prompt]) # If using 1.5-flash with audio, pass audio blob
    response_text = chat_response.text
    
    # 3. Generate TTS
    audio_bytes = text_to_speech(response_text)
    
    # Save response audio locally for archive
    with open(os.path.join(session_dir, f"system_{int(time.time())}.mp3"), "wb") as f:
        f.write(audio_bytes)

    return JSONResponse(content={
        "response_text": response_text,
        "audio_base64": None # Frontend handles blob, we stream or return file? 
        # Better: return file directly or base64. 
        # We will return the text and let frontend request audio or send base64 here.
    })

@app.get("/audio_response")
async def get_audio(text: str):
    """Helper to get audio if not bundled"""
    audio = text_to_speech(text)
    return FileResponse(io.BytesIO(audio), media_type="audio/mpeg")

@app.post("/analyze")
async def analyze_session(session_id: str = Form(...)):
    """
    The Core Diagnostic Logic.
    1. Aggregates all user audio.
    2. Trims Silence.
    3. Runs via HuggingFace Model (Shields/Wav2Vec2).
    4. Asks Gemini for Diagnosis.
    """
    session_dir = os.path.join(SESSION_STORAGE, session_id)

    # Validate session directory
    if not os.path.isdir(session_dir):
        return JSONResponse(status_code=404, content={
            "error": "session_not_found",
            "message": f"No session directory for '{session_id}'"
        })

    # 1. Aggregate Audio
    combined = AudioSegment.empty()
    files = sorted([f for f in os.listdir(session_dir) if f.startswith("user_")])
    if not files:
        return JSONResponse(status_code=400, content={
            "error": "no_audio_files",
            "message": "No user_*.wav files found in session"
        })

    for f in files:
        try:
            combined += AudioSegment.from_file(os.path.join(session_dir, f))
        except Exception as e:
            print(f"Failed to load audio file {f}: {e}")

    full_audio_path = os.path.join(session_dir, "full_user_audio.wav")
    combined.export(full_audio_path, format="wav")

    # 2. Trim Silence
    try:
        trimmed_path = trim_silence(full_audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "trim_failed",
            "message": str(e)
        })

    # 3. Run local inference (preferred) to extract features, transcript and score
    local_result = run_local_inference(trimmed_path)
    if local_result is None:
        return JSONResponse(status_code=500, content={"error": "local_inference_failed"})

    # 4. Ask Gemini to explain the diagnosis reasoning using the features & transcript
    explanation_prompt = f"""
    You are a medical expert in neurology and speech pathology. A local model analyzed a user's speech and returned this result:

    Transcript: {local_result.get('transcript')}
    Acoustic Feature Mean (First 10 dims): {local_result.get('features_mean')}
    Acoustic Feature Variance (First 10 dims): {local_result.get('features_std')}
    Local Score: {local_result.get('score')} (label: {local_result.get('label')}, confidence: {local_result.get('confidence')})

    Based on the transcript and these acoustic markers, provide a concise explanation of how the diagnosis was reached. Include:
    - A label ("Dementia" or "Healthy")
    - A score (0-100)
    - A confidence level (High/Medium/Low)
    - A short explanation referencing possible markers (e.g., pauses, filler words, acoustic jitter, language errors) and how the DementiaBank dataset informed this.

    Return output as strict JSON only with keys: label, score, confidence, explanation
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        resp = model.generate_content([explanation_prompt])
        text = resp.text.replace("```json", "").replace("```", "").strip()
        try:
            parsed_explanation = json.loads(text)
        except Exception:
            # If parsing fails, return the raw explanation under 'explanation_text'
            parsed_explanation = {"explanation_text": text}

        # Combine local inference and Gemini explanation
        out = {
            "local_inference": local_result,
            "gemini_explanation": parsed_explanation,
        }
        return JSONResponse(content=out)
    except Exception as e:
        print(f"Explanation generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": "explanation_failed", "message": str(e), "local_inference": local_result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)