import os
import time
import shutil
import io
import json
import torch
import librosa
import numpy as np
import requests
import pyaudio
import wave
import threading
import re
from dotenv import load_dotenv
import whisper
from pydub import AudioSegment, silence
from elevenlabs.client import ElevenLabs
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# --- WEB SERVER SETUP ---
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.abspath(__file__))
# Adjust these paths if necessary
ffmpeg_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
ffprobe_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe")
temp_folder = os.path.join(project_root, "temp")
os.makedirs(temp_folder, exist_ok=True)
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

load_dotenv()

# --- AUDIO SETTINGS ---
THRESHOLD_DB = -40
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_LIMIT = 2.0
IDLE_LIMIT = 30.0

# --- API/MODEL CONFIG ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "pheem49/Luna:4b"
eleven_client = ElevenLabs(api_key=os.getenv("TTS"))
DEMENTIA_MODEL_ID = "shields/wav2vec2-xl-960h-dementiabank"

# Global State
state = {
    "is_playing_audio": False,
    "history": [],
    "clips_collected": [],
    "session_id": f"local_{int(time.time())}",
    "status": "Idle"
}

stop_audio_event = threading.Event()
barge_in_enabled = False
bargingAllowed = False

# --- MODELS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Models on {device}...")
processor = Wav2Vec2Processor.from_pretrained(DEMENTIA_MODEL_ID)
wav_model = Wav2Vec2Model.from_pretrained(DEMENTIA_MODEL_ID).to(device)
whisper_model = whisper.load_model("base", device=device)

# --- HELPER FUNCTIONS ---
def rms_to_db(rms_value):
    if rms_value <= 0: return -np.inf
    return 20 * np.log10(rms_value / 32768.0)

def get_rms(audio_chunk):
    data = np.frombuffer(audio_chunk, dtype=np.int16)
    if len(data) == 0: return 0
    return np.sqrt(np.mean(np.square(data.astype(np.float32))))

def is_sound_detected(audio_chunk, threshold_db=THRESHOLD_DB):
    return rms_to_db(get_rms(audio_chunk)) > threshold_db

def trim_silence_pydub(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = silence.split_on_silence(audio, min_silence_len=600, silence_thresh=audio.dBFS - 16)
        output = AudioSegment.empty()
        for chunk in chunks: output += chunk
        trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
        output.export(trimmed_path, format="wav")
        return trimmed_path
    except: return audio_path

def extract_acoustic_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(): outputs = wav_model(**inputs)
        embeddings = outputs.last_hidden_state.cpu()
        mean_feats = torch.mean(embeddings, dim=1).numpy().tolist()[0]
        std_feats = torch.std(embeddings, dim=1).numpy().tolist()[0]
        return mean_feats[:15], std_feats[:15]
    except Exception as e:
        print(f"Feature Error: {e}")
        return None, None

def detect_breathing_patterns(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0: return None
        frame_length, hop_length = 2048, 512
        env = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        win = 5
        env_smooth = np.convolve(env, np.ones(win) / win, mode="same") if len(env) >= win else env
        thresh = max(env_smooth.mean() * 1.1, np.percentile(env_smooth, 60) * 0.5)
        
        peaks = []
        for i in range(1, len(env_smooth) - 1):
            if env_smooth[i] > env_smooth[i - 1] and env_smooth[i] > env_smooth[i + 1] and env_smooth[i] > thresh:
                peaks.append(i)
        
        times = [p * hop_length / sr for p in peaks]
        intervals = np.diff(times) if len(times) >= 2 else np.array([])
        
        breath_rate_bpm = 60.0 / float(np.mean(intervals)) if len(intervals) > 0 and np.mean(intervals) > 0 else 0.0
        cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 0 and np.mean(intervals) > 0 else 0
        
        return {
            "breath_rate_bpm": round(breath_rate_bpm, 2),
            "interval_cv": round(cv, 3) if cv else 0,
            "num_breath_events": len(times),
            "irregular": (cv > 0.35) or (len(times) < 3),
            "laboured": (breath_rate_bpm > 24 or env_smooth.mean() > (env.mean() * 2.0))
        }
    except: return None

def perform_health_analysis(session_clips):
    if not session_clips: return None
    try:
        combined_audio = AudioSegment.empty()
        for clip_path in session_clips:
            try: combined_audio += AudioSegment.from_file(clip_path)
            except: continue
        
        combined_path = os.path.join(temp_folder, "combined_analysis.wav")
        combined_audio.export(combined_path, format="wav")
        trimmed_path = trim_silence_pydub(combined_path)
        means, stds = extract_acoustic_features(trimmed_path)
        breathing_report = detect_breathing_patterns(trimmed_path)
        
        if means is None: return None

        prompt = f"""Act as a Neuro-Speech Pathologist. 
        Breathing Summary: {breathing_report}
        Determine likelihood of cognitive impairment. 
        Respond in JSON: {{ "label": "Dementia"|"Healthy", "score": 0-100, "confidence": "High"|"Medium"|"Low", "explanation": "string" }}
        ONLY JSON."""

        try:
            response = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
            raw_text = response.json().get("response", "")
            
            # Clean JSON from Markdown blocks
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                diagnosis_result = json.loads(json_match.group(0))
            else:
                diagnosis_result = {"explanation": raw_text}
        except Exception as e:
            diagnosis_result = {"explanation": str(e)}

        report = {"dementia_assessment": diagnosis_result, "breathing": breathing_report}
        socketio.emit('health_report', report) # SEND TO WEB
        return report
    except Exception as e:
        print(f"Analysis Error: {e}")
        return None

def play_audio(audio_bytes):
    global barge_in_enabled
    state["is_playing_audio"] = True
    barge_in_enabled = False
    stop_audio_event.clear()
    
    socketio.emit('status_update', {'status': 'AI Speaking'})
    
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels, rate=audio.frame_rate, output=True)
    
    time.sleep(0.5)
    barge_in_enabled = bargingAllowed
    
    data = audio.raw_data
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        if stop_audio_event.is_set(): break
        stream.write(data[i:i+chunk_size])
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    state["is_playing_audio"] = False
    barge_in_enabled = False
    socketio.emit('status_update', {'status': 'Listening'})

def audio_loop():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print(">>> System Ready.")
    socketio.emit('status_update', {'status': 'Idle'})
    idle_timeout = 0
    no_response_count = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Barge-in
        if state["is_playing_audio"] and barge_in_enabled and is_sound_detected(data, THRESHOLD_DB + 5):
            stop_audio_event.set()
            socketio.emit('status_update', {'status': 'Interrupted'})
            time.sleep(0.1)

        # Trigger
        if is_sound_detected(data, THRESHOLD_DB) and not state["is_playing_audio"]:
            idle_timeout = 0
            socketio.emit('status_update', {'status': 'Recording'})
            frames = []
            silent_chunks = 0
            max_silent_chunks = int(SILENCE_LIMIT * RATE / CHUNK)
            
            while silent_chunks < max_silent_chunks:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                if is_sound_detected(data, THRESHOLD_DB): silent_chunks = 0
                else: silent_chunks += 1

            temp_wav = os.path.join(temp_folder, f"temp_{len(state['clips_collected'])}.wav")
            wf = wave.open(temp_wav, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            state['clips_collected'].append(temp_wav)

            socketio.emit('status_update', {'status': 'Transcribing'})
            try:
                result = whisper_model.transcribe(temp_wav)
                user_text = result["text"].strip()
                
                if user_text:
                    no_response_count = 0
                    socketio.emit('chat_message', {'role': 'user', 'text': user_text})
                    
                    full_prompt = f"User: {user_text}\nHistory: {state['history']}"
                    resp = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False})
                    reply = resp.json().get("response", "")
                    
                    state['history'].append({"u": user_text, "a": reply})
                    socketio.emit('chat_message', {'role': 'ai', 'text': reply})
                    
                    audio_stream = eleven_client.text_to_speech.convert(text=reply, voice_id="hpp4J3VqNfWAUOO0d1Us", model_id="eleven_flash_v2_5")
                    audio_bytes = b"".join(chunk for chunk in audio_stream)
                    threading.Thread(target=play_audio, args=(audio_bytes,)).start()
                else:
                    no_response_count += 1
            except Exception as e:
                print(e)
        
        else:
            idle_timeout += CHUNK / RATE
            if idle_timeout >= IDLE_LIMIT:
                socketio.emit('status_update', {'status': 'Analyzing'})
                perform_health_analysis(state['clips_collected'])
                
                # Cleanup
                state['clips_collected'] = []
                state['history'] = []
                state['session_id'] = f"local_{int(time.time())}"
                idle_timeout = 0
                socketio.emit('session_reset', {'session_id': state['session_id']})
                socketio.emit('status_update', {'status': 'Idle'})

def run_audio_thread():
    t = threading.Thread(target=audio_loop)
    t.daemon = True
    t.start()

if __name__ == "__main__":
    run_audio_thread()
    socketio.run(app, port=5000)