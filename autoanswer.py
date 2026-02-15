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
from dotenv import load_dotenv
import whisper
from pydub import AudioSegment, silence
from elevenlabs.client import ElevenLabs
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
ffprobe_path = os.path.join(project_root, r"bin\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe")
temp_folder = os.path.join(project_root, "temp")
os.makedirs(temp_folder, exist_ok=True)  # Create temp folder if it doesn't exist
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

load_dotenv()

# --- AUDIO SETTINGS ---
THRESHOLD_DB = -40     # dB threshold: -40 dB is moderate speech level
CHUNK = 1024           # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000           # Wav2Vec2 likes 16kHz
SILENCE_LIMIT = 2.0    # Seconds of silence to split clips or trigger response

# --- API/MODEL CONFIG ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "pheem49/Luna:4b"
OLLAMA_MODEL = "gemma3:1b"
eleven_client = ElevenLabs(api_key=os.getenv("TTS"))
DEMENTIA_MODEL_ID = "shields/wav2vec2-xl-960h-dementiabank"

# Global State
is_playing_audio = False
stop_audio_event = threading.Event()
no_response_count = 0
history = []
session_id = f"local_{int(time.time())}"
clips_collected = []  # Store audio clips for final analysis
idle_timeout = 0  # Track seconds of idle silence
IDLE_LIMIT = 60.0  # 10 seconds of silence before ending call
all_calls = []  # Store all call sessions
barge_in_enabled = False  # Flag to prevent barge-in on TTS startup
bargingAllowed = False

# --- MODELS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Models on {device}...")
processor = Wav2Vec2Processor.from_pretrained(DEMENTIA_MODEL_ID)
wav_model = Wav2Vec2Model.from_pretrained(DEMENTIA_MODEL_ID).to(device)

print("Loading Whisper base model...")
whisper_model = whisper.load_model("base", device=device)

# --- HELPER FUNCTIONS ---

def rms_to_db(rms_value):
    """Convert RMS amplitude to dB. Handles zero/near-zero values safely."""
    if rms_value <= 0:
        return -np.inf
    return 20 * np.log10(rms_value / 32768.0)  # Normalize by max int16 value


def get_rms(audio_chunk):
    """Calculate RMS from audio chunk safely."""
    data = np.frombuffer(audio_chunk, dtype=np.int16)
    if len(data) == 0:
        return 0
    return np.sqrt(np.mean(np.square(data.astype(np.float32))))


def is_sound_detected(audio_chunk, threshold_db=THRESHOLD_DB):
    """Check if audio exceeds dB threshold."""
    rms = get_rms(audio_chunk)
    db = rms_to_db(rms)
    return db > threshold_db

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
        
        # Return a subset (first 15 dimensions) to provide as context
        return mean_feats[:15], std_feats[:15]
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None, None

def perform_dementia_analysis(session_clips):
    """Perform dementia analysis on collected audio clips."""
    if not session_clips:
        return None
    
    try:
        # Combine all clips
        combined_audio = AudioSegment.empty()
        for clip_path in session_clips:
            try:
                combined_audio += AudioSegment.from_file(clip_path)
            except Exception as e:
                print(f"Warning: Could not load {clip_path}: {e}")
                continue
        
        if len(combined_audio) == 0:
            return None
        
        # Export combined audio temporarily
        combined_path = os.path.join(temp_folder, "combined_analysis.wav")
        combined_audio.export(combined_path, format="wav")
        
        # Trim silence
        trimmed_path = trim_silence_pydub(combined_path)
        
        # Extract features
        means, stds = extract_acoustic_features(trimmed_path)
        
        if means is None or stds is None:
            return None
        
        # Query Ollama for diagnosis
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
        
        try:
            response = requests.post(OLLAMA_API_URL,
                                   json={"model": OLLAMA_MODEL, "prompt": diagnosis_prompt, "stream": False},
                                   timeout=60)
            diagnosis_response = response.json().get("response", "") if response.status_code == 200 else ""
        except Exception as e:
            print(f"Ollama Error: {e}")
            diagnosis_response = ""
        
        # Try to parse JSON
        diagnosis_result = {}
        try:
            parsed = json.loads(diagnosis_response)
            if isinstance(parsed, dict):
                diagnosis_result = parsed
        except:
            diagnosis_result = {"explanation": diagnosis_response}
        
        # Cleanup temp files
        try:
            if os.path.exists(combined_path):
                os.remove(combined_path)
            if os.path.exists(trimmed_path):
                os.remove(trimmed_path)
        except:
            pass
        
        return diagnosis_result
    except Exception as e:
        print(f"Analysis Error: {e}")
        return None

def save_and_reset_call():
    """Save current call data and reset for next call."""
    global session_id, history, clips_collected, idle_timeout, no_response_count
    
    # Perform dementia analysis before resetting
    diagnosis = perform_dementia_analysis(clips_collected)
    
    # Save current call to all_calls list
    call_data = {
        "session_id": session_id,
        "clips": clips_collected.copy(),
        "history": history.copy(),
        "no_response_count": no_response_count,
        "timestamp": time.time(),
        "diagnosis": diagnosis
    }
    all_calls.append(call_data)
    
    # Delete all audio clips from temp folder
    for clip_path in clips_collected:
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
                print(f"[Deleted: {os.path.basename(clip_path)}]")
        except Exception as e:
            print(f"Error deleting {clip_path}: {e}")
    
    # Reset for next call
    session_id = f"local_{int(time.time())}"
    history = []
    clips_collected = []
    idle_timeout = 0
    no_response_count = 0

# --- 1. SOUND OUTPUT (SPEAKERS) ---
def play_audio(audio_bytes):
    """Plays audio bytes to system speakers with interrupt support."""
    global is_playing_audio, barge_in_enabled
    is_playing_audio = True
    barge_in_enabled = False  # Disable barge-in initially
    stop_audio_event.clear()
    
    # Convert ElevenLabs MP3 to raw PCM for PyAudio
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                    channels=audio.channels,
                    rate=audio.frame_rate,
                    output=True)
    
    # Wait 0.5 seconds before enabling barge-in to avoid detecting TTS startup noise
    time.sleep(1)
    barge_in_enabled = bargingAllowed
    
    data = audio.raw_data
    chunk_size = 1024
    for i in range(0, len(data), chunk_size):
        if stop_audio_event.is_set(): # "Barge-in" check
            break
        stream.write(data[i:i+chunk_size])
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    is_playing_audio = False
    barge_in_enabled = False

# --- 2. CORE LOGIC ---
def get_ai_response(text):
    base_prompt = "You are a friendly companion. Keep it brief."
    full_prompt = f"{base_prompt}\nUser: {text}\nHistory: {history}"
    
    try:
        response = requests.post(OLLAMA_API_URL, 
                                 json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False})
        return response.json().get("response", "")
    except:
        return "I'm listening. Tell me more."

# --- 3. MICROPHONE STREAMING LOOP ---
def start_voice_system():
    global no_response_count, is_playing_audio, history, clips_collected, idle_timeout
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)
    
    print("\n>>> System Idle. Speak to begin.")
    idle_timeout = 0

    while True:
        # Read microphone
        data = stream.read(CHUNK, exception_on_overflow=False)

        # BARGE-IN: If AI is talking and you shout, AI stops talking
        if is_playing_audio and barge_in_enabled and is_sound_detected(data, THRESHOLD_DB + 5):  # Higher threshold for interrupt
            stop_audio_event.set()
            print("\n[Barge-in detected]")
            time.sleep(0.1)  # Brief pause before listening

        # TRIGGER CONVERSATION: If sound exceeds threshold and AI isn't talking
        if is_sound_detected(data, THRESHOLD_DB) and not is_playing_audio:
            idle_timeout = 0  # Reset idle timer on sound detection
            print("[Listening...]")
            frames = []
            silent_chunks = 0
            max_silent_chunks = int(SILENCE_LIMIT * RATE / CHUNK)
            
            # Record until 2 seconds of silence
            while silent_chunks < max_silent_chunks:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                if is_sound_detected(data, THRESHOLD_DB):
                    silent_chunks = 0  # Reset silence timer
                else:
                    silent_chunks += 1

            # Process the recorded clip
            print("[Processing...]")
            temp_wav = os.path.join(temp_folder, f"temp_segment_{len(clips_collected)}.wav")
            wf = wave.open(temp_wav, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Store clip for final analysis
            clips_collected.append(temp_wav)

            # Transcribe using Whisper
            try:
                result = whisper_model.transcribe(temp_wav, language="en", fp16=torch.cuda.is_available())
                user_text = result["text"].strip()
                
                if user_text and len(user_text) > 0:
                    print(f"User: {user_text}")
                    no_response_count = 0
                    
                    # Get AI Text
                    reply = get_ai_response(user_text)
                    print(f"AI: {reply}")
                    history.append({"u": user_text, "a": reply})

                    # ElevenLabs TTS
                    audio_stream = eleven_client.text_to_speech.convert(
                        text=reply, voice_id="21m00Tcm4TlvDq8ikWAM", model_id="eleven_turbo_v2_5"
                    )
                    audio_bytes = b"".join(chunk for chunk in audio_stream)
                    
                    # Play audio in a separate thread so we can still listen for barge-ins
                    threading.Thread(target=play_audio, args=(audio_bytes,)).start()
                else:
                    no_response_count += 1
                    print(f"No speech recognized ({no_response_count}/2)")
                    idle_timeout = 0  # Reset idle counter on silence detection
            except Exception as e:
                print(f"Transcription Error: {e}")
                no_response_count += 1
                idle_timeout = 0  # Reset idle counter on error
        
        # IDLE TIMEOUT: Increment idle counter if no sound, check for 10-second silence
        else:
            idle_timeout += CHUNK / RATE  # Accumulate silence duration
            if idle_timeout >= IDLE_LIMIT:
                # Announce call ending
                print("\n[10 seconds of silence detected]")
                print(">>> Announcing call ending...")
                try:
                    audio_stream = eleven_client.text_to_speech.convert(
                        text="Thank you for the call. Goodbye.", 
                        voice_id="21m00Tcm4TlvDq8ikWAM", 
                        model_id="eleven_turbo_v2_5"
                    )
                    audio_bytes = b"".join(chunk for chunk in audio_stream)
                    play_audio(audio_bytes)
                except Exception as e:
                    print(f"TTS Error: {e}")
                
                # Print final analysis for this call
                print("\n>>> CALL ENDED")
                print("\n=== CALL ANALYSIS ===")
                print(f"Session ID: {session_id}")
                print(f"Total Clips Collected: {len(clips_collected)}")
                print(f"Conversation History:")
                for i, exchange in enumerate(history, 1):
                    print(f"  {i}. User: {exchange['u']}")
                    print(f"     AI: {exchange['a']}")
                print(f"No-Response Count: {no_response_count}")
                
                # Show dementia analysis if available
                if len(all_calls) > 0 and all_calls[-1].get("diagnosis"):
                    diagnosis = all_calls[-1]["diagnosis"]
                    print("\n>>> DEMENTIA ANALYSIS ===")
                    if "label" in diagnosis:
                        print(f"Assessment: {diagnosis.get('label', 'Unknown')}")
                        print(f"Score: {diagnosis.get('score', 'N/A')}/100")
                        print(f"Confidence: {diagnosis.get('confidence', 'Unknown')}")
                    if "explanation" in diagnosis:
                        print(f"Explanation: {diagnosis['explanation']}")
                    print("=== END DEMENTIA ANALYSIS ===")
                
                print("=== END CALL ANALYSIS ===\n")
                
                # Save call data and reset for next call
                save_and_reset_call()
                print(">>> System Idle. Waiting for next call...")
                time.sleep(1)  # Brief pause before listening again

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    start_voice_system()
    
    # Print overall session summary when manually stopped
    print("\n\n=== OVERALL SESSION SUMMARY ===")
    print(f"Total Calls: {len(all_calls)}")
    for i, call in enumerate(all_calls, 1):
        print(f"\nCall {i} (Session: {call['session_id']}):")
        print(f"  Clips: {len(call['clips'])}")
        print(f"  Exchanges: {len(call['history'])}")
        print(f"  No-Response Count: {call['no_response_count']}")
        if call.get("diagnosis"):
            diagnosis = call["diagnosis"]
            print(f"  Diagnosis: {diagnosis.get('label', 'Unknown')} (Score: {diagnosis.get('score', 'N/A')}, Confidence: {diagnosis.get('confidence', 'Unknown')})")
    print("=== END SESSION SUMMARY ===\n")