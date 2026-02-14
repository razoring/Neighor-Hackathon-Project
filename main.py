import sys
import os
import traceback
from dotenv import load_dotenv as load
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import uvicorn

load()
app = FastAPI()
dataset = os.getenv("DATASET")
#print(model)

try:
    extract = AutoFeatureExtractor.from_pretrained(dataset)
    model = AutoModelForAudioClassification.from_pretrained(dataset)
    model.eval()
except Exception as e: traceback.print_exc()

@app.post("/predict")
async def train():
    file:File = "AbeBurrows_sample.wav"
    with open(file,"wb") as buffer: buffer.write(file.read())
    speech, rate = librosa.load(file, sr=16000) #16000 current sample rate
    inputs = extract(speech, sampling_rate=rate, return_tensors="pt", padding=True, max_length=rate*30, truncation=True) #!!! rate*30 = 16000*30 seconds of processing MAX

# DO NOT TOUCH BRO
if __name__ == "__main__": uvicorn.run("main:app", host="localhost", port=6967)