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
async def train(file:UploadFile = File(...)):
    temp = f"{file.filename}_TEMP"
    with open(temp,"wb") as buffer: buffer.write(await file.read())

    speech, rate = librosa.load(file, sr=16000) #16000 current sample rate
    inputs = extract(speech, sampling_rate=rate, return_tensors="pt", padding=True, max_length=rate*30, truncation=True) #!!! rate*30 = 16000*30 seconds of processing MAX
    with torch.no_grad(): infer = model(**inputs).logits

    prediction = torch.nn.functional.softmax(infer, dim=1)
    item = torch.argmax(prediction, dim=1).item()
    label = model.config.id2label[item]
    confidence = float(prediction[0][item].item())

    return {"prediction":label, "scores": {model.config.id2label[i]: float(prediction[0][i]) for i in range(len(prediction[0]))}, "confidence":confidence}


# DO NOT TOUCH BRO
if __name__ == "__main__": uvicorn.run("main:app", host="localhost", port=6967, reload=True)