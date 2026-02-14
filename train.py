import sys
import os
import traceback
from dotenv import load_dotenv as load
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

load()
app = FastAPI()
dataset = os.getenv("DATASET")
#print(model)

try:
    extract = AutoFeatureExtractor.from_pretrained(dataset)
    model = AutoModelForAudioClassification.from_pretrained(dataset)
    model.eval()
except Exception as e: traceback.print_exc()