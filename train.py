import sys
import os
from dotenv import load_dotenv as load
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

load()
app = FastAPI()
model = os.getenv("DATASET")