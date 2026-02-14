import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
