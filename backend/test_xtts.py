import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.api import TTS

print("Loading XTTSv2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"XTTSv2 loaded successfully on {device}!")
