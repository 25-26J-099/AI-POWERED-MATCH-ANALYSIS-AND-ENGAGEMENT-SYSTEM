#!/usr/bin/env python3
"""Quick test to verify Qwen3-TTS voice cloning works with one reference sample."""

import os
import sys
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== Qwen3-TTS Voice Clone Test ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda"):
    # Fallback to float16 if bfloat16 is not supported by the GPU architecture
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32

print(f"\nLoading Qwen3-TTS-0.6B-Base on {device} (dtype={dtype})...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map=device,
    torch_dtype=dtype,  # Changed from dtype=dtype to torch_dtype=dtype
)
print("Model loaded successfully!")

# Test with Neutral.mp3
ref_audio = os.path.join(SCRIPT_DIR, "TTS", "Neutral.mp3")
ref_text = (
    "England tactical formation seems decent, Belgium players are gathering "
    "around for a strategic move, the opposition seems to adapt to this properly as well"
)

print(f"\nBuilding voice clone prompt from: {ref_audio}")
prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
)
print("Voice clone prompt built successfully!")

test_text = "And the ball is passed to the midfielder, a great tactical play by the team."
print(f"\nGenerating speech for: '{test_text}'")
wavs, sr = model.generate_voice_clone(
    text=test_text,
    language="English",
    voice_clone_prompt=prompt,
)

output_path = os.path.join(SCRIPT_DIR, "test_qwen_tts_output.wav")
sf.write(output_path, wavs[0], sr)
print(f"\nOutput saved to: {output_path}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(wavs[0]) / sr:.2f}s")

if torch.cuda.is_available():
    print(f"GPU memory used: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.0f} MB")

print("\n=== TEST PASSED ===")
