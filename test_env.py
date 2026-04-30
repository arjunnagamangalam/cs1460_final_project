import torch
import transformers
import datasets

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")

# Check for hardware acceleration
if torch.cuda.is_available():
    print("Device: NVIDIA CUDA is available! 🚀")
elif torch.backends.mps.is_available():
    print("Device: Apple Silicon MPS is available! 🚀")
else:
    print("Device: CPU only (Fine for debugging, slow for training) 🐢")