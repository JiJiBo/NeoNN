import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")  # 使用 MPS
elif torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用 CUDA
else:
    device = torch.device("cpu")  # 使用 CPU

print(f"Using device: {device}")
