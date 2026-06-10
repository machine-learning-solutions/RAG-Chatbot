#!/usr/bin/env bash
# Verify NVIDIA GPU passthrough and CUDA compute inside Docker.
set -euo pipefail

echo "=== Host ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== Docker passthrough (nvidia-smi) ==="
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L

echo ""
echo "=== CUDA runtime (cuInit) ==="
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 bash -c '
  apt-get update -qq && apt-get install -y -qq python3 >/dev/null
  python3 << "PY"
import ctypes
cuda = ctypes.CDLL("libcuda.so.1")
rc = cuda.cuInit(0)
count = ctypes.c_int()
cuda.cuDeviceGetCount(ctypes.byref(count))
print(f"cuInit={rc} (0=OK, 999=driver broken) devices={count.value}")
PY
'

echo ""
echo "=== PyTorch (cu121) ==="
docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime \
  python3 -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
