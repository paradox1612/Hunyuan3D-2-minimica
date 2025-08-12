FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y \
    python3 python3-pip git build-essential wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 libgstreamer-plugins-bad1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
RUN pip install flash-attn==2.8.0.post2

# Copy source code
COPY . .
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda
# Build custom rasterizer
RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install

# Pre-download BOTH models
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
import os; \
print('📥 Downloading Hunyuan3D-2mv...'); \
snapshot_download(repo_id='tencent/Hunyuan3D-2mv', allow_patterns='hunyuan3d-dit-v2-mv/*', local_dir='./model-cache', local_dir_use_symlinks=False); \
print('📥 Downloading Hunyuan3D-2.1...'); \
snapshot_download(repo_id='tencent/Hunyuan3D-2.1', allow_patterns='hunyuan3d-dit-v2-1/*', local_dir='./model-cache', local_dir_use_symlinks=False); \
[os.makedirs(f'./model-cache/{model_dir}/hunyuan3d-dit-v2-0', exist_ok=True) for model_dir in ['hunyuan3d-dit-v2-mv', 'hunyuan3d-dit-v2-1']]; \
[[os.rename(f'./model-cache/{model_dir}/{file}', f'./model-cache/{model_dir}/hunyuan3d-dit-v2-0/{file}') for file in ['config.yaml', 'model.fp16.ckpt', 'model.fp16.safetensors'] if os.path.exists(f'./model-cache/{model_dir}/{file}')] for model_dir in ['hunyuan3d-dit-v2-mv', 'hunyuan3d-dit-v2-1']]; \
print('✓ All models cached')"

# Pre-download rembg model
RUN python3 -c "from hy3dgen.rembg import BackgroundRemover; BackgroundRemover(); print('✓ Rembg model cached')"

# Set environment variables
ENV HY3DGEN_MODELS=/app/model-cache

# Expose port (if needed for API)
EXPOSE 8000

# Default command
CMD ["python3", "api.py"]