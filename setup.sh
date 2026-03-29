# Create and activate conda environment
conda create -n craft-env python=3.10 -y
conda activate craft-env

# Install PyTorch 2.4.0 with CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core pinned dependencies
pip install \
    diffusers==0.27.2 \
    transformers==4.37.2 \
    peft==0.12.0 \
    accelerate==0.31.0 \
    tokenizers==0.15.2 \
    safetensors \
    huggingface_hub

# Install everything else unpinned
pip install \
    pillow \
    opencv-python \
    imageio \
    numpy \
    scipy \
    omegaconf \
    einops \
    ftfy \
    sentencepiece \
    tensorboard \
    tqdm \
    loralib \
    pytorch-wavelets \
    PyWavelets \
    pyiqa \
    basicsr \
    facexlib \
    ptflops \
    numba \
    shapely \
    safetensors

# Install xformers last — must match torch version
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"