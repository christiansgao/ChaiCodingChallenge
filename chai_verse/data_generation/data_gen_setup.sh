# create env (Python 3.10 is stable with vLLM + Flash-Attn)
conda create -n chai python=3.10 -y
conda activate chai

nvidia-smi
#!/bin/bash
set -e

# ---- Core system build deps ----
conda update -n base -c conda-forge conda
conda install -c conda-forge cmake>=3.25 ninja -y
conda install -c conda-forge pyarrow sentencepiece -y

pip install \
    torch==2.1.2 \
    transformers==4.41.2 \
    vllm==0.4.0 \
    pydantic==2.6.4 \
    uvicorn==0.29.0 \
    nvidia-ml-py3==7.352.0 \
    prometheus-client==0.20.0 \
    py-cpuinfo==9.0.0 \
    lark-parser==0.12.0 \
    blobfile==2.1.1 \
    outlines==0.0.34 \
    numpy==1.26.4 \
    datasets==2.19.1 \
    accelerate==0.30.1 \
    peft==0.11.1 \
    openai==1.34.0


sudo amazon-linux-extras install epel -y
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y
git lfs install
git lfs version

# ---- Optional: FlashAttention2 for faster training/inference ----
pip install flash-attn --no-build-isolation || echo "FlashAttention2 build skipped (check CUDA compat)"

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"


huggingface-cli login

export HUGGINGFACE_HUB_TOKEN=
conda activate chai
nvidia-smi | grep 'python' | awk '{ print $5 }' | sudo xargs -n1 kill -9
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --served-model-name gen7b --port 8000

# Judge

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model teknium/OpenHermes-2.5-Mistral-7B \
  --dtype bfloat16 --max-model-len 2048 \
  --trust-remote-code --served-model-name judge7b --port 8001










