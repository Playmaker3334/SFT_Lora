#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends fonts-lmodern

pip install --ignore-installed cryptography
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y torchvision torchaudio
pip install --no-build-isolation --no-cache-dir "causal-conv1d==1.7.0"
pip install -e .
python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
