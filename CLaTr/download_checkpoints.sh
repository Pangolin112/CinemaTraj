#!/usr/bin/env bash
# CLaTr checkpoints from E.T. the Exceptional Trajectories (Courant et al., ECCV'24).
# Only the encoders are needed for the CLaTr Score in src/eval/eval.py.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q gdown
python -m gdown "https://drive.google.com/uc?id=1FqN-pa955Wvu3utGViUKiVfza6cL_W0D"  # clatr-e100
python -m gdown "https://drive.google.com/uc?id=1YVrh7nhnujYMYbOQOUUek5ZTRn64K2gd"  # clatr-text_encoder
python -m gdown "https://drive.google.com/uc?id=1LkwrknkQ7bURHl9Bqj2mDqEsGZJtCkpx"  # clatr-traj_encoder
python -m gdown "https://drive.google.com/uc?id=1E-pui3CGMdW2Z7e85RYbHdMHl6W13-D0"  # clatr-traj_decoder
mkdir -p checkpoints
mv clatr-*.ckpt checkpoints/
echo "✓ CLaTr checkpoints in $(pwd)/checkpoints"
