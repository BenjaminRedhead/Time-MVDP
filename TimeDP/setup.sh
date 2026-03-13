#!/bin/bash
# Created for internship coding assessment to set up base environment
set -e

# ---- Install Miniconda ----
apt-get update && apt-get install -y wget
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
rm /tmp/miniconda.sh

# Add conda to PATH for this session
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# ---- Accept Terms of Service ----
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- Setup project ----
cd /workspace/TimeCraft/TimeDP
conda env create -f environment.yml
conda activate timedp

bash run_experiment.sh