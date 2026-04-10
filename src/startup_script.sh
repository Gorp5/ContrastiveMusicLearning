#!/bin/bash
set -e  # exit on error

# Argument (must pass this when running script)
ID_BASE=$1

if [ -z "$ID_BASE" ]; then
  echo "Usage: bash setup_and_run.sh <ID_NUMBER_MULTIPLE_OF_16>"
  exit 1
fi

echo "Starting setup for ID base: $ID_BASE"

# System setup
sudo apt update
sudo apt install -y python3.10-venv

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip (good practice)
pip install --upgrade pip

# Install requirements
pip install -r ContrastiveMusicLearning/src/requirements.txt

sudo mkdir dataset
sudo mkdir output

# Run training script
bash ContrastiveMusicLearning/src/8xnode_training.sh $ID_BASE