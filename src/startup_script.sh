#!/bin/bash
set -e  # exit on error



# Argument (must pass this when running script)
ID_BASE=$1
NUM_MODELS=$2

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

[ -d "dataset" ] && echo "exists" || sudo mkdir dataset
[ -d "output" ] && echo "exists" || sudo mkdir output
# Run training script
bash ContrastiveMusicLearning/src/individual_ablation_training.sh $ID_BASE $NUM_MODELS

ID_BASE=$1
OUTPUT="/home/pordanjhillips/output"
GCS_BUCKET="gs://mtg-jamendo/SongsDataset/models"

# Create archive directory and move contents
sudo mkdir -p "${OUTPUT}-${ID_BASE}"
sudo mv ${OUTPUT}/* "${OUTPUT}-${ID_BASE}/"

# Upload to GCS
gsutil -m cp -r "${OUTPUT}-${ID_BASE}" "${GCS_BUCKET}"