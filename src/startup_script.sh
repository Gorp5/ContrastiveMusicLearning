#!/bin/bash
set -e  # exit on error

# Argument (must pass this when running script)
ID_BASE=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/ABLATION_ID)

NUM_MODELS=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/NUM_MODELS)

if [ -z "$ID_BASE" ]; then
  echo "Usage: bash setup_and_run.sh <ID_NUMBER_MULTIPLE_OF_16>"
  exit 1
fi

echo "Starting setup for ID base: $ID_BASE"

if ! mountpoint -q /mnt/ssd; then
  sudo mkfs.ext4 -F /dev/nvme0n1
fi

sudo mkdir -p /mnt/ssd

if ! mountpoint -q /mnt/ssd; then
  sudo mount /dev/nvme0n1 /mnt/ssd
fi
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

sudo mkdir -p /mnt/ssd/dataset
sudo mkdir -p /mnt/ssd/output

# Run training script
bash ContrastiveMusicLearning/src/individual_ablation_training.sh $ID_BASE $NUM_MODELS

OUTPUT="/mnt/ssd/output"
GCS_BUCKET="gs://mtg-jamendo-dataset/outputs"
END_ID=$((ID_BASE + NUM_MODELS))

# Create archive directory and move contents
sudo mkdir -p "${OUTPUT}-${ID_BASE}"

if compgen -G "${OUTPUT}/*" > /dev/null; then
  sudo mv ${OUTPUT}/* "${OUTPUT}-${ID_BASE}/"
else
  echo "No output files to move"
fi

# Upload to GCS
gcloud storage cp -r ${OUTPUT}-${ID_BASE} ${GCS_BUCKET}-${ID_BASE}-${END_ID} --no-clobber

sudo shutdown -h now