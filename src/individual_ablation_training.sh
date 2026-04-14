#!/bin/bash
set -e

# ===============================
# CONFIGURATION
# ===============================

GCS_BUCKET="gs://mtg-jamendo-dataset/SongsDataset"
BATCH_SIZE=512
EPOCHS=128

ABLATION_ID=$1
NUM_MODELS=$2

DATASET="/mnt/ssd/dataset"
OUTPUT="/mnt/ssd/output"

# Cache dataset locally
sudo mkdir -p "$DATASET"
sudo mkdir -p "$OUTPUT"

sudo chown -R $USER:$USER "$DATASET" "$OUTPUT"

echo "Checking dataset cache..."

DATASET_MARKER="${DATASET}/.cache_complete"

if [ -f "$DATASET_MARKER" ]; then
  echo "Dataset already cached (marker found)."
else
  echo "Caching dataset..."

  gcloud storage cp -r "${GCS_BUCKET}/train*.bin" "$DATASET/"
  gcloud storage cp "${GCS_BUCKET}/index.npy" "$DATASET/"

  # Mark cache as complete
  touch "$DATASET_MARKER"
fi

# Run training
echo "Starting training for ablation $ABLATION_ID..."
python3 ContrastiveMusicLearning/src/individual_ablation_training.py \
    --id $ABLATION_ID \
    --num_models ${NUM_MODELS} \
    --save_dir ${OUTPUT} \
    --train_data_dir ${DATASET} \
    --chunk_length 2048 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS}