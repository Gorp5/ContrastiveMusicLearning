#!/bin/bash
set -e

# ===============================
# CONFIGURATION
# ===============================
GCS_BUCKET="gs://mtg-jamendo/SongsDataset"
BATCH_SIZE=512
EPOCHS=128
ABLATION_ID=$1
DATASET="/home/pordanjhillips/dataset"
OUTPUT="/home/pordanjhillips/output"

# Cache dataset locally
echo "Caching dataset locally..."
sudo chown -R $USER:$USER /home/pordanjhillips/dataset

if [ ! -d "$DATASET" ] || [ -z "$(ls -A "$DATASET")" ]; then
  echo "Caching dataset..."
  gsutil -m cp -r ${GCS_BUCKET}/train*.bin ${DATASET}
  gsutil -m cp -r ${GCS_BUCKET}/index.npy ${DATASET}
else
  echo "Dataset already cached."
fi

# Run training
echo "Starting training for ablation $ABLATION_ID..."
python3 ContrastiveMusicLearning/src/individual_ablation_training.py \
    --id $ABLATION_ID \
    --num_models 16 \
    --save_dir ${OUTPUT} \
    --train_data_dir ${DATASET} \
    --chunk_length 256 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS}

# Shutdown VM
shutdown -h now