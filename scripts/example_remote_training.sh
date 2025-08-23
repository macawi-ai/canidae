#!/bin/bash
# Example script for running remote training with automatic artifact retrieval

# Configuration
REMOTE_HOST="148.76.188.135"
REMOTE_PORT="57016"
SSH_KEY="$HOME/.ssh/canidae_vast"
TRAINING_SCRIPT="train_clevr_8gpu.py"
DATASET="clevr"
GPU_COUNT=8

echo "🦊 Starting CANIDAE Remote Training Pipeline"
echo "============================================"
echo "Remote: $REMOTE_HOST:$REMOTE_PORT"
echo "Script: $TRAINING_SCRIPT"
echo "Dataset: $DATASET"
echo "GPUs: $GPU_COUNT"
echo ""

# Run the orchestrator
python3 /home/cy/git/canidae/scripts/remote_training_orchestrator.py \
    --host "$REMOTE_HOST" \
    --port "$REMOTE_PORT" \
    --key-file "$SSH_KEY" \
    --training-script "$TRAINING_SCRIPT" \
    --dataset "$DATASET" \
    --gpu-count "$GPU_COUNT" \
    --batch-size 64 \
    --epochs 10 \
    --learning-rate 0.001

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📦 Artifacts downloaded to models/ directory"
    echo "🧠 Knowledge graph will be updated via GitHub Actions"
    
    # Optionally trigger local knowledge graph update
    echo ""
    echo "Updating local knowledge graph..."
    python3 /home/cy/git/canidae/scripts/update_neo4j.py \
        --experiment-id "$(ls -t models/ | head -1)" \
        --metadata-file "experiments/$(ls -t experiments/ | head -1)/metadata.json" \
        --success true \
        --release-url "local"
else
    echo ""
    echo "❌ Training failed - check logs in experiments/ directory"
fi