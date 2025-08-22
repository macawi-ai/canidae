#!/bin/bash
# Test MLOps pipeline locally before remote deployment

echo "🦊 Testing CANIDAE MLOps Pipeline Locally"
echo "=========================================="

# Configuration
EXPERIMENT_ID="local_test_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/home/cy/git/canidae/models/$EXPERIMENT_ID"

echo "Experiment ID: $EXPERIMENT_ID"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "/home/cy/git/canidae/experiments/$EXPERIMENT_ID"

# Run training (just 2 epochs for testing)
echo "Starting training..."
python3 /home/cy/git/canidae/train_dsprites_mlops.py \
    --experiment-id "$EXPERIMENT_ID" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 2 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --device cuda

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed!"
    
    # List generated files
    echo ""
    echo "Generated files:"
    ls -lah "$OUTPUT_DIR"
    
    # Create experiment metadata for Neo4j update
    cat > "/home/cy/git/canidae/experiments/$EXPERIMENT_ID/metadata.json" <<EOF
{
    "experiment_id": "$EXPERIMENT_ID",
    "timestamp": "$(date -Iseconds)",
    "dataset": "dsprites",
    "gpu_config": "1x3090",
    "training_success": true,
    "local_test": true
}
EOF
    
    # Update Neo4j knowledge graph
    echo ""
    echo "Updating knowledge graph..."
    python3 /home/cy/git/canidae/scripts/update_neo4j.py \
        --experiment-id "$EXPERIMENT_ID" \
        --metadata-file "/home/cy/git/canidae/experiments/$EXPERIMENT_ID/metadata.json" \
        --success true \
        --release-url "local_test"
    
    echo ""
    echo "🎉 MLOps pipeline test successful!"
else
    echo ""
    echo "❌ Training failed"
    exit 1
fi