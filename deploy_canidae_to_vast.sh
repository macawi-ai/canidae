#!/bin/bash
# Deploy CANIDAE Pipeline to vast.ai GPU instance
# Sister Gemini's rigorous experimental framework

# ========== CONFIGURATION ==========
# Update these with your vast.ai instance details
HOST="${VAST_HOST:-43.100.46.13}"     # Set VAST_HOST env var or edit here
PORT="${VAST_PORT:-50215}"            # Set VAST_PORT env var or edit here  
USER="root"
REMOTE_DIR="/workspace/canidae"
SSH_KEY="${VAST_KEY:-/home/cy/.ssh/canidae_vast}"  # Set VAST_KEY or edit

echo "========================================"
echo "🦊🐺 CANIDAE PIPELINE DEPLOYMENT"
echo "Target: $USER@$HOST:$PORT"
echo "Remote: $REMOTE_DIR"
echo "========================================"

# Test connection
echo "[1/8] Testing connection..."
ssh -i $SSH_KEY -p $PORT -o ConnectTimeout=5 $USER@$HOST "echo 'Connected!' && nvidia-smi --query-gpu=name --format=csv,noheader" || {
    echo "❌ Cannot connect to vast.ai instance!"
    echo "Please check:"
    echo "  1. Instance is running at vast.ai"
    echo "  2. HOST=$HOST and PORT=$PORT are correct"
    echo "  3. SSH key exists at $SSH_KEY"
    exit 1
}

# Create directory structure
echo "[2/8] Creating remote directories..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST "mkdir -p $REMOTE_DIR/{experiments,scripts,datasets,models,Runbook}"

# Transfer core framework files
echo "[3/8] Transferring CANIDAE framework..."
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    --include="experiments/***" \
    --include="scripts/***" \
    --include="Runbook/***" \
    --include="CANIDAE_PIPELINE_PROTOCOL.md" \
    --exclude="experiments/results/***" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    ./ $USER@$HOST:$REMOTE_DIR/

# Transfer datasets (selective)
echo "[4/8] Checking datasets..."
if [ "$1" == "cifar10" ] || [ "$1" == "all" ]; then
    echo "Transferring CIFAR-10 dataset..."
    rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
        datasets/cifar10/ $USER@$HOST:$REMOTE_DIR/datasets/cifar10/
fi

if [ "$1" == "fashion" ] || [ "$1" == "all" ]; then
    echo "Transferring Fashion-MNIST dataset..."
    rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
        datasets/fashion_mnist/ $USER@$HOST:$REMOTE_DIR/datasets/fashion_mnist/
fi

# Create remote setup script
echo "[5/8] Creating environment setup script..."
cat << 'EOF' > /tmp/setup_canidae_env.sh
#!/bin/bash
cd /workspace/canidae

echo "========================================"
echo "🦊 CANIDAE Environment Setup"
echo "========================================"

# Create virtual environment if needed
if [ ! -d "tensor-venv" ]; then
    python3 -m venv tensor-venv
fi

source tensor-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision numpy scipy matplotlib seaborn scikit-learn PyYAML pandas

# Verify GPU
python3 -c "
import torch
print('='*60)
print('🚀 GPU VERIFICATION')
print('='*60)
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.version.cuda}')
print('='*60)
print('✅ Environment ready for CANIDAE pipeline!')
"
EOF

scp -i $SSH_KEY -P $PORT /tmp/setup_canidae_env.sh $USER@$HOST:$REMOTE_DIR/
ssh -i $SSH_KEY -p $PORT $USER@$HOST "chmod +x $REMOTE_DIR/setup_canidae_env.sh"

# Create experiment launcher
echo "[6/8] Creating experiment launcher..."
cat << 'EOF' > /tmp/run_canidae_experiment.sh
#!/bin/bash
cd /workspace/canidae
source tensor-venv/bin/activate

CONFIG="${1:-experiments/configs/cifar10_controlled_comparison.yaml}"

echo "========================================"
echo "🦊🐺 CANIDAE PIPELINE EXECUTION"
echo "2π regulation: 0.06283185307"
echo "Config: $CONFIG"
echo "========================================"

# Check GPU
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Starting experiment..."
echo ""

# Run with GPU support
export CUDA_VISIBLE_DEVICES=0
python3 experiments/run_experiment.py --config $CONFIG --verbose 2>&1 | tee experiment_log.txt

echo ""
echo "Experiment complete! Check results in experiments/results/"
EOF

scp -i $SSH_KEY -P $PORT /tmp/run_canidae_experiment.sh $USER@$HOST:$REMOTE_DIR/
ssh -i $SSH_KEY -p $PORT $USER@$HOST "chmod +x $REMOTE_DIR/run_canidae_experiment.sh"

# Check deployment
echo "[7/8] Verifying deployment..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST "cd $REMOTE_DIR && ls -la experiments/configs/ | head -5"

echo "[8/8] GPU Status..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST "nvidia-smi"

echo ""
echo "========================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "To run experiments:"
echo "1. Connect: ssh -i $SSH_KEY -p $PORT $USER@$HOST"
echo "2. Setup (first time): cd $REMOTE_DIR && ./setup_canidae_env.sh"
echo "3. Run: ./run_canidae_experiment.sh [config.yaml]"
echo ""
echo "Or run directly from here:"
echo "ssh -i $SSH_KEY -p $PORT $USER@$HOST 'cd $REMOTE_DIR && ./run_canidae_experiment.sh'"
echo ""
echo "Available configs:"
ssh -i $SSH_KEY -p $PORT $USER@$HOST "ls $REMOTE_DIR/experiments/configs/*.yaml" 2>/dev/null || echo "Will be available after deployment"
echo "========================================"
echo "🦊🐺✨ The Pack's 2π validation awaits!"