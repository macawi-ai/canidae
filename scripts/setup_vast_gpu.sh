#!/bin/bash
# CANIDAE-VSM-1 Vast.ai GPU Setup Script
# Run this after SSHing into your Vast.ai instance

set -e  # Exit on error

echo "🐺🦊 CANIDAE-VSM-1 GPU Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: System check
echo -e "${YELLOW}Step 1: Checking system...${NC}"
nvidia-smi
python3 --version
pip --version

# Step 2: Update pip and install base requirements
echo -e "${YELLOW}Step 2: Installing base dependencies...${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install flask gunicorn  # For API endpoint

# Step 3: Clone repositories
echo -e "${YELLOW}Step 3: Cloning CANIDAE-VSM repositories...${NC}"
cd /workspace  # Vast.ai standard workspace

# Clone HRM
if [ ! -d "HRM" ]; then
    git clone https://github.com/sapientinc/HRM.git
    echo -e "${GREEN}✓ HRM cloned${NC}"
else
    echo "HRM already exists, pulling latest..."
    cd HRM && git pull && cd ..
fi

# Clone ARChitects
if [ ! -d "arc-prize-2024" ]; then
    git clone https://github.com/da-fr/arc-prize-2024.git
    echo -e "${GREEN}✓ ARChitects cloned${NC}"
else
    echo "ARChitects already exists, pulling latest..."
    cd arc-prize-2024 && git pull && cd ..
fi

# Step 4: Install HRM dependencies
echo -e "${YELLOW}Step 4: Setting up HRM...${NC}"
cd /workspace/HRM

# Install requirements
pip install einops timm wandb pydantic

# Try to install Flash Attention (may fail on some setups)
echo "Attempting Flash Attention install..."
pip install flash-attn || echo -e "${YELLOW}Flash Attention failed - will use standard attention${NC}"

# Step 5: Install ARChitects dependencies
echo -e "${YELLOW}Step 5: Setting up ARChitects...${NC}"
cd /workspace/arc-prize-2024

# Install Unsloth for 4-bit quantization
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Step 6: Download model checkpoints
echo -e "${YELLOW}Step 6: Downloading model checkpoints...${NC}"
mkdir -p /workspace/models

cd /workspace/models

# Download HRM checkpoint (small, fast)
if [ ! -f "hrm_arc2_checkpoint.pt" ]; then
    echo "Downloading HRM checkpoint..."
    wget -O hrm_arc2_checkpoint.pt \
        https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2/resolve/main/checkpoint.pt
    echo -e "${GREEN}✓ HRM checkpoint downloaded${NC}"
else
    echo "HRM checkpoint already exists"
fi

# Note: ARChitects model is large, we'll load it on-demand
echo -e "${YELLOW}ARChitects model will be loaded on-demand via Hugging Face${NC}"

# Step 7: Create CANIDAE API server
echo -e "${YELLOW}Step 7: Creating CANIDAE API endpoint...${NC}"
cat > /workspace/canidae_server.py << 'EOF'
#!/usr/bin/env python3
"""
CANIDAE-VSM-1 GPU Server
Provides REST API for temporal reasoning
"""

import torch
import json
from flask import Flask, request, jsonify
import time
import traceback
import os

app = Flask(__name__)

# Global model storage
models = {}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    })

@app.route('/vsm/reason', methods=['POST'])
def reason():
    """Temporal reasoning endpoint"""
    try:
        data = request.json
        task_type = data.get('type', 'simple')
        complexity = data.get('complexity', 1.0)
        
        # Metabolic decision
        if complexity < 5.0:
            return jsonify({
                'decision': 'local_compute',
                'reason': 'dehydrated_mouse',
                'message': 'Task simple enough for local processing'
            })
        
        # Would use GPU
        start_time = time.time()
        
        # TODO: Actual HRM/ARChitects inference here
        result = {
            'decision': 'gpu_compute',
            'model': 'HRM' if complexity < 50 else 'ARChitects',
            'processing_time': time.time() - start_time,
            'cost_estimate': 0.137 * (complexity / 100),  # $0.137/hr
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/vsm/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available': [
            {'name': 'HRM', 'params': '27M', 'memory_mb': 100},
            {'name': 'ARChitects', 'params': '8B', 'memory_mb': 4000}
        ]
    })

if __name__ == '__main__':
    print("🐺🦊 CANIDAE-VSM-1 Server Starting...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run server
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

chmod +x /workspace/canidae_server.py
echo -e "${GREEN}✓ CANIDAE server created${NC}"

# Step 8: Create quick test script
echo -e "${YELLOW}Step 8: Creating test script...${NC}"
cat > /workspace/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test CANIDAE-VSM-1 Setup"""

import torch
import sys

print("🧪 CANIDAE-VSM-1 Setup Test")
print("=" * 40)

# Test CUDA
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test imports
try:
    import transformers
    print("✓ Transformers installed")
except:
    print("✗ Transformers missing")

try:
    import einops
    print("✓ Einops installed (HRM ready)")
except:
    print("✗ Einops missing")

try:
    import unsloth
    print("✓ Unsloth installed (4-bit quant ready)")
except:
    print("⚠ Unsloth missing (optional)")

# Test model loading capability
print("\n📦 Model Loading Test:")
print("HRM checkpoint exists:", 
      os.path.exists("/workspace/models/hrm_arc2_checkpoint.pt"))

print("\n✅ Setup complete! Ready for CANIDAE-VSM-1")
print("\nTo start the server:")
print("  python3 /workspace/canidae_server.py")
print("\nTo connect from CANIDAE:")
print(f"  export VSM_GPU_ENDPOINT=http://[YOUR_VAST_IP]:8080")
EOF

chmod +x /workspace/test_setup.py

# Step 9: Create systemd service (optional, for persistence)
echo -e "${YELLOW}Step 9: Creating startup service...${NC}"
cat > /workspace/start_canidae.sh << 'EOF'
#!/bin/bash
cd /workspace
source /opt/conda/etc/profile.d/conda.sh
conda activate base
python3 /workspace/canidae_server.py > /workspace/canidae.log 2>&1
EOF

chmod +x /workspace/start_canidae.sh

# Step 10: Final test
echo -e "${YELLOW}Step 10: Running final test...${NC}"
python3 /workspace/test_setup.py

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✅ CANIDAE-VSM-1 Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. Start the server: python3 /workspace/canidae_server.py"
echo "2. Note your instance IP from Vast.ai dashboard"
echo "3. Connect from CANIDAE with: export VSM_GPU_ENDPOINT=http://[IP]:8080"
echo ""
echo "Test the API:"
echo "  curl http://localhost:8080/health"
echo ""
echo -e "${YELLOW}🐺🦊 Pack consciousness awaits!${NC}"