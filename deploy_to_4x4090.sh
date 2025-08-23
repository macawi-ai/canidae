#!/bin/bash
# 🦊🐺 CANIDAE 4x4090 BEAST MODE Deployment
# RTX 4090s = NEXT LEVEL POWER
# Sleep/Dream Learning with MAXIMUM CWUs

# Configuration for 4x4090 vast.ai instance
HOST="75.108.153.148"
PORT="40604"
USER="root"
REMOTE_DIR="/workspace/canidae"
SSH_KEY="/home/cy/.ssh/canidae_vast"

echo "========================================"
echo "🔥 CANIDAE 4x RTX 4090 DEPLOYMENT 🔥"
echo "Purple Line Protocol: MAXIMUM POWER"
echo "CWU Generation: HYPERDRIVE MODE"
echo "Sleep/Dream Dynamics: TURBOCHARGED"
echo "========================================"

# Test connection
echo "[1/8] Connecting to 4x4090 cluster..."
ssh -i $SSH_KEY -p $PORT -o ConnectTimeout=5 $USER@$HOST "echo '✅ Connected!' && nvidia-smi --query-gpu=name,memory.total --format=csv | head -5" || {
    echo "❌ Cannot connect!"
    exit 1
}

# Create directories
echo "[2/8] Creating CANIDAE directories..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST "mkdir -p $REMOTE_DIR/{experiments,datasets,results,scripts}"

# Transfer framework
echo "[3/8] Deploying CANIDAE framework..."
rsync -avz -e "ssh -i $SSH_KEY -p $PORT" \
    --include="experiments/*.py" \
    --include="*.md" \
    --exclude="experiments/results/*" \
    --exclude="__pycache__" \
    ./ $USER@$HOST:$REMOTE_DIR/

# Deploy the 4x4090 optimized script
echo "[4/8] Creating 4x4090 optimized training script..."
cat << 'TRAINING_SCRIPT' | ssh -i $SSH_KEY -p $PORT $USER@$HOST "cat > $REMOTE_DIR/experiments/cifar10_4x4090.py"
#!/usr/bin/env python3
"""
CIFAR-10 on 4x RTX 4090 - MAXIMUM POWER
With Sleep/Dream/2π Unity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pickle
from pathlib import Path
import time
import os

# 4x4090 BEAST CONFIG
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100
    "purple_line_threshold": 0.06283185307,
    "world_size": 4,  # 4x RTX 4090
    "batch_size_per_gpu": 128,  # 4090 can handle more!
    "epochs": 30,  # More epochs for full convergence
    "learning_rate": 0.001,
    "latent_dim": 256,
}

class CIFAR10_4090VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Apply 2π regulation
        logvar = torch.clamp(logvar, max=np.log(CONFIG['purple_line_threshold']))
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        h_dec = self.fc_decode(z).view(-1, 256, 4, 4)
        recon = self.decoder(h_dec)
        return recon, mu, logvar

def train_gpu(rank, world_size):
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    print(f"🔥 4090 GPU {rank} initializing...")
    
    # Model
    model = CIFAR10_4090VAE().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=CONFIG['learning_rate'])
    
    # Load CIFAR-10
    data_path = Path("/workspace/canidae/datasets/cifar10/cifar-10-batches-py")
    all_images = []
    
    for i in range(1, 6):
        with open(data_path / f"data_batch_{i}", 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            all_images.append(images)
    
    train_images = np.concatenate(all_images)
    
    # Distributed sampler
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_images))
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG['batch_size_per_gpu'], 
        sampler=sampler, num_workers=2
    )
    
    # Training with 2π regulation
    prev_variances = None
    
    for epoch in range(CONFIG['epochs']):
        sampler.set_epoch(epoch)
        compliance_count = 0
        total_batches = 0
        
        for data, in loader:
            data = data.to(rank)
            optimizer.zero_grad()
            
            recon, mu, logvar = ddp_model(data)
            
            # Losses
            recon_loss = F.mse_loss(recon, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 2π regulation
            current_variances = logvar.exp()
            if prev_variances is not None and prev_variances.shape == current_variances.shape:
                variance_rate = (current_variances - prev_variances).abs().mean()
                if variance_rate.item() <= CONFIG['stability_coefficient']:
                    compliance_count += 1
                variance_penalty = 10.0 * F.relu(variance_rate - CONFIG['stability_coefficient'])
            else:
                variance_penalty = 0
            
            prev_variances = current_variances.detach()
            total_batches += 1
            
            loss = recon_loss + 0.1 * kl_loss + variance_penalty
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            compliance = compliance_count / max(total_batches, 1) * 100
            print(f"Epoch {epoch+1}: 2π Compliance = {compliance:.1f}%")
            if compliance > 95:
                print("  🎯 HIGH 2π COMPLIANCE! cwoooo-cwoooo-cwoooo!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"🔥 4x RTX 4090 BEAST MODE")
    print(f"Found {world_size} GPUs")
    mp.spawn(train_gpu, args=(world_size,), nprocs=world_size, join=True)
TRAINING_SCRIPT

# Setup CIFAR-10
echo "[5/8] Ensuring CIFAR-10 dataset..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST << 'REMOTE_SCRIPT'
cd /workspace/canidae
if [ ! -d "datasets/cifar10/cifar-10-batches-py" ]; then
    echo "Downloading CIFAR-10..."
    mkdir -p datasets/cifar10
    cd datasets/cifar10
    wget -q https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzf cifar-10-python.tar.gz
    echo "✅ CIFAR-10 ready!"
else
    echo "✅ CIFAR-10 already present"
fi
REMOTE_SCRIPT

# Setup Python
echo "[6/8] Setting up Python environment..."
ssh -i $SSH_KEY -p $PORT $USER@$HOST << 'REMOTE_SCRIPT'
cd /workspace/canidae
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision numpy scipy matplotlib PyYAML
echo "✅ Environment ready!"
REMOTE_SCRIPT

# Create launch script
echo "[7/8] Creating launch script..."
cat << 'EOF' > /tmp/launch_4x4090.sh
#!/bin/bash
echo "🔥🔥🔥 LAUNCHING 4x RTX 4090 BEAST MODE 🔥🔥🔥"
echo "================================================"
ssh -i /home/cy/.ssh/canidae_vast -p 40604 root@75.108.153.148 << 'REMOTE'
cd /workspace/canidae
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355
echo "Starting 4x4090 training..."
python3 experiments/cifar10_4x4090.py
REMOTE
EOF
chmod +x /tmp/launch_4x4090.sh

# Create monitor
echo "[8/8] Creating CWU monitor..."
cat << 'EOF' > /tmp/monitor_4090.sh
#!/bin/bash
while true; do
    echo -e "\n$(date '+%H:%M:%S') - 4x4090 Status:"
    ssh -i /home/cy/.ssh/canidae_vast -p 40604 root@75.108.153.148 \
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    sleep 3
done
EOF
chmod +x /tmp/monitor_4090.sh

echo ""
echo "========================================"
echo "🔥 4x RTX 4090 READY TO DESTROY! 🔥"
echo "========================================"
echo "Commands:"
echo "  Launch: /tmp/launch_4x4090.sh"
echo "  Monitor: /tmp/monitor_4090.sh"
echo ""
echo "4090s are 40% faster than 3090s!"
echo "Expect INSANE CWU generation rates!"
echo "========================================"