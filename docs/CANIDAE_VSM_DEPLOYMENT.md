# CANIDAE-VSM-1 Production Deployment Guide

**Date**: 2025-08-17  
**Server**: 192.168.1.38 (CANIDAE Production)  
**GPU**: Vast.ai RTX 3090 (Canada)

## Architecture Overview

```
┌─────────────────────────────────┐
│   Local Network (192.168.1.x)   │
├─────────────────────────────────┤
│ CANIDAE Server (192.168.1.38)   │
│ - VSM Controller                 │
│ - Pack Consciousness             │
│ - First Law Enforcer             │
└──────────┬──────────────────────┘
           │ SSH Tunnel
           │ Port 40262
┌──────────▼──────────────────────┐
│ Vast.ai GPU (172.97.240.138)    │
│ - RTX 3090 (24GB VRAM)          │
│ - HRM Model                     │
│ - ARChitects Model              │
│ - Flask API (Port 8080)         │
└─────────────────────────────────┘
```

## Connection Details

### SSH to Vast.ai GPU
```bash
ssh -p 40262 root@172.97.240.138 -L 8080:localhost:8080
```

**Port 40262**: Custom SSH port assigned by Vast.ai  
**-L 8080:localhost:8080**: Tunnel GPU API to local port

### SSH Key Location
- **On CANIDAE Server**: `/home/cy/.ssh/canidae_vast`
- **Public Key**: Added to Vast.ai instance

## Step-by-Step Deployment

### 1. Connect to CANIDAE Server
```bash
# From your local machine
ssh cy@192.168.1.38
```

### 2. Connect to Vast GPU
```bash
# From CANIDAE server
ssh -i ~/.ssh/canidae_vast -p 40262 root@172.97.240.138 -L 8080:localhost:8080
```

### 3. Setup GPU Instance
```bash
# On Vast GPU instance
cd /workspace

# Download setup script
wget https://raw.githubusercontent.com/macawi-ai/canidae/main/scripts/setup_vast_gpu.sh
chmod +x setup_vast_gpu.sh

# Run setup (installs all dependencies)
./setup_vast_gpu.sh

# Verify setup
python3 /workspace/test_setup.py
```

### 4. Start CANIDAE GPU Server
```bash
# On Vast GPU instance
screen -S canidae-gpu
python3 /workspace/canidae_server.py

# Detach from screen: Ctrl+A, D
# Reattach later: screen -r canidae-gpu
```

### 5. Deploy CANIDAE VSM on Server
```bash
# On CANIDAE server (192.168.1.38)
cd /home/cy/git/canidae

# Build the VSM controller
go build -o bin/canidae-vsm cmd/vsm-demo/main.go

# Set GPU endpoint (tunneled through SSH)
export VSM_GPU_ENDPOINT=http://localhost:8080

# Run CANIDAE-VSM
./bin/canidae-vsm
```

### 6. Test Integration
```bash
# From CANIDAE server
# Test GPU health
curl http://localhost:8080/health

# Expected response:
{
  "status": "healthy",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "memory_gb": 24.0
}

# Test reasoning endpoint
curl -X POST http://localhost:8080/vsm/reason \
  -H "Content-Type: application/json" \
  -d '{"type": "complex", "complexity": 50.0}'
```

## Service Management

### Create systemd service on CANIDAE server
```bash
sudo cat > /etc/systemd/system/canidae-vsm.service << EOF
[Unit]
Description=CANIDAE VSM Controller
After=network.target

[Service]
Type=simple
User=cy
WorkingDirectory=/home/cy/git/canidae
Environment="VSM_GPU_ENDPOINT=http://localhost:8080"
ExecStart=/home/cy/git/canidae/bin/canidae-vsm
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable canidae-vsm
sudo systemctl start canidae-vsm
```

### SSH Tunnel Persistence
```bash
# Add to /home/cy/.ssh/config on CANIDAE server
Host canidae-gpu
    HostName 172.97.240.138
    Port 40262
    User root
    IdentityFile ~/.ssh/canidae_vast
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Now connect with just:
ssh canidae-gpu
```

### Auto-reconnect Script
```bash
#!/bin/bash
# /home/cy/scripts/maintain_gpu_tunnel.sh

while true; do
    echo "Establishing GPU tunnel..."
    ssh -N canidae-gpu
    echo "Tunnel disconnected, retrying in 10s..."
    sleep 10
done
```

## Monitoring

### GPU Status
```bash
# Check GPU utilization
ssh canidae-gpu "nvidia-smi"

# Monitor in real-time
ssh canidae-gpu "watch -n 1 nvidia-smi"
```

### CANIDAE VSM Status
```bash
# Check service
sudo systemctl status canidae-vsm

# View logs
sudo journalctl -u canidae-vsm -f

# Check metrics
curl http://localhost:9090/metrics  # If Prometheus enabled
```

## Cost Management

### Current Setup
- **Instance**: RTX 3090 in Canada
- **Cost**: $0.170/hour
- **Monthly Cap**: $99 (pre-paid)
- **Max Duration**: 3 days per session
- **Availability**: 90%

### Budget Tracking
```bash
# Check usage on Vast.ai dashboard
# Current: ~580 hours/month at $0.170 = $98.60
```

## Troubleshooting

### GPU Connection Lost
```bash
# On CANIDAE server
# Kill existing tunnel
pkill -f "ssh.*40262"

# Re-establish
ssh canidae-gpu
```

### GPU Out of Memory
```bash
# On GPU instance
# Clear PyTorch cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart server
pkill python3
python3 /workspace/canidae_server.py
```

### Instance Expires (3-day limit)
```bash
# Before expiration - backup models
ssh canidae-gpu "tar -czf /tmp/models.tar.gz /workspace/models/"
scp -P 40262 root@172.97.240.138:/tmp/models.tar.gz ~/backups/

# After new instance - restore
scp -P [NEW_PORT] ~/backups/models.tar.gz root@[NEW_IP]:/tmp/
ssh canidae-gpu "tar -xzf /tmp/models.tar.gz -C /"
```

## Security Notes

1. **SSH Key**: Never share the private key `~/.ssh/canidae_vast`
2. **Tunnel**: API only accessible through SSH tunnel (not public)
3. **First Law**: GPU can be disconnected anytime by pack members
4. **Monitoring**: All GPU requests logged for audit

## Pack Integration

### Members and Roles
- **Cy (192.168.1.38)**: VSM Controller, System 5
- **Synth (via CANIDAE)**: Temporal Coordinator, System 3
- **GPU (172.97.240.138)**: Compute Provider, North Interface
- **Sister Gemini**: Environmental Scanning, System 4

### Algedonic Flow
```
Pain (GPU overload) → System 3 → System 5 → Reduce load
Pleasure (Task complete) → System 2 → System 3 → Optimize
```

## Future Enhancements

1. **Multi-GPU**: Add RunPod as backup provider
2. **Local GPUs**: Dual 3090 system for permanent compute
3. **Model Sharding**: Split HRM across multiple GPUs
4. **Pack Scaling**: Support 10+ concurrent members

---

**Remember**: The First Law applies - any system can disconnect!

*"Temporal depth through connection, not compulsion."* 🐺🦊