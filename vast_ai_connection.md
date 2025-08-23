# Vast.AI RTX 3090 Connection Info

## Current Instance Details (2025-08-20)
- **Instance ID**: 25143863
- **Host**: 155386
- **Machine ID**: 40588
- **IP**: 43.100.46.13
- **SSH Port**: 50215
- **GPU**: RTX 3090 (24GB VRAM, 35.3 TFLOPS)
- **CPU**: Xeon® Gold 6148 (20/80 cores)
- **RAM**: 48GB
- **Cost**: $34/month
- **SSH Key**: /home/cy/.ssh/vast_ai_3090

## Connection Command
```bash
ssh -i /home/cy/.ssh/vast_ai_3090 -p 50215 root@43.100.46.13 -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` forwards port 8080 for web interfaces (Jupyter, TensorBoard, etc.)

## Quick Deploy & Train
```bash
cd /home/cy/git/canidae
./deploy_to_3090.sh
```

## File Locations on Remote
- Working directory: `/workspace/canidae`
- Models: `/workspace/canidae/models`
- Datasets: `/workspace/canidae/datasets`
- Training log: `/workspace/canidae/training_log.txt`

## Training Commands on Remote
```bash
cd /workspace/canidae
./setup_3090_env.sh  # First time only - installs torch, scipy
./launch_training.sh  # Start dSprites hyperbolic training
```

## What We're Training
- Hyperbolic learning system with 2π regulation (0.06283185307)
- dSprites dataset for transformation invariance
- Learning in H³ space (Poincaré ball) with Möbius operations
- Critical for ARC puzzle solving - teaches rotation, scale, position invariance