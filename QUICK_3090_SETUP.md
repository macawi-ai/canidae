# Quick RTX 3090 Setup Commands

Once connected to vast.ai instance, run these commands:

```bash
# 1. Create directories
mkdir -p /workspace/canidae/models /workspace/canidae/datasets/phase2

# 2. Install Python packages
pip install torch torchvision numpy scipy --upgrade

# 3. Test GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 4. Start training (after files are transferred)
cd /workspace/canidae
python3 train_dsprites_3090.py
```

## Files to transfer (from local to remote):
- hyperbolic_learning_system.py
- train_dsprites_3090.py  
- conscious_2pi_modulation.py
- embodied_sensory_layer.py
- embodied_hyperbolic_consciousness.py
- datasets/phase2/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

## The 2π hyperbolic learning awaits! 🦊🐺✨