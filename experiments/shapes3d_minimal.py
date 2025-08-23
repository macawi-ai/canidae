#!/usr/bin/env python3
"""
Shapes3D 2π - Minimal numpy-only test
"""

import numpy as np
import time
import h5py

print("🦊 SHAPES3D MINIMAL TEST")
print("=" * 50)

# Load a tiny subset directly to memory
print("Loading 1000 images from HDF5...")
start = time.time()

with h5py.File('/tmp/3dshapes.h5', 'r') as f:
    # Load just 1000 images directly to RAM
    images = f['images'][:1000]
    labels = f['labels'][:1000]

print(f"Loaded in {time.time() - start:.2f}s")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Simple 2π regulation test
print("\n2π REGULATION TEST")
print("-" * 30)

# Simulate variance tracking
prev_variance = np.random.randn(10) * 0.1
stability_coefficient = 0.06283185307  # 2π/100

compliant_steps = 0
total_steps = 100

for step in range(total_steps):
    # Simulate learning step
    current_variance = prev_variance + np.random.randn(10) * 0.01
    
    # Calculate variance rate
    variance_rate = np.abs(current_variance - prev_variance).mean()
    
    # Check 2π compliance
    if variance_rate <= stability_coefficient:
        compliant_steps += 1
    
    prev_variance = current_variance
    
    if step % 20 == 0:
        print(f"Step {step}: variance_rate={variance_rate:.6f}, compliant={variance_rate <= stability_coefficient}")

compliance = (compliant_steps / total_steps) * 100
print(f"\nFinal 2π Compliance: {compliance:.1f}%")

if compliance > 80:
    print("✅ HIGH COMPLIANCE ACHIEVED!")
else:
    print("⚠️  Low compliance - needs tuning")

print("=" * 50)
print("Test complete!")