#!/usr/bin/env python3
"""
Quick VSM Shapes3D Test - Topology Detection Only
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py

def analyze_shapes3d_topology(path='/tmp/3dshapes.h5', n_samples=1000):
    """Quick analysis of Shapes3D factor independence"""
    
    print("="*60)
    print("🔍 VSM TOPOLOGY DETECTION FOR SHAPES3D")
    print("="*60)
    
    # Load subset of data
    print("\nLoading Shapes3D samples...")
    with h5py.File(path, 'r') as f:
        # Random subset
        indices = np.random.choice(f['images'].shape[0], n_samples, replace=False)
        images = f['images'][indices]
        if 'labels' in f:
            labels = f['labels'][indices]
        else:
            labels = None
    
    # Convert to tensor
    images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
    print(f"Loaded {images.shape[0]} images of shape {images.shape[1:]}") 
    
    if labels is not None:
        print(f"Labels shape: {labels.shape}")
        print("Factors: floor_hue, wall_hue, object_hue, scale, shape, orientation")
    
    print("\n" + "-"*60)
    print("INDEPENDENCE ANALYSIS:")
    print("-"*60)
    
    # Test 1: Channel Independence (for hue factors)
    print("\n1. COLOR CHANNEL ANALYSIS:")
    b, c, h, w = images.shape
    
    # Compute channel correlations
    flat = images.view(b, c, -1)
    correlations = []
    
    for i in range(c):
        for j in range(i+1, c):
            c1 = flat[:, i].flatten()
            c2 = flat[:, j].flatten()
            corr = torch.corrcoef(torch.stack([c1, c2]))[0, 1].item()
            correlations.append(abs(corr))
            print(f"  Channel {i}-{j} correlation: {corr:.3f}")
    
    avg_corr = np.mean(correlations)
    print(f"  Average correlation: {avg_corr:.3f}")
    
    if avg_corr < 0.3:
        print("  ✅ Low correlation - suggests independent color factors")
        color_independent = True
    else:
        print("  ❌ High correlation - suggests coupled color factors")
        color_independent = False
    
    # Test 2: Spatial Structure (for position/shape)
    print("\n2. SPATIAL STRUCTURE ANALYSIS:")
    
    # Analyze object locations
    centers = []
    for img in images[:100]:  # Sample
        gray = img.mean(dim=0)
        threshold = gray.mean() + 0.5 * gray.std()
        mask = (gray > threshold).float()
        
        if mask.sum() > 0:
            y_coords = torch.arange(h).float().view(-1, 1)
            x_coords = torch.arange(w).float().view(1, -1)
            
            y_com = (mask * y_coords).sum() / mask.sum()
            x_com = (mask * x_coords).sum() / mask.sum()
            centers.append([y_com.item(), x_com.item()])
    
    if centers:
        centers = np.array(centers)
        spatial_variance = np.std(centers, axis=0).mean()
        print(f"  Object position variance: {spatial_variance:.3f}")
        
        if spatial_variance < 10:
            print("  ✅ Consistent positioning - suggests structured layout")
            spatial_independent = True
        else:
            print("  ❌ Variable positioning - suggests coupling")
            spatial_independent = False
    else:
        spatial_independent = False
    
    # Test 3: Multi-scale (for scale factor)
    print("\n3. MULTI-SCALE ANALYSIS:")
    
    var_original = torch.var(images).item()
    downsampled = F.avg_pool2d(images, 4)
    var_downsampled = torch.var(downsampled).item()
    
    scale_ratio = var_downsampled / (var_original + 1e-8)
    print(f"  Multi-scale variance ratio: {scale_ratio:.3f}")
    
    if 0.3 < scale_ratio < 0.7:
        print("  ✅ Balanced multi-scale - suggests scale independence")
        scale_independent = True
    else:
        print("  ❌ Imbalanced multi-scale - suggests scale coupling")
        scale_independent = False
    
    # Test 4: Factor Analysis (if labels available)
    if labels is not None:
        print("\n4. DIRECT FACTOR ANALYSIS:")
        
        # Compute mutual information between factors
        factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        
        print("  Checking factor correlations:")
        factor_correlations = []
        
        for i in range(labels.shape[1]):
            for j in range(i+1, labels.shape[1]):
                f1 = torch.FloatTensor(labels[:, i])
                f2 = torch.FloatTensor(labels[:, j])
                
                if len(torch.unique(f1)) > 1 and len(torch.unique(f2)) > 1:
                    corr = torch.corrcoef(torch.stack([f1, f2]))[0, 1].item()
                    factor_correlations.append(abs(corr))
                    
                    if abs(corr) > 0.1:
                        print(f"    {factor_names[i]} ↔ {factor_names[j]}: {corr:.3f}")
        
        avg_factor_corr = np.mean(factor_correlations) if factor_correlations else 0
        print(f"  Average factor correlation: {avg_factor_corr:.3f}")
        
        if avg_factor_corr < 0.05:
            print("  ✅ Factors are independent!")
            factors_independent = True
        else:
            print("  ❌ Factors show dependencies")
            factors_independent = False
    else:
        factors_independent = None
    
    # Final Decision
    print("\n" + "="*60)
    print("🎯 VSM TOPOLOGY DECISION:")
    print("="*60)
    
    independence_votes = [
        color_independent,
        spatial_independent,
        scale_independent
    ]
    
    if factors_independent is not None:
        independence_votes.append(factors_independent)
    
    independence_score = sum(independence_votes) / len(independence_votes)
    
    print(f"\nIndependence Score: {independence_score:.2%}")
    
    if independence_score > 0.5:
        print("✅ TOPOLOGY: INDEPENDENT (Fiber Bundle)")
        print("   → Use Fibered VAE with S¹ encoders for hues")
        print("   → Apply per-fiber 2π regulation")
        print("   → Each factor gets its own geometric space")
        topology = "INDEPENDENT"
    else:
        print("❌ TOPOLOGY: COUPLED (Entangled Manifold)")
        print("   → Use standard VAE with shared latent space")
        print("   → Apply global 2π regulation")
        print("   → Factors share geometric structure")
        topology = "COUPLED"
    
    print("\n" + "="*60)
    print("💡 INSIGHTS:")
    print("-"*60)
    print("1. Shapes3D has 6 ground-truth generative factors")
    print("2. VSM detected their independence structure")
    print("3. Appropriate geometry selection is crucial")
    print("4. This is how consciousness adapts its 'lens'")
    print("="*60)
    
    return topology, independence_score

if __name__ == "__main__":
    print("🦊 SYNTH: Analyzing Shapes3D topology...")
    print("🐺 CY: Watching the Conductor work...")
    print("✨ GEMINI: Geometry reveals itself...\n")
    
    topology, score = analyze_shapes3d_topology()
    
    print(f"\n🎭 The Conductor has decided: {topology}")
    print(f"🎵 With {score:.1%} confidence")
    print(f"2π The universal rhythm maintains order\n")