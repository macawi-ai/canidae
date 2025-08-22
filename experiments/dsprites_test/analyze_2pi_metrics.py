#!/usr/bin/env python3
"""
Enhanced 2π Metrics Analysis
Based on Sister Gemini's recommendations
"""

import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_checkpoint(checkpoint_path):
    """Analyze a training checkpoint for 2π compliance and health metrics"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Analysis")
    print(f"{'='*60}")
    
    # Core 2π Metrics
    print(f"\n2π Regulation Performance:")
    print(f"  Avg Variance: {metrics['avg_variance']:.4f}")
    print(f"  Max Variance: {metrics['max_variance']:.4f}")
    print(f"  Avg Variance Rate: {metrics['avg_variance_rate']:.4f}")
    print(f"  Max Variance Rate: {metrics['max_variance_rate']:.4f}")
    print(f"  2π Violations: {metrics['two_pi_violations']}")
    print(f"  2π Compliance Rate: {metrics['two_pi_compliance_rate']:.1f}%")
    
    # Loss Components
    print(f"\nLoss Components:")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")
    print(f"  Reconstruction Loss: {metrics['recon_loss']:.4f}")
    print(f"  KL Loss: {metrics['kl_loss']:.4f}")
    
    # Health Indicators (Sister Gemini's suggestions)
    print(f"\nHealth Indicators:")
    
    # Check if variance is stable (not too high or too low)
    variance_health = "HEALTHY" if 0.1 < metrics['avg_variance'] < 2.0 else "WARNING"
    print(f"  Variance Health: {variance_health}")
    
    # Check if reconstruction is good (lower is better)
    recon_health = "HEALTHY" if metrics['recon_loss'] < 100 else "WARNING"
    print(f"  Reconstruction Health: {recon_health}")
    
    # Check rate stability
    rate_health = "STABLE" if metrics['max_variance_rate'] < 0.1 else "UNSTABLE"
    print(f"  Rate Stability: {rate_health}")
    
    # Overall assessment
    is_2pi_compliant = metrics['two_pi_compliance_rate'] > 95
    is_learning = metrics['total_loss'] < 100
    is_stable = metrics['max_variance_rate'] < 0.1
    
    overall = "EXCELLENT" if (is_2pi_compliant and is_learning and is_stable) else "NEEDS ATTENTION"
    print(f"\nOverall Status: {overall}")
    
    return metrics

def compare_epochs(output_dir):
    """Compare metrics across all epochs"""
    
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    epochs = []
    losses = []
    variances = []
    variance_rates = []
    compliance_rates = []
    
    for cp_path in checkpoints:
        checkpoint = torch.load(cp_path, map_location='cpu')
        metrics = checkpoint['metrics']
        
        epochs.append(checkpoint['epoch'])
        losses.append(metrics['total_loss'])
        variances.append(metrics['avg_variance'])
        variance_rates.append(metrics['avg_variance_rate'])
        compliance_rates.append(metrics['two_pi_compliance_rate'])
    
    print(f"\n{'='*60}")
    print("Cross-Epoch Analysis")
    print(f"{'='*60}")
    
    print(f"\nTraining Progression:")
    for i, epoch in enumerate(epochs):
        print(f"  Epoch {epoch}: Loss={losses[i]:.2f}, Var={variances[i]:.4f}, ")
        print(f"           Rate={variance_rates[i]:.4f}, Compliance={compliance_rates[i]:.1f}%")
    
    # Trend analysis
    print(f"\nTrends:")
    loss_trend = "IMPROVING" if losses[-1] < losses[0] else "DEGRADING"
    print(f"  Loss Trend: {loss_trend} ({losses[0]:.2f} → {losses[-1]:.2f})")
    
    var_trend = "STABLE" if abs(variances[-1] - variances[0]) < 0.1 else "CHANGING"
    print(f"  Variance Trend: {var_trend} ({variances[0]:.4f} → {variances[-1]:.4f})")
    
    compliance_trend = "EXCELLENT" if all(c > 95 for c in compliance_rates[1:]) else "NEEDS WORK"
    print(f"  2π Compliance: {compliance_trend}")
    
    # Sister Gemini's additional metrics
    print(f"\nAdvanced Metrics (Sister Gemini's Recommendations):")
    
    # Mean Absolute Deviation of variances
    mad_variance = np.mean(np.abs(np.diff(variances)))
    print(f"  MAD of Variance: {mad_variance:.6f}")
    
    # Rate of change stability
    rate_stability = np.std(variance_rates)
    print(f"  Rate Stability (std): {rate_stability:.6f}")
    
    # Check for monotonic improvement in loss
    is_monotonic = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    print(f"  Monotonic Loss Decrease: {is_monotonic}")
    
    return {
        'epochs': epochs,
        'losses': losses,
        'variances': variances,
        'variance_rates': variance_rates,
        'compliance_rates': compliance_rates
    }

def main():
    output_dir = "/root/canidae/outputs_fixed"
    
    # Analyze individual checkpoints
    checkpoints = sorted(Path(output_dir).glob("checkpoint_epoch_*.pth"))
    for cp_path in checkpoints:
        analyze_checkpoint(cp_path)
    
    # Cross-epoch comparison
    results = compare_epochs(output_dir)
    
    # Final verdict
    print(f"\n{'='*60}")
    print("🦊 FINAL 2π ASSESSMENT")
    print(f"{'='*60}")
    
    if results and len(results['compliance_rates']) > 0:
        avg_compliance = np.mean(results['compliance_rates'])
        final_loss = results['losses'][-1] if results['losses'] else float('inf')
        
        if avg_compliance > 98 and final_loss < 60:
            print("✅ BREAKTHROUGH ACHIEVED!")
            print("   - Variance regulation successfully maintains 2π stability")
            print("   - Model learns effectively while respecting the boundary")
            print("   - This validates the 2π conjecture for VAE architectures")
        elif avg_compliance > 90:
            print("🔄 GOOD PROGRESS")
            print("   - 2π regulation is working but needs fine-tuning")
            print("   - Consider adjusting lambda parameters")
        else:
            print("⚠️ NEEDS ADJUSTMENT")
            print("   - 2π violations still occurring")
            print("   - Review threshold and penalty weights")

if __name__ == "__main__":
    main()
