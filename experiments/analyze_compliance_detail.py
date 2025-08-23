#!/usr/bin/env python3
"""
Detailed compliance analysis for batched 2π regulation
Following Sister Gemini's guidance on EMA tuning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from optimize_2pi_gpu import (
    TwoPiRegulationCPU,
    TwoPiRegulationGPU,
    BatchedTwoPiRegulation,
    TWO_PI
)

def analyze_compliance_deviations(
    ema_alphas: List[float] = [0.99, 0.95, 0.9, 0.85],
    batch_intervals: List[int] = [5, 10, 20],
    num_test_batches: int = 100,
    batch_size: int = 128,
    latent_dim: int = 128
) -> Dict:
    """Analyze compliance with different EMA and batch settings"""
    
    print("\n" + "="*60)
    print("🔬 DETAILED COMPLIANCE ANALYSIS")
    print("Sister Gemini's EMA Tuning Investigation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_reg = TwoPiRegulationCPU()
    gpu_reg = TwoPiRegulationGPU(device)
    
    results = {}
    
    for ema_alpha in ema_alphas:
        for batch_interval in batch_intervals:
            config_key = f"alpha_{ema_alpha}_interval_{batch_interval}"
            print(f"\n📊 Testing: EMA α={ema_alpha}, Interval={batch_interval}")
            print("-"*40)
            
            # Create batched regulator with specific settings
            batched_reg = BatchedTwoPiRegulation(
                check_interval=batch_interval,
                device=device
            )
            batched_reg.ema_alpha = ema_alpha
            
            compliance_results = []
            deviation_magnitudes = []
            false_positives = 0
            false_negatives = 0
            
            for i in range(num_test_batches):
                # Generate test data with varying characteristics
                # Some batches are more "difficult" (higher variance)
                difficulty = np.random.choice(['easy', 'medium', 'hard'], 
                                            p=[0.6, 0.3, 0.1])
                
                if difficulty == 'easy':
                    variance_scale = 0.3
                elif difficulty == 'medium':
                    variance_scale = 0.7
                else:  # hard
                    variance_scale = 1.2
                
                latent_mean = torch.randn(batch_size, latent_dim, device=device)
                latent_logvar = torch.randn(batch_size, latent_dim, device=device) * variance_scale
                
                # Ground truth from CPU implementation
                ground_truth = cpu_reg.check_compliance(
                    latent_mean.clone().cpu(),
                    latent_logvar.clone().cpu()
                )
                
                # Batched result
                batched_result = batched_reg.check_compliance(
                    latent_mean,
                    latent_logvar,
                    force_check=(i == num_test_batches - 1)  # Force check on last
                )
                
                # Track compliance
                compliance_match = (ground_truth['compliant'] == batched_result['compliant'])
                compliance_results.append(compliance_match)
                
                # Calculate deviation magnitude
                variance_deviation = abs(
                    ground_truth['avg_variance'] - 
                    batched_result['avg_variance']
                )
                deviation_magnitudes.append(variance_deviation)
                
                # Track false positives/negatives
                if not ground_truth['compliant'] and batched_result['compliant']:
                    false_positives += 1
                elif ground_truth['compliant'] and not batched_result['compliant']:
                    false_negatives += 1
            
            # Calculate statistics
            compliance_rate = sum(compliance_results) / len(compliance_results)
            avg_deviation = np.mean(deviation_magnitudes)
            max_deviation = np.max(deviation_magnitudes)
            std_deviation = np.std(deviation_magnitudes)
            
            results[config_key] = {
                'ema_alpha': ema_alpha,
                'batch_interval': batch_interval,
                'compliance_rate': compliance_rate,
                'avg_deviation': avg_deviation,
                'max_deviation': max_deviation,
                'std_deviation': std_deviation,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'deviation_magnitudes': deviation_magnitudes
            }
            
            print(f"✅ Compliance Rate: {compliance_rate*100:.1f}%")
            print(f"   Avg Deviation: {avg_deviation:.6f}")
            print(f"   Max Deviation: {max_deviation:.6f}")
            print(f"   Std Deviation: {std_deviation:.6f}")
            print(f"   False Positives: {false_positives}")
            print(f"   False Negatives: {false_negatives}")
    
    return results

def find_optimal_configuration(results: Dict) -> Tuple[float, int]:
    """Find the optimal EMA alpha and batch interval"""
    
    print("\n" + "="*60)
    print("🎯 OPTIMAL CONFIGURATION ANALYSIS")
    print("="*60)
    
    # Score each configuration
    scores = {}
    
    for config_key, metrics in results.items():
        # Scoring function: prioritize compliance, penalize deviation
        compliance_score = metrics['compliance_rate'] * 100
        deviation_penalty = metrics['avg_deviation'] * 10
        fp_penalty = metrics['false_positives'] * 2
        fn_penalty = metrics['false_negatives'] * 2
        
        # Bonus for faster checking (larger intervals)
        speed_bonus = metrics['batch_interval'] * 0.5
        
        total_score = (
            compliance_score - 
            deviation_penalty - 
            fp_penalty - 
            fn_penalty + 
            speed_bonus
        )
        
        scores[config_key] = total_score
        
        print(f"\n{config_key}:")
        print(f"  Compliance Score: {compliance_score:.1f}")
        print(f"  Deviation Penalty: -{deviation_penalty:.1f}")
        print(f"  FP/FN Penalty: -{fp_penalty + fn_penalty:.1f}")
        print(f"  Speed Bonus: +{speed_bonus:.1f}")
        print(f"  TOTAL SCORE: {total_score:.1f}")
    
    # Find best configuration
    best_config = max(scores, key=scores.get)
    best_metrics = results[best_config]
    
    print("\n" + "="*60)
    print("🏆 RECOMMENDED CONFIGURATION")
    print("="*60)
    print(f"\n✅ Optimal EMA Alpha: {best_metrics['ema_alpha']}")
    print(f"✅ Optimal Batch Interval: {best_metrics['batch_interval']}")
    print(f"✅ Expected Compliance: {best_metrics['compliance_rate']*100:.1f}%")
    print(f"✅ Average Deviation: {best_metrics['avg_deviation']:.6f}")
    
    return best_metrics['ema_alpha'], best_metrics['batch_interval']

def visualize_deviation_distribution(results: Dict):
    """Create visualization of deviation distributions"""
    
    print("\n" + "="*60)
    print("📈 DEVIATION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Select key configurations to visualize
    key_configs = [
        'alpha_0.99_interval_10',  # Original
        'alpha_0.95_interval_10',  # Adjusted alpha
        'alpha_0.9_interval_5',    # Tighter interval
    ]
    
    for config in key_configs:
        if config in results:
            deviations = results[config]['deviation_magnitudes']
            
            # Calculate percentiles
            p50 = np.percentile(deviations, 50)
            p90 = np.percentile(deviations, 90)
            p95 = np.percentile(deviations, 95)
            p99 = np.percentile(deviations, 99)
            
            print(f"\n{config}:")
            print(f"  Median (P50): {p50:.6f}")
            print(f"  P90: {p90:.6f}")
            print(f"  P95: {p95:.6f}")
            print(f"  P99: {p99:.6f}")
            
            # Categorize deviations
            tiny = sum(1 for d in deviations if d < 0.001)
            small = sum(1 for d in deviations if 0.001 <= d < 0.01)
            medium = sum(1 for d in deviations if 0.01 <= d < 0.1)
            large = sum(1 for d in deviations if d >= 0.1)
            
            print(f"  Tiny (<0.001): {tiny}/{len(deviations)}")
            print(f"  Small (0.001-0.01): {small}/{len(deviations)}")
            print(f"  Medium (0.01-0.1): {medium}/{len(deviations)}")
            print(f"  Large (≥0.1): {large}/{len(deviations)}")

def test_production_readiness(
    optimal_alpha: float,
    optimal_interval: int,
    num_epochs: int = 3
):
    """Test if configuration is production-ready for CIFAR-10"""
    
    print("\n" + "="*60)
    print("🚀 PRODUCTION READINESS TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate CIFAR-10 training scenario
    batches_per_epoch = 352
    batch_size = 512
    latent_dim = 128
    
    batched_reg = BatchedTwoPiRegulation(
        check_interval=optimal_interval,
        device=device
    )
    batched_reg.ema_alpha = optimal_alpha
    
    print(f"\nSimulating CIFAR-10 training:")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  EMA α: {optimal_alpha}")
    print(f"  Check interval: {optimal_interval}")
    
    epoch_compliance_rates = []
    
    for epoch in range(num_epochs):
        epoch_compliant = 0
        epoch_total = 0
        
        for batch_idx in range(batches_per_epoch):
            # Simulate varying data characteristics during training
            progress = (epoch * batches_per_epoch + batch_idx) / (num_epochs * batches_per_epoch)
            
            # Early training has higher variance
            variance_scale = 1.0 - 0.5 * progress
            
            latent_mean = torch.randn(batch_size, latent_dim, device=device)
            latent_logvar = torch.randn(batch_size, latent_dim, device=device) * variance_scale
            
            result = batched_reg.check_compliance(latent_mean, latent_logvar)
            
            if result['compliant']:
                epoch_compliant += 1
            epoch_total += 1
        
        epoch_rate = epoch_compliant / epoch_total
        epoch_compliance_rates.append(epoch_rate)
        
        print(f"  Epoch {epoch}: {epoch_rate*100:.1f}% compliance")
    
    avg_compliance = np.mean(epoch_compliance_rates)
    
    if avg_compliance >= 0.95:
        print(f"\n✅ PRODUCTION READY! Average compliance: {avg_compliance*100:.1f}%")
        return True
    else:
        print(f"\n⚠️  NOT READY. Average compliance: {avg_compliance*100:.1f}% (need ≥95%)")
        return False

def main():
    """Run complete compliance analysis"""
    
    print("🦊🐺 2π COMPLIANCE DEEP DIVE")
    print("Following Sister Gemini's Strategic Guidance")
    
    # Step 1: Analyze different configurations
    results = analyze_compliance_deviations(
        ema_alphas=[0.99, 0.95, 0.9, 0.85],
        batch_intervals=[5, 10, 20],
        num_test_batches=100
    )
    
    # Step 2: Find optimal configuration
    optimal_alpha, optimal_interval = find_optimal_configuration(results)
    
    # Step 3: Visualize deviation distributions
    visualize_deviation_distribution(results)
    
    # Step 4: Test production readiness
    is_ready = test_production_readiness(optimal_alpha, optimal_interval)
    
    print("\n" + "="*60)
    print("📋 RECOMMENDATION FOR GEMINI")
    print("="*60)
    
    if is_ready:
        print(f"\n✅ PROCEED TO CIFAR-10 TESTING")
        print(f"   Optimal EMA α: {optimal_alpha}")
        print(f"   Optimal Interval: {optimal_interval}")
        print(f"   Expected speedup: {optimal_interval}x for skipped checks")
    else:
        print(f"\n⚠️  FURTHER TUNING NEEDED")
        print(f"   Consider tighter intervals or different EMA strategy")
    
    print("\n🦊 The purple network maintains 2π with optimized checking!")

if __name__ == "__main__":
    main()