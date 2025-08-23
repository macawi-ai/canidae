#!/usr/bin/env python3
"""
Quick optimization test for CIFAR-10 data loading
Testing Gemini's recommendations before patent meeting
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

def benchmark_dataloader(batch_size=128, num_workers=4, pin_memory=False, 
                         persistent_workers=False, prefetch_factor=2):
    """Benchmark different DataLoader configurations"""
    
    # Create dummy CIFAR-10 sized data
    print(f"\n{'='*60}")
    print(f"Testing configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Persistent workers: {persistent_workers}")
    print(f"  Prefetch factor: {prefetch_factor}")
    print(f"{'='*60}")
    
    # Simulate CIFAR-10 data
    num_samples = 50000
    data = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(data, labels)
    
    # Create DataLoader with specified config
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark iteration speed
    start_time = time.time()
    num_batches = 0
    
    # Simulate one epoch
    for batch_idx, (data_batch, label_batch) in enumerate(loader):
        if device.type == 'cuda':
            data_batch = data_batch.to(device, non_blocking=pin_memory)
            label_batch = label_batch.to(device, non_blocking=pin_memory)
        
        # Simulate some computation (minimal to focus on data loading)
        _ = data_batch.mean()
        
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}", end='\r')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    samples_per_sec = num_samples / total_time
    time_per_batch = total_time / num_batches
    
    if device.type == 'cuda':
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        max_memory_mb = 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Samples/sec: {samples_per_sec:.1f}")
    print(f"  Time/batch: {time_per_batch*1000:.1f} ms")
    print(f"  Max GPU memory: {max_memory_mb:.1f} MB")
    print(f"  Batches: {num_batches}")
    print(f"{'='*60}\n")
    
    return {
        'config': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor
        },
        'results': {
            'total_time': total_time,
            'samples_per_sec': samples_per_sec,
            'time_per_batch_ms': time_per_batch * 1000,
            'max_memory_mb': max_memory_mb,
            'num_batches': num_batches
        }
    }

def main():
    """Run optimization benchmarks"""
    
    print("\n" + "="*60)
    print("🦊 CIFAR-10 DATA LOADING OPTIMIZATION BENCHMARK")
    print("Testing Sister Gemini's recommendations")
    print("="*60)
    
    results = []
    
    # Test 1: Baseline (current config)
    print("\n📊 TEST 1: BASELINE (Current Configuration)")
    results.append(benchmark_dataloader(
        batch_size=128,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2
    ))
    
    # Test 2: Optimized data loading (no batch size change)
    print("\n📊 TEST 2: OPTIMIZED DATA LOADING")
    results.append(benchmark_dataloader(
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    ))
    
    # Test 3: Larger batch size
    print("\n📊 TEST 3: LARGER BATCH SIZE")
    results.append(benchmark_dataloader(
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    ))
    
    # Test 4: Maximum batch size (memory permitting)
    print("\n📊 TEST 4: MAXIMUM BATCH SIZE")
    results.append(benchmark_dataloader(
        batch_size=1024,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    ))
    
    # Summary comparison
    print("\n" + "="*60)
    print("📈 OPTIMIZATION SUMMARY")
    print("="*60)
    
    baseline = results[0]['results']
    print(f"\n{'Configuration':<30} {'Samples/sec':>12} {'Speedup':>10} {'Memory (MB)':>12}")
    print("-"*65)
    
    for result in results:
        config = result['config']
        metrics = result['results']
        speedup = metrics['samples_per_sec'] / baseline['samples_per_sec']
        
        config_str = f"BS={config['batch_size']}, W={config['num_workers']}"
        if config['pin_memory']:
            config_str += ", PIN"
        if config['persistent_workers']:
            config_str += ", PERSIST"
            
        print(f"{config_str:<30} {metrics['samples_per_sec']:>12.1f} {speedup:>10.2f}x {metrics['max_memory_mb']:>12.1f}")
    
    print("\n" + "="*60)
    print("✅ RECOMMENDATIONS:")
    print("="*60)
    
    best_result = max(results, key=lambda x: x['results']['samples_per_sec'])
    best_config = best_result['config']
    best_speedup = best_result['results']['samples_per_sec'] / baseline['samples_per_sec']
    
    print(f"\nOptimal configuration achieves {best_speedup:.1f}x speedup:")
    print(f"  - Batch size: {best_config['batch_size']}")
    print(f"  - Workers: {best_config['num_workers']}")
    print(f"  - Pin memory: {best_config['pin_memory']}")
    print(f"  - Persistent workers: {best_config['persistent_workers']}")
    print(f"  - Prefetch factor: {best_config['prefetch_factor']}")
    
    print("\n⚠️  IMPORTANT: Test with actual 2π model to ensure compliance!")
    print("🦊🐺 The Pack's optimization preserves the eigenvalue boundary!")

if __name__ == "__main__":
    main()