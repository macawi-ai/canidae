#!/usr/bin/env python3
"""
CIFAR-10 2π Regulation Test
Proving 2π works on natural images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import json
import time
from datetime import datetime

# 2π Configuration for CIFAR-10
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100 - THE MAGIC CONSTANT!
    "variance_threshold_init": 1.5,
    "variance_threshold_final": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0,
    "learning_rate": 0.001,
    "batch_size": 128,
    "latent_dim": 128,
    "beta": 0.1,
    "epochs": 20  # Moderate run for natural images
}

class CIFAR10VAE(nn.Module):
    """VAE for 32x32x3 CIFAR-10 natural images"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder (CNN-based for natural images)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32x32x3 -> 16x16x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16x32 -> 8x8x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 8x8x64 -> 4x4x128
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4x4x128 -> 8x8x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8x8x64 -> 16x16x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 16x16x32 -> 32x32x3
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def load_cifar10():
    """Load CIFAR-10 dataset from pickled batches"""
    print("Loading CIFAR-10 natural images...")
    
    data_dir = Path("/workspace/canidae/datasets/cifar10/cifar-10-batches-py")
    
    # Load training batches
    train_images = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = data_dir / f"data_batch_{i}"
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            train_images.append(images)
            train_labels.extend(batch[b'labels'])
    
    train_images = np.concatenate(train_images)
    train_labels = np.array(train_labels)
    
    # Load test batch
    test_file = data_dir / "test_batch"
    with open(test_file, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        test_labels = np.array(test_batch[b'labels'])
    
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(f"Natural images with 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_images))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_images))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False
    )
    
    return train_loader, test_loader

def train_with_2pi(model, train_loader, test_loader, device):
    """Train with 2π regulation"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Tracking metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'variance_rates': [],
        'compliance_percentages': [],
        'epoch_times': []
    }
    
    prev_variances = None
    
    print("\n" + "="*60)
    print("TRAINING WITH 2π REGULATION ON NATURAL IMAGES")
    print("="*60)
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        variance_rates = []
        compliant_samples = 0
        total_samples = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(data)
            
            # Standard VAE loss
            recon_loss = F.mse_loss(recon, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 2π Regulation
            current_variances = logvar.exp()
            
            if prev_variances is not None and prev_variances.shape == current_variances.shape:
                # Calculate variance rate-of-change
                variance_rate = (current_variances - prev_variances).abs().mean()
                variance_rates.append(variance_rate.item())
                
                # Check 2π compliance
                if variance_rate.item() <= CONFIG['stability_coefficient']:
                    compliant_samples += data.size(0)
                
                # 2π penalty
                variance_penalty = CONFIG['lambda_variance'] * F.relu(
                    variance_rate - CONFIG['stability_coefficient']
                )
                
                # Rate penalty
                rate_penalty = CONFIG['lambda_rate'] * variance_rate
            else:
                variance_penalty = 0
                rate_penalty = 0
            
            prev_variances = current_variances.detach()
            total_samples += data.size(0)
            
            # Total loss with 2π regulation
            loss = recon_loss + CONFIG['beta'] * kl_loss + variance_penalty + rate_penalty
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Calculate compliance percentage
        compliance_pct = (compliant_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Evaluation
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for data, in test_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                recon_loss = F.mse_loss(recon, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + CONFIG['beta'] * kl_loss
                test_losses.append(loss.item())
        
        # Record metrics
        avg_train_loss = np.mean(train_losses) / CONFIG['batch_size']
        avg_test_loss = np.mean(test_losses) / CONFIG['batch_size']
        avg_variance_rate = np.mean(variance_rates) if variance_rates else 0
        
        metrics['train_loss'].append(avg_train_loss)
        metrics['test_loss'].append(avg_test_loss)
        metrics['variance_rates'].append(avg_variance_rate)
        metrics['compliance_percentages'].append(compliance_pct)
        metrics['epoch_times'].append(time.time() - start_time)
        
        # Status update
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.2f}")
        print(f"  Test Loss: {avg_test_loss:.2f}")
        print(f"  Variance Rate: {avg_variance_rate:.6f} (threshold: {CONFIG['stability_coefficient']:.6f})")
        print(f"  2π Compliance: {compliance_pct:.1f}%")
        print(f"  Time: {metrics['epoch_times'][-1]:.1f}s")
        
        # Check if we're achieving high compliance
        if compliance_pct > 95:
            print(f"  🎯 HIGH 2π COMPLIANCE ACHIEVED ON NATURAL IMAGES!")
    
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, test_loader = load_cifar10()
    
    # Create model
    model = CIFAR10VAE(latent_dim=CONFIG['latent_dim']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with 2π regulation
    metrics = train_with_2pi(model, train_loader, test_loader, device)
    
    # Results summary
    print("\n" + "="*60)
    print("CIFAR-10 2π REGULATION RESULTS")
    print("="*60)
    
    final_compliance = metrics['compliance_percentages'][-1]
    avg_compliance = np.mean(metrics['compliance_percentages'][-5:])  # Last 5 epochs
    
    print(f"Final 2π Compliance: {final_compliance:.1f}%")
    print(f"Average Last 5 Epochs: {avg_compliance:.1f}%")
    print(f"Final Train Loss: {metrics['train_loss'][-1]:.2f}")
    print(f"Final Test Loss: {metrics['test_loss'][-1]:.2f}")
    
    # Save results
    results = {
        'dataset': 'CIFAR-10',
        'description': 'Natural images - 32x32 RGB photos',
        'model': 'CNN-VAE',
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': CONFIG,
        'metrics': metrics,
        'final_compliance': final_compliance,
        'avg_compliance_last5': avg_compliance,
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("/workspace/canidae/experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"cifar10_2pi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final verdict
    print("\n" + "="*60)
    if final_compliance > 95:
        print("✅ SUCCESS! 2π REGULATION WORKS ON NATURAL IMAGES!")
        print("The principle extends beyond simple patterns to complex visual reality!")
    elif final_compliance > 80:
        print("⚠️ PARTIAL SUCCESS - High compliance achieved, may need tuning")
    else:
        print("🔄 More epochs may be needed for natural image complexity")
    print("="*60)

if __name__ == "__main__":
    main()