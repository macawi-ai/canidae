#!/usr/bin/env python3
"""
Shapes3D 2π Pipeline - Fast Preloaded Version
Loads dataset into RAM for maximum speed
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import h5py
import json
import time
from datetime import datetime

# Configuration
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100
    "learning_rate": 0.001,
    "batch_size": 128,
    "latent_dim": 10,
    "beta": 4.0,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class SimpleVAE(nn.Module):
    """Simplified VAE for Shapes3D"""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8->4
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        h_dec = self.fc_decode(z).view(-1, 256, 4, 4)
        recon = self.decoder(h_dec)
        
        return recon, mu, logvar

def train_epoch(model, dataloader, optimizer, device, prev_variances=None):
    """Train one epoch with 2π regulation"""
    model.train()
    
    losses = []
    variance_rates = []
    compliant_batches = 0
    total_batches = 0
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = model(data)
        
        # Losses
        recon_loss = F.mse_loss(recon, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 2π Regulation
        current_variances = logvar.exp()
        
        if prev_variances is not None and prev_variances.shape == current_variances.shape:
            variance_rate = (current_variances - prev_variances).abs().mean()
            variance_rates.append(variance_rate.item())
            
            if variance_rate.item() <= CONFIG['stability_coefficient']:
                compliant_batches += 1
            
            # 2π penalty
            variance_penalty = 10.0 * F.relu(variance_rate - CONFIG['stability_coefficient'])
        else:
            variance_penalty = 0
        
        prev_variances = current_variances.detach()
        total_batches += 1
        
        # Total loss
        loss = recon_loss + CONFIG['beta'] * kl_loss + variance_penalty
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item()/data.size(0):.4f}")
    
    compliance = (compliant_batches / total_batches * 100) if total_batches > 0 else 0
    avg_loss = np.mean(losses) / CONFIG['batch_size']
    avg_variance_rate = np.mean(variance_rates) if variance_rates else 0
    
    return avg_loss, avg_variance_rate, compliance, prev_variances

def compute_disentanglement(model, dataloader, device):
    """Compute disentanglement metric"""
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            _, mu, _ = model(data)
            all_z.append(mu.cpu())
            all_labels.append(labels)
    
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Simple importance matrix: variance of z when varying each factor
    importance = np.zeros((6, 10))  # 6 factors, 10 latent dims
    
    for factor_idx in range(6):
        factor_values = all_labels[:, factor_idx]
        unique_values = np.unique(factor_values)
        
        if len(unique_values) > 1:
            for latent_idx in range(10):
                # Variance of latent dim when this factor changes
                variances = []
                for val in unique_values:
                    mask = factor_values == val
                    if mask.sum() > 0:
                        variances.append(all_z[mask, latent_idx].var())
                importance[factor_idx, latent_idx] = np.mean(variances)
    
    # Disentanglement: each latent should encode primarily one factor
    disentanglement_score = 0
    for latent_idx in range(10):
        scores = importance[:, latent_idx]
        if scores.sum() > 0:
            normalized = scores / scores.sum()
            # Higher score if one factor dominates
            disentanglement_score += normalized.max()
    
    return disentanglement_score / 10, importance

def main():
    print(f"🦊 SHAPES3D 2π FAST VERSION")
    print(f"Device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset into RAM
    print("\nLoading Shapes3D dataset into RAM...")
    start_time = time.time()
    
    with h5py.File('/tmp/3dshapes.h5', 'r') as f:
        # Load 50K images for reasonable speed
        num_samples = 50000
        images = f['images'][:num_samples]
        labels = f['labels'][:num_samples]
    
    print(f"Loaded {num_samples} images in {time.time() - start_time:.1f}s")
    
    # Convert to tensors
    print("Converting to tensors...")
    images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
    labels = torch.FloatTensor(labels)
    
    # Split train/test
    train_size = 45000
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]
    
    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = SimpleVAE(latent_dim=CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*60)
    print("TRAINING WITH 2π REGULATION ON SHAPES3D")
    print("="*60)
    
    prev_variances = None
    metrics = {
        'train_loss': [],
        'variance_rates': [],
        'compliance': [],
        'disentanglement': []
    }
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        train_loss, variance_rate, compliance, prev_variances = train_epoch(
            model, train_loader, optimizer, CONFIG['device'], prev_variances
        )
        
        # Compute disentanglement every few epochs
        if epoch % 3 == 0 or epoch == CONFIG['epochs'] - 1:
            print("  Computing disentanglement...")
            disentanglement, importance = compute_disentanglement(
                model, test_loader, CONFIG['device']
            )
            metrics['disentanglement'].append(disentanglement)
            print(f"  Disentanglement Score: {disentanglement:.3f}")
        
        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['variance_rates'].append(variance_rate)
        metrics['compliance'].append(compliance)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Variance Rate: {variance_rate:.6f} (threshold: {CONFIG['stability_coefficient']:.6f})")
        print(f"  2π Compliance: {compliance:.1f}%")
        print(f"  Time: {time.time() - start_time:.1f}s")
        
        if compliance > 95:
            print("  🎯 HIGH 2π COMPLIANCE ACHIEVED!")
    
    # Save results
    results = {
        'dataset': 'Shapes3D (50K subset)',
        'samples': train_size,
        'config': CONFIG,
        'metrics': metrics,
        'final_compliance': metrics['compliance'][-1],
        'final_disentanglement': metrics['disentanglement'][-1],
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = Path("/workspace/canidae/results/shapes3d_fast_results.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Save model
    model_file = Path("/workspace/canidae/results/shapes3d_model.pt")
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': CONFIG['epochs'],
        'metrics': metrics
    }, model_file)
    print(f"Model saved to: {model_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("SHAPES3D 2π REGULATION RESULTS")
    print("="*60)
    print(f"Final Compliance: {metrics['compliance'][-1]:.1f}%")
    print(f"Final Disentanglement: {metrics['disentanglement'][-1]:.3f}")
    print(f"Final Variance Rate: {metrics['variance_rates'][-1]:.6f}")
    print(f"Final Loss: {metrics['train_loss'][-1]:.4f}")
    
    if metrics['compliance'][-1] > 80 and metrics['disentanglement'][-1] > 0.5:
        print("\n✅ SUCCESS! 2π regulation preserves disentanglement!")
    elif metrics['compliance'][-1] > 80:
        print("\n✅ 2π regulation achieved, disentanglement needs tuning")
    else:
        print("\n⚠️  Need to tune hyperparameters")
    
    print("="*60)

if __name__ == "__main__":
    main()