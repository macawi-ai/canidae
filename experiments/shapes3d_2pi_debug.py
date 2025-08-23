#!/usr/bin/env python3
"""
Shapes3D 2π Pipeline - Debug Version
Minimal test with verbose output
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import json
import time
from datetime import datetime
import sys

# Configuration
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100
    "learning_rate": 0.001,
    "batch_size": 32,  # Even smaller batch
    "latent_dim": 10,
    "beta": 4.0,
    "epochs": 2,  # Just 2 epochs for testing
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class Shapes3DDataset(Dataset):
    """Memory-efficient dataset that reads from HDF5 on demand"""
    
    def __init__(self, h5_path, indices=None):
        print(f"Loading dataset from {h5_path}")
        self.h5_path = h5_path
        self.h5_file = None
        self.indices = indices
        
        # Get dataset info
        with h5py.File(h5_path, 'r') as f:
            self.total_len = len(f['images'])
            print(f"Total images in dataset: {self.total_len}")
            if indices is not None:
                self.len = len(indices)
            else:
                self.len = self.total_len
                self.indices = np.arange(self.len)
            print(f"Using {self.len} images")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        real_idx = self.indices[idx]
        image = self.h5_file['images'][real_idx]
        label = self.h5_file['labels'][real_idx]
        
        # Convert to tensor and normalize
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        label = torch.FloatTensor(label)
        
        return image, label

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
    
    print(f"Starting epoch with {len(dataloader)} batches")
    sys.stdout.flush()
    
    for batch_idx, (data, _) in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}/{len(dataloader)}...")
            sys.stdout.flush()
        
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
            print(f"    Loss={loss.item()/data.size(0):.4f}")
            sys.stdout.flush()
    
    compliance = (compliant_batches / total_batches * 100) if total_batches > 0 else 0
    avg_loss = np.mean(losses) / CONFIG['batch_size']
    avg_variance_rate = np.mean(variance_rates) if variance_rates else 0
    
    return avg_loss, avg_variance_rate, compliance, prev_variances

def main():
    print(f"\n🦊 SHAPES3D 2π DEBUG VERSION")
    print(f"Device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    data_path = Path("/workspace/canidae/datasets/shapes3d/3dshapes.h5")
    
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        return
    
    # Use tiny subset for debug (first 5K images)
    print("\nCreating datasets...")
    train_indices = np.arange(0, 4000)
    test_indices = np.arange(4000, 5000)
    
    train_dataset = Shapes3DDataset(data_path, train_indices)
    test_dataset = Shapes3DDataset(data_path, test_indices)
    
    print(f"\nDataset created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    print("\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=0)  # No multiprocessing for debug
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0)
    
    print("Data loaders created")
    
    # Test loading one batch
    print("\nTesting data loading...")
    for i, (batch, labels) in enumerate(train_loader):
        print(f"  Loaded batch shape: {batch.shape}")
        print(f"  Labels shape: {labels.shape}")
        break
    
    # Create model
    print("\nCreating model...")
    model = SimpleVAE(latent_dim=CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        test_batch = batch.to(CONFIG['device'])
        recon, mu, logvar = model(test_batch)
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
    
    print("\n" + "="*60)
    print("TRAINING WITH 2π REGULATION")
    print("="*60)
    
    prev_variances = None
    metrics = {
        'train_loss': [],
        'variance_rates': [],
        'compliance': []
    }
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        print(f"\n[Epoch {epoch+1}/{CONFIG['epochs']}]")
        
        # Train
        train_loss, variance_rate, compliance, prev_variances = train_epoch(
            model, train_loader, optimizer, CONFIG['device'], prev_variances
        )
        
        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['variance_rates'].append(variance_rate)
        metrics['compliance'].append(compliance)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Variance Rate: {variance_rate:.6f} (threshold: {CONFIG['stability_coefficient']:.6f})")
        print(f"  2π Compliance: {compliance:.1f}%")
        print(f"  Time: {time.time() - start_time:.1f}s")
        
        if compliance > 95:
            print("  🎯 HIGH 2π COMPLIANCE ACHIEVED!")
    
    # Save results
    results = {
        'dataset': 'Shapes3D (debug subset)',
        'samples': len(train_dataset),
        'config': CONFIG,
        'metrics': metrics,
        'final_compliance': metrics['compliance'][-1] if metrics['compliance'] else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = Path("/workspace/canidae/results/shapes3d_debug_results.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("DEBUG RUN COMPLETE")
    print("="*60)
    print(f"Final Compliance: {metrics['compliance'][-1] if metrics['compliance'] else 0:.1f}%")
    print(f"Final Variance Rate: {metrics['variance_rates'][-1] if metrics['variance_rates'] else 0:.6f}")
    print(f"Final Loss: {metrics['train_loss'][-1] if metrics['train_loss'] else 0:.4f}")
    
    if metrics['compliance'] and metrics['compliance'][-1] > 80:
        print("\n✅ SUCCESS! 2π regulation works on Shapes3D!")
    
    print("="*60)

if __name__ == "__main__":
    main()