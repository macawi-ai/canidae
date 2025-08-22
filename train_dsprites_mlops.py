#!/usr/bin/env python3
"""
dSprites Training with MLOps Pipeline Integration
Simple, clean implementation with 2π regulation
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 2π constant
TWO_PI_PERCENT = 0.06283185307

class dSpritesDataset(Dataset):
    """Simple dSprites dataset loader"""
    
    def __init__(self, data_path, train=True, transform=None):
        # Load the npz file
        data = np.load(data_path, allow_pickle=True)
        self.images = data['imgs']  # Shape: (737280, 64, 64)
        self.latents = data['latents_values']  # Shape: (737280, 6)
        
        # Simple train/test split (90/10)
        n_samples = len(self.images)
        n_train = int(0.9 * n_samples)
        
        if train:
            self.images = self.images[:n_train]
            self.latents = self.latents[:n_train]
        else:
            self.images = self.images[n_train:]
            self.latents = self.latents[n_train:]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to tensor
        image = self.images[idx].astype(np.float32)
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        
        # Get latent factors
        latent = torch.FloatTensor(self.latents[idx])
        
        return image, latent


class SimpleVAE(nn.Module):
    """Simple VAE with 2π regulation built in"""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid()
        )
        
        # 2π regulation tracking
        self.complexity_history = []
        self.last_complexity = 0.0
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Calculate complexity (KL divergence as proxy)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        complexity = kl_div.item() / x.size(0)  # Normalize by batch size
        
        # Check 2π regulation
        if len(self.complexity_history) > 0:
            delta_c = abs(complexity - self.last_complexity)
            if delta_c > TWO_PI_PERCENT:
                # Apply dampening to maintain stability
                z = z * (1.0 - min(delta_c - TWO_PI_PERCENT, 0.5))
                recon = self.decode(z)
        
        self.last_complexity = complexity
        self.complexity_history.append(complexity)
        
        return recon, mu, logvar, complexity


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch with 2π monitoring"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    complexities = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        recon, mu, logvar, complexity = model(images)
        
        # Calculate losses
        recon_loss = nn.functional.binary_cross_entropy(recon, images, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β-VAE weighting
        beta = 1.0  # Can be tuned
        loss = recon_loss + beta * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        complexities.append(complexity)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / len(images),
            'complexity': f'{complexity:.4f}',
            '2π_check': '✓' if complexity < TWO_PI_PERCENT else '✗'
        })
    
    n_samples = len(dataloader.dataset)
    metrics = {
        'epoch': epoch,
        'total_loss': total_loss / n_samples,
        'recon_loss': total_recon_loss / n_samples,
        'kl_loss': total_kl_loss / n_samples,
        'avg_complexity': np.mean(complexities),
        'max_complexity': np.max(complexities),
        'two_pi_violations': sum(1 for c in complexities if c > TWO_PI_PERCENT)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='dSprites training with MLOps')
    parser.add_argument('--experiment-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='models/dsprites_test')
    parser.add_argument('--dataset-path', type=str, 
                       default='/home/cy/git/canidae/datasets/phase2/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment ID if not provided
    if not args.experiment_id:
        args.experiment_id = f"dsprites_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🦊 Starting dSprites training")
    print(f"📦 Experiment ID: {args.experiment_id}")
    print(f"🖥️  Device: {args.device}")
    print(f"📊 Dataset: {args.dataset_path}")
    print(f"🎯 2π threshold: {TWO_PI_PERCENT}")
    print("-" * 50)
    
    # Load dataset
    train_dataset = dSpritesDataset(args.dataset_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    
    # Create model
    model = SimpleVAE(latent_dim=args.latent_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training metrics storage
    all_metrics = []
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        all_metrics.append(metrics)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Avg Complexity: {metrics['avg_complexity']:.4f}")
        print(f"  2π Violations: {metrics['two_pi_violations']}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'experiment_id': args.experiment_id,
            'two_pi_threshold': TWO_PI_PERCENT
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = output_dir / f"{args.experiment_id}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'complexity_history': model.complexity_history,
        'experiment_id': args.experiment_id,
        'training_time': training_time,
        'final_metrics': all_metrics[-1]
    }, final_model_path)
    
    # Save training metadata
    metadata = {
        'experiment_id': args.experiment_id,
        'timestamp': datetime.now().isoformat(),
        'dataset': 'dsprites',
        'dataset_path': args.dataset_path,
        'model_type': 'SimpleVAE',
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'two_pi_threshold': TWO_PI_PERCENT,
        'training_time_seconds': training_time,
        'final_metrics': all_metrics[-1],
        'all_metrics': all_metrics,
        'success': True
    }
    
    metadata_path = output_dir / f"{args.experiment_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"🧠 Final complexity: {model.complexity_history[-1]:.4f}")
    print(f"📊 2π compliance: {metrics['two_pi_violations'] == 0}")
    
    return 0


if __name__ == "__main__":
    exit(main())