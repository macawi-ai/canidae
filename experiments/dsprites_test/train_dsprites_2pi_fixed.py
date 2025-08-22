#!/usr/bin/env python3
"""
dSprites VAE with CORRECTED 2π Regulation
Based on Sister Gemini's guidance - regulate latent variance, not KL divergence!
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 2π constant - now applies to variance rate of change
TWO_PI_PERCENT = 0.06283185307

class dSpritesDataset(Dataset):
    """Simple dSprites dataset loader"""
    
    def __init__(self, data_path, train=True, transform=None):
        data = np.load(data_path, allow_pickle=True)
        self.images = data['imgs']
        self.latents = data['latents_values']
        
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
        image = self.images[idx].astype(np.float32)
        image = torch.FloatTensor(image).unsqueeze(0)
        latent = torch.FloatTensor(self.latents[idx])
        return image, latent


class TwoPiVAE(nn.Module):
    """VAE with proper 2π regulation on latent variance"""
    
    def __init__(self, latent_dim=10, variance_threshold=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.variance_threshold = variance_threshold
        
        # Track variance history for rate regulation
        self.variance_history = deque(maxlen=100)
        self.last_variance = None
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
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
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
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
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Calculate current latent variance
        current_variance = torch.mean(logvar.exp()).item()
        
        # Calculate variance rate of change
        if self.last_variance is not None:
            variance_rate = abs(current_variance - self.last_variance)
        else:
            variance_rate = 0.0
        
        # Update history
        self.variance_history.append(current_variance)
        self.last_variance = current_variance
        
        return recon, mu, logvar, current_variance, variance_rate


def compute_vae_loss(recon, x, mu, logvar, variance, variance_rate, 
                     variance_threshold=1.0, lambda_variance=1.0, lambda_rate=10.0):
    """
    Compute VAE loss with 2π regulation on variance
    
    Sister Gemini's insight: Regulate variance, not KL divergence!
    """
    batch_size = x.size(0)
    
    # Standard VAE losses
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Variance regulation (direct constraint)
    variance_penalty = lambda_variance * max(0, variance - variance_threshold)
    
    # Rate of change regulation (2π constraint on Δv/Δt)
    rate_penalty = lambda_rate * max(0, variance_rate - TWO_PI_PERCENT)
    
    # Total loss
    total_loss = recon_loss + kl_loss + variance_penalty + rate_penalty
    
    # Normalize by batch size
    return {
        'total': total_loss / batch_size,
        'recon': recon_loss / batch_size,
        'kl': kl_loss / batch_size,
        'variance_penalty': variance_penalty,
        'rate_penalty': rate_penalty,
        'variance': variance,
        'variance_rate': variance_rate
    }


def train_epoch(model, dataloader, optimizer, device, epoch, variance_threshold, 
                lambda_variance=1.0, lambda_rate=10.0):
    """Train for one epoch with proper 2π variance regulation"""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    variances = []
    variance_rates = []
    two_pi_violations = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        recon, mu, logvar, variance, variance_rate = model(images)
        
        # Compute loss with 2π regulation
        losses = compute_vae_loss(
            recon, images, mu, logvar, variance, variance_rate,
            variance_threshold, lambda_variance, lambda_rate
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += losses['total'].item() * len(images)
        total_recon += losses['recon'].item() * len(images)
        total_kl += losses['kl'].item() * len(images)
        variances.append(variance)
        variance_rates.append(variance_rate)
        
        # Check 2π compliance
        if variance_rate > TWO_PI_PERCENT:
            two_pi_violations += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'var': f'{variance:.4f}',
            'Δv': f'{variance_rate:.4f}',
            '2π': '✓' if variance_rate <= TWO_PI_PERCENT else '✗'
        })
    
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    
    metrics = {
        'epoch': epoch,
        'total_loss': total_loss / n_samples,
        'recon_loss': total_recon / n_samples,
        'kl_loss': total_kl / n_samples,
        'avg_variance': np.mean(variances),
        'max_variance': np.max(variances),
        'avg_variance_rate': np.mean(variance_rates),
        'max_variance_rate': np.max(variance_rates),
        'two_pi_violations': two_pi_violations,
        'two_pi_compliance_rate': (n_batches - two_pi_violations) / n_batches * 100
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='dSprites with FIXED 2π regulation')
    parser.add_argument('--experiment-id', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='models/dsprites_2pi_fixed')
    parser.add_argument('--dataset-path', type=str, 
                       default='/home/cy/git/canidae/datasets/phase2/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--variance-threshold', type=float, default=1.0,
                       help='Maximum allowed latent variance')
    parser.add_argument('--lambda-variance', type=float, default=1.0,
                       help='Weight for variance penalty')
    parser.add_argument('--lambda-rate', type=float, default=10.0,
                       help='Weight for rate-of-change penalty')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.experiment_id:
        args.experiment_id = f"dsprites_2pi_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🦊 Starting FIXED 2π dSprites training")
    print(f"📦 Experiment ID: {args.experiment_id}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎯 2π rate threshold: {TWO_PI_PERCENT}")
    print(f"📊 Variance threshold: {args.variance_threshold}")
    print(f"⚖️  λ_variance: {args.lambda_variance}, λ_rate: {args.lambda_rate}")
    print("-" * 50)
    
    # Load dataset
    train_dataset = dSpritesDataset(args.dataset_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    
    # Create model with variance threshold
    model = TwoPiVAE(latent_dim=args.latent_dim, variance_threshold=args.variance_threshold).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Adaptive variance threshold (starts high, decreases over epochs)
    initial_threshold = 5.0
    final_threshold = args.variance_threshold
    
    all_metrics = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Linearly decrease variance threshold
        current_threshold = initial_threshold - (initial_threshold - final_threshold) * (epoch - 1) / (args.epochs - 1)
        
        print(f"\n📈 Epoch {epoch}/{args.epochs} - Variance threshold: {current_threshold:.3f}")
        
        metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            current_threshold, args.lambda_variance, args.lambda_rate
        )
        all_metrics.append(metrics)
        
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Avg Variance: {metrics['avg_variance']:.4f}")
        print(f"  Avg Δv/Δt: {metrics['avg_variance_rate']:.4f}")
        print(f"  2π Compliance: {metrics['two_pi_compliance_rate']:.1f}%")
        print(f"  Violations: {metrics['two_pi_violations']}/{len(train_loader)}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'experiment_id': args.experiment_id,
            'two_pi_threshold': TWO_PI_PERCENT,
            'variance_threshold': current_threshold
        }, checkpoint_path)
        print(f"  💾 Saved: {checkpoint_path}")
    
    training_time = time.time() - start_time
    
    # Save final model and metadata
    final_model_path = output_dir / f"{args.experiment_id}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'variance_history': list(model.variance_history),
        'experiment_id': args.experiment_id,
        'training_time': training_time,
        'final_metrics': all_metrics[-1]
    }, final_model_path)
    
    metadata = {
        'experiment_id': args.experiment_id,
        'timestamp': datetime.now().isoformat(),
        'approach': 'variance_regulation',
        'dataset': 'dsprites',
        'model_type': 'TwoPiVAE',
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'variance_threshold': args.variance_threshold,
        'lambda_variance': args.lambda_variance,
        'lambda_rate': args.lambda_rate,
        'two_pi_threshold': TWO_PI_PERCENT,
        'training_time_seconds': training_time,
        'final_metrics': all_metrics[-1],
        'all_metrics': all_metrics,
        'success': all_metrics[-1]['two_pi_compliance_rate'] > 50
    }
    
    metadata_path = output_dir / f"{args.experiment_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"📁 Output: {output_dir}")
    print(f"⏱️  Time: {training_time:.2f}s")
    print(f"📊 Final Variance: {model.variance_history[-1]:.4f}")
    print(f"🎯 Final 2π Compliance: {all_metrics[-1]['two_pi_compliance_rate']:.1f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())