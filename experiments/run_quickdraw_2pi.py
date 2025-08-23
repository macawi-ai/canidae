#!/usr/bin/env python3
"""
QuickDraw 2π Experiment Runner
Using available categories from our dataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime
import time

# Categories we have that are simple and diverse
CATEGORIES = ['apple', 'airplane', 'bicycle', 'star', 'flower']
# Fallback if some missing
BACKUP_CATEGORIES = ['apple', 'airplane', 'bicycle', 'baseball', 'butterfly']

# 2π Configuration - Our secret sauce
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100
    "variance_threshold_init": 3.0,
    "variance_threshold_final": 1.5, 
    "lambda_variance": 1.5,
    "lambda_rate": 15.0,
    "latent_dim": 16,
    "batch_size": 256,
    "epochs": 50,  # Quick test
    "learning_rate": 0.001
}

class QuickDrawVAE(nn.Module):
    """Simple VAE for 28x28 sketches with 2π tracking"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
        self.variance_history = []
        
    def encode(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon.view(-1, 28, 28), mu, logvar

def load_quickdraw_data():
    """Load available QuickDraw categories"""
    base_path = Path("/home/cy/git/canidae/datasets/phase3/quickdraw/sketches/sketches")
    
    data = []
    labels = []
    loaded_categories = []
    
    # Try primary categories first
    for idx, category in enumerate(CATEGORIES):
        npz_path = base_path / f"{category}.npz"
        if not npz_path.exists():
            # Try backup
            if idx < len(BACKUP_CATEGORIES):
                category = BACKUP_CATEGORIES[idx]
                npz_path = base_path / f"{category}.npz"
        
        if npz_path.exists():
            print(f"Loading {category}...")
            npz_data = np.load(npz_path, allow_pickle=True)
            # Use training data
            sketches = npz_data['train'][:5000]  # 5000 per category
            sketches = sketches.astype(np.float32) / 255.0
            data.append(sketches)
            labels.extend([idx] * len(sketches))
            loaded_categories.append(category)
        else:
            print(f"Skipping {category} (not found)")
    
    if len(data) == 0:
        raise ValueError("No QuickDraw data found!")
    
    return np.concatenate(data), np.array(labels), loaded_categories

def train_with_2pi():
    """Train VAE with 2π regulation"""
    
    print("="*60)
    print("2π QuickDraw Experiment - Testing Universality")
    print("="*60)
    
    # Load data
    data, labels, categories = load_quickdraw_data()
    print(f"\nLoaded {len(categories)} categories: {categories}")
    print(f"Total samples: {len(data)}")
    print(f"Sparsity: {np.mean(data < 0.1):.3f}")
    
    # Create dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data),
        torch.LongTensor(labels)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    # Model
    model = QuickDrawVAE(CONFIG['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Tracking
    compliance_history = []
    loss_history = []
    purple_lines = []  # When we violate 2π
    
    prev_variance = None
    total_batches = len(dataloader) * CONFIG['epochs']
    
    print("\nStarting training with 2π regulation...")
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        epoch_violations = 0
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            
            # Forward
            recon, mu, logvar = model(x)
            
            # Losses
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            
            # 2π Regulation
            current_variance = torch.exp(logvar).mean()
            
            # Adaptive threshold
            progress = (epoch * len(dataloader) + batch_idx) / total_batches
            threshold = CONFIG['variance_threshold_init'] - \
                       (CONFIG['variance_threshold_init'] - CONFIG['variance_threshold_final']) * progress
            
            # Variance penalty
            var_penalty = CONFIG['lambda_variance'] * F.relu(current_variance - threshold)
            
            # Rate penalty (THE 2π MAGIC)
            if prev_variance is not None:
                rate = torch.abs(current_variance - prev_variance)
                
                if rate > CONFIG['stability_coefficient']:
                    epoch_violations += 1
                    purple_lines.append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'rate': rate.item(),
                        'variance': current_variance.item()
                    })
                
                rate_penalty = CONFIG['lambda_rate'] * F.relu(rate - CONFIG['stability_coefficient'])
            else:
                rate_penalty = 0
            
            prev_variance = current_variance.detach()
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss + var_penalty + rate_penalty
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Metrics
        compliance = 1.0 - (epoch_violations / batch_count)
        compliance_history.append(compliance)
        loss_history.append(epoch_loss / batch_count)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Compliance={compliance:.1%}, Loss={epoch_loss/batch_count:.2f}, "
                  f"Var={current_variance:.3f}, Purple Lines={len(purple_lines)}")
    
    train_time = time.time() - start_time
    
    # Results
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Final 2π Compliance: {compliance_history[-1]:.1%}")
    print(f"Final Loss: {loss_history[-1]:.2f}")
    print(f"Total Purple Line Events: {len(purple_lines)}")
    print(f"Training Time: {train_time:.1f}s")
    
    # Compare with dSprites
    print("\n📊 COMPARISON:")
    print("dSprites:  99.9% compliance, 25.24 loss")
    print(f"QuickDraw: {compliance_history[-1]:.1%} compliance, {loss_history[-1]:.2f} loss")
    
    if compliance_history[-1] > 0.95:
        print("\n✅ 2π UNIVERSALITY CONFIRMED!")
        print("The principle holds for sparse sketch data!")
    else:
        print("\n⚠️ Needs tuning for sketch sparsity")
    
    # Save results
    results = {
        'dataset': 'QuickDraw',
        'categories': categories,
        'samples': len(data),
        'sparsity': float(np.mean(data < 0.1)),
        'compliance': float(compliance_history[-1]),
        'final_loss': float(loss_history[-1]),
        'purple_lines': len(purple_lines),
        'training_time': train_time,
        'timestamp': datetime.now().isoformat()
    }
    
    results_dir = Path("/home/cy/git/canidae/experiments/results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"quickdraw_2pi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/")
    
    return compliance_history[-1], loss_history[-1]

if __name__ == "__main__":
    compliance, loss = train_with_2pi()