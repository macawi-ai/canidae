#!/usr/bin/env python3
"""
Test 2π Regulation on MNIST
Proving universality across datasets
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import time

# 2π Configuration for MNIST (standard dense images)
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100
    "variance_threshold_init": 1.5,
    "variance_threshold_final": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0,
    "learning_rate": 0.001,
    "batch_size": 256,
    "latent_dim": 10,
    "beta": 0.1,
    "epochs": 50
}

class MNISTVAE(nn.Module):
    """Simple VAE for 28x28 MNIST digits"""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
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

def load_mnist():
    """Load MNIST dataset"""
    print("Loading MNIST...")
    
    # Load from NPZ file
    data_path = Path("/home/cy/git/canidae/datasets/mnist/mnist.npz")
    if not data_path.exists():
        raise FileNotFoundError(f"MNIST not found at {data_path}")
    
    data = np.load(data_path)
    x_train = data['x_train'].astype(np.float32) / 255.0
    x_test = data['x_test'].astype(np.float32) / 255.0
    
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test))
    
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
    
    history = {
        'compliance': [],
        'train_loss': [],
        'test_loss': [],
        'violations': []
    }
    
    prev_variance = None
    best_compliance = 0
    
    print("\n🎯 Training MNIST with 2π Regulation...")
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        epoch_violations = 0
        epoch_batches = 0
        epoch_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward
            recon, mu, logvar = model(data)
            
            # Losses
            recon_loss = F.binary_cross_entropy(recon, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            
            # 2π Regulation
            current_variance = torch.exp(logvar).mean()
            
            # Adaptive threshold
            progress = (epoch * len(train_loader) + batch_idx) / (CONFIG['epochs'] * len(train_loader))
            threshold = CONFIG['variance_threshold_init'] - \
                       (CONFIG['variance_threshold_init'] - CONFIG['variance_threshold_final']) * progress
            
            # Variance penalty
            var_penalty = CONFIG['lambda_variance'] * F.relu(current_variance - threshold)
            
            # Rate penalty (2π magic)
            if prev_variance is not None:
                rate = torch.abs(current_variance - prev_variance)
                
                if rate.item() > CONFIG['stability_coefficient']:
                    epoch_violations += 1
                
                rate_penalty = CONFIG['lambda_rate'] * F.relu(rate - CONFIG['stability_coefficient'])
            else:
                rate_penalty = 0
            
            prev_variance = current_variance.detach()
            
            # Total loss
            loss = recon_loss + CONFIG['beta'] * kl_loss + var_penalty + rate_penalty
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Calculate compliance
        compliance = 1.0 - (epoch_violations / epoch_batches)
        history['compliance'].append(compliance)
        history['train_loss'].append(epoch_loss / epoch_batches)
        history['violations'].append(epoch_violations)
        
        # Test evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, in test_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                test_loss += F.binary_cross_entropy(recon, data, reduction='sum').item()
        
        test_loss /= len(test_loader.dataset)
        history['test_loss'].append(test_loss)
        
        # Save best model
        if compliance > best_compliance:
            best_compliance = compliance
            torch.save(model.state_dict(), '/home/cy/git/canidae/models/mnist_best_2pi.pth')
        
        # Progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Compliance={compliance:.1%}, "
                  f"Train Loss={epoch_loss/epoch_batches:.2f}, Test Loss={test_loss:.2f}")
    
    return history, best_compliance

def main():
    """Run MNIST 2π experiment"""
    
    print("="*60)
    print("🔢 MNIST 2π REGULATION TEST")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist()
    
    # Create model
    model = MNISTVAE(CONFIG['latent_dim']).to(device)
    
    # Train
    start_time = time.time()
    history, best_compliance = train_with_2pi(model, train_loader, test_loader, device)
    train_time = time.time() - start_time
    
    # Results
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Best 2π Compliance: {best_compliance:.1%}")
    print(f"Final Train Loss: {history['train_loss'][-1]:.2f}")
    print(f"Final Test Loss: {history['test_loss'][-1]:.2f}")
    print(f"Total Violations: {sum(history['violations'])}")
    print(f"Training Time: {train_time/60:.1f} minutes")
    
    # Compare with dSprites
    print("\n📊 COMPARISON:")
    print("dSprites:  99.9% compliance, 25.24 loss")
    print(f"MNIST:     {best_compliance:.1%} compliance, {history['test_loss'][-1]:.2f} loss")
    
    # Save results
    results = {
        'dataset': 'MNIST',
        'best_compliance': float(best_compliance),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_test_loss': float(history['test_loss'][-1]),
        'total_violations': sum(history['violations']),
        'training_time_min': train_time/60,
        'config': CONFIG,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    results_path = Path('/home/cy/git/canidae/experiments/results/mnist_2pi_results.json')
    results_path.parent.mkdir(exist_ok=True, parents=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Update metadata if successful
    if best_compliance > 0.95:
        print("\n✅ 2π UNIVERSALITY CONFIRMED ON MNIST!")
        # Would update metadata.yaml here
    
    return best_compliance

if __name__ == "__main__":
    compliance = main()