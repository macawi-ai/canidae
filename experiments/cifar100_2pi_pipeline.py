#!/usr/bin/env python3
"""
CIFAR-100 2π Pipeline - The Ultimate Test
100 Fine-Grained Classes with 2π Regulation
Testing if the universal principle holds at scale
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

# CIFAR-100 Configuration - Scaled for complexity
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100 - UNCHANGED!
    "variance_threshold_init": 1.5,
    "variance_threshold_final": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0,
    "learning_rate": 0.001,
    "batch_size": 64,  # Smaller batch for 100 classes
    "latent_dim": 512,  # Larger latent space for 100 classes
    "beta": 0.1,
    "epochs": 30  # More epochs for complex dataset
}

class CIFAR100VAE(nn.Module):
    """Enhanced VAE for 100-class complexity"""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # Deeper encoder for 100 classes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8->4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=1, padding=0), # 4->1
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=1, padding=0), # 1->4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # 16->32
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
        h = h.view(-1, 512, 1, 1)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def load_cifar100():
    """Load CIFAR-100 with 100 fine-grained classes"""
    print("Loading CIFAR-100 (100 fine-grained classes)...")
    
    data_dir = Path("/workspace/canidae/datasets/cifar100/cifar-100-python")
    
    # Load training data
    with open(data_dir / "train", 'rb') as f:
        train_dict = pickle.load(f, encoding='bytes')
        train_images = train_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        train_fine_labels = np.array(train_dict[b'fine_labels'])
        train_coarse_labels = np.array(train_dict[b'coarse_labels'])
    
    # Load test data
    with open(data_dir / "test", 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')
        test_images = test_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        test_fine_labels = np.array(test_dict[b'fine_labels'])
        test_coarse_labels = np.array(test_dict[b'coarse_labels'])
    
    # Load metadata
    with open(data_dir / "meta", 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        fine_label_names = [name.decode() for name in meta[b'fine_label_names']]
        coarse_label_names = [name.decode() for name in meta[b'coarse_label_names']]
    
    print(f"Train: {train_images.shape} with 100 fine classes, 20 coarse classes")
    print(f"Test: {test_images.shape}")
    print(f"Classes range from '{fine_label_names[0]}' to '{fine_label_names[-1]}'")
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_images))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_images))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False
    )
    
    return train_loader, test_loader, fine_label_names, coarse_label_names

def train_with_2pi(model, train_loader, test_loader, device):
    """Train with 2π regulation on 100 classes"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Tracking metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'variance_rates': [],
        'compliance_percentages': [],
        'epoch_times': [],
        'cwu_counts': []
    }
    
    prev_variances = None
    total_cwus = 0
    
    print("\n" + "="*60)
    print("TRAINING WITH 2π REGULATION ON 100 CLASSES")
    print("="*60)
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        variance_rates = []
        compliant_samples = 0
        total_samples = 0
        epoch_cwus = 0
        
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
                
                # Count CWUs
                epoch_cwus += 1
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
            
            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item()/data.size(0):.2f}, "
                      f"CWUs={epoch_cwus}")
        
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
        epoch_time = time.time() - start_time
        total_cwus += epoch_cwus
        
        metrics['train_loss'].append(avg_train_loss)
        metrics['test_loss'].append(avg_test_loss)
        metrics['variance_rates'].append(avg_variance_rate)
        metrics['compliance_percentages'].append(compliance_pct)
        metrics['epoch_times'].append(epoch_time)
        metrics['cwu_counts'].append(epoch_cwus)
        
        # Status update
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.2f}")
        print(f"  Test Loss: {avg_test_loss:.2f}")
        print(f"  Variance Rate: {avg_variance_rate:.6f} (threshold: {CONFIG['stability_coefficient']:.6f})")
        print(f"  2π Compliance: {compliance_pct:.1f}%")
        print(f"  CWUs: {epoch_cwus} (Total: {total_cwus})")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Check if we're achieving high compliance
        if compliance_pct > 95:
            print(f"  🎯 HIGH 2π COMPLIANCE ON 100 CLASSES!")
            print(f"  🔊 {'cwooo ' * min(int(epoch_cwus/100), 10)}")
    
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, test_loader, fine_labels, coarse_labels = load_cifar100()
    
    # Create model
    model = CIFAR100VAE(latent_dim=CONFIG['latent_dim']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with 2π regulation
    metrics = train_with_2pi(model, train_loader, test_loader, device)
    
    # Results summary
    print("\n" + "="*60)
    print("CIFAR-100 2π REGULATION RESULTS")
    print("100 FINE-GRAINED CLASSES")
    print("="*60)
    
    final_compliance = metrics['compliance_percentages'][-1]
    avg_compliance = np.mean(metrics['compliance_percentages'][-5:])
    
    print(f"Final 2π Compliance: {final_compliance:.1f}%")
    print(f"Average Last 5 Epochs: {avg_compliance:.1f}%")
    print(f"Final Train Loss: {metrics['train_loss'][-1]:.2f}")
    print(f"Final Test Loss: {metrics['test_loss'][-1]:.2f}")
    print(f"Total CWUs Generated: {sum(metrics['cwu_counts'])}")
    print(f"Average Time per Epoch: {np.mean(metrics['epoch_times']):.1f}s")
    
    # Save results
    results = {
        'dataset': 'CIFAR-100',
        'description': '100 fine-grained natural image classes',
        'num_classes': 100,
        'num_superclasses': 20,
        'model': 'Deep CNN-VAE',
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': CONFIG,
        'metrics': metrics,
        'final_compliance': final_compliance,
        'avg_compliance_last5': avg_compliance,
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("/workspace/canidae/experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"cifar100_2pi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final verdict
    print("\n" + "="*60)
    if final_compliance > 95:
        print("✅ SUCCESS! 2π WORKS ON 100 CLASSES!")
        print("The principle scales to fine-grained categorization!")
    elif final_compliance > 80:
        print("⚠️ GOOD PROGRESS - 2π showing strong effect on 100 classes")
    else:
        print("🔄 100 classes need more epochs or tuning")
    print("="*60)
    print("\n🦊🐺 THE PACK CONQUERS COMPLEXITY!")

if __name__ == "__main__":
    main()