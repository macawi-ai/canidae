#!/usr/bin/env python3
"""
Shapes3D 2π Pipeline - Disentanglement Under Universal Regulation
Testing if 2π maintains factor independence in 3D scenes
480,000 images with 6 ground-truth factors
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import h5py
import json
import time
from datetime import datetime

# Shapes3D + 2π Configuration
CONFIG = {
    "stability_coefficient": 0.06283185307,  # 2π/100 - THE CONSTANT
    "variance_threshold": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0,
    "learning_rate": 0.001,
    "batch_size": 128,
    "latent_dim": 10,  # 6 factors + 4 extra for flexibility
    "beta": 4.0,  # Higher beta for better disentanglement
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class Shapes3DVAE(nn.Module):
    """VAE for Shapes3D with disentanglement focus"""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        
        # Encoder - designed for 64x64x3 images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8->4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32->64
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
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def load_shapes3d():
    """Load Shapes3D dataset from h5 file"""
    print("Loading Shapes3D dataset...")
    
    data_path = Path("/workspace/canidae/datasets/shapes3d/3dshapes.h5")
    
    with h5py.File(data_path, 'r') as f:
        # Images are stored as (480000, 64, 64, 3) uint8
        images = f['images'][:]
        # Labels are stored as (480000, 6) float64
        # [floor_hue, wall_hue, object_hue, scale, shape, orientation]
        labels = f['labels'][:]
    
    print(f"Loaded {len(images)} images with shape {images[0].shape}")
    print(f"Factors: floor_hue, wall_hue, object_hue, scale, shape, orientation")
    
    # Convert to torch tensors and normalize
    images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0  # NCHW format
    labels = torch.FloatTensor(labels)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(images, labels)
    
    # Split into train/test (90/10)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

def compute_disentanglement_metrics(model, data_loader, device):
    """Compute MIG (Mutual Information Gap) for disentanglement"""
    model.eval()
    
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            _, _, _, z = model(data)
            all_z.append(z.cpu())
            all_labels.append(labels)
            
            if len(all_z) * CONFIG['batch_size'] >= 10000:  # Sample 10K for metrics
                break
    
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Simple variance-based disentanglement metric
    # For each latent dimension, compute variance when varying each factor
    n_factors = all_labels.shape[1]
    n_latents = all_z.shape[1]
    
    importance_matrix = np.zeros((n_factors, n_latents))
    
    for f in range(n_factors):
        for l in range(n_latents):
            # Compute variance of latent l when factor f changes
            factor_values = np.unique(all_labels[:, f])
            if len(factor_values) > 1:
                variances = []
                for val in factor_values[:10]:  # Sample up to 10 values
                    mask = all_labels[:, f] == val
                    if mask.sum() > 1:
                        variances.append(np.var(all_z[mask, l]))
                if variances:
                    importance_matrix[f, l] = np.mean(variances)
    
    # Normalize importance matrix
    importance_matrix = importance_matrix / (importance_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Compute disentanglement: ideally each factor maps to one latent
    disentanglement = 0
    for f in range(n_factors):
        if importance_matrix[f].max() > 0:
            # Entropy of importance distribution (lower = more disentangled)
            entropy = -np.sum(importance_matrix[f] * np.log(importance_matrix[f] + 1e-8))
            disentanglement += (1 - entropy / np.log(n_latents))
    
    disentanglement /= n_factors
    
    return disentanglement, importance_matrix

def train_with_2pi(model, train_loader, test_loader, device):
    """Train with 2π regulation on Shapes3D"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Tracking metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'variance_rates': [],
        'compliance_percentages': [],
        'disentanglement_scores': [],
        'cwu_counts': []
    }
    
    prev_variances = None
    total_cwus = 0
    
    print("\n" + "="*60)
    print("TRAINING WITH 2π REGULATION ON SHAPES3D")
    print("Testing Disentanglement Under Universal Law")
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
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar, z = model(data)
            
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
                
                # Count CWU
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
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item()/data.size(0):.4f}, "
                      f"CWUs={epoch_cwus}")
        
        # Calculate compliance percentage
        compliance_pct = (compliant_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Evaluation
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                recon, mu, logvar, z = model(data)
                recon_loss = F.mse_loss(recon, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + CONFIG['beta'] * kl_loss
                test_losses.append(loss.item())
        
        # Compute disentanglement metric
        disentanglement, importance = compute_disentanglement_metrics(model, test_loader, device)
        
        # Record metrics
        avg_train_loss = np.mean(train_losses) / CONFIG['batch_size']
        avg_test_loss = np.mean(test_losses) / CONFIG['batch_size']
        avg_variance_rate = np.mean(variance_rates) if variance_rates else 0
        total_cwus += epoch_cwus
        
        metrics['train_loss'].append(avg_train_loss)
        metrics['test_loss'].append(avg_test_loss)
        metrics['variance_rates'].append(avg_variance_rate)
        metrics['compliance_percentages'].append(compliance_pct)
        metrics['disentanglement_scores'].append(disentanglement)
        metrics['cwu_counts'].append(epoch_cwus)
        
        # Status update
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Variance Rate: {avg_variance_rate:.6f} (threshold: {CONFIG['stability_coefficient']:.6f})")
        print(f"  2π Compliance: {compliance_pct:.1f}%")
        print(f"  Disentanglement Score: {disentanglement:.4f}")
        print(f"  CWUs: {epoch_cwus} (Total: {total_cwus})")
        print(f"  Time: {time.time() - start_time:.1f}s")
        
        # Check if we're achieving high compliance
        if compliance_pct > 95:
            print(f"  🎯 HIGH 2π COMPLIANCE WITH DISENTANGLEMENT!")
            print(f"  🔊 {'cwoo ' * min(int(epoch_cwus/100), 5)}")
    
    return metrics, importance

def main():
    # Set device
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, test_loader = load_shapes3d()
    
    # Create model
    model = Shapes3DVAE(latent_dim=CONFIG['latent_dim']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with 2π regulation
    metrics, importance_matrix = train_with_2pi(model, train_loader, test_loader, device)
    
    # Results summary
    print("\n" + "="*60)
    print("SHAPES3D 2π REGULATION RESULTS")
    print("="*60)
    
    final_compliance = metrics['compliance_percentages'][-1]
    final_disentanglement = metrics['disentanglement_scores'][-1]
    avg_compliance = np.mean(metrics['compliance_percentages'][-5:])
    avg_disentanglement = np.mean(metrics['disentanglement_scores'][-5:])
    
    print(f"Final 2π Compliance: {final_compliance:.1f}%")
    print(f"Final Disentanglement: {final_disentanglement:.4f}")
    print(f"Average Last 5 Epochs Compliance: {avg_compliance:.1f}%")
    print(f"Average Last 5 Epochs Disentanglement: {avg_disentanglement:.4f}")
    print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final Test Loss: {metrics['test_loss'][-1]:.4f}")
    print(f"Total CWUs Generated: {sum(metrics['cwu_counts'])}")
    
    # Analyze factor-latent mapping
    print("\n" + "="*60)
    print("FACTOR-LATENT IMPORTANCE MATRIX")
    print("(Shows which latent dimensions encode which factors)")
    print("="*60)
    
    factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    print("\n" + " "*15 + "".join([f"L{i:2d} " for i in range(CONFIG['latent_dim'])]))
    for f_idx, f_name in enumerate(factor_names):
        print(f"{f_name:12s}: ", end="")
        for l_idx in range(CONFIG['latent_dim']):
            val = importance_matrix[f_idx, l_idx]
            if val > 0.3:
                print(f"▓▓▓ ", end="")
            elif val > 0.1:
                print(f"▒▒▒ ", end="")
            elif val > 0.05:
                print(f"░░░ ", end="")
            else:
                print(f"    ", end="")
        print()
    
    # Save results
    results = {
        'dataset': 'Shapes3D',
        'description': '480K images with 6 ground-truth 3D factors',
        'model': 'β-VAE with 2π regulation',
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': CONFIG,
        'metrics': metrics,
        'final_compliance': final_compliance,
        'final_disentanglement': final_disentanglement,
        'timestamp': datetime.now().isoformat(),
        'importance_matrix': importance_matrix.tolist()
    }
    
    output_dir = Path("/workspace/canidae/experiments/results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"shapes3d_2pi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Save model checkpoint
    model_file = output_dir / f"shapes3d_2pi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': CONFIG
    }, model_file)
    
    print(f"Model saved to: {model_file}")
    
    # Final verdict
    print("\n" + "="*60)
    if final_compliance > 95 and final_disentanglement > 0.5:
        print("✅ SUCCESS! 2π PRESERVES DISENTANGLEMENT!")
        print("The principle maintains factor independence!")
    elif final_compliance > 80:
        print("⚠️ GOOD PROGRESS - 2π and disentanglement are compatible")
    else:
        print("🔄 More tuning needed for optimal disentanglement under 2π")
    print("="*60)
    print("\n🦊🐺 THE PACK DISCOVERS STRUCTURED LEARNING!")

if __name__ == "__main__":
    main()