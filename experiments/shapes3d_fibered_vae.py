#!/usr/bin/env python3
"""
Fibered VAE for Shapes3D with Topological Structure
Phase 1: Circular S¹ encoding for hue factors
Based on Sister Gemini's fiber bundle insights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import h5py
from torch.utils.data import Dataset, DataLoader
import json
import time

# The magic 2π constant - now per-fiber!
TWO_PI = 0.06283185307
PER_FIBER_2PI = TWO_PI / np.sqrt(6)  # ~2.565% per factor for 6 factors

class CircularEncoder(nn.Module):
    """Encode values on S¹ (circle) manifold"""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        
        # Pre-process to circle representation
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output 2D circle coordinates
        self.to_circle = nn.Linear(hidden_dim, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to unit circle (cos θ, sin θ)"""
        h = self.pre_net(x)
        circle_coords = self.to_circle(h)
        # Normalize to unit circle
        return F.normalize(circle_coords, p=2, dim=-1)
    
    def decode_angle(self, z: torch.Tensor) -> torch.Tensor:
        """Recover angle from circle coordinates"""
        return torch.atan2(z[..., 1], z[..., 0])


class LinearEncoder(nn.Module):
    """Standard linear encoder for Euclidean factors"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiberedEncoder(nn.Module):
    """
    Encoder that respects the natural geometry of each factor:
    - floor_hue: S¹ (circular)
    - wall_hue: S¹ (circular)  
    - object_hue: S¹ (circular)
    - scale: R+ (positive reals)
    - shape: Discrete manifold (4 shapes)
    - orientation: SO(3) simplified to S¹ for now
    """
    def __init__(self, input_channels: int = 3, base_dim: int = 32):
        super().__init__()
        
        # Shared convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU()
        )
        
        # Flatten features
        self.flatten_size = 64 * 4 * 4  # After 4 conv layers with stride 2
        
        # Factor-specific encoders with proper geometry
        self.floor_hue_encoder = CircularEncoder(self.flatten_size, base_dim)  # S¹
        self.wall_hue_encoder = CircularEncoder(self.flatten_size, base_dim)   # S¹
        self.object_hue_encoder = CircularEncoder(self.flatten_size, base_dim) # S¹
        self.scale_encoder = LinearEncoder(self.flatten_size, 1, base_dim)     # R+
        self.shape_encoder = LinearEncoder(self.flatten_size, 4, base_dim)     # Discrete
        self.orientation_encoder = CircularEncoder(self.flatten_size, base_dim) # S¹ (simplified)
        
        # Store which factors use circular encoding
        self.circular_factors = [0, 1, 2, 5]  # Indices of circular factors
        self.factor_dims = [2, 2, 2, 1, 4, 2]  # Dimensions per factor
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract shared features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Encode each factor with proper geometry
        latents = {
            'floor_hue': self.floor_hue_encoder(features),      # 2D circle
            'wall_hue': self.wall_hue_encoder(features),        # 2D circle
            'object_hue': self.object_hue_encoder(features),    # 2D circle
            'scale': torch.relu(self.scale_encoder(features)),  # R+ (positive)
            'shape': self.shape_encoder(features),              # 4D discrete
            'orientation': self.orientation_encoder(features)   # 2D circle
        }
        
        return latents


class FiberedDecoder(nn.Module):
    """Decoder that reconstructs from fibered latent space"""
    def __init__(self, output_channels: int = 3):
        super().__init__()
        
        # Total latent dimension: 2+2+2+1+4+2 = 13
        self.latent_dim = 13
        
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1)
        )
        
    def forward(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all latents
        z = torch.cat([
            latents['floor_hue'],
            latents['wall_hue'],
            latents['object_hue'],
            latents['scale'],
            latents['shape'],
            latents['orientation']
        ], dim=-1)
        
        h = self.net(z)
        h = h.view(h.size(0), 64, 4, 4)
        return self.deconv(h)


class PerFiberVarianceRegulator:
    """
    Regulate variance per fiber bundle
    Each factor gets 2π/√n variety budget
    """
    def __init__(self, n_factors: int = 6, device: str = 'cuda'):
        self.n_factors = n_factors
        self.per_fiber_limit = TWO_PI / np.sqrt(n_factors)
        self.device = device
        
        # Track variance history per factor
        self.variance_history = {
            f'factor_{i}': [] for i in range(n_factors)
        }
        
    def compute_fiber_variance(self, latents: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute variance for each fiber"""
        variances = {}
        for name, z in latents.items():
            if z.dim() > 1:
                var = torch.var(z, dim=0).mean().item()
            else:
                var = torch.var(z).item()
            variances[name] = var
        return variances
    
    def compute_variance_rates(self, latents: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute rate of change of variance per fiber"""
        current_vars = self.compute_fiber_variance(latents)
        rates = {}
        
        for i, (name, var) in enumerate(current_vars.items()):
            key = f'factor_{i}'
            if len(self.variance_history[key]) > 0:
                prev_var = self.variance_history[key][-1]
                rate = abs(var - prev_var) / (prev_var + 1e-8)
            else:
                rate = 0.0
            
            rates[name] = rate
            self.variance_history[key].append(var)
            
        return rates
    
    def get_regulation_loss(self, latents: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute per-fiber 2π regulation loss"""
        rates = self.compute_variance_rates(latents)
        
        total_loss = 0.0
        violations = {}
        
        for name, rate in rates.items():
            if rate > self.per_fiber_limit:
                # Penalize violation of per-fiber bound
                violation = (rate - self.per_fiber_limit) ** 2
                total_loss += violation
                violations[name] = rate
                
        metrics = {
            'per_fiber_limit': self.per_fiber_limit,
            'variance_rates': rates,
            'violations': violations,
            'max_rate': max(rates.values()) if rates else 0.0
        }
        
        return torch.tensor(total_loss, device=self.device), metrics


class FiberedVAE(nn.Module):
    """VAE with fiber bundle structure and per-fiber 2π regulation"""
    def __init__(self, beta: float = 4.0, device: str = 'cuda'):
        super().__init__()
        self.encoder = FiberedEncoder()
        self.decoder = FiberedDecoder()
        self.regulator = PerFiberVarianceRegulator(n_factors=6, device=device)
        self.beta = beta
        self.device = device
        
    def reparameterize(self, latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reparameterization trick - respecting each manifold's geometry"""
        sampled = {}
        
        for name, z in latents.items():
            if name in ['floor_hue', 'wall_hue', 'object_hue', 'orientation']:
                # For circular factors, add noise on the tangent space
                # then project back to circle
                noise = torch.randn_like(z) * 0.1
                z_noisy = z + noise
                sampled[name] = F.normalize(z_noisy, p=2, dim=-1)
            elif name == 'scale':
                # For positive reals, use log-normal noise
                log_z = torch.log(z + 1e-8)
                noise = torch.randn_like(log_z) * 0.1
                sampled[name] = torch.exp(log_z + noise)
            else:
                # Standard Gaussian noise for other factors
                noise = torch.randn_like(z) * 0.1
                sampled[name] = z + noise
                
        return sampled
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Encode with proper geometry
        latents = self.encoder(x)
        
        # Reparameterize
        z_sampled = self.reparameterize(latents)
        
        # Decode
        recon = self.decoder(z_sampled)
        
        # Compute losses
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence (simplified for now)
        kl_loss = 0.0
        for name, z in latents.items():
            kl_loss += torch.mean(0.5 * torch.sum(z**2, dim=-1))
        
        # Per-fiber 2π regulation
        reg_loss, reg_metrics = self.regulator.get_regulation_loss(latents)
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss + 10.0 * reg_loss
        
        metrics = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'reg_loss': reg_loss.item(),
            **reg_metrics
        }
        
        return loss, metrics


def compute_mutual_information(latents: Dict[str, torch.Tensor]) -> float:
    """
    Compute mutual information between latent factors
    Lower MI = better disentanglement
    """
    # Concatenate all latents
    z_list = []
    for z in latents.values():
        if z.dim() > 1:
            z_list.append(z)
        else:
            z_list.append(z.unsqueeze(-1))
    
    Z = torch.cat(z_list, dim=-1)
    
    # Compute correlation matrix
    Z_centered = Z - Z.mean(dim=0)
    cov = torch.mm(Z_centered.t(), Z_centered) / (Z.size(0) - 1)
    
    # Mutual information approximation via determinant
    # MI ≈ -0.5 * log(det(correlation_matrix))
    corr = cov / (torch.sqrt(torch.diag(cov)).unsqueeze(0) * torch.sqrt(torch.diag(cov)).unsqueeze(1))
    
    # Add small epsilon for numerical stability
    det = torch.det(corr + 1e-6 * torch.eye(corr.size(0), device=corr.device))
    mi = -0.5 * torch.log(torch.abs(det) + 1e-8)
    
    return mi.item()


class Shapes3DDataset(Dataset):
    """Shapes3D dataset with lazy loading"""
    def __init__(self, path: str, subset_size: int = None):
        self.path = path
        self.h5file = None
        self.subset_size = subset_size
        
        # Get dataset size
        with h5py.File(path, 'r') as f:
            self.total_size = f['images'].shape[0]
            
        if subset_size:
            self.size = min(subset_size, self.total_size)
        else:
            self.size = self.total_size
            
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.h5file is None:
            self.h5file = h5py.File(self.path, 'r')
            
        image = self.h5file['images'][idx]
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # Also return factor values if available
        factors = {}
        if 'labels' in self.h5file:
            labels = self.h5file['labels'][idx]
            factors = {
                'floor_hue': labels[0],
                'wall_hue': labels[1],
                'object_hue': labels[2],
                'scale': labels[3],
                'shape': labels[4],
                'orientation': labels[5]
            }
            
        return image, factors


def train_epoch(model, dataloader, optimizer, device='cuda'):
    """Train one epoch with fibered VAE"""
    model.train()
    epoch_metrics = {
        'loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'reg_loss': 0.0,
        'mutual_info': 0.0,
        'per_fiber_compliance': {}
    }
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        optimizer.zero_grad()
        loss, metrics = model(images)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        for key in ['loss', 'recon_loss', 'kl_loss', 'reg_loss']:
            epoch_metrics[key] += metrics[key]
            
        # Compute mutual information periodically
        if batch_idx % 10 == 0:
            with torch.no_grad():
                latents = model.encoder(images)
                mi = compute_mutual_information(latents)
                epoch_metrics['mutual_info'] += mi
                
    # Average metrics
    n_batches = len(dataloader)
    for key in epoch_metrics:
        if isinstance(epoch_metrics[key], (int, float)):
            epoch_metrics[key] /= n_batches
            
    return epoch_metrics


def main():
    """Run Phase 1: Circular encoding experiment"""
    
    print("="*60)
    print("FIBERED VAE - PHASE 1: CIRCULAR ENCODING")
    print("Implementing Sister Gemini's fiber bundle insights")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load dataset
    dataset_path = '/tmp/3dshapes.h5'  # Assuming copied to local SSD
    dataset = Shapes3DDataset(dataset_path, subset_size=50000)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} samples")
    
    # Create model
    model = FiberedVAE(beta=4.0, device=device).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Per-fiber 2π limit: {model.regulator.per_fiber_limit:.6f}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 10
    
    results = {
        'config': {
            'architecture': 'FiberedVAE',
            'per_fiber_2pi': PER_FIBER_2PI,
            'n_factors': 6,
            'circular_factors': ['floor_hue', 'wall_hue', 'object_hue', 'orientation'],
            'beta': 4.0,
            'epochs': n_epochs
        },
        'epochs': []
    }
    
    print("\nTraining with per-fiber geometry:")
    print("-"*40)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        metrics = train_epoch(model, dataloader, optimizer, device)
        epoch_time = time.time() - start_time
        
        # Check per-fiber compliance
        compliance_count = 0
        for name, rate in metrics.get('variance_rates', {}).items():
            if rate <= model.regulator.per_fiber_limit:
                compliance_count += 1
                
        per_fiber_compliance = compliance_count / 6 * 100
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"  Loss: {metrics['loss']:.2f}")
        print(f"  Recon: {metrics['recon_loss']:.2f}")
        print(f"  KL: {metrics['kl_loss']:.2f}")
        print(f"  Reg: {metrics['reg_loss']:.4f}")
        print(f"  MI: {metrics['mutual_info']:.4f}")
        print(f"  Per-fiber compliance: {per_fiber_compliance:.1f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        results['epochs'].append({
            'epoch': epoch + 1,
            **metrics,
            'per_fiber_compliance': per_fiber_compliance,
            'time': epoch_time
        })
    
    # Save results
    results_path = '/home/cy/git/canidae/results/shapes3d_fibered_phase1.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save model
    model_path = '/home/cy/git/canidae/results/models/shapes3d_fibered_phase1.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE: Circular encoding implemented!")
    print("Next: Compare disentanglement vs baseline")
    print("="*60)


if __name__ == "__main__":
    main()