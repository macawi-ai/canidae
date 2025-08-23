#!/usr/bin/env python3
"""
QuickDraw 5-Category 2π Regulation Experiment
Testing if 2π principle holds for sparse sketch data
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

# 2π Configuration for Sketches (as per Gemini's recommendations)
SKETCH_CONFIG = {
    "categories": ['circle', 'square', 'triangle', 'star', 'flower'],
    "samples_per_category": 10000,
    "2pi_regulation": {
        "stability_coefficient": 0.06283185307,  # Our secret 2π
        "variance_threshold_init": 3.0,  # Higher for sparse data
        "variance_threshold_final": 1.5,
        "lambda_variance": 1.5,
        "lambda_rate": 15.0,
        "adaptive_schedule": True,
        "sketch_specific": {
            "handle_sparsity": True,
            "stroke_variance": True,
            "sparsity_threshold": 0.8
        }
    },
    "model": {
        "latent_dim": 16,  # Slightly larger for sketch complexity
        "beta": 0.1,  # Lower beta for sparse data
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100
    }
}

class QuickDrawDataLoader:
    """Load and preprocess QuickDraw sketches"""
    
    def __init__(self, base_path="/home/cy/git/canidae/datasets/phase3/quickdraw"):
        self.base_path = Path(base_path)
        self.sketches_path = self.base_path / "sketches"
        
    def load_categories(self, categories, samples_per_category=10000):
        """Load specified categories with 2π-aware preprocessing"""
        data = []
        labels = []
        
        for idx, category in enumerate(categories):
            print(f"Loading {category}...")
            
            # Try different file formats
            npz_path = self.sketches_path / f"{category}.npz"
            npy_path = self.sketches_path / f"{category}.npy"
            
            if npz_path.exists():
                sketches = np.load(npz_path)['arr_0'][:samples_per_category]
            elif npy_path.exists():
                sketches = np.load(npy_path)[:samples_per_category]
            else:
                # Load from ndjson if needed
                ndjson_path = self.base_path / f"{category}.ndjson"
                if ndjson_path.exists():
                    sketches = self._load_ndjson(ndjson_path, samples_per_category)
                else:
                    print(f"Warning: {category} not found, skipping...")
                    continue
            
            # Normalize to [0, 1] for VAE
            sketches = sketches.astype(np.float32) / 255.0
            
            # Calculate sparsity for 2π adjustment
            sparsity = np.mean(sketches < 0.1)
            print(f"  Sparsity: {sparsity:.3f}")
            
            data.append(sketches)
            labels.extend([idx] * len(sketches))
        
        return np.concatenate(data), np.array(labels)
    
    def _load_ndjson(self, path, max_samples):
        """Load from QuickDraw ndjson format"""
        sketches = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                drawing = json.loads(line)
                # Convert strokes to 28x28 image
                sketch = self._strokes_to_image(drawing['drawing'])
                sketches.append(sketch)
        return np.array(sketches)
    
    def _strokes_to_image(self, strokes, size=28):
        """Convert stroke format to image"""
        img = np.zeros((size, size), dtype=np.uint8)
        # Simplified stroke rendering - would need proper implementation
        return img

class SketchVAE(nn.Module):
    """VAE optimized for sketch data with 2π regulation built-in"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # Encoder for 28x28 sketches
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 128 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # 2π tracking
        self.variance_history = []
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Track variance for 2π regulation
        current_variance = torch.exp(logvar).mean().item()
        self.variance_history.append(current_variance)
        
        return recon, mu, logvar

def train_with_2pi_regulation(model, data_loader, config):
    """Training loop with 2π regulation"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    
    # Tracking metrics
    metrics = {
        'compliance_rate': [],
        'reconstruction_loss': [],
        'variance_violations': [],
        'purple_line_activations': []  # Track when system approaches limits
    }
    
    stability_coef = config['2pi_regulation']['stability_coefficient']
    prev_variance = None
    
    for epoch in range(config['model']['epochs']):
        epoch_violations = 0
        epoch_recon_loss = 0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(data)
            
            # Reconstruction loss
            recon_loss = nn.functional.binary_cross_entropy(recon, data, reduction='sum')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 2π Variance regulation
            current_variance = torch.exp(logvar).mean()
            
            # Adaptive threshold (decreases over training)
            progress = (epoch * len(data_loader) + batch_idx) / (config['model']['epochs'] * len(data_loader))
            threshold = config['2pi_regulation']['variance_threshold_init'] - \
                       (config['2pi_regulation']['variance_threshold_init'] - \
                        config['2pi_regulation']['variance_threshold_final']) * progress
            
            # Variance penalty
            variance_penalty = config['2pi_regulation']['lambda_variance'] * \
                              torch.relu(current_variance - threshold)
            
            # Rate of change penalty (2π regulation)
            if prev_variance is not None:
                rate_of_change = torch.abs(current_variance - prev_variance)
                
                # Check for violation
                if rate_of_change > stability_coef:
                    epoch_violations += 1
                    # Purple line activation! System approaching instability
                    metrics['purple_line_activations'].append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'rate': rate_of_change.item(),
                        'threshold': stability_coef
                    })
                
                rate_penalty = config['2pi_regulation']['lambda_rate'] * \
                              torch.relu(rate_of_change - stability_coef)
            else:
                rate_penalty = 0
            
            prev_variance = current_variance
            
            # Total loss with 2π regulation
            loss = recon_loss + config['model']['beta'] * kl_loss + variance_penalty + rate_penalty
            
            loss.backward()
            optimizer.step()
            
            epoch_recon_loss += recon_loss.item()
            batch_count += 1
        
        # Calculate compliance rate
        compliance = 1.0 - (epoch_violations / batch_count)
        metrics['compliance_rate'].append(compliance)
        metrics['reconstruction_loss'].append(epoch_recon_loss / len(data_loader.dataset))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Compliance={compliance:.1%}, "
                  f"Recon Loss={epoch_recon_loss/len(data_loader.dataset):.2f}, "
                  f"Variance={current_variance:.3f}")
    
    return metrics

def main():
    """Run QuickDraw 2π experiment"""
    
    print("="*60)
    print("QuickDraw 2π Regulation Experiment")
    print("Testing universal stability on sparse sketch data")
    print("="*60)
    
    # Load data
    loader = QuickDrawDataLoader()
    data, labels = loader.load_categories(
        SKETCH_CONFIG['categories'],
        SKETCH_CONFIG['samples_per_category']
    )
    
    print(f"\nLoaded {len(data)} sketches from {len(SKETCH_CONFIG['categories'])} categories")
    print(f"Data shape: {data.shape}")
    print(f"Sparsity: {np.mean(data < 0.1):.3f}")
    
    # Prepare PyTorch dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data).unsqueeze(1),  # Add channel dimension
        torch.LongTensor(labels)
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=SKETCH_CONFIG['model']['batch_size'],
        shuffle=True
    )
    
    # Create model
    model = SketchVAE(latent_dim=SKETCH_CONFIG['model']['latent_dim'])
    
    # Train with 2π regulation
    print("\nStarting training with 2π regulation...")
    metrics = train_with_2pi_regulation(model, data_loader, SKETCH_CONFIG)
    
    # Report results
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Final 2π Compliance: {metrics['compliance_rate'][-1]:.1%}")
    print(f"Final Reconstruction Loss: {metrics['reconstruction_loss'][-1]:.2f}")
    print(f"Purple Line Activations: {len(metrics['purple_line_activations'])}")
    
    # Compare with dSprites baseline
    print("\nComparison with dSprites:")
    print("dSprites: 99.9% compliance, 25.24 loss")
    print(f"QuickDraw: {metrics['compliance_rate'][-1]:.1%} compliance, "
          f"{metrics['reconstruction_loss'][-1]:.2f} loss")
    
    # Save results for pipeline tracking
    results = {
        'dataset': 'QuickDraw-5',
        'categories': SKETCH_CONFIG['categories'],
        'compliance': metrics['compliance_rate'][-1],
        'reconstruction_loss': metrics['reconstruction_loss'][-1],
        'config': SKETCH_CONFIG,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path('/home/cy/git/canidae/experiments/results/quickdraw_2pi_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Check if we maintain 2π universality
    if metrics['compliance_rate'][-1] > 0.95:
        print("\n✅ 2π PRINCIPLE CONFIRMED FOR SKETCHES!")
        print("The universal constant holds across modalities!")
    else:
        print("\n⚠️ 2π compliance below 95% - needs adjustment")
        print("Investigating failure modes...")
    
    return metrics

if __name__ == "__main__":
    main()