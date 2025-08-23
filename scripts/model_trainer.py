#!/usr/bin/env python3
"""
Model Trainer Module for 2π Validation Pipeline
Handles training of 2π regulated models and baselines for controlled comparison

Features:
- 2π regulated training with compliance monitoring
- Baseline model training (identical architecture, no 2π)
- Multiple model training in parallel
- Training curve logging and visualization
- Model checkpointing and recovery
- Fair comparison protocols
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FashionVAE(nn.Module):
    """VAE architecture for Fashion-MNIST (and similar 28x28 datasets)"""
    
    def __init__(self, latent_dim: int = 10, use_2pi: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_2pi = use_2pi
        
        # Encoder - slightly deeper for texture patterns
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
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

class StandardVAE(nn.Module):
    """Standard VAE without 2π regulation (baseline)"""
    
    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Identical architecture to FashionVAE
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
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

class TwoPiTrainer:
    """Trainer for 2π regulated models with compliance monitoring"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.two_pi_config = config.get('two_pi_regulation', {})
        
        # 2π regulation parameters
        self.stability_coefficient = self.two_pi_config.get('stability_coefficient', 0.06283185307)
        self.variance_threshold_init = self.two_pi_config.get('variance_threshold_init', 1.5)
        self.variance_threshold_final = self.two_pi_config.get('variance_threshold_final', 1.0)
        self.lambda_variance = self.two_pi_config.get('lambda_variance', 1.0)
        self.lambda_rate = self.two_pi_config.get('lambda_rate', 10.0)
        
        # Training parameters
        hyperparams = config['hyperparameters']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.epochs = hyperparams['epochs']
        self.beta = hyperparams.get('beta', 0.1)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training state
        self.prev_variance = None
        self.training_history = {
            'compliance': [],
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [], 
            'var_penalty': [],
            'rate_penalty': [],
            'violations': []
        }
        
        logger.info(f"TwoPiTrainer initialized with stability_coefficient={self.stability_coefficient}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train one epoch with 2π regulation"""
        
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0, 
            'kl': 0.0,
            'var_penalty': 0.0,
            'rate_penalty': 0.0
        }
        epoch_violations = 0
        epoch_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch_data, (list, tuple)):
                data = batch_data[0].to(self.device)
            else:
                data = batch_data.to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(data)
            
            # Standard VAE losses
            recon_loss = F.binary_cross_entropy(recon, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            
            # 2π Regulation
            current_variance = torch.exp(logvar).mean()
            
            # Adaptive threshold (decreases during training)
            total_batches = self.epochs * len(train_loader)
            current_batch = epoch_batches
            progress = current_batch / total_batches if total_batches > 0 else 0
            threshold = self.variance_threshold_init - \
                       (self.variance_threshold_init - self.variance_threshold_final) * progress
            
            # Variance penalty
            var_penalty = self.lambda_variance * F.relu(current_variance - threshold)
            
            # Rate penalty (THE 2π MAGIC!)
            rate_penalty = torch.tensor(0.0, device=self.device)
            if self.prev_variance is not None:
                rate = torch.abs(current_variance - self.prev_variance)
                
                # Check for violation
                if rate.item() > self.stability_coefficient:
                    epoch_violations += 1
                
                rate_penalty = self.lambda_rate * F.relu(rate - self.stability_coefficient)
            
            self.prev_variance = current_variance.detach()
            
            # Total loss
            total_loss = recon_loss + self.beta * kl_loss + var_penalty + rate_penalty
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item() 
            epoch_losses['var_penalty'] += var_penalty.item()
            epoch_losses['rate_penalty'] += rate_penalty.item()
            epoch_batches += 1
        
        # Calculate averages
        for key in epoch_losses:
            epoch_losses[key] /= epoch_batches
        
        # Calculate compliance
        compliance = 1.0 - (epoch_violations / epoch_batches) if epoch_batches > 0 else 0.0
        
        return {
            **epoch_losses,
            'compliance': compliance,
            'violations': epoch_violations,
            'batches': epoch_batches
        }
    
    def train(self, train_loader, val_loader=None) -> Dict[str, Any]:
        """Full training loop with 2π regulation"""
        
        logger.info(f"Starting 2π regulated training for {self.epochs} epochs...")
        
        best_compliance = 0.0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Train epoch
            epoch_results = self.train_epoch(train_loader)
            
            # Validation (if provided)
            val_results = {}
            if val_loader is not None:
                val_results = self.validate(val_loader)
            
            # Update history
            self.training_history['compliance'].append(epoch_results['compliance'])
            self.training_history['train_loss'].append(epoch_results['total'])
            self.training_history['recon_loss'].append(epoch_results['recon'])
            self.training_history['kl_loss'].append(epoch_results['kl'])
            self.training_history['var_penalty'].append(epoch_results['var_penalty'])
            self.training_history['rate_penalty'].append(epoch_results['rate_penalty'])
            self.training_history['violations'].append(epoch_results['violations'])
            
            # Save best model
            if epoch_results['compliance'] > best_compliance:
                best_compliance = epoch_results['compliance']
                best_model_state = self.model.state_dict().copy()
            
            # Logging
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(f"Epoch {epoch:3d}: Compliance={epoch_results['compliance']:.1%}, "
                           f"Loss={epoch_results['total']:.2f}, "
                           f"Violations={epoch_results['violations']}")
        
        training_time = time.time() - start_time
        
        # Final results
        results = {
            'best_compliance': best_compliance,
            'final_loss': self.training_history['train_loss'][-1],
            'total_violations': sum(self.training_history['violations']),
            'training_time_min': training_time / 60,
            'training_history': self.training_history,
            'best_model_state': best_model_state
        }
        
        logger.info(f"2π training completed: {best_compliance:.1%} compliance, "
                   f"{training_time/60:.1f} minutes")
        
        return results
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0].to(self.device)
                else:
                    data = batch_data.to(self.device)
                
                recon, mu, logvar = self.model(data)
                loss = F.binary_cross_entropy(recon, data, reduction='sum')
                val_loss += loss.item()
        
        val_loss /= len(val_loader.dataset)
        return {'val_loss': val_loss}

class StandardTrainer:
    """Trainer for standard models (baselines) without 2π regulation"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        
        # Training parameters (identical to 2π trainer for fair comparison)
        hyperparams = config['hyperparameters']
        self.learning_rate = hyperparams['learning_rate']
        self.epochs = hyperparams['epochs']
        self.beta = hyperparams.get('beta', 0.1)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        logger.info("StandardTrainer initialized (no 2π regulation)")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train one epoch with standard VAE loss"""
        
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0
        }
        epoch_batches = 0
        
        for batch_data in train_loader:
            # Handle different batch formats
            if isinstance(batch_data, (list, tuple)):
                data = batch_data[0].to(self.device)
            else:
                data = batch_data.to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(data)
            
            # Standard VAE losses ONLY
            recon_loss = F.binary_cross_entropy(recon, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            
            # Total loss (no 2π regulation!)
            total_loss = recon_loss + self.beta * kl_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_batches += 1
        
        # Calculate averages
        for key in epoch_losses:
            epoch_losses[key] /= epoch_batches
        
        return epoch_losses
    
    def train(self, train_loader, val_loader=None) -> Dict[str, Any]:
        """Full training loop without 2π regulation"""
        
        logger.info(f"Starting standard training for {self.epochs} epochs...")
        
        best_loss = float('inf')
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Train epoch
            epoch_results = self.train_epoch(train_loader)
            
            # Validation (if provided)
            val_results = {}
            if val_loader is not None:
                val_results = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(epoch_results['total'])
            self.training_history['recon_loss'].append(epoch_results['recon'])
            self.training_history['kl_loss'].append(epoch_results['kl'])
            
            # Save best model
            if epoch_results['total'] < best_loss:
                best_loss = epoch_results['total']
                best_model_state = self.model.state_dict().copy()
            
            # Logging
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(f"Epoch {epoch:3d}: Loss={epoch_results['total']:.2f}")
        
        training_time = time.time() - start_time
        
        # Final results
        results = {
            'best_loss': best_loss,
            'final_loss': self.training_history['train_loss'][-1], 
            'training_time_min': training_time / 60,
            'training_history': self.training_history,
            'best_model_state': best_model_state,
            'compliance': 0.0  # No compliance tracking for standard models
        }
        
        logger.info(f"Standard training completed: {best_loss:.2f} loss, "
                   f"{training_time/60:.1f} minutes")
        
        return results
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step for standard trainer"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0].to(self.device)
                else:
                    data = batch_data.to(self.device)
                
                recon, mu, logvar = self.model(data)
                loss = F.binary_cross_entropy(recon, data, reduction='sum')
                val_loss += loss.item()
        
        val_loss /= len(val_loader.dataset)
        return {'val_loss': val_loss}

class ModelTrainer:
    """Main interface for training multiple models (2π + baselines)"""
    
    MODEL_REGISTRY = {
        'FashionVAE': FashionVAE,
        'StandardVAE': StandardVAE
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config.get('output', {}).get('base_path', 'outputs'))
        
        logger.info(f"ModelTrainer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        device_config = self.config.get('resources', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        return device
    
    def _create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration"""
        
        model_name = model_config['architecture']
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model architecture: {model_name}")
        
        model_class = self.MODEL_REGISTRY[model_name]
        
        # Model parameters
        latent_dim = model_config.get('latent_dim', 10)
        use_2pi = model_config.get('use_2pi_regulation', True)
        
        # Create model
        if 'use_2pi' in model_class.__init__.__code__.co_varnames:
            model = model_class(latent_dim=latent_dim, use_2pi=use_2pi)
        else:
            model = model_class(latent_dim=latent_dim)
        
        return model.to(self.device)
    
    def train_models(self, train_loader, val_loader=None, test_loader=None) -> Dict[str, Any]:
        """Train primary model and all baselines"""
        
        results = {}
        
        # Train primary model (2π regulated)
        logger.info("="*60)
        logger.info("Training PRIMARY MODEL (2π regulated)")
        logger.info("="*60)
        
        primary_model = self._create_model(self.config['model'])
        trainer = TwoPiTrainer(self.config, primary_model, self.device)
        primary_results = trainer.train(train_loader, val_loader)
        
        # Save primary model
        model_path = self.output_dir / 'models' / 'primary_2pi_model.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(primary_results['best_model_state'], model_path)
        
        results['primary_2pi'] = {
            **primary_results,
            'model_path': str(model_path),
            'architecture': self.config['model']['architecture'],
            'regulation': '2π'
        }
        
        # Train baselines
        if 'baselines' in self.config:
            logger.info("="*60)
            logger.info("Training BASELINE MODELS")
            logger.info("="*60)
            
            for baseline_config in self.config['baselines']:
                baseline_name = baseline_config['name']
                logger.info(f"\nTraining baseline: {baseline_name}")
                
                # Create baseline model
                baseline_model = self._create_model(baseline_config)
                
                # Use appropriate trainer
                if baseline_config.get('use_2pi_regulation', False):
                    # Special case: 2π with different parameters
                    baseline_trainer = TwoPiTrainer(
                        {**self.config, **baseline_config}, 
                        baseline_model, 
                        self.device
                    )
                else:
                    # Standard training
                    baseline_trainer = StandardTrainer(
                        {**self.config, **baseline_config}, 
                        baseline_model, 
                        self.device
                    )
                
                baseline_results = baseline_trainer.train(train_loader, val_loader)
                
                # Save baseline model
                model_path = self.output_dir / 'models' / f'{baseline_name}_model.pth'
                torch.save(baseline_results['best_model_state'], model_path)
                
                results[baseline_name] = {
                    **baseline_results,
                    'model_path': str(model_path),
                    'architecture': baseline_config['architecture'],
                    'regulation': '2π' if baseline_config.get('use_2pi_regulation', False) else 'standard'
                }
        
        # Summary comparison
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        for model_name, model_results in results.items():
            if 'compliance' in model_results:
                logger.info(f"{model_name:20s}: {model_results.get('compliance', 0):.1%} compliance, "
                           f"{model_results['final_loss']:.2f} loss")
            else:
                logger.info(f"{model_name:20s}: {model_results['final_loss']:.2f} loss")
        
        return results

# Utility functions

def train_models(config: Dict[str, Any], train_loader, val_loader=None, test_loader=None) -> Dict[str, Any]:
    """Main entry point for model training"""
    
    trainer = ModelTrainer(config)
    return trainer.train_models(train_loader, val_loader, test_loader)

def save_training_curves(results: Dict[str, Any], output_dir: Path):
    """Save training curves for visualization"""
    
    curves_data = {}
    for model_name, model_results in results.items():
        if 'training_history' in model_results:
            curves_data[model_name] = model_results['training_history']
    
    # Save to JSON for visualization scripts
    curves_file = output_dir / 'results' / 'training_curves.json'
    curves_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(curves_file, 'w') as f:
        json.dump(curves_data, f, indent=2, default=str)
    
    logger.info(f"Training curves saved to {curves_file}")

if __name__ == "__main__":
    # Simple test of model trainer
    test_config = {
        'model': {
            'architecture': 'FashionVAE',
            'use_2pi_regulation': True,
            'latent_dim': 10
        },
        'baselines': [
            {
                'name': 'standard_baseline',
                'architecture': 'StandardVAE',
                'use_2pi_regulation': False
            }
        ],
        'two_pi_regulation': {
            'stability_coefficient': 0.06283185307,
            'variance_threshold_init': 1.5,
            'variance_threshold_final': 1.0,
            'lambda_variance': 1.0,
            'lambda_rate': 10.0
        },
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,
            'beta': 0.1
        },
        'resources': {
            'device': 'auto'
        },
        'output': {
            'base_path': '/tmp/test_training'
        }
    }
    
    print("🧪 Testing Model Trainer...")
    
    try:
        # Create dummy data loader for testing (normalized to [0,1] for BCE)
        dummy_data = torch.rand(100, 28, 28)  # Random data in [0,1] range
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=32)
        
        results = train_models(test_config, dummy_loader)
        
        print("✅ Model trainer test successful!")
        for name, result in results.items():
            print(f"{name}: {result.get('compliance', 0):.1%} compliance, {result['final_loss']:.3f} loss")
        
    except Exception as e:
        print(f"❌ Model trainer test failed: {e}")
        import traceback
        traceback.print_exc()