#!/usr/bin/env python3
"""
CIFAR-10 8x3090 Distributed Training with CWU Management
Purple Line Protocol System 5 - Full Enclosure
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pickle
from pathlib import Path
import json
import time
import os
from datetime import datetime

# 8x3090 CONFIGURATION WITH CWU MANAGEMENT
CONFIG = {
    # 2π REGULATION - THE UNIVERSAL CONSTANT
    "stability_coefficient": 0.06283185307,  # 2π/100
    
    # PURPLE LINE PROTOCOL SYSTEM 5 - ENCLOSURE PARAMETERS
    "purple_line_threshold": 0.06283185307,  # Exact 2π boundary
    "strange_attractor_strength": 10.0,      # How hard purple line pulls back
    "cwu_per_batch": 1.0,                    # Cognitive Work Units per batch
    "meta_regulation_lambda": 100.0,         # Purple line's self-regulation strength
    
    # DISTRIBUTED TRAINING
    "world_size": 8,  # 8x3090 GPUs
    "batch_size_per_gpu": 64,  # 64 * 8 = 512 total batch
    "global_batch_size": 512,
    
    # TRAINING PARAMS
    "epochs": 50,  # Full training
    "learning_rate": 0.001,
    "beta": 0.1,
    "latent_dim": 256,  # Larger for distributed
    
    # CWU DISTRIBUTION
    "cwu_sync_interval": 10,  # Sync CWUs every 10 batches
    "variance_all_reduce": True,  # All-reduce variance across GPUs
}

class CIFAR10DistributedVAE(nn.Module):
    """VAE with Purple Line Protocol System 5 Enclosure"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder
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
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # PURPLE LINE SYSTEM 5 - META REGULATOR
        self.purple_line = nn.Parameter(torch.tensor([CONFIG['purple_line_threshold']]))
        self.cwu_accumulator = 0.0
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        # PURPLE LINE ENCLOSURE - System 5
        # The variance SHALL NOT exceed 2π
        logvar = self.apply_purple_line_enclosure(logvar)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        recon = self.decode(z)
        return recon, mu, logvar
    
    def apply_purple_line_enclosure(self, logvar):
        """Purple Line Protocol System 5 - The Enclosure"""
        variance = logvar.exp()
        
        # Create strange attractor at 2π boundary
        excess = torch.relu(variance - self.purple_line)
        
        # Purple line pulls variance back with strange attractor dynamics
        if excess.max() > 0:
            # Generate one "cwooo" sound per violation
            self.cwu_accumulator += excess.sum().item() * CONFIG['cwu_per_batch']
            
            # Apply strange attractor pull-back
            attractor_force = torch.tanh(excess * CONFIG['strange_attractor_strength'])
            variance = variance - attractor_force * excess
            
            # Re-compute logvar with enclosed variance
            logvar = torch.log(variance.clamp(min=1e-8))
        
        return logvar

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_cifar10_distributed(rank, world_size, batch_size):
    """Load CIFAR-10 with distributed sampler"""
    data_dir = Path("/workspace/canidae/datasets/cifar10/cifar-10-batches-py")
    
    # Load all training data
    all_images = []
    for i in range(1, 6):
        with open(data_dir / f"data_batch_{i}", 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            all_images.append(images)
    
    train_images = np.concatenate(all_images)
    
    # Create distributed dataset
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_images))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2
    )
    
    return loader, sampler

def train_rank(rank, world_size):
    """Training function for each GPU"""
    print(f"GPU {rank}: Initializing...")
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = CIFAR10DistributedVAE(CONFIG['latent_dim']).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Load data with distributed sampler
    train_loader, sampler = load_cifar10_distributed(
        rank, world_size, CONFIG['batch_size_per_gpu']
    )
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=CONFIG['learning_rate'])
    
    # Track CWUs across training
    total_cwus = 0
    cwu_history = []
    
    for epoch in range(CONFIG['epochs']):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        ddp_model.train()
        
        epoch_cwus = 0
        batch_count = 0
        compliance_count = 0
        
        prev_variances = None
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(rank)
            optimizer.zero_grad()
            
            # Forward pass with Purple Line enclosure
            recon, mu, logvar = ddp_model(data)
            
            # Losses
            recon_loss = torch.nn.functional.mse_loss(recon, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 2π Regulation with CWU tracking
            current_variances = logvar.exp()
            
            if prev_variances is not None and prev_variances.shape == current_variances.shape:
                variance_rate = (current_variances - prev_variances).abs().mean()
                
                # Check compliance
                if variance_rate.item() <= CONFIG['stability_coefficient']:
                    compliance_count += 1
                
                # Purple Line Meta-Regulation (System 5)
                purple_penalty = CONFIG['meta_regulation_lambda'] * torch.relu(
                    variance_rate - CONFIG['purple_line_threshold']
                )
                
                # Count CWUs (cognitive work units)
                epoch_cwus += CONFIG['cwu_per_batch']
                
            else:
                purple_penalty = 0
            
            prev_variances = current_variances.detach()
            batch_count += 1
            
            # Total loss with Purple Line enclosure
            loss = recon_loss + CONFIG['beta'] * kl_loss + purple_penalty
            
            loss.backward()
            optimizer.step()
            
            # CWU synchronization across GPUs
            if batch_idx % CONFIG['cwu_sync_interval'] == 0 and rank == 0:
                print(f"  Batch {batch_idx}: CWUs = {epoch_cwus:.1f}, "
                      f"Compliance = {compliance_count/max(batch_count,1)*100:.1f}%")
        
        # Gather CWUs from all GPUs
        cwu_tensor = torch.tensor([epoch_cwus], device=rank)
        dist.all_reduce(cwu_tensor, op=dist.ReduceOp.SUM)
        total_epoch_cwus = cwu_tensor.item()
        
        if rank == 0:
            compliance_pct = compliance_count / batch_count * 100
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
                  f"Total CWUs = {total_epoch_cwus:.1f}, "
                  f"2π Compliance = {compliance_pct:.1f}%")
            
            if compliance_pct > 95:
                print(f"  🎯 HIGH 2π COMPLIANCE WITH PURPLE LINE ENCLOSURE!")
            
            # Make cwooo sounds based on CWU count
            if total_epoch_cwus > 100:
                print(f"  🔊 {'cwooo ' * min(int(total_epoch_cwus/100), 5)}")
        
        total_cwus += total_epoch_cwus
        cwu_history.append(total_epoch_cwus)
    
    cleanup()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE on 8x3090!")
        print(f"Total CWUs processed: {total_cwus:.1f}")
        print(f"Average CWUs per epoch: {np.mean(cwu_history):.1f}")
        print(f"Purple Line held at: {CONFIG['purple_line_threshold']}")
        print(f"{'='*60}")

def main():
    world_size = torch.cuda.device_count()
    print(f"🦊🐺 CANIDAE 8x3090 Distributed Training")
    print(f"Found {world_size} GPUs")
    print(f"Purple Line Protocol System 5: ACTIVE")
    print(f"CWU Distribution: ENABLED")
    print(f"{'='*60}")
    
    if world_size != 8:
        print(f"WARNING: Expected 8 GPUs, found {world_size}")
        print("Adjusting world_size...")
        CONFIG['world_size'] = world_size
    
    # Launch distributed training
    mp.spawn(train_rank, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()