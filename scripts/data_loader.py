#!/usr/bin/env python3
"""
Data Loader Module for 2π Validation Pipeline
Handles loading and preprocessing of various datasets

Supports:
- MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- Image datasets (PNG, JPG)
- NPZ files
- Train/validation/test splits
- Data augmentation and preprocessing
- Batch creation with proper normalization
"""

import numpy as np
import torch
import torch.utils.data
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
import gzip
import pickle

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Flexible dataset loader supporting multiple formats and datasets"""
    
    SUPPORTED_DATASETS = {
        'MNIST': {'channels': 1, 'size': (28, 28), 'classes': 10},
        'Fashion-MNIST': {'channels': 1, 'size': (28, 28), 'classes': 10},
        'CIFAR-10': {'channels': 3, 'size': (32, 32), 'classes': 10},
        'CIFAR-100': {'channels': 3, 'size': (32, 32), 'classes': 100}
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader with dataset configuration"""
        self.config = config
        self.dataset_config = config['dataset']
        self.dataset_name = self.dataset_config['name']
        
        # Set random seed for reproducibility
        if 'reproducibility' in config and 'random_seed' in config['reproducibility']:
            torch.manual_seed(config['reproducibility']['random_seed'])
            np.random.seed(config['reproducibility']['random_seed'])
        
        logger.info(f"DataLoader initialized for {self.dataset_name}")
    
    def load_dataset(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load dataset based on configuration and return data loaders"""
        
        if self.dataset_name in self.SUPPORTED_DATASETS:
            return self._load_standard_dataset()
        else:
            return self._load_custom_dataset()
    
    def _load_standard_dataset(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load standard datasets (MNIST, Fashion-MNIST, etc.)"""
        
        if self.dataset_name == 'Fashion-MNIST':
            return self._load_fashion_mnist()
        elif self.dataset_name == 'MNIST':
            return self._load_mnist()
        elif self.dataset_name == 'CIFAR-10':
            return self._load_cifar10()
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not yet implemented")
    
    def _load_fashion_mnist(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load Fashion-MNIST from NPZ file"""
        
        data_path = Path(self.dataset_config['train_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Fashion-MNIST data not found at {data_path}")
        
        # Load NPZ file
        data = np.load(data_path)
        x_train = data['x_train'].astype(np.float32) / 255.0
        y_train = data['y_train'] if 'y_train' in data else None
        x_test = data['x_test'].astype(np.float32) / 255.0  
        y_test = data['y_test'] if 'y_test' in data else None
        
        logger.info(f"Loaded Fashion-MNIST: Train {x_train.shape}, Test {x_test.shape}")
        
        # Create validation split if needed
        val_split = self.dataset_config.get('val_split', 0.1)
        if val_split > 0:
            n_val = int(len(x_train) * val_split)
            indices = np.random.permutation(len(x_train))
            
            x_val = x_train[indices[:n_val]]
            x_train = x_train[indices[n_val:]]
            
            if y_train is not None:
                y_val = y_train[indices[:n_val]]
                y_train = y_train[indices[n_val:]]
            else:
                y_val = None
        else:
            x_val, y_val = None, None
        
        # Create datasets
        train_dataset = self._create_tensor_dataset(x_train, y_train)
        val_dataset = self._create_tensor_dataset(x_val, y_val) if x_val is not None else None
        test_dataset = self._create_tensor_dataset(x_test, y_test)
        
        # Create data loaders
        batch_size = self.config['hyperparameters']['batch_size']
        num_workers = self.config.get('resources', {}).get('num_workers', 2)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created data loaders - Train: {len(train_loader)} batches, "
                   f"Val: {len(val_loader) if val_loader else 0} batches, "
                   f"Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def _load_mnist(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load MNIST dataset"""
        # Similar implementation to Fashion-MNIST
        raise NotImplementedError("MNIST loader to be implemented")
    
    def _load_cifar10(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load CIFAR-10 dataset"""
        # Implementation for CIFAR-10
        raise NotImplementedError("CIFAR-10 loader to be implemented")
    
    def _create_tensor_dataset(self, x_data: np.ndarray, y_data: Optional[np.ndarray] = None) -> torch.utils.data.Dataset:
        """Create PyTorch TensorDataset from numpy arrays"""
        
        if x_data is None:
            return None
        
        x_tensor = torch.FloatTensor(x_data)
        
        if y_data is not None:
            y_tensor = torch.LongTensor(y_data)
            return torch.utils.data.TensorDataset(x_tensor, y_tensor)
        else:
            # For unsupervised tasks (VAEs, etc.)
            return torch.utils.data.TensorDataset(x_tensor)
    
    def _load_custom_dataset(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Load custom datasets from file paths"""
        raise NotImplementedError("Custom dataset loading to be implemented")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset"""
        
        info = {
            'name': self.dataset_name,
            'type': self.dataset_config.get('type', 'unknown'),
            'num_classes': self.dataset_config.get('num_classes', 0),
            'input_shape': self.dataset_config.get('input_shape', None)
        }
        
        if self.dataset_name in self.SUPPORTED_DATASETS:
            standard_info = self.SUPPORTED_DATASETS[self.dataset_name]
            info.update(standard_info)
        
        return info

def load_data(config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Main entry point for data loading"""
    
    loader = DatasetLoader(config)
    return loader.load_dataset()

def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get dataset information without loading data"""
    
    loader = DatasetLoader(config)
    return loader.get_dataset_info()

# Utility functions for common preprocessing tasks

def apply_data_augmentation(config: Dict[str, Any]) -> torch.nn.Module:
    """Create data augmentation pipeline based on config"""
    
    import torchvision.transforms as transforms
    
    augmentations = []
    
    aug_config = config.get('data_augmentation', {})
    
    if aug_config.get('random_flip', False):
        augmentations.append(transforms.RandomHorizontalFlip(0.5))
    
    if aug_config.get('random_rotation', 0) > 0:
        augmentations.append(transforms.RandomRotation(aug_config['random_rotation']))
    
    if aug_config.get('random_crop', False):
        size = aug_config.get('crop_size', 32)
        padding = aug_config.get('crop_padding', 4)
        augmentations.append(transforms.RandomCrop(size, padding=padding))
    
    if aug_config.get('color_jitter', False):
        augmentations.append(transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ))
    
    # Always normalize
    if 'normalization' in config.get('preprocessing', {}):
        norm_config = config['preprocessing']['normalization']
        mean = norm_config.get('mean', [0.5])
        std = norm_config.get('std', [0.5])
        augmentations.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(augmentations)

def calculate_dataset_statistics(data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Calculate mean and std of dataset for normalization"""
    
    mean = torch.zeros(3)  # Assume RGB for now
    std = torch.zeros(3)
    total_samples = 0
    
    for data_batch in data_loader:
        if isinstance(data_batch, (list, tuple)):
            images = data_batch[0]
        else:
            images = data_batch
            
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples
    }

if __name__ == "__main__":
    # Simple test of data loader
    test_config = {
        'dataset': {
            'name': 'Fashion-MNIST',
            'train_path': '/home/cy/git/canidae/datasets/fashion_mnist/fashion_mnist.npz',
            'val_split': 0.1
        },
        'hyperparameters': {
            'batch_size': 32
        },
        'resources': {
            'num_workers': 2
        },
        'reproducibility': {
            'random_seed': 42
        }
    }
    
    try:
        train_loader, val_loader, test_loader = load_data(test_config)
        print(f"✅ Data loader test successful!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader) if val_loader else 0}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for batch_data in train_loader:
            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
                print(f"Batch shape: {images.shape}")
            else:
                print(f"Batch shape: {batch_data.shape}")
            break
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")