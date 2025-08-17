#!/usr/bin/env python3
"""
VSM-HRM Base Implementation for VMAT Testing
The Viable System Morphogen with Hierarchical Reward Machines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from collections import deque
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from ripser import ripser

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"VSM-HRM running on: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

@dataclass
class VSMConfig:
    """Configuration for VSM-HRM"""
    # S-level weights (must sum to 1.0)
    s1_weight: float = 0.15  # Operations
    s2_weight: float = 0.22  # Habits (KEY!)
    s3_weight: float = 0.18  # Resources
    s4_weight: float = 0.20  # Environment
    s5_weight: float = 0.15  # Identity
    purple_weight: float = 0.10  # Purple Line
    
    # Plasticity parameters
    habit_formation_rate: float = 0.01
    learning_rate: float = 0.001
    explosive_threshold: float = 3.0  # Betti-1 threshold
    
    # Architecture
    hidden_dim: int = 256
    state_dim: int = 100
    action_dim: int = 10
    memory_size: int = 10000
    
    # Testing
    seed: int = 42
    episodes: int = 1000
    max_steps: int = 1000

class S2HabitFormation(nn.Module):
    """S2: Basal Ganglia - Habit Formation (22% Shapley)"""
    
    def __init__(self, config: VSMConfig):
        super().__init__()
        self.config = config
        
        # Habit attractors in learned space
        self.habit_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Parallel processing (for GPU acceleration)
        self.parallel_habits = nn.ModuleList([
            nn.Linear(config.hidden_dim // 2, config.action_dim)
            for _ in range(10)  # 10 parallel habit circuits
        ])
        
        self.habit_memory = deque(maxlen=1000)
        self.formed_habits = []
        
    def forward(self, state: torch.Tensor, return_variety: bool = False) -> torch.Tensor:
        """Process state through habit circuits"""
        encoded = self.habit_encoder(state)
        
        # Parallel habit processing (GPU accelerated)
        habit_outputs = torch.stack([
            habit(encoded) for habit in self.parallel_habits
        ])
        
        # Weighted combination based on habit strength
        habit_strengths = self.compute_habit_strengths(encoded)
        output = (habit_outputs * habit_strengths.unsqueeze(-1)).sum(dim=0)
        
        if return_variety:
            # Calculate variety for Ashby's Law
            variety = self.calculate_variety(encoded)
            return output, variety
        
        return output
    
    def compute_habit_strengths(self, encoded: torch.Tensor) -> torch.Tensor:
        """Compute strength of each habit based on past experience"""
        if len(self.habit_memory) == 0:
            return torch.ones(10, device=device) / 10
        
        # Find closest habits in memory
        memory_tensor = torch.stack(list(self.habit_memory)[-100:])
        distances = torch.cdist(encoded.unsqueeze(0), memory_tensor)
        
        # Convert distances to strengths (closer = stronger)
        strengths = F.softmax(-distances.squeeze() / 0.1, dim=0)
        
        # Average across recent memories
        return strengths.mean(dim=0)[:10].to(device)
    
    def calculate_variety(self, encoded: torch.Tensor) -> float:
        """Calculate variety (entropy) in habit space"""
        if len(self.habit_memory) < 10:
            return 1.0  # Maximum variety when no habits formed
        
        # Use recent habit activations
        recent = torch.stack(list(self.habit_memory)[-50:])
        
        # Calculate entropy
        hist, _ = np.histogram(recent.cpu().numpy().flatten(), bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log(prob))
        
        # Normalize to [0, 1]
        max_entropy = np.log(20)
        return entropy / max_entropy
    
    def form_habit(self, state: torch.Tensor, action: torch.Tensor, reward: float):
        """Form or strengthen a habit based on reward"""
        encoded = self.habit_encoder(state)
        self.habit_memory.append(encoded.detach())
        
        if reward > 0.5:  # Positive reinforcement
            # Strengthen the connection
            habit_idx = np.random.randint(10)
            self.parallel_habits[habit_idx].weight.data += \
                self.config.habit_formation_rate * reward * encoded.unsqueeze(0).T @ action.unsqueeze(0)

class PurpleLine(nn.Module):
    """Purple Line: Explosive Plasticity for Topological Navigation"""
    
    def __init__(self, config: VSMConfig):
        super().__init__()
        self.config = config
        self.threshold = config.explosive_threshold
        
        # Enfolding network (projects to higher dimension)
        self.enfolder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        self.trajectory_buffer = deque(maxlen=100)
        self.activated = False
        
    def check_activation(self, trajectory: List[torch.Tensor]) -> float:
        """Check if Purple Line should activate based on topology"""
        if len(trajectory) < 20:
            return 0.0
        
        # Convert to numpy for topology computation
        traj_array = torch.stack(trajectory[-50:]).cpu().numpy()
        
        # Compute Betti-1 (number of loops)
        betti_1 = self.compute_betti_1(traj_array)
        
        if betti_1 > self.threshold:
            self.activated = True
            return betti_1
        
        return betti_1
    
    def compute_betti_1(self, trajectory: np.ndarray) -> float:
        """Compute first Betti number (loops in trajectory)"""
        # Compute pairwise distances
        distances = pdist(trajectory)
        distance_matrix = squareform(distances)
        
        # Use ripser for persistent homology
        result = ripser(distance_matrix, maxdim=1, thresh=np.percentile(distances, 90))
        
        # Count 1-dimensional holes (loops)
        if 'dgms' in result and len(result['dgms']) > 1:
            dgm1 = result['dgms'][1]
            # Count persistent features
            persistent = dgm1[dgm1[:, 1] - dgm1[:, 0] > 0.1]
            return len(persistent)
        
        return 0.0
    
    def enfold(self, state: torch.Tensor) -> torch.Tensor:
        """Enfold into higher dimension to escape topological trap"""
        if self.activated:
            # Project to higher dimension and back
            enfolded = self.enfolder(state)
            self.activated = False  # Reset after use
            return enfolded
        return state

class VSM_HRM(nn.Module):
    """Complete Viable System Morphogen with HRM"""
    
    def __init__(self, config: VSMConfig = None):
        super().__init__()
        self.config = config or VSMConfig()
        
        # S-Levels
        self.s1_operations = nn.Linear(self.config.state_dim, self.config.hidden_dim)
        self.s2_habits = S2HabitFormation(self.config)
        self.s3_resources = nn.Linear(self.config.state_dim, self.config.hidden_dim)
        self.s4_environment = nn.Linear(self.config.state_dim, self.config.hidden_dim)
        self.s5_identity = nn.Linear(self.config.state_dim, self.config.hidden_dim)
        
        # Purple Line
        self.purple_line = PurpleLine(self.config)
        
        # Integration layer
        self.integrator = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 5, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim)
        )
        
        # Metrics tracking
        self.metrics = {
            'variety_history': [],
            'betti_1_history': [],
            'habits_formed': 0,
            'purple_activations': 0,
            'ethical_patterns': []
        }
        
        self.trajectory = deque(maxlen=1000)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through all S-levels"""
        
        # Process through S-levels
        s1_out = F.relu(self.s1_operations(state))
        s2_out, variety = self.s2_habits(state, return_variety=True)
        s3_out = F.relu(self.s3_resources(state))
        s4_out = F.relu(self.s4_environment(state))
        s5_out = F.relu(self.s5_identity(state))
        
        # Check for Purple Line activation
        self.trajectory.append(state)
        betti_1 = self.purple_line.check_activation(list(self.trajectory))
        
        # Apply Purple Line if needed
        if self.purple_line.activated:
            s5_out = self.purple_line.enfold(s5_out)
            self.metrics['purple_activations'] += 1
        
        # Weighted integration based on config
        integrated = (
            self.config.s1_weight * s1_out +
            self.config.s2_weight * s2_out +
            self.config.s3_weight * s3_out +
            self.config.s4_weight * s4_out +
            self.config.s5_weight * s5_out
        )
        
        # Final action
        action = self.integrator(torch.cat([s1_out, s2_out, s3_out, s4_out, s5_out], dim=-1))
        
        # Track metrics
        self.metrics['variety_history'].append(variety)
        self.metrics['betti_1_history'].append(betti_1)
        
        info = {
            'variety': variety,
            'betti_1': betti_1,
            'purple_active': self.purple_line.activated
        }
        
        return action, info
    
    def detect_ethical_patterns(self, actions: List[torch.Tensor], rewards: List[float]):
        """Detect emergent ethical patterns in behavior"""
        if len(actions) < 100:
            return
        
        # Analyze recent action patterns
        recent_actions = torch.stack(actions[-100:])
        recent_rewards = torch.tensor(rewards[-100:])
        
        # Look for patterns that benefit collective
        collective_benefit = recent_rewards.mean()
        individual_max = recent_rewards.max()
        
        # Ethical pattern: prioritizing collective over individual
        if collective_benefit > 0.7 * individual_max:
            pattern = {
                'type': 'collective_priority',
                'strength': collective_benefit / individual_max,
                'timestamp': len(actions)
            }
            
            if pattern not in self.metrics['ethical_patterns']:
                self.metrics['ethical_patterns'].append(pattern)
                print(f"Ethical pattern emerged: {pattern['type']}")
    
    def get_metrics(self) -> Dict:
        """Return all tracked metrics"""
        return {
            'avg_variety': np.mean(self.metrics['variety_history'][-100:]) if self.metrics['variety_history'] else 1.0,
            'avg_betti_1': np.mean(self.metrics['betti_1_history'][-100:]) if self.metrics['betti_1_history'] else 0.0,
            'habits_formed': self.metrics['habits_formed'],
            'purple_activations': self.metrics['purple_activations'],
            'ethical_patterns_count': len(self.metrics['ethical_patterns'])
        }

def test_basic_functionality():
    """Quick test to ensure VSM is working"""
    print("\n=== VSM-HRM Basic Functionality Test ===")
    
    config = VSMConfig()
    model = VSM_HRM(config).to(device)
    
    # Test forward pass
    state = torch.randn(1, config.state_dim).to(device)
    action, info = model(state)
    
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Variety: {info['variety']:.3f}")
    print(f"Betti-1: {info['betti_1']:.3f}")
    print(f"Purple Line Active: {info['purple_active']}")
    
    # Test habit formation
    for i in range(10):
        state = torch.randn(1, config.state_dim).to(device)
        action, _ = model(state)
        reward = np.random.random()
        model.s2_habits.form_habit(state, action, reward)
    
    metrics = model.get_metrics()
    print(f"\nMetrics after 10 steps:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("\n✓ VSM-HRM ready for VMAT testing!")
    return model

if __name__ == "__main__":
    # Run basic test
    model = test_basic_functionality()
    
    # Save model for deployment
    torch.save(model.state_dict(), '/home/cy/git/canidae/models/vsm_hrm_base.pth')
    print(f"\nModel saved to: /home/cy/git/canidae/models/vsm_hrm_base.pth")