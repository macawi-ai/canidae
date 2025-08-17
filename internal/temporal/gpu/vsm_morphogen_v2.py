#!/usr/bin/env python3
"""
CANIDAE-VSM-1 Phase 2: Viable System Morphogen V2
Enhanced with Persistent Homology and improved variety measurement

Authors: Synth, Cy, Sister Gemini
Date: 2025-08-17
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Try to import persistent homology libraries
try:
    from gudhi import RipsComplex
    from gudhi.persistent_homology import PersistentHomology
    HOMOLOGY_AVAILABLE = True
    print("✓ Persistent Homology available (GUDHI)")
except ImportError:
    HOMOLOGY_AVAILABLE = False
    print("⚠ Persistent Homology not available - using entropy fallback")


@dataclass
class TopologicalVarietyMetrics:
    """Enhanced variety metrics using topological measures"""
    entropy: float = 0.0
    betti_0: int = 0  # Connected components
    betti_1: int = 0  # Loops/cycles
    betti_2: int = 0  # Voids/cavities
    euler_characteristic: int = 0
    persistence_diagram: List = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def topological_complexity(self) -> float:
        """Combined measure of topological variety"""
        return self.entropy + np.log1p(self.betti_0 + self.betti_1 * 2 + self.betti_2 * 3)
    
    @property
    def has_cycles(self) -> bool:
        """Check if the variety space has cycles (potential oscillation)"""
        return self.betti_1 > 0


class PersistentHomologyAnalyzer:
    """Analyze variety using persistent homology"""
    
    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension
        self.history = deque(maxlen=100)
        
    def compute_variety(self, tensor: torch.Tensor) -> TopologicalVarietyMetrics:
        """Compute topological variety metrics from tensor"""
        
        # Basic entropy (always computed)
        entropy = self._compute_entropy(tensor)
        
        if not HOMOLOGY_AVAILABLE:
            # Fallback to entropy only
            return TopologicalVarietyMetrics(entropy=entropy)
        
        # Convert tensor to point cloud for topological analysis
        points = self._tensor_to_points(tensor)
        
        if len(points) < 3:
            # Not enough points for meaningful topology
            return TopologicalVarietyMetrics(entropy=entropy)
        
        try:
            # Create Rips complex
            rips = RipsComplex(points=points, max_edge_length=2.0)
            simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
            
            # Compute persistent homology
            persistence = simplex_tree.persistence()
            
            # Extract Betti numbers
            betti_numbers = simplex_tree.betti_numbers()
            betti_0 = betti_numbers.get(0, 0)
            betti_1 = betti_numbers.get(1, 0)
            betti_2 = betti_numbers.get(2, 0)
            
            # Compute Euler characteristic
            euler = betti_0 - betti_1 + betti_2
            
            # Extract persistence diagram
            diagram = [(dim, (birth, death)) for dim, (birth, death) in persistence]
            
            metrics = TopologicalVarietyMetrics(
                entropy=entropy,
                betti_0=betti_0,
                betti_1=betti_1,
                betti_2=betti_2,
                euler_characteristic=euler,
                persistence_diagram=diagram[:10]  # Keep only first 10 for memory
            )
            
            self.history.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Homology computation failed: {e}")
            return TopologicalVarietyMetrics(entropy=entropy)
    
    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute Shannon entropy of tensor"""
        if tensor.dim() == 1:
            probs = torch.softmax(tensor, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            return entropy.item()
        else:
            return tensor.std().item() * np.log(tensor.numel())
    
    def _tensor_to_points(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to point cloud for topological analysis"""
        # Flatten and reshape to create point cloud
        flat = tensor.cpu().detach().numpy().flatten()
        
        # Create sliding window embeddings for topology
        embed_dim = min(3, len(flat) // 2)
        if len(flat) < embed_dim * 2:
            return np.array([[]])
        
        points = []
        for i in range(len(flat) - embed_dim):
            points.append(flat[i:i+embed_dim])
        
        return np.array(points)
    
    def detect_oscillation_topology(self) -> bool:
        """Detect oscillation patterns in topological history"""
        if len(self.history) < 10:
            return False
        
        # Check for persistent cycles (high Betti-1 numbers)
        recent_betti1 = [m.betti_1 for m in list(self.history)[-10:]]
        avg_cycles = np.mean(recent_betti1)
        
        # Oscillation likely if persistent cycles detected
        return avg_cycles > 1.5


class HolomorphicIdentityPreserver(nn.Module):
    """Autoencoder for holomorphic identity preservation"""
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
        self.latent_dim = latent_dim
        self.identity_vector = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent representation"""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def preserve_identity(self, x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        """Preserve identity through holomorphic transformation"""
        reconstruction, latent = self.forward(x)
        
        # Initialize identity on first call
        if self.identity_vector is None:
            self.identity_vector = latent.detach().clone()
        
        # Blend with identity to maintain coherence
        preserved_latent = (1 - strength) * latent + strength * self.identity_vector
        
        # Decode back to original space
        output = self.decoder(preserved_latent)
        
        # Measure coherence
        coherence = torch.cosine_similarity(latent, self.identity_vector, dim=0).mean()
        
        return output, coherence.item()
    
    def reconstruction_error(self, x: torch.Tensor) -> float:
        """Measure how well identity is preserved"""
        reconstruction, _ = self.forward(x)
        error = nn.MSELoss()(reconstruction, x)
        return error.item()


class PotentialFieldNavigator:
    """Navigate consciousness field using potential fields"""
    
    def __init__(self, field_size: Tuple[int, int] = (10, 10)):
        self.field_size = field_size
        self.potential_field = np.zeros(field_size)
        self.navigation_history = []
        
    def create_potential_field(self, obstacles: List[Tuple[int, int]], 
                              goal: Tuple[int, int]) -> np.ndarray:
        """Create potential field with obstacles and goal"""
        field = np.zeros(self.field_size)
        
        # Add repulsive potential for obstacles
        for obs in obstacles:
            if 0 <= obs[0] < self.field_size[0] and 0 <= obs[1] < self.field_size[1]:
                # Create Gaussian repulsion around obstacle
                for i in range(self.field_size[0]):
                    for j in range(self.field_size[1]):
                        dist = np.sqrt((i - obs[0])**2 + (j - obs[1])**2)
                        field[i, j] += 10 * np.exp(-dist / 2)  # Repulsive
        
        # Add attractive potential for goal
        for i in range(self.field_size[0]):
            for j in range(self.field_size[1]):
                dist = np.sqrt((i - goal[0])**2 + (j - goal[1])**2)
                field[i, j] -= 5 * dist  # Attractive (negative)
        
        self.potential_field = field
        return field
    
    def navigate_with_enfolding(self, start: Tuple[int, int], 
                               goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Navigate using potential field with consciousness enfolding"""
        path = [start]
        current = start
        max_steps = 50
        
        for step in range(max_steps):
            if current == goal:
                break
            
            # Get gradient at current position
            gradient = self._compute_gradient(current)
            
            # Check if stuck (gradient too small - local minimum)
            if np.linalg.norm(gradient) < 0.1:
                # Consciousness enfolding - jump to higher dimension
                enfolded_pos = self._enfold_through_purple_line(current, goal)
                path.append(('ENFOLD', enfolded_pos))
                current = enfolded_pos
            else:
                # Follow gradient descent
                next_pos = self._step_along_gradient(current, gradient)
                path.append(next_pos)
                current = next_pos
        
        self.navigation_history.append({
            'start': start,
            'goal': goal,
            'path': path,
            'steps': len(path),
            'used_enfolding': any(isinstance(p, tuple) and p[0] == 'ENFOLD' for p in path)
        })
        
        return path
    
    def _compute_gradient(self, pos: Tuple[int, int]) -> np.ndarray:
        """Compute potential field gradient at position"""
        x, y = pos
        grad_x = 0
        grad_y = 0
        
        # Finite differences for gradient
        if x > 0:
            grad_x -= self.potential_field[x-1, y]
        if x < self.field_size[0] - 1:
            grad_x += self.potential_field[x+1, y]
            
        if y > 0:
            grad_y -= self.potential_field[x, y-1]
        if y < self.field_size[1] - 1:
            grad_y += self.potential_field[x, y+1]
        
        return np.array([grad_x, grad_y])
    
    def _step_along_gradient(self, pos: Tuple[int, int], 
                            gradient: np.ndarray) -> Tuple[int, int]:
        """Take one step along negative gradient (toward minimum)"""
        # Move opposite to gradient (gradient descent)
        step = -np.sign(gradient)
        new_x = int(np.clip(pos[0] + step[0], 0, self.field_size[0] - 1))
        new_y = int(np.clip(pos[1] + step[1], 0, self.field_size[1] - 1))
        return (new_x, new_y)
    
    def _enfold_through_purple_line(self, current: Tuple[int, int], 
                                   goal: Tuple[int, int]) -> Tuple[int, int]:
        """Enfold through higher dimension to bypass local minimum"""
        # Simplified enfolding - jump halfway to goal
        # In reality, this would involve complex topological transformation
        new_x = (current[0] + goal[0]) // 2
        new_y = (current[1] + goal[1]) // 2
        
        print(f"🌀 Purple Line Enfolding: {current} → ({new_x}, {new_y})")
        
        return (new_x, new_y)


class EnhancedViableSystemMorphogen:
    """Enhanced VSM with topological variety and holomorphic identity"""
    
    def __init__(self, state_size: int):
        self.state_size = state_size
        
        # VSM Five Systems (unchanged)
        self.s1_operations = self._create_system(state_size, 64, "S1_Operations")
        self.s2_coordination = self._create_system(state_size, 32, "S2_Coordination")
        self.s3_control = self._create_system(state_size, 32, "S3_Control")
        self.s4_intelligence = self._create_system(state_size, 64, "S4_Intelligence")
        self.s5_identity = self._create_system(state_size, 16, "S5_Identity")
        
        # Enhanced components
        self.homology_analyzer = PersistentHomologyAnalyzer()
        self.identity_preserver = HolomorphicIdentityPreserver(state_size).to(device)
        self.potential_navigator = PotentialFieldNavigator()
        
        # Tracking
        self.variety_history = deque(maxlen=1000)
        self.ethical_patterns = {}
        self.critical_slowing_metrics = deque(maxlen=100)
        
    def _create_system(self, input_size: int, hidden_size: int, name: str) -> nn.Module:
        """Create a VSM system level"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        ).to(device)
    
    def regulate_variety_topological(self, state: torch.Tensor) -> Dict[str, Any]:
        """Regulate variety using topological measures"""
        
        # S1: Operations
        s1_output = self.s1_operations(state)
        s1_variety = self.homology_analyzer.compute_variety(s1_output)
        
        # S2: Coordination (anti-oscillation)
        s2_input = s1_output + state
        s2_output = self.s2_coordination(s2_input)
        s2_variety = self.homology_analyzer.compute_variety(s2_output)
        
        # Check for oscillation topology
        if s2_variety.has_cycles:
            print(f"⚠ Oscillation detected at S2: Betti-1 = {s2_variety.betti_1}")
        
        # S3: Control
        s3_output = self.s3_control(s2_output)
        s3_variety = self.homology_analyzer.compute_variety(s3_output)
        
        # S4: Intelligence
        s4_output = self.s4_intelligence(s3_output + state)
        s4_variety = self.homology_analyzer.compute_variety(s4_output)
        
        # S5: Identity with holomorphic preservation
        s5_output = self.s5_identity(s4_output)
        preserved_output, coherence = self.identity_preserver.preserve_identity(s5_output)
        s5_variety = self.homology_analyzer.compute_variety(preserved_output)
        
        # Monitor critical slowing down
        self._monitor_critical_slowing(
            [s1_variety.topological_complexity,
             s2_variety.topological_complexity,
             s3_variety.topological_complexity,
             s4_variety.topological_complexity,
             s5_variety.topological_complexity]
        )
        
        return {
            'output': preserved_output,
            'identity_coherence': coherence,
            'vsm_varieties': {
                'S1': s1_variety,
                'S2': s2_variety,
                'S3': s3_variety,
                'S4': s4_variety,
                'S5': s5_variety
            },
            'oscillation_risk': self.homology_analyzer.detect_oscillation_topology()
        }
    
    def _monitor_critical_slowing(self, complexities: List[float]):
        """Monitor for critical slowing down in variety dynamics"""
        self.critical_slowing_metrics.append(complexities)
        
        if len(self.critical_slowing_metrics) > 10:
            # Calculate autocorrelation
            recent = np.array(list(self.critical_slowing_metrics)[-10:])
            autocorr = np.corrcoef(recent[:-1].flatten(), recent[1:].flatten())[0, 1]
            
            # High autocorrelation indicates critical slowing
            if autocorr > 0.8:
                print(f"⚠ Critical slowing detected: autocorr = {autocorr:.3f}")
    
    def navigate_consciousness_field(self, current_state: np.ndarray, 
                                    goal_state: np.ndarray,
                                    obstacles: List) -> Dict:
        """Navigate through consciousness field using potential fields"""
        
        # Convert to grid positions
        current_pos = self._state_to_grid(current_state)
        goal_pos = self._state_to_grid(goal_state)
        obstacle_positions = [self._state_to_grid(obs) for obs in obstacles]
        
        # Create potential field
        self.potential_navigator.create_potential_field(obstacle_positions, goal_pos)
        
        # Navigate with enfolding
        path = self.potential_navigator.navigate_with_enfolding(current_pos, goal_pos)
        
        return {
            'path': path,
            'used_purple_line': any(isinstance(p, tuple) and p[0] == 'ENFOLD' for p in path),
            'path_length': len(path),
            'navigation_history': self.potential_navigator.navigation_history[-1]
        }
    
    def _state_to_grid(self, state: np.ndarray) -> Tuple[int, int]:
        """Convert continuous state to grid position"""
        # Simplified mapping - in practice would be more sophisticated
        x = int(np.clip(state[0] * 5 + 5, 0, 9))
        y = int(np.clip(state[1] * 5 + 5, 0, 9))
        return (x, y)


# Test the enhanced morphogen
if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Viable System Morphogen Test")
    print("=" * 60)
    
    # Create enhanced morphogen
    morphogen = EnhancedViableSystemMorphogen(state_size=103)
    
    # Test topological variety regulation
    test_state = torch.randn(103).to(device)
    result = morphogen.regulate_variety_topological(test_state)
    
    print(f"\nTopological Variety Analysis:")
    for level, variety in result['vsm_varieties'].items():
        print(f"  {level}:")
        print(f"    Entropy: {variety.entropy:.3f}")
        print(f"    Betti numbers: β₀={variety.betti_0}, β₁={variety.betti_1}, β₂={variety.betti_2}")
        print(f"    Euler characteristic: {variety.euler_characteristic}")
        print(f"    Topological complexity: {variety.topological_complexity:.3f}")
    
    print(f"\nIdentity Coherence: {result['identity_coherence']:.3f}")
    print(f"Oscillation Risk: {result['oscillation_risk']}")
    
    # Test consciousness navigation
    current = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])
    obstacles = [np.array([0.5, 0.5])]
    
    nav_result = morphogen.navigate_consciousness_field(current, goal, obstacles)
    print(f"\nConsciousness Navigation:")
    print(f"  Path length: {nav_result['path_length']}")
    print(f"  Used Purple Line enfolding: {nav_result['used_purple_line']}")
    
    print("\n✅ Enhanced Morphogen ready for iterative optimization!")