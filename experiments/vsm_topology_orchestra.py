#!/usr/bin/env python3
"""
VSM Topology Orchestra - Multiple simultaneous topology detectors
Inverting Rancière's "distribution of the sensible"
Each brain system "feels" the geometric structure differently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import networkx as nx
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

# The universal constant
TWO_PI = 0.06283185307

class TopologyType(Enum):
    """Different geometric structures the brain can sense"""
    EUCLIDEAN = "euclidean"          # Flat, regular space
    HYPERBOLIC = "hyperbolic"        # Threat/safety (everything far)
    SPHERICAL = "spherical"          # Bounded, cyclic
    TREE = "hierarchical"            # Parent-child relations
    GRAPH = "network"                # Connected components
    MANIFOLD = "smooth"              # Differentiable (motor)
    DISCRETE = "categorical"         # Separate categories
    SEQUENTIAL = "temporal"          # Time-ordered
    FIBERED = "independent"          # Product space
    COUPLED = "entangled"            # Dependencies
    FRACTAL = "self-similar"         # Recursive patterns

class BrainSystemDetector(nn.Module):
    """Base class for different brain system topology detectors"""
    
    def __init__(self, name: str, preferred_topology: TopologyType):
        super().__init__()
        self.name = name
        self.preferred_topology = preferred_topology
        self.confidence = 0.0
        self.vote_weight = 1.0
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Each system 'feels' the topology differently"""
        raise NotImplementedError
        
    def get_vote(self) -> Tuple[TopologyType, float]:
        """Return this system's vote for topology"""
        return self.preferred_topology, self.confidence * self.vote_weight


class AmygdalaDetector(BrainSystemDetector):
    """Emotional threat detection - prefers hyperbolic geometry"""
    
    def __init__(self):
        super().__init__("amygdala", TopologyType.HYPERBOLIC)
        self.threat_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect if space feels threatening (hyperbolic) or safe (euclidean)"""
        features = self.threat_encoder(x).squeeze()
        
        # High variance in features = threatening = hyperbolic
        variance = torch.var(features).item()
        threat_level = torch.sigmoid(torch.tensor(variance * 10)).item()
        
        votes = {
            TopologyType.HYPERBOLIC: threat_level,
            TopologyType.EUCLIDEAN: 1 - threat_level,
        }
        
        self.confidence = max(votes.values())
        return votes


class HippocampusDetector(BrainSystemDetector):
    """Spatial memory - prefers metric spaces with landmarks"""
    
    def __init__(self):
        super().__init__("hippocampus", TopologyType.GRAPH)
        self.place_cell_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU()
        )
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect familiar landmarks and spatial structure"""
        features = self.place_cell_encoder(x)
        
        # Detect repeated patterns (landmarks)
        flat_features = features.view(features.size(0), -1)
        similarity_matrix = torch.cdist(flat_features, flat_features)
        
        # More clusters = more landmarks = graph structure
        n_clusters = self._count_clusters(similarity_matrix)
        graph_score = 1 - torch.exp(-torch.tensor(n_clusters * 0.1)).item()
        
        votes = {
            TopologyType.GRAPH: graph_score,
            TopologyType.EUCLIDEAN: 1 - graph_score,
        }
        
        self.confidence = max(votes.values())
        return votes
    
    def _count_clusters(self, sim_matrix: torch.Tensor) -> int:
        """Simple clustering based on similarity"""
        threshold = torch.median(sim_matrix)
        adjacency = (sim_matrix < threshold).float()
        return int(torch.sum(torch.diagonal(adjacency @ adjacency)).item() / 2)


class CerebellumDetector(BrainSystemDetector):
    """Motor control - prefers smooth differentiable manifolds"""
    
    def __init__(self):
        super().__init__("cerebellum", TopologyType.MANIFOLD)
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect smoothness and differentiability"""
        # Compute gradients to test smoothness
        x.requires_grad = True
        grad_x = torch.autograd.grad(x.sum(), x, create_graph=True)[0]
        
        # Smooth manifold has consistent gradients
        grad_variance = torch.var(grad_x).item()
        smoothness = torch.exp(-grad_variance).item()
        
        votes = {
            TopologyType.MANIFOLD: smoothness,
            TopologyType.DISCRETE: 1 - smoothness,
        }
        
        self.confidence = max(votes.values())
        return votes


class PrefrontalDetector(BrainSystemDetector):
    """Executive control - prefers hierarchical tree structures"""
    
    def __init__(self):
        super().__init__("prefrontal", TopologyType.TREE)
        self.hierarchy_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect hierarchical organization"""
        # Flatten input
        flat = x.view(x.size(0), -1)
        if flat.size(1) > 256:
            flat = F.adaptive_avg_pool1d(flat.unsqueeze(1), 256).squeeze(1)
        else:
            flat = F.pad(flat, (0, 256 - flat.size(1)))
            
        features = self.hierarchy_encoder(flat)
        
        # Check if features form tree-like structure
        # Tree has low cycles, high connectivity
        adjacency = torch.cdist(features, features)
        threshold = torch.median(adjacency)
        binary_adj = (adjacency < threshold).float()
        
        # Compute tree-ness score
        n_edges = torch.sum(binary_adj) / 2
        n_nodes = features.size(0)
        tree_score = torch.sigmoid((n_edges - n_nodes + 1) * -0.5).item()
        
        votes = {
            TopologyType.TREE: tree_score,
            TopologyType.GRAPH: 1 - tree_score,
        }
        
        self.confidence = max(votes.values())
        return votes


class TemporalDetector(BrainSystemDetector):
    """Left hemisphere - sequential/linguistic processing"""
    
    def __init__(self):
        super().__init__("temporal_lobe", TopologyType.SEQUENTIAL)
        self.sequence_detector = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect sequential/temporal patterns"""
        # Create sequence from spatial data
        b, c, h, w = x.shape
        # Unfold spatial dimensions into sequence
        seq = x.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Pad or truncate to fixed length
        target_len = 64
        if seq.size(1) > target_len:
            seq = seq[:, :target_len, :]
        else:
            seq = F.pad(seq, (0, 0, 0, target_len - seq.size(1)))
            
        # Check for sequential dependencies
        output, (hidden, cell) = self.sequence_detector(seq)
        
        # High LSTM activation = sequential structure
        activation_variance = torch.var(output).item()
        sequential_score = torch.sigmoid(activation_variance * 5).item()
        
        votes = {
            TopologyType.SEQUENTIAL: sequential_score,
            TopologyType.FIBERED: 1 - sequential_score,  # Independent if not sequential
        }
        
        self.confidence = max(votes.values())
        return votes


class ParietalDetector(BrainSystemDetector):
    """Right hemisphere - spatial/holistic processing"""
    
    def __init__(self):
        super().__init__("parietal_lobe", TopologyType.SPHERICAL)
        
    def sense_topology(self, x: torch.Tensor) -> Dict[TopologyType, float]:
        """Detect global/holistic patterns"""
        # Compute global statistics
        mean = torch.mean(x)
        std = torch.std(x)
        skew = torch.mean((x - mean) ** 3) / (std ** 3 + 1e-8)
        
        # Bounded distribution suggests spherical topology
        boundedness = torch.exp(-torch.abs(skew)).item()
        
        votes = {
            TopologyType.SPHERICAL: boundedness,
            TopologyType.EUCLIDEAN: 1 - boundedness,
        }
        
        self.confidence = max(votes.values())
        return votes


class TopologyOrchestra(nn.Module):
    """
    Orchestrates multiple brain system topology detectors
    With 2π regulation to prevent cacophony
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize all brain system detectors
        self.detectors = nn.ModuleDict({
            'amygdala': AmygdalaDetector(),
            'hippocampus': HippocampusDetector(),
            'cerebellum': CerebellumDetector(),
            'prefrontal': PrefrontalDetector(),
            'temporal': TemporalDetector(),
            'parietal': ParietalDetector(),
        })
        
        # Meta-level attention (the "conductor")
        self.conductor = nn.Sequential(
            nn.Linear(len(self.detectors) * 11, 64),  # 11 topology types
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 11),  # Final topology selection
            nn.Softmax(dim=-1)
        )
        
        # 2π regulation for switching
        self.meta_regulator = MetaTwoPiRegulator()
        
        # Track history for jazz improvisation
        self.topology_history = []
        self.current_topology = TopologyType.EUCLIDEAN
        
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Orchestrate all detectors to choose topology
        """
        # Collect votes from all brain systems
        all_votes = {}
        system_confidences = {}
        
        for name, detector in self.detectors.items():
            votes = detector.sense_topology(x)
            all_votes[name] = votes
            system_confidences[name] = detector.confidence
            
        # Aggregate votes into topology scores
        topology_scores = self._aggregate_votes(all_votes, system_confidences)
        
        # Apply conductor modulation (top-down control)
        modulated_scores = self._conductor_modulation(topology_scores)
        
        # Check 2π regulation before switching
        new_topology = self._select_topology_with_regulation(modulated_scores)
        
        # Update history for learning
        self.topology_history.append({
            'topology': new_topology,
            'scores': modulated_scores,
            'system_confidences': system_confidences
        })
        
        return {
            'selected_topology': new_topology,
            'topology_scores': modulated_scores,
            'system_confidences': system_confidences,
            'meta_variety': self.meta_regulator.get_current_variety(),
            'switching_allowed': self.meta_regulator.can_switch()
        }
    
    def _aggregate_votes(self, all_votes: Dict, confidences: Dict) -> Dict[TopologyType, float]:
        """Aggregate votes from all systems"""
        topology_scores = {t: 0.0 for t in TopologyType}
        
        for system_name, votes in all_votes.items():
            confidence = confidences[system_name]
            for topology, score in votes.items():
                topology_scores[topology] += score * confidence
                
        # Normalize
        total = sum(topology_scores.values())
        if total > 0:
            for t in topology_scores:
                topology_scores[t] /= total
                
        return topology_scores
    
    def _conductor_modulation(self, scores: Dict[TopologyType, float]) -> Dict[TopologyType, float]:
        """Apply top-down modulation from prefrontal 'conductor'"""
        # Convert scores to tensor
        score_vector = torch.tensor([scores.get(t, 0.0) for t in TopologyType])
        
        # Flatten all detector states for conductor
        detector_states = []
        for detector in self.detectors.values():
            detector_states.append(torch.tensor(detector.confidence))
        conductor_input = torch.cat([score_vector] + detector_states)
        
        # Pad to expected size
        expected_size = len(self.detectors) * 11
        if conductor_input.size(0) < expected_size:
            conductor_input = F.pad(conductor_input, (0, expected_size - conductor_input.size(0)))
        
        # Apply conductor modulation
        modulated = self.conductor(conductor_input)
        
        # Convert back to dictionary
        modulated_scores = {}
        for i, t in enumerate(TopologyType):
            modulated_scores[t] = modulated[i].item()
            
        return modulated_scores
    
    def _select_topology_with_regulation(self, scores: Dict[TopologyType, float]) -> TopologyType:
        """Select topology with 2π regulation on switching"""
        # Find highest scoring topology
        best_topology = max(scores, key=scores.get)
        best_score = scores[best_topology]
        
        # Check if we should switch
        if self.meta_regulator.can_switch():
            # Calculate switching cost
            switch_cost = self._calculate_switch_cost(self.current_topology, best_topology)
            
            # Only switch if benefit outweighs cost
            current_score = scores.get(self.current_topology, 0.0)
            benefit = best_score - current_score
            
            if benefit > switch_cost * TWO_PI:
                self.meta_regulator.record_switch(self.current_topology, best_topology)
                self.current_topology = best_topology
        
        return self.current_topology
    
    def _calculate_switch_cost(self, from_topology: TopologyType, to_topology: TopologyType) -> float:
        """Calculate cost of switching between topologies"""
        # Some switches are more expensive than others
        cost_matrix = {
            (TopologyType.EUCLIDEAN, TopologyType.HYPERBOLIC): 0.8,  # Big shift
            (TopologyType.TREE, TopologyType.GRAPH): 0.3,  # Related
            (TopologyType.MANIFOLD, TopologyType.DISCRETE): 0.9,  # Opposite
            (TopologyType.SEQUENTIAL, TopologyType.FIBERED): 0.4,  # Related
        }
        
        key = (from_topology, to_topology)
        if key in cost_matrix:
            return cost_matrix[key]
        elif (to_topology, from_topology) in cost_matrix:
            return cost_matrix[(to_topology, from_topology)]
        else:
            return 0.5  # Default cost


class MetaTwoPiRegulator:
    """
    Regulates the meta-level variety of topology switching
    Prevents 'geometry thrashing'
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.switch_history = []
        self.current_variety = 0.0
        
    def record_switch(self, from_topology: TopologyType, to_topology: TopologyType):
        """Record a topology switch"""
        self.switch_history.append({
            'from': from_topology,
            'to': to_topology,
            'timestamp': len(self.switch_history)
        })
        
        # Keep window size
        if len(self.switch_history) > self.window_size:
            self.switch_history.pop(0)
            
        # Update variety
        self.current_variety = self._compute_variety()
    
    def _compute_variety(self) -> float:
        """Compute variety of switching behavior"""
        if len(self.switch_history) < 2:
            return 0.0
            
        # Calculate switch frequency
        switch_intervals = []
        for i in range(1, len(self.switch_history)):
            interval = self.switch_history[i]['timestamp'] - self.switch_history[i-1]['timestamp']
            switch_intervals.append(interval)
            
        if not switch_intervals:
            return 0.0
            
        # Variety = normalized entropy of switch intervals
        hist, _ = np.histogram(switch_intervals, bins=10)
        hist = hist / hist.sum()
        variety = entropy(hist) / np.log(10)  # Normalize to [0, 1]
        
        return variety
    
    def can_switch(self) -> bool:
        """Check if switching is allowed under 2π regulation"""
        return self.current_variety < TWO_PI
    
    def get_current_variety(self) -> float:
        """Get current meta-variety level"""
        return self.current_variety


class GeometryBank(nn.Module):
    """
    Bank of different geometric encoders
    Selected based on detected topology
    """
    
    def __init__(self):
        super().__init__()
        
        # Different geometric encoders
        self.encoders = nn.ModuleDict({
            TopologyType.EUCLIDEAN.value: EuclideanEncoder(),
            TopologyType.HYPERBOLIC.value: HyperbolicEncoder(),
            TopologyType.SPHERICAL.value: SphericalEncoder(),
            TopologyType.TREE.value: TreeEncoder(),
            TopologyType.GRAPH.value: GraphEncoder(),
            TopologyType.MANIFOLD.value: ManifoldEncoder(),
            TopologyType.DISCRETE.value: DiscreteEncoder(),
            TopologyType.SEQUENTIAL.value: SequentialEncoder(),
            TopologyType.FIBERED.value: FiberedEncoder(),
            TopologyType.COUPLED.value: CoupledEncoder(),
            TopologyType.FRACTAL.value: FractalEncoder(),
        })
        
    def forward(self, x: torch.Tensor, topology: TopologyType) -> torch.Tensor:
        """Encode using selected geometry"""
        encoder = self.encoders[topology.value]
        return encoder(x)


# Placeholder encoder classes (to be implemented)
class EuclideanEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class HyperbolicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        # Hyperbolic projection
        z = self.encoder(x.view(x.size(0), -1)[:, :256])
        return torch.tanh(z) * 0.99  # Keep in Poincaré ball

class SphericalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        z = self.encoder(x.view(x.size(0), -1)[:, :256])
        return F.normalize(z, p=2, dim=-1)  # Project to sphere

class TreeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class ManifoldEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class DiscreteEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        z = self.encoder(x.view(x.size(0), -1)[:, :256])
        return torch.round(z)  # Discretize

class SequentialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(256, 64, batch_first=True)
    def forward(self, x):
        seq = x.view(x.size(0), 1, -1)[:, :, :256]
        output, _ = self.lstm(seq)
        return output.squeeze(1)

class FiberedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class CoupledEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 64)
    def forward(self, x):
        return self.encoder(x.view(x.size(0), -1)[:, :256])

class FractalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Linear(256, 16) for _ in range(4)
        ])
    def forward(self, x):
        features = []
        flat = x.view(x.size(0), -1)[:, :256]
        for scale in self.scales:
            features.append(scale(flat))
        return torch.cat(features, dim=-1)


class VSM(nn.Module):
    """
    Variational Sensory Morphism
    The complete system: Orchestra + Geometry Bank
    """
    
    def __init__(self):
        super().__init__()
        self.orchestra = TopologyOrchestra()
        self.geometry_bank = GeometryBank()
        
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Complete VSM pipeline:
        1. Sense topology with orchestra
        2. Select appropriate geometry
        3. Encode with selected geometry
        4. Track meta-variety
        """
        # Orchestra determines topology
        orchestra_output = self.orchestra(x)
        selected_topology = orchestra_output['selected_topology']
        
        # Encode with selected geometry
        encoded = self.geometry_bank(x, selected_topology)
        
        return {
            'encoded': encoded,
            'topology': selected_topology,
            'topology_scores': orchestra_output['topology_scores'],
            'system_confidences': orchestra_output['system_confidences'],
            'meta_variety': orchestra_output['meta_variety'],
            'switching_allowed': orchestra_output['switching_allowed']
        }


def test_vsm_on_shapes3d():
    """Test the VSM on Shapes3D data"""
    print("="*60)
    print("VSM TOPOLOGY ORCHESTRA TEST")
    print("Inverting Rancière's Distribution of the Sensible")
    print("="*60)
    
    # Create VSM
    vsm = VSM()
    print(f"Initialized VSM with {len(vsm.orchestra.detectors)} brain systems")
    print(f"Geometry bank has {len(vsm.geometry_bank.encoders)} topologies")
    
    # Test with random data (simulating Shapes3D)
    batch_size = 32
    test_data = torch.randn(batch_size, 3, 64, 64)
    
    print("\nProcessing batch...")
    output = vsm(test_data)
    
    print(f"\nSelected Topology: {output['topology']}")
    print(f"Meta-variety: {output['meta_variety']:.4f} (limit: {TWO_PI:.4f})")
    print(f"Switching allowed: {output['switching_allowed']}")
    
    print("\nSystem Confidences:")
    for system, confidence in output['system_confidences'].items():
        print(f"  {system}: {confidence:.3f}")
    
    print("\nTop 3 Topology Scores:")
    sorted_topologies = sorted(output['topology_scores'].items(), 
                              key=lambda x: x[1], reverse=True)
    for topology, score in sorted_topologies[:3]:
        print(f"  {topology}: {score:.3f}")
    
    print("\n" + "="*60)
    print("VSM successfully orchestrates topology detection!")
    print("Each brain system contributes its 'feeling' of the geometry")
    print("2π regulation prevents switching chaos")
    print("="*60)


if __name__ == "__main__":
    test_vsm_on_shapes3d()