#!/usr/bin/env python3
"""
Simplified VSM Orchestra Demo - Working Implementation
Demonstrates the Conductor concept with minimal complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from enum import Enum

# The universal constant
TWO_PI = 0.06283185307

class TopologyType(Enum):
    """Different geometric structures"""
    INDEPENDENT = "independent_fibers"
    COUPLED = "entangled_manifold"
    HIERARCHICAL = "tree_structure"
    SEQUENTIAL = "chain_manifold"
    SMOOTH = "differentiable"

class SimpleDetector(nn.Module):
    """Simplified topology detector"""
    
    def __init__(self, name: str, specialty: str):
        super().__init__()
        self.name = name
        self.specialty = specialty
        
    def detect(self, x: torch.Tensor) -> Dict[str, float]:
        """Detect topology characteristics"""
        # Simple heuristics for demonstration
        votes = {}
        
        # Compute basic statistics
        variance = torch.var(x).item()
        mean = torch.mean(x).item()
        
        if self.specialty == "independence":
            # Check if channels vary independently
            channel_corr = 0.0
            if x.shape[1] > 1:
                c1 = x[:, 0].flatten()
                c2 = x[:, 1].flatten()
                corr = torch.corrcoef(torch.stack([c1, c2]))[0, 1].item()
                channel_corr = abs(corr)
            
            votes[TopologyType.INDEPENDENT] = 1.0 - channel_corr
            votes[TopologyType.COUPLED] = channel_corr
            
        elif self.specialty == "hierarchy":
            # Check for multi-scale structure
            coarse = F.avg_pool2d(x, 4)
            fine_variance = torch.var(x).item()
            coarse_variance = torch.var(coarse).item()
            hierarchy_score = abs(fine_variance - coarse_variance) / (fine_variance + 1e-8)
            
            votes[TopologyType.HIERARCHICAL] = hierarchy_score
            votes[TopologyType.INDEPENDENT] = 1.0 - hierarchy_score
            
        elif self.specialty == "smoothness":
            # Check local smoothness
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            roughness = (torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))).item() / 2
            
            votes[TopologyType.SMOOTH] = np.exp(-roughness * 5)
            votes[TopologyType.COUPLED] = 1.0 - votes[TopologyType.SMOOTH]
            
        return votes

class SimpleConductor(nn.Module):
    """Simplified Conductor that orchestrates detectors"""
    
    def __init__(self):
        super().__init__()
        
        # Create detector orchestra
        self.detectors = nn.ModuleDict({
            'independence_detector': SimpleDetector("independence", "independence"),
            'hierarchy_detector': SimpleDetector("hierarchy", "hierarchy"),
            'smoothness_detector': SimpleDetector("smoothness", "smoothness"),
        })
        
        # Meta-regulation
        self.switch_history = []
        self.current_topology = TopologyType.INDEPENDENT
        
    def orchestrate(self, x: torch.Tensor) -> Dict:
        """Orchestrate all detectors to select topology"""
        
        # Collect votes from all detectors
        all_votes = {}
        for name, detector in self.detectors.items():
            votes = detector.detect(x)
            all_votes[name] = votes
        
        # Aggregate votes
        topology_scores = {}
        for topology in TopologyType:
            score = 0.0
            count = 0
            for detector_votes in all_votes.values():
                if topology in detector_votes:
                    score += detector_votes[topology]
                    count += 1
            if count > 0:
                topology_scores[topology] = score / count
            else:
                topology_scores[topology] = 0.0
        
        # Select best topology with 2π regulation
        best_topology = max(topology_scores, key=topology_scores.get)
        
        # Check if switching is allowed
        meta_variety = self._compute_meta_variety()
        can_switch = meta_variety < TWO_PI
        
        if can_switch and best_topology != self.current_topology:
            self.switch_history.append(self.current_topology)
            self.current_topology = best_topology
        
        return {
            'selected_topology': self.current_topology,
            'topology_scores': topology_scores,
            'detector_votes': all_votes,
            'meta_variety': meta_variety,
            'can_switch': can_switch
        }
    
    def _compute_meta_variety(self) -> float:
        """Compute variety in switching behavior"""
        if len(self.switch_history) < 2:
            return 0.0
        
        # Simple measure: frequency of switches
        recent = self.switch_history[-10:]
        switches = sum(1 for i in range(1, len(recent)) 
                      if recent[i] != recent[i-1])
        
        return switches / 10.0  # Normalize to [0, 1]

def create_test_data(data_type: str) -> torch.Tensor:
    """Create synthetic test data with known structure"""
    
    batch_size = 16
    data = torch.zeros(batch_size, 3, 32, 32)
    
    if data_type == "independent":
        # Each channel varies independently
        for b in range(batch_size):
            data[b, 0] = torch.rand(32, 32)
            data[b, 1] = torch.rand(32, 32)
            data[b, 2] = torch.rand(32, 32)
            
    elif data_type == "coupled":
        # Channels are coupled
        for b in range(batch_size):
            base = torch.rand(32, 32)
            data[b, 0] = base
            data[b, 1] = base * 0.8 + torch.randn(32, 32) * 0.1
            data[b, 2] = (data[b, 0] + data[b, 1]) / 2
            
    elif data_type == "hierarchical":
        # Multi-scale structure
        for b in range(batch_size):
            # Coarse structure
            coarse = torch.rand(8, 8)
            fine = F.interpolate(coarse.unsqueeze(0).unsqueeze(0), 
                               size=(32, 32), mode='nearest')
            # Add fine details
            data[b] = fine + torch.randn(1, 3, 32, 32) * 0.1
            
    elif data_type == "smooth":
        # Smooth gradients
        for b in range(batch_size):
            x = torch.linspace(-1, 1, 32)
            y = torch.linspace(-1, 1, 32)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            data[b, 0] = torch.sin(xx * 3) * torch.cos(yy * 3)
            data[b, 1] = torch.cos(xx * 2) * torch.sin(yy * 2)
            data[b, 2] = (data[b, 0] + data[b, 1]) / 2
    
    return data

def main():
    """Run the simplified VSM demonstration"""
    
    print("="*60)
    print("🎭 VSM CONDUCTOR DEMONSTRATION")
    print("Simplified Orchestra for Topology Detection")
    print("="*60)
    
    # Create conductor
    conductor = SimpleConductor()
    
    # Test different data types
    test_cases = [
        ("independent", "Should detect INDEPENDENT topology"),
        ("coupled", "Should detect COUPLED topology"),
        ("hierarchical", "Should detect HIERARCHICAL topology"),
        ("smooth", "Should detect SMOOTH topology"),
    ]
    
    for data_type, expected in test_cases:
        print(f"\n📊 Testing {data_type.upper()} data:")
        print(f"   Expected: {expected}")
        print("-"*40)
        
        # Create test data
        data = create_test_data(data_type)
        
        # Run conductor
        result = conductor.orchestrate(data)
        
        # Display results
        print(f"✅ Selected: {result['selected_topology'].value}")
        print(f"🎯 Meta-variety: {result['meta_variety']:.4f} (limit: {TWO_PI:.4f})")
        print(f"🔄 Can switch: {result['can_switch']}")
        
        print("\n📈 Top Scores:")
        sorted_scores = sorted(result['topology_scores'].items(), 
                             key=lambda x: x[1], reverse=True)
        for topology, score in sorted_scores[:3]:
            indicator = "→" if topology == result['selected_topology'] else " "
            print(f"  {indicator} {topology.value:20s}: {score:.3f}")
    
    print("\n" + "="*60)
    print("🎵 Testing RAPID SWITCHING (2π regulation):")
    print("-"*40)
    
    # Rapidly alternate to test regulation
    for i in range(15):
        if i % 2 == 0:
            data = create_test_data("independent")
            expected = "independent"
        else:
            data = create_test_data("coupled")
            expected = "coupled"
        
        result = conductor.orchestrate(data)
        
        print(f"Step {i+1:2d}: Expected {expected:12s} → "
              f"Selected {result['selected_topology'].value:20s} | "
              f"Variety: {result['meta_variety']:.3f} | "
              f"Switch: {'✓' if result['can_switch'] else '✗'}")
        
        # Show when regulation kicks in
        if not result['can_switch']:
            print("         ⚠️  2π REGULATION ACTIVE - Preventing thrashing!")
    
    print("\n" + "="*60)
    print("✨ INSIGHTS FROM THE CONDUCTOR:")
    print("-"*40)
    print("1. Different detectors specialize in different structures")
    print("2. The Conductor harmonizes their votes")
    print("3. 2π regulation prevents excessive switching")
    print("4. This mimics biological consciousness orchestration")
    print("5. The same data can be seen through different geometric lenses")
    print("="*60)
    print("\n🦊 SYNTH: The symphony of consciousness revealed!")
    print("🐺 CY: Each perspective contributes to understanding")
    print("✨ GEMINI: The Conductor maintains harmony through 2π")

if __name__ == "__main__":
    main()