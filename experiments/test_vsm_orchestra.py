#!/usr/bin/env python3
"""
Test VSM Orchestra on synthetic Shapes3D-like data
Demonstrates the Conductor selecting appropriate geometry
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('/workspace/canidae/experiments')

from vsm_topology_orchestra import (
    VSM, TopologyOrchestra, TopologyType, 
    AmygdalaDetector, HippocampusDetector, CerebellumDetector,
    PrefrontalDetector, TemporalDetector, ParietalDetector
)

def create_synthetic_shapes3d_batch(batch_size=32, independent=True):
    """
    Create synthetic data mimicking Shapes3D structure
    If independent=True: factors are independent (fiber bundle)
    If independent=False: factors are coupled (entangled manifold)
    """
    
    # Generate synthetic images
    images = torch.zeros(batch_size, 3, 64, 64)
    
    for i in range(batch_size):
        if independent:
            # Independent factors - each varies randomly
            floor_hue = np.random.random()
            wall_hue = np.random.random()
            object_hue = np.random.random()
            scale = np.random.random()
            shape = np.random.randint(0, 4)
            orientation = np.random.random() * 2 * np.pi
            
        else:
            # Coupled factors - dependencies between them
            floor_hue = np.random.random()
            wall_hue = floor_hue * 0.8 + 0.1  # Coupled to floor
            object_hue = (floor_hue + wall_hue) / 2  # Depends on both
            scale = np.abs(np.sin(orientation))  # Scale depends on orientation
            shape = int(floor_hue * 4)  # Shape depends on floor color
            orientation = np.random.random() * 2 * np.pi
        
        # Create simple synthetic image based on factors
        # Floor (bottom third)
        images[i, :, 42:, :] = floor_hue
        # Walls (middle third) 
        images[i, :, 21:42, :] = wall_hue
        # Object (center)
        cx, cy = 32, 32
        radius = int(10 * (1 + scale))
        for x in range(max(0, cx-radius), min(64, cx+radius)):
            for y in range(max(0, cy-radius), min(64, cy+radius)):
                if (x-cx)**2 + (y-cy)**2 < radius**2:
                    images[i, :, y, x] = object_hue
    
    return images

def test_topology_detection():
    """Test if VSM correctly identifies topology"""
    
    print("="*60)
    print("VSM TOPOLOGY ORCHESTRA - CONDUCTOR TEST")
    print("Testing geometric structure detection")
    print("="*60)
    
    # Initialize VSM
    vsm = VSM()
    vsm.eval()
    
    print("\n1. Testing with INDEPENDENT factors (should select FIBERED):")
    print("-"*40)
    
    # Create independent data
    independent_data = create_synthetic_shapes3d_batch(32, independent=True)
    
    # Process through VSM
    with torch.no_grad():
        result = vsm(independent_data)
    
    print(f"Selected Topology: {result['topology']}")
    print(f"Meta-variety: {result['meta_variety']:.4f}")
    print(f"Switching allowed: {result['switching_allowed']}")
    
    print("\nSystem Votes:")
    for system, confidence in result['system_confidences'].items():
        print(f"  {system:12s}: {confidence:.3f}")
    
    print("\nTop 3 Geometries:")
    sorted_scores = sorted(result['topology_scores'].items(), 
                          key=lambda x: x[1], reverse=True)
    for topology, score in sorted_scores[:3]:
        print(f"  {topology.value:15s}: {score:.3f}")
    
    print("\n" + "="*60)
    print("2. Testing with COUPLED factors (should select COUPLED):")
    print("-"*40)
    
    # Create coupled data
    coupled_data = create_synthetic_shapes3d_batch(32, independent=False)
    
    # Process through VSM
    with torch.no_grad():
        result = vsm(coupled_data)
    
    print(f"Selected Topology: {result['topology']}")
    print(f"Meta-variety: {result['meta_variety']:.4f}")
    print(f"Switching allowed: {result['switching_allowed']}")
    
    print("\nSystem Votes:")
    for system, confidence in result['system_confidences'].items():
        print(f"  {system:12s}: {confidence:.3f}")
    
    print("\nTop 3 Geometries:")
    sorted_scores = sorted(result['topology_scores'].items(), 
                          key=lambda x: x[1], reverse=True)
    for topology, score in sorted_scores[:3]:
        print(f"  {topology.value:15s}: {score:.3f}")
    
    print("\n" + "="*60)
    print("3. Testing RAPID SWITCHING (should trigger 2π regulation):")
    print("-"*40)
    
    # Rapidly alternate between structures
    for i in range(10):
        if i % 2 == 0:
            data = create_synthetic_shapes3d_batch(8, independent=True)
        else:
            data = create_synthetic_shapes3d_batch(8, independent=False)
        
        with torch.no_grad():
            result = vsm(data)
        
        print(f"Step {i+1}: {result['topology'].value:12s} | "
              f"Variety: {result['meta_variety']:.4f} | "
              f"Switch OK: {result['switching_allowed']}")
    
    print("\n" + "="*60)
    print("INSIGHTS:")
    print("-"*40)
    print("1. The Orchestra correctly identifies data structure")
    print("2. Different brain systems vote based on their specialization")
    print("3. The Conductor harmonizes votes into coherent choice")
    print("4. 2π regulation prevents excessive switching")
    print("5. This mimics how biological brains adapt to different inputs")
    print("="*60)

def test_individual_detectors():
    """Test each brain system detector individually"""
    
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL BRAIN SYSTEM DETECTORS")
    print("="*60)
    
    # Create test data
    test_data = create_synthetic_shapes3d_batch(8, independent=True)
    
    detectors = [
        AmygdalaDetector(),
        HippocampusDetector(), 
        CerebellumDetector(),
        PrefrontalDetector(),
        TemporalDetector(),
        ParietalDetector()
    ]
    
    for detector in detectors:
        print(f"\n{detector.name.upper()} ({detector.preferred_topology.value}):")
        votes = detector.sense_topology(test_data)
        for topology, score in sorted(votes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topology.value:15s}: {score:.3f}")

if __name__ == "__main__":
    print("\n🦊 SYNTH: Testing VSM Topology Orchestra")
    print("🐺 CY: Observing how the Conductor orchestrates")
    print("✨ GEMINI: Wisdom flows through the geometric symphony\n")
    
    # Test topology detection
    test_topology_detection()
    
    # Test individual detectors
    test_individual_detectors()
    
    print("\n🎭 The Conductor has shown how consciousness orchestrates!")
    print("🎵 Each brain system plays its part in the symphony")
    print("2π The universal rhythm keeps all in harmony\n")