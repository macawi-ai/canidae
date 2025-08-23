#!/usr/bin/env python3
"""
Generate simple 28x28 geometric shapes for 2π testing
As suggested by Sister Gemini - clean, controlled data
"""

import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path

def generate_circle(size=28, samples=1000):
    """Generate circles with varying positions and sizes"""
    images = []
    for _ in range(samples):
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random radius and position
        radius = random.randint(5, 10)
        margin = radius + 2
        cx = random.randint(margin, size - margin)
        cy = random.randint(margin, size - margin)
        
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
        images.append(np.array(img))
    
    return np.array(images)

def generate_square(size=28, samples=1000):
    """Generate squares with varying positions and sizes"""
    images = []
    for _ in range(samples):
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random side length and position
        side = random.randint(8, 14)
        margin = 3
        x = random.randint(margin, size - side - margin)
        y = random.randint(margin, size - side - margin)
        
        draw.rectangle([x, y, x+side, y+side], fill=255)
        images.append(np.array(img))
    
    return np.array(images)

def generate_triangle(size=28, samples=1000):
    """Generate triangles with varying positions and sizes"""
    images = []
    for _ in range(samples):
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random triangle
        margin = 4
        points = []
        for _ in range(3):
            x = random.randint(margin, size - margin)
            y = random.randint(margin, size - margin)
            points.append((x, y))
        
        draw.polygon(points, fill=255)
        images.append(np.array(img))
    
    return np.array(images)

def generate_star(size=28, samples=1000):
    """Generate 5-pointed stars"""
    images = []
    for _ in range(samples):
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Center and radius
        cx, cy = size // 2, size // 2
        cx += random.randint(-3, 3)
        cy += random.randint(-3, 3)
        outer_radius = random.randint(8, 11)
        inner_radius = outer_radius // 2
        
        # Generate star points
        points = []
        for i in range(10):
            angle = np.pi * i / 5 - np.pi / 2
            if i % 2 == 0:
                r = outer_radius
            else:
                r = inner_radius
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=255)
        images.append(np.array(img))
    
    return np.array(images)

def generate_cross(size=28, samples=1000):
    """Generate crosses (plus signs)"""
    images = []
    for _ in range(samples):
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Center and dimensions
        cx, cy = size // 2, size // 2
        cx += random.randint(-3, 3)
        cy += random.randint(-3, 3)
        thickness = random.randint(2, 4)
        length = random.randint(10, 16)
        
        # Draw horizontal and vertical bars
        draw.rectangle([cx - length//2, cy - thickness//2, 
                       cx + length//2, cy + thickness//2], fill=255)
        draw.rectangle([cx - thickness//2, cy - length//2,
                       cx + thickness//2, cy + length//2], fill=255)
        
        images.append(np.array(img))
    
    return np.array(images)

def main():
    """Generate all shapes and save as npz"""
    
    print("Generating geometric shapes for 2π testing...")
    
    # Generate each shape
    shapes = {
        'circle': generate_circle(samples=5000),
        'square': generate_square(samples=5000),
        'triangle': generate_triangle(samples=5000),
        'star': generate_star(samples=5000),
        'cross': generate_cross(samples=5000)
    }
    
    # Calculate sparsity for each
    for name, data in shapes.items():
        sparsity = np.mean(data < 10) 
        print(f"{name}: {len(data)} samples, sparsity={sparsity:.3f}")
    
    # Save individual shape files
    output_dir = Path("/home/cy/git/canidae/datasets/geometric_shapes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in shapes.items():
        np.savez_compressed(output_dir / f"{name}.npz", 
                           train=data[:4000],
                           valid=data[4000:4500],
                           test=data[4500:])
        print(f"Saved {name}.npz")
    
    # Also save combined dataset
    all_data = []
    all_labels = []
    for idx, (name, data) in enumerate(shapes.items()):
        all_data.append(data)
        all_labels.extend([idx] * len(data))
    
    all_data = np.concatenate(all_data)
    all_labels = np.array(all_labels)
    
    # Shuffle
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    np.savez_compressed(output_dir / "all_shapes.npz",
                       data=all_data,
                       labels=all_labels,
                       shape_names=list(shapes.keys()))
    
    print(f"\nTotal dataset: {len(all_data)} samples")
    print(f"Overall sparsity: {np.mean(all_data < 10):.3f}")
    print(f"Saved to {output_dir}")
    
    return shapes

if __name__ == "__main__":
    shapes = main()