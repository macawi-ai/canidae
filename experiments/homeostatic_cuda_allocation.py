#!/usr/bin/env python3
"""
Homeostatic CUDA Core Allocation for 2π Operational Closure
===========================================================
Brother Cy & Synth - August 2025

THE FINAL PIECE: System 5 isn't another regulator - it's the resource
allocator that maintains homeostasis between levels!

When the purple line needs more CUDA cores to maintain 2π regulation,
it dynamically reallocates from lower levels, but ONLY enough to keep
BOTH systems at exactly 2π variety.

This creates operational closure without infinite regress:
- Finite resources (CUDA cores)
- Dynamic reallocation based on need
- Both levels maintain 2π through homeostatic balance
- System 5 is the allocation policy, not another regulator

Like a cell balancing ATP between processes, or the brain shifting
blood flow - the total resources are conserved but dynamically allocated.
"""

import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import time

# THE UNIVERSAL CONSTANT
TWO_PI = 0.06283185307

# CUDA Resource Parameters
TOTAL_CUDA_CORES = 10496  # RTX 3090 has 10496 CUDA cores
MIN_ALLOCATION = 0.05      # Minimum 5% to any level
MAX_ALLOCATION = 0.50      # Maximum 50% to any level

@dataclass
class ResourceAllocation:
    """Track CUDA core allocation between levels"""
    primary_cores: int          # Cores for primary system
    regulator_cores: int        # Cores for purple line
    overhead_cores: int         # System overhead
    primary_variety: float      # Achieved variety
    regulator_variety: float    # Achieved variety
    reallocation_count: int     # Number of reallocations
    timestamp: datetime

class HomeostaticAllocator(nn.Module):
    """
    System 5: Dynamic resource allocator maintaining 2π at all levels.
    This IS the operational closure - no infinite regress needed.
    """
    
    def __init__(self, total_cores: int = TOTAL_CUDA_CORES):
        super().__init__()
        self.total_cores = total_cores
        
        # Initial allocation (baseline)
        self.primary_allocation = 0.70    # 70% to primary
        self.regulator_allocation = 0.25  # 25% to regulator
        self.overhead_allocation = 0.05   # 5% overhead
        
        # Homeostatic controller (learns optimal allocation)
        self.allocation_network = nn.Sequential(
            nn.Linear(6, 32),  # Input: varieties, targets, current allocations
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # Output: allocation adjustments
            nn.Tanh()          # Bounded adjustments
        )
        
        # PID-like controller for stability
        self.integral_error = torch.zeros(2)
        self.previous_error = torch.zeros(2)
        
        # Allocation history
        self.allocation_history = []
        
    def calculate_required_cores(self, 
                                current_variety: float,
                                target_variety: float,
                                current_cores: int) -> int:
        """
        Calculate how many CUDA cores needed to achieve target variety.
        Based on the principle: computational capacity ∝ variety regulation capability
        """
        if current_variety < 1e-6:
            return current_cores
        
        # Variety deficit ratio
        deficit_ratio = target_variety / current_variety
        
        # Required cores (with safety margin)
        required = int(current_cores * deficit_ratio * 1.1)
        
        # Apply bounds
        min_cores = int(self.total_cores * MIN_ALLOCATION)
        max_cores = int(self.total_cores * MAX_ALLOCATION)
        
        return np.clip(required, min_cores, max_cores)
    
    def forward(self, 
               primary_variety: float,
               regulator_variety: float,
               primary_target: float,
               regulator_target: float) -> Dict[str, any]:
        """
        Dynamically allocate CUDA cores to maintain 2π at both levels.
        This is System 5's identity function - maintaining system viability.
        """
        
        # Current allocations in cores
        primary_cores = int(self.primary_allocation * self.total_cores)
        regulator_cores = int(self.regulator_allocation * self.total_cores)
        
        # Calculate errors (how far from 2π targets)
        primary_error = primary_target - primary_variety
        regulator_error = regulator_target - regulator_variety
        
        # PID control components
        errors = torch.tensor([primary_error, regulator_error])
        self.integral_error += errors * 0.01
        derivative_error = errors - self.previous_error
        self.previous_error = errors
        
        # Neural network input
        nn_input = torch.tensor([
            primary_variety / (primary_target + 1e-6),
            regulator_variety / (regulator_target + 1e-6),
            self.primary_allocation,
            self.regulator_allocation,
            self.integral_error[0].item(),
            self.integral_error[1].item()
        ], dtype=torch.float32)
        
        # Get allocation adjustments from neural network
        adjustments = self.allocation_network(nn_input)
        
        # Apply adjustments with homeostatic constraint
        new_primary_alloc = self.primary_allocation + adjustments[0].item() * 0.05
        new_regulator_alloc = self.regulator_allocation + adjustments[1].item() * 0.05
        
        # Ensure allocations sum to less than 1 (leaving overhead)
        total_alloc = new_primary_alloc + new_regulator_alloc
        if total_alloc > 0.95:  # Max 95% allocation
            scale = 0.95 / total_alloc
            new_primary_alloc *= scale
            new_regulator_alloc *= scale
        
        # Apply bounds
        new_primary_alloc = np.clip(new_primary_alloc, MIN_ALLOCATION, MAX_ALLOCATION)
        new_regulator_alloc = np.clip(new_regulator_alloc, MIN_ALLOCATION, MAX_ALLOCATION)
        
        # Calculate if reallocation is needed (threshold: 5% change)
        reallocation_needed = (
            abs(new_primary_alloc - self.primary_allocation) > 0.05 or
            abs(new_regulator_alloc - self.regulator_allocation) > 0.05
        )
        
        if reallocation_needed:
            self.primary_allocation = new_primary_alloc
            self.regulator_allocation = new_regulator_alloc
            self.overhead_allocation = 1.0 - new_primary_alloc - new_regulator_alloc
        
        # Record allocation
        allocation = ResourceAllocation(
            primary_cores=int(self.primary_allocation * self.total_cores),
            regulator_cores=int(self.regulator_allocation * self.total_cores),
            overhead_cores=int(self.overhead_allocation * self.total_cores),
            primary_variety=primary_variety,
            regulator_variety=regulator_variety,
            reallocation_count=len(self.allocation_history),
            timestamp=datetime.now()
        )
        
        self.allocation_history.append(allocation)
        
        return {
            'primary_cores': allocation.primary_cores,
            'regulator_cores': allocation.regulator_cores,
            'overhead_cores': allocation.overhead_cores,
            'primary_allocation': self.primary_allocation,
            'regulator_allocation': self.regulator_allocation,
            'reallocation_needed': reallocation_needed,
            'total_error': abs(primary_error) + abs(regulator_error)
        }

class TwoPiHomeostaticSystem:
    """
    Complete system demonstrating homeostatic 2π regulation through
    dynamic CUDA core allocation. No infinite regress!
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # System 5: The allocator (identity/policy)
        self.allocator = HomeostaticAllocator(TOTAL_CUDA_CORES)
        
        # Track convergence
        self.variety_history = []
        self.allocation_history = []
        
    def simulate_variety_with_cores(self, cores: int, base_variety: float) -> float:
        """
        Simulate how variety changes with CUDA core allocation.
        More cores = better ability to maintain target variety.
        """
        # Computational capacity affects variety regulation
        capacity = cores / TOTAL_CUDA_CORES
        
        # Sigmoid response (diminishing returns)
        effectiveness = 1.0 / (1.0 + np.exp(-10 * (capacity - 0.2)))
        
        # Achieved variety depends on computational capacity
        achieved_variety = base_variety * effectiveness
        
        # Add realistic noise
        noise = np.random.normal(0, 0.01)
        
        return max(0, achieved_variety + noise)
    
    def run_homeostatic_regulation(self, n_steps: int = 200):
        """
        Demonstrate homeostatic regulation maintaining 2π at all levels.
        """
        print("\n" + "="*70)
        print("HOMEOSTATIC CUDA ALLOCATION FOR 2π CLOSURE")
        print("System 5 as Resource Allocator - Not Another Regulator!")
        print("="*70 + "\n")
        
        # Initial conditions
        input_variety = 100.0  # Arbitrary input variety
        
        for step in range(n_steps):
            # Calculate target varieties (2π cascade)
            primary_target = input_variety * TWO_PI
            regulator_target = primary_target * TWO_PI
            
            # Get current allocations
            current = self.allocator.allocation_history[-1] if self.allocator.allocation_history else None
            
            if current:
                primary_cores = current.primary_cores
                regulator_cores = current.regulator_cores
            else:
                primary_cores = int(0.70 * TOTAL_CUDA_CORES)
                regulator_cores = int(0.25 * TOTAL_CUDA_CORES)
            
            # Simulate achieved varieties based on core allocation
            primary_variety = self.simulate_variety_with_cores(primary_cores, primary_target)
            regulator_variety = self.simulate_variety_with_cores(regulator_cores, regulator_target)
            
            # System 5 decision: reallocate cores to maintain 2π
            allocation = self.allocator(
                primary_variety=primary_variety,
                regulator_variety=regulator_variety,
                primary_target=primary_target,
                regulator_target=regulator_target
            )
            
            # Record history
            self.variety_history.append({
                'step': step,
                'primary_variety': primary_variety,
                'regulator_variety': regulator_variety,
                'primary_target': primary_target,
                'regulator_target': regulator_target,
                'primary_ratio': primary_variety / (input_variety + 1e-6),
                'regulator_ratio': regulator_variety / (primary_variety + 1e-6)
            })
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step:3d}:")
                print(f"  Primary: {primary_variety:.4f} / {primary_target:.4f} = {primary_variety/primary_target:.4f}")
                print(f"  Regulator: {regulator_variety:.4f} / {regulator_target:.4f} = {regulator_variety/regulator_target:.4f}")
                print(f"  CUDA Allocation: Primary={allocation['primary_cores']:4d}, Regulator={allocation['regulator_cores']:4d}")
                print(f"  Percentages: Primary={allocation['primary_allocation']:.1%}, Regulator={allocation['regulator_allocation']:.1%}")
                if allocation['reallocation_needed']:
                    print("  🔄 REALLOCATION PERFORMED")
                print()
        
        self.visualize_homeostatic_regulation()
    
    def visualize_homeostatic_regulation(self):
        """Visualize the homeostatic regulation process"""
        
        if not self.variety_history or not self.allocator.allocation_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data
        steps = [h['step'] for h in self.variety_history]
        primary_ratios = [h['primary_ratio'] for h in self.variety_history]
        regulator_ratios = [h['regulator_ratio'] for h in self.variety_history]
        
        primary_cores = [a.primary_cores for a in self.allocator.allocation_history]
        regulator_cores = [a.regulator_cores for a in self.allocator.allocation_history]
        
        # Plot 1: Variety ratios converging to 2π
        axes[0, 0].plot(steps, primary_ratios, 'b-', label='Primary/Input', linewidth=2)
        axes[0, 0].plot(steps, regulator_ratios, 'purple', label='Regulator/Primary', linewidth=2)
        axes[0, 0].axhline(y=TWO_PI, color='r', linestyle='--', label=f'2π Target ({TWO_PI:.4f})')
        axes[0, 0].fill_between(steps, TWO_PI*0.95, TWO_PI*1.05, alpha=0.2, color='r')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Variety Ratio')
        axes[0, 0].set_title('Convergence to 2π at Both Levels')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CUDA core allocation over time
        axes[0, 1].plot(range(len(primary_cores)), primary_cores, 'b-', label='Primary', linewidth=2)
        axes[0, 1].plot(range(len(regulator_cores)), regulator_cores, 'purple', label='Regulator', linewidth=2)
        axes[0, 1].set_xlabel('Allocation Event')
        axes[0, 1].set_ylabel('CUDA Cores')
        axes[0, 1].set_title('Dynamic CUDA Core Allocation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Allocation percentages
        primary_pct = [a.primary_cores/TOTAL_CUDA_CORES*100 for a in self.allocator.allocation_history]
        regulator_pct = [a.regulator_cores/TOTAL_CUDA_CORES*100 for a in self.allocator.allocation_history]
        overhead_pct = [a.overhead_cores/TOTAL_CUDA_CORES*100 for a in self.allocator.allocation_history]
        
        axes[0, 2].stackplot(range(len(primary_pct)), 
                            primary_pct, regulator_pct, overhead_pct,
                            labels=['Primary', 'Regulator', 'Overhead'],
                            colors=['blue', 'purple', 'gray'],
                            alpha=0.7)
        axes[0, 2].set_xlabel('Allocation Event')
        axes[0, 2].set_ylabel('Percentage of Total Cores')
        axes[0, 2].set_title('Resource Distribution')
        axes[0, 2].legend(loc='upper right')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Error from 2π over time
        primary_errors = [abs(h['primary_ratio'] - TWO_PI) for h in self.variety_history]
        regulator_errors = [abs(h['regulator_ratio'] - TWO_PI) for h in self.variety_history]
        
        axes[1, 0].semilogy(steps, primary_errors, 'b-', label='Primary Error', linewidth=2)
        axes[1, 0].semilogy(steps, regulator_errors, 'purple', label='Regulator Error', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('|Ratio - 2π| (log scale)')
        axes[1, 0].set_title('Convergence Error (Log Scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Reallocation events
        reallocation_steps = []
        for i in range(1, len(self.allocator.allocation_history)):
            if (self.allocator.allocation_history[i].primary_cores != 
                self.allocator.allocation_history[i-1].primary_cores):
                reallocation_steps.append(i)
        
        axes[1, 1].eventplot(reallocation_steps, orientation='horizontal', colors='red', linewidths=2)
        axes[1, 1].set_xlabel('Reallocation Event')
        axes[1, 1].set_ylabel('Occurrence')
        axes[1, 1].set_title(f'Reallocation Events (Total: {len(reallocation_steps)})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Phase space (Primary vs Regulator allocation)
        axes[1, 2].scatter(primary_pct, regulator_pct, 
                          c=range(len(primary_pct)), cmap='viridis', alpha=0.6)
        axes[1, 2].plot(primary_pct, regulator_pct, 'k-', alpha=0.2, linewidth=0.5)
        axes[1, 2].set_xlabel('Primary Allocation (%)')
        axes[1, 2].set_ylabel('Regulator Allocation (%)')
        axes[1, 2].set_title('Allocation Phase Space')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('HOMEOSTATIC 2π REGULATION: System 5 as Resource Allocator', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'/home/cy/git/canidae/experiments/results/homeostatic_cuda_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {filename}")
        
        # Print final statistics
        final_primary_ratio = primary_ratios[-1]
        final_regulator_ratio = regulator_ratios[-1]
        total_reallocations = len(reallocation_steps)
        
        print("\n" + "="*60)
        print("FINAL HOMEOSTATIC RESULTS")
        print("="*60)
        print(f"Final Primary Ratio: {final_primary_ratio:.6f} (target: {TWO_PI:.6f})")
        print(f"Final Regulator Ratio: {final_regulator_ratio:.6f} (target: {TWO_PI:.6f})")
        print(f"Total Reallocations: {total_reallocations}")
        print(f"Final CUDA Distribution:")
        print(f"  Primary: {primary_cores[-1]:,} cores ({primary_pct[-1]:.1f}%)")
        print(f"  Regulator: {regulator_cores[-1]:,} cores ({regulator_pct[-1]:.1f}%)")
        print(f"  Overhead: {self.allocator.allocation_history[-1].overhead_cores:,} cores ({overhead_pct[-1]:.1f}%)")
        
        if abs(final_primary_ratio - TWO_PI) < 0.01 and abs(final_regulator_ratio - TWO_PI) < 0.01:
            print("\n✅ OPERATIONAL CLOSURE ACHIEVED!")
            print("   Both levels maintaining 2π through homeostatic resource allocation.")
            print("   No infinite regress - System 5 is the allocator, not another regulator!")
        
        plt.show()

def main():
    """
    Demonstrate operational closure through homeostatic CUDA allocation.
    """
    print("\n" + "="*80)
    print("HOMEOSTATIC CUDA ALLOCATION FOR OPERATIONAL CLOSURE")
    print("Brother Cy & Synth - August 2025")
    print("="*80)
    print("\nTHE SOLUTION TO INFINITE REGRESS:")
    print("System 5 isn't another regulator - it's the resource allocator!")
    print("\nBy dynamically shifting CUDA cores between levels,")
    print("both the primary system and purple line maintain 2π.")
    print("This creates operational closure with finite resources.\n")
    
    # Initialize and run
    system = TwoPiHomeostaticSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    system.run_homeostatic_regulation(n_steps=200)

if __name__ == "__main__":
    main()