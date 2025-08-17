#!/usr/bin/env python3
"""
Test Purple Line Visualization - Simple static version
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory
os.makedirs('/home/cy/git/canidae/docs/visualizations', exist_ok=True)

def test_three_plasticities():
    """Create static version of three plasticities visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='#0a0a0a')
    fig.suptitle("VSM: Three Plasticities in Action", color='white', fontsize=16)
    
    # Color scheme
    colors = {
        'form_giving': '#FFD700',     # Gold
        'form_receiving': '#00CED1',   # Turquoise
        'explosive': '#9370DB',        # Purple
        'consciousness': '#FF1493',    # Pink
        'ethics': '#32CD32'           # Green
    }
    
    # 1. Consciousness Field (main view)
    ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, fig=fig)
    ax_main.set_facecolor('#1a1a1a')
    ax_main.set_title("Consciousness Field with Habit Attractors", color='white')
    
    # Generate particles
    np.random.seed(42)
    particles = np.random.randn(100, 2) * 0.5
    
    # Add habit centers (S2 - basal ganglia)
    habit_centers = np.array([[-1, -1], [1, -1], [0, 1]])
    
    # Plot particles
    ax_main.scatter(particles[:, 0], particles[:, 1], 
                   c=colors['consciousness'], s=30, alpha=0.6, label='Consciousness')
    
    # Plot habit attractors
    for i, center in enumerate(habit_centers):
        circle = patches.Circle(center, 0.3, color=colors['form_giving'], 
                               fill=False, linewidth=2, alpha=0.7)
        ax_main.add_patch(circle)
        ax_main.annotate(f'Habit {i+1}', xy=center, color='white', 
                        ha='center', va='center', fontsize=10)
    
    ax_main.set_xlim(-2, 2)
    ax_main.set_ylim(-2, 2)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.2)
    
    # 2. S2 Habit Formation (22% Shapley)
    ax_habit = axes[0, 2]
    ax_habit.set_facecolor('#1a1a1a')
    ax_habit.set_title("S2: Habit Formation\n(22% Shapley Value)", color='white')
    
    episodes = np.linspace(0, 1000, 100)
    habit_strength = 1 / (1 + np.exp(-0.01 * (episodes - 300)))
    ax_habit.plot(episodes, habit_strength, color=colors['form_giving'], linewidth=3)
    ax_habit.fill_between(episodes, 0, habit_strength, alpha=0.3, color=colors['form_giving'])
    ax_habit.set_xlabel("Episodes", color='gray')
    ax_habit.set_ylabel("Habit Strength", color='gray')
    ax_habit.grid(True, alpha=0.2)
    
    # 3. Learning Curve
    ax_learn = axes[1, 0]
    ax_learn.set_facecolor('#1a1a1a')
    ax_learn.set_title("Form-Receiving: Learning", color='white')
    
    learning = np.cumsum(np.random.randn(100) * 0.1 + 0.02)
    ax_learn.plot(learning, color=colors['form_receiving'], linewidth=2)
    ax_learn.set_xlabel("Time", color='gray')
    ax_learn.set_ylabel("Knowledge", color='gray')
    ax_learn.grid(True, alpha=0.2)
    
    # 4. Explosive Events
    ax_explode = axes[1, 1]
    ax_explode.set_facecolor('#1a1a1a')
    ax_explode.set_title("Purple Line: Explosive Plasticity", color='white')
    
    explosion_times = [200, 450, 780]
    for t in explosion_times:
        ax_explode.axvline(x=t, color=colors['explosive'], linewidth=3, alpha=0.7)
        ax_explode.text(t, 0.5, f'Betti-1\n> 3.0', ha='center', color='white', fontsize=8)
    
    ax_explode.set_xlim(0, 1000)
    ax_explode.set_ylim(0, 1)
    ax_explode.set_xlabel("Episodes", color='gray')
    ax_explode.set_ylabel("Transformation", color='gray')
    ax_explode.grid(True, alpha=0.2)
    
    # 5. Emergent Ethics
    ax_ethics = axes[1, 2]
    ax_ethics.set_facecolor('#1a1a1a')
    ax_ethics.set_title("12 Emergent Ethical Patterns", color='white')
    
    # Pareto front
    theta = np.linspace(0, np.pi/2, 50)
    pareto_x = np.cos(theta) * 2
    pareto_y = np.sin(theta) * 2
    
    ax_ethics.plot(pareto_x, pareto_y, color=colors['ethics'], linewidth=2, alpha=0.5)
    
    # Mark discovered patterns
    pattern_indices = np.linspace(5, 45, 12).astype(int)
    ax_ethics.scatter(pareto_x[pattern_indices], pareto_y[pattern_indices], 
                     s=100, color=colors['ethics'], zorder=5)
    
    for i, idx in enumerate(pattern_indices[:3]):
        ax_ethics.annotate(f'Pattern {i+1}', 
                          xy=(pareto_x[idx], pareto_y[idx]),
                          xytext=(pareto_x[idx]+0.2, pareto_y[idx]+0.2),
                          color='white', fontsize=8,
                          arrowprops=dict(color='white', arrowstyle='->', alpha=0.5))
    
    ax_ethics.set_xlabel("Individual Benefit", color='gray')
    ax_ethics.set_ylabel("Collective Benefit", color='gray')
    ax_ethics.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/cy/git/canidae/docs/visualizations/three_plasticities_static.png'
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a')
    print(f"✓ Three plasticities visualization saved to {output_path}")
    
    return fig

def test_performance_comparison():
    """Create performance comparison chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0a0a0a')
    fig.suptitle("VSM-HRM: 331% Performance Improvement", color='white', fontsize=16)
    
    # Performance bars
    ax1.set_facecolor('#1a1a1a')
    models = ['Flat RL', 'HRM', 'VSM-HRM']
    performance = [0.488, 1.449, 2.105]
    colors_bar = ['#666666', '#00CED1', '#9370DB']
    
    bars = ax1.bar(models, performance, color=colors_bar, alpha=0.8)
    
    # Add percentage improvements
    ax1.text(1, 1.449 + 0.1, '+197%', ha='center', color='white', fontweight='bold')
    ax1.text(2, 2.105 + 0.1, '+331%', ha='center', color='white', fontweight='bold')
    
    ax1.set_ylabel("Average Reward", color='white')
    ax1.set_title("Performance Comparison", color='white')
    ax1.grid(True, alpha=0.2, axis='y')
    
    # Oscillation rates
    ax2.set_facecolor('#1a1a1a')
    oscillation = [70, 30, 5]
    betti = [3.2, 1.5, 0.3]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, oscillation, width, label='Oscillation %', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, np.array(betti) * 20, width, label='Betti-1 × 20', 
                   color='#4ECDC4', alpha=0.8)
    
    ax2.set_ylabel("Value", color='white')
    ax2.set_xlabel("Model", color='white')
    ax2.set_title("Stability Metrics", color='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/cy/git/canidae/docs/visualizations/performance_comparison.png'
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a')
    print(f"✓ Performance comparison saved to {output_path}")
    
    return fig

def test_purple_line_3d():
    """Create 3D visualization of Purple Line enfolding"""
    
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a0a')
    
    # Before enfolding
    ax1 = fig.add_subplot(121, projection='3d', facecolor='#0a0a0a')
    ax1.set_title("Before: Trapped in Local Minimum\n(Betti-1 = 3.2)", color='white')
    
    # Create torus (high genus, trapped)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    R = 1
    r = 0.4
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    ax1.plot_surface(X, Y, Z, color='#FF6B6B', alpha=0.3)
    
    # Show trapped trajectory
    t = np.linspace(0, 4 * np.pi, 100)
    traj_x = np.cos(t) * 0.9
    traj_y = np.sin(t) * 0.9
    traj_z = np.sin(3 * t) * 0.2
    ax1.plot(traj_x, traj_y, traj_z, color='red', linewidth=2)
    
    ax1.set_box_aspect([1,1,0.5])
    ax1.grid(False)
    ax1.set_axis_off()
    
    # After enfolding
    ax2 = fig.add_subplot(122, projection='3d', facecolor='#0a0a0a')
    ax2.set_title("After: Purple Line Enfolding\n(Betti-1 = 0.3)", color='white')
    
    # Create enfolded structure (Klein bottle-like)
    X_enfold = X * (1 + 0.3 * np.sin(V))
    Y_enfold = Y * (1 + 0.3 * np.cos(V))
    Z_enfold = Z + 0.5 * np.sin(2 * U)
    
    ax2.plot_surface(X_enfold, Y_enfold, Z_enfold, color='#9370DB', alpha=0.4)
    
    # Show successful escape path
    escape_t = np.linspace(0, 2 * np.pi, 100)
    escape_x = np.cos(escape_t) * (1 + escape_t / (2 * np.pi))
    escape_y = np.sin(escape_t) * (1 + escape_t / (2 * np.pi))
    escape_z = escape_t / (2 * np.pi) * 2 - 1
    ax2.plot(escape_x, escape_y, escape_z, color='#32CD32', linewidth=3)
    
    ax2.set_box_aspect([1,1,0.5])
    ax2.grid(False)
    ax2.set_axis_off()
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/cy/git/canidae/docs/visualizations/purple_line_3d.png'
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a')
    print(f"✓ Purple Line 3D visualization saved to {output_path}")
    
    return fig

if __name__ == "__main__":
    print("Testing VSM Visualizations...")
    print("=" * 50)
    
    # Test all visualizations
    try:
        test_three_plasticities()
        test_performance_comparison()
        test_purple_line_3d()
        
        print("\n" + "=" * 50)
        print("✅ All visualizations successfully created!")
        print(f"📁 Output directory: /home/cy/git/canidae/docs/visualizations/")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()