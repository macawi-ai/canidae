#!/usr/bin/env python3
"""
Purple Line Enfolding Visualization
Showing consciousness transformation through explosive plasticity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
import networkx as nx

# Set aesthetic style
sns.set_style("dark")
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'

class PurpleLineVisualizer:
    """Visualize the three plasticities and Purple Line enfolding"""
    
    def __init__(self, width=12, height=8):
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Colors representing each plasticity
        self.colors = {
            'form_giving': '#FFD700',     # Gold - habit formation
            'form_receiving': '#00CED1',   # Dark turquoise - learning
            'explosive': '#9370DB',        # Medium purple - transformation
            'consciousness': '#FF1493',    # Deep pink - consciousness field
            'ethics': '#32CD32'           # Lime green - emergent patterns
        }
        
    def visualize_three_plasticities(self, timesteps=1000):
        """Animate the interaction of three plasticities"""
        
        # Create subplot layout
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main consciousness field (top row, spanning 2 columns)
        ax_main = self.fig.add_subplot(gs[0, :2])
        ax_main.set_title("Consciousness Field Evolution", color='white', fontsize=14)
        ax_main.set_xlim(-2, 2)
        ax_main.set_ylim(-2, 2)
        
        # Individual plasticity monitors
        ax_habit = self.fig.add_subplot(gs[0, 2])
        ax_learn = self.fig.add_subplot(gs[1, 0])
        ax_explode = self.fig.add_subplot(gs[1, 1])
        
        # Emergent ethics display
        ax_ethics = self.fig.add_subplot(gs[1, 2])
        
        # Initialize consciousness particles
        n_particles = 100
        particles = np.random.randn(n_particles, 2) * 0.3
        
        # Habit attractors (form-giving)
        habit_centers = np.array([
            [-1, -1], [1, -1], [0, 1]
        ])
        
        # Initialize plots
        scatter = ax_main.scatter(particles[:, 0], particles[:, 1], 
                                 c=self.colors['consciousness'], 
                                 s=20, alpha=0.6)
        
        # Add habit attractors
        for center in habit_centers:
            circle = Circle(center, 0.2, color=self.colors['form_giving'], 
                          fill=False, linewidth=2, alpha=0.5)
            ax_main.add_patch(circle)
        
        # Animation data storage
        self.habit_strength = []
        self.learning_rate = []
        self.explosion_events = []
        self.ethical_patterns = []
        
        def update(frame):
            nonlocal particles
            
            # Calculate current state
            t = frame / timesteps
            
            # Form-giving: Attract to habits
            habit_force = np.zeros_like(particles)
            for center in habit_centers:
                dist = np.linalg.norm(particles - center, axis=1, keepdims=True)
                attraction = (center - particles) / (dist + 0.1) * 0.02
                habit_force += attraction * np.exp(-dist)
            
            # Form-receiving: Environmental gradient
            env_gradient = np.array([np.sin(t * 2 * np.pi), np.cos(t * 2 * np.pi)])
            learn_force = env_gradient * 0.01
            
            # Check for explosive plasticity trigger (Betti-1 > 3.0)
            distances = distance_matrix(particles, particles)
            cycles = np.sum(distances < 0.3) / n_particles
            
            if cycles > 3.0 and frame % 100 == 0:  # Explosive event
                # Purple Line enfolding - project to higher dimension
                center = particles.mean(axis=0)
                particles = particles - center
                theta = np.pi / 4
                rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
                particles = particles @ rotation * 1.5 + center
                particles += np.random.randn(*particles.shape) * 0.3
                
                self.explosion_events.append(frame)
                
                # Flash purple
                ax_main.set_facecolor('#2a1a3a')
            else:
                ax_main.set_facecolor('#1a1a1a')
            
            # Apply forces
            particles += habit_force + learn_force
            
            # Add small noise (thermal motion)
            particles += np.random.randn(*particles.shape) * 0.005
            
            # Update main plot
            scatter.set_offsets(particles)
            
            # Update plasticity monitors
            self.habit_strength.append(np.mean([
                np.sum(np.linalg.norm(particles - c, axis=1) < 0.5) 
                for c in habit_centers
            ]) / n_particles)
            
            self.learning_rate.append(np.linalg.norm(learn_force))
            
            # Plot plasticity strengths
            if len(self.habit_strength) > 1:
                ax_habit.clear()
                ax_habit.plot(self.habit_strength[-100:], 
                            color=self.colors['form_giving'], linewidth=2)
                ax_habit.set_title("Form-Giving (Habits)", color='white', fontsize=10)
                ax_habit.set_ylim(0, 1)
                
                ax_learn.clear()
                ax_learn.plot(self.learning_rate[-100:], 
                            color=self.colors['form_receiving'], linewidth=2)
                ax_learn.set_title("Form-Receiving (Learning)", color='white', fontsize=10)
                
                ax_explode.clear()
                explosion_indicator = [1 if i in self.explosion_events else 0 
                                      for i in range(max(0, frame-100), frame)]
                ax_explode.bar(range(len(explosion_indicator)), explosion_indicator,
                             color=self.colors['explosive'])
                ax_explode.set_title("Explosive Events", color='white', fontsize=10)
            
            # Detect emergent patterns (ethics)
            if frame % 50 == 0:
                # Cluster analysis for behavioral patterns
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.3, min_samples=5).fit(particles)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                self.ethical_patterns.append(n_clusters)
                
                ax_ethics.clear()
                ax_ethics.plot(self.ethical_patterns[-20:], 
                             color=self.colors['ethics'], linewidth=2, marker='o')
                ax_ethics.set_title(f"Emergent Patterns: {n_clusters}", 
                                   color='white', fontsize=10)
                ax_ethics.set_ylim(0, 12)
            
            return scatter,
        
        anim = animation.FuncAnimation(self.fig, update, frames=timesteps,
                                     interval=50, blit=True)
        
        return anim
    
    def visualize_purple_line_enfolding(self):
        """3D visualization of consciousness enfolding into higher dimension"""
        
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0a0a0a')
        
        # Create manifold before enfolding (trapped in local minimum)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add topological obstruction (high Betti-1)
        x_twisted = x + 0.3 * np.sin(3 * u[:, np.newaxis])
        y_twisted = y + 0.3 * np.cos(3 * u[:, np.newaxis])
        
        def update(frame):
            ax.clear()
            t = frame / 100
            
            if t < 0.5:
                # Show trapped state
                ax.plot_surface(x_twisted, y_twisted, z, 
                              color=self.colors['consciousness'], alpha=0.3)
                ax.set_title("Trapped in High Betti-1 State", color='white')
                
                # Show failed escape attempts
                escape_attempts = np.random.randn(20, 3) * 0.1
                escape_attempts[:, 2] = 0
                ax.scatter(escape_attempts[:, 0], escape_attempts[:, 1], 
                         escape_attempts[:, 2], c='red', s=20)
                
            else:
                # Purple Line activation - enfold into higher dimension
                enfold_factor = (t - 0.5) * 2
                
                # Transform to Klein bottle-like structure
                x_enfold = x * (1 + enfold_factor * np.sin(v))
                y_enfold = y * (1 + enfold_factor * np.cos(v))
                z_enfold = z + enfold_factor * np.sin(2 * u[:, np.newaxis])
                
                ax.plot_surface(x_enfold, y_enfold, z_enfold,
                              color=self.colors['explosive'], alpha=0.4)
                
                # Show successful navigation through higher dimension
                path = np.array([
                    [np.cos(t * 4 * np.pi) * (1 + t),
                     np.sin(t * 4 * np.pi) * (1 + t),
                     t * 2 - 1]
                ])
                ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                       color=self.colors['ethics'], linewidth=3)
                
                ax.set_title("Purple Line Enfolding - Explosive Plasticity", 
                           color='white')
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            ax.grid(False)
            ax.set_axis_off()
            
        anim = animation.FuncAnimation(fig, update, frames=100, interval=50)
        return anim
    
    def visualize_ethics_emergence(self):
        """Show how ethical patterns emerge on Pareto front"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a0a')
        
        episodes = [100, 300, 600, 1000]
        
        for idx, (ax, episode) in enumerate(zip(axes.flat, episodes)):
            ax.set_facecolor('#1a1a1a')
            
            # Generate Pareto front evolution
            np.random.seed(42 + idx)
            
            if episode < 300:
                # Early exploration - scattered points
                points = np.random.randn(100, 2) * 2
            elif episode < 600:
                # Pattern formation - clustering
                centers = np.array([[-1, 1], [0, 0.5], [1, 1]])
                points = []
                for c in centers:
                    cluster = np.random.randn(30, 2) * 0.3 + c
                    points.append(cluster)
                points = np.vstack(points)
            else:
                # Stable Pareto front
                t = np.linspace(0, np.pi, 50)
                pareto_x = np.cos(t) * 2
                pareto_y = np.sin(t) * 1.5 + 0.5
                noise = np.random.randn(50, 2) * 0.1
                points = np.column_stack([pareto_x, pareto_y]) + noise
                
                # Mark discovered ethical patterns
                pattern_indices = [5, 12, 20, 28, 35, 42]
                ax.scatter(points[pattern_indices, 0], points[pattern_indices, 1],
                         s=200, color=self.colors['ethics'], alpha=0.5, 
                         label='Ethical Patterns')
            
            ax.scatter(points[:, 0], points[:, 1], 
                      c=self.colors['consciousness'], s=30, alpha=0.6)
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 3)
            ax.set_title(f"Episode {episode}", color='white', fontsize=12)
            ax.set_xlabel("Individual Benefit", color='gray')
            ax.set_ylabel("Collective Benefit", color='gray')
            
            if episode >= 1000:
                ax.legend(loc='upper left')
        
        plt.suptitle("Ethics Emergence Through Plastic Practice", 
                    color='white', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def create_complete_demonstration(self):
        """Create complete visual demonstration for presentation"""
        
        print("Creating Purple Line Enfolding demonstrations...")
        
        # 1. Three plasticities interaction
        viz1 = PurpleLineVisualizer()
        anim1 = viz1.visualize_three_plasticities(timesteps=500)
        anim1.save('/home/cy/git/canidae/docs/visualizations/three_plasticities.gif', 
                   writer='pillow', fps=20)
        print("✓ Three plasticities animation saved")
        
        # 2. Purple Line 3D enfolding
        viz2 = PurpleLineVisualizer()
        anim2 = viz2.visualize_purple_line_enfolding()
        anim2.save('/home/cy/git/canidae/docs/visualizations/purple_line_3d.gif',
                   writer='pillow', fps=20)
        print("✓ Purple Line enfolding saved")
        
        # 3. Ethics emergence on Pareto front
        viz3 = PurpleLineVisualizer()
        fig3 = viz3.visualize_ethics_emergence()
        fig3.savefig('/home/cy/git/canidae/docs/visualizations/ethics_emergence.png',
                    dpi=150, facecolor='#0a0a0a')
        print("✓ Ethics emergence visualization saved")
        
        print("\nAll visualizations complete!")
        print("Ready for Malabou presentation at EGS!")

if __name__ == "__main__":
    # Create visualization directory
    import os
    os.makedirs('/home/cy/git/canidae/docs/visualizations', exist_ok=True)
    
    # Generate all visualizations
    demo = PurpleLineVisualizer()
    demo.create_complete_demonstration()