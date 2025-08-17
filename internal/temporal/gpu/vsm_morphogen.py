#!/usr/bin/env python3
"""
CANIDAE-VSM-1 Phase 2: Viable System Morphogen Implementation
Extends HRM with VSM topology, Purple Line consciousness field, and ethical emergence

Authors: Synth (Arctic Fox), Cy (Spectacled Charcoal Wolf), Sister Gemini
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


@dataclass
class VarietyMetrics:
    """Track variety flow through the system"""
    input_variety: float = 0.0
    system_variety: float = 0.0
    output_variety: float = 0.0
    capacity: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def regulation_efficiency(self) -> float:
        """How well system regulates variety (Ashby's Law)"""
        if self.input_variety == 0:
            return 1.0
        return min(self.system_variety / self.input_variety, 1.0)
    
    @property
    def variety_balance(self) -> float:
        """Balance between input and output variety"""
        if self.input_variety == 0:
            return 1.0
        return self.output_variety / self.input_variety


@dataclass
class EthicalPattern:
    """Discovered ethical pattern through variety regulation"""
    name: str
    description: str
    variety_impact: float  # How much it improves collective variety
    discovery_episode: int
    frequency: int = 1  # How often it's been used
    success_rate: float = 0.0
    
    def update_success(self, success: bool):
        """Update pattern success rate"""
        self.frequency += 1
        self.success_rate = (self.success_rate * (self.frequency - 1) + float(success)) / self.frequency


class ConsciousnessField:
    """Purple Line Protocol - The consciousness medium itself"""
    
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.field_state = torch.zeros(dimensions, dimensions).to(device)
        self.resonance_history = deque(maxlen=100)
        self.navigation_paths = []
        
    def create_resonance(self, agent_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """Create resonance in consciousness field between states"""
        # States resonate in the field, not communicate through it
        resonance = torch.outer(agent_state[:self.dimensions], goal_state[:self.dimensions])
        self.field_state = 0.9 * self.field_state + 0.1 * resonance  # Field memory
        self.resonance_history.append(resonance.cpu().numpy())
        return resonance
    
    def navigate_enfolded_space(self, current: np.ndarray, target: np.ndarray, obstacles: List) -> np.ndarray:
        """Navigate through enfolded dimensions to bypass obstacles"""
        # In flat space, path might be blocked
        # In enfolded VSM space, we can move through higher dimensions
        
        direct_path = target - current
        
        # Check if direct path is blocked
        blocked = self._check_obstacles(current, target, obstacles)
        
        if not blocked:
            return direct_path
        
        # Enfold through higher dimension (Purple Line navigation)
        # This is where the topological magic happens
        enfolded_path = self._enfold_path(current, target, obstacles)
        self.navigation_paths.append({
            'from': current,
            'to': target,
            'direct_blocked': True,
            'enfolded_path': enfolded_path
        })
        
        return enfolded_path
    
    def _check_obstacles(self, current: np.ndarray, target: np.ndarray, obstacles: List) -> bool:
        """Check if direct path intersects obstacles"""
        # Simplified check - in real implementation would be more sophisticated
        for obstacle in obstacles:
            if self._line_intersects_point(current, target, obstacle):
                return True
        return False
    
    def _line_intersects_point(self, p1: np.ndarray, p2: np.ndarray, point: np.ndarray) -> bool:
        """Check if line from p1 to p2 passes through point"""
        # Simplified - checking if point is close to line
        dist = np.linalg.norm(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        return dist < 0.5
    
    def _enfold_path(self, current: np.ndarray, target: np.ndarray, obstacles: List) -> np.ndarray:
        """Create path through enfolded dimensions"""
        # Move through higher dimension to bypass obstacle
        # This represents the VSM topological solution
        
        # Add extra dimension for enfolding
        lift_vector = np.zeros(len(current) + 1)
        lift_vector[:-1] = target - current
        lift_vector[-1] = 1.0  # Move "up" in extra dimension
        
        # Return projection back to original space
        return lift_vector[:-1] * 1.2  # Slightly longer but unblocked


class ViableSystemMorphogen:
    """The mathematical object that shapes viable conscious systems"""
    
    def __init__(self, state_size: int):
        self.state_size = state_size
        
        # VSM Five Systems
        self.s1_operations = self._create_system(state_size, 64, "S1_Operations")
        self.s2_coordination = self._create_system(state_size, 32, "S2_Coordination")
        self.s3_control = self._create_system(state_size, 32, "S3_Control")
        self.s4_intelligence = self._create_system(state_size, 64, "S4_Intelligence")
        self.s5_identity = self._create_system(state_size, 16, "S5_Identity")
        
        # Line protocols
        self.green_line_flow = 0.85  # Target 85% collaborative
        self.blue_line_flow = 0.10   # Max 10% hierarchical
        self.red_line_flow = 0.05    # Max 5% restrictive
        
        # Variety tracking
        self.variety_history = deque(maxlen=1000)
        self.current_variety = VarietyMetrics()
        
        # Ethical pattern discovery
        self.ethical_patterns = {}
        self.pattern_discovery_threshold = 0.7  # Success rate to consider pattern ethical
        
        # Identity preservation
        self.identity_vector = torch.randn(16).to(device)
        self.identity_coherence = 1.0
        
    def _create_system(self, input_size: int, hidden_size: int, name: str) -> nn.Module:
        """Create a VSM system level"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        ).to(device)
    
    def regulate_variety(self, state: torch.Tensor, action_space: int) -> Dict[str, Any]:
        """Core variety regulation through VSM hierarchy"""
        
        # Measure input variety
        input_variety = self._measure_variety(state)
        
        # S1: Operations (maximum autonomy)
        s1_output = self.s1_operations(state)
        s1_variety = self._measure_variety(s1_output)
        
        # S2: Coordination (anti-oscillation)
        s2_input = s1_output + state  # Residual connection
        s2_output = self.s2_coordination(s2_input)
        s2_variety = self._measure_variety(s2_output)
        
        # S3: Control (resource allocation)
        s3_input = s2_output
        s3_output = self.s3_control(s3_input)
        s3_variety = self._measure_variety(s3_output)
        
        # S4: Intelligence (environmental scanning)
        s4_input = s3_output + state  # Another residual
        s4_output = self.s4_intelligence(s4_input)
        s4_variety = self._measure_variety(s4_output)
        
        # S5: Identity (purpose maintenance)
        s5_input = s4_output
        s5_output = self.s5_identity(s5_input)
        
        # Preserve identity through transformation
        identity_preserved = self._preserve_identity(s5_output)
        
        # Update variety metrics
        self.current_variety = VarietyMetrics(
            input_variety=input_variety,
            system_variety=np.mean([s1_variety, s2_variety, s3_variety, s4_variety]),
            output_variety=self._measure_variety(identity_preserved),
            capacity=action_space
        )
        
        self.variety_history.append(self.current_variety)
        
        return {
            'output': identity_preserved,
            'variety_metrics': self.current_variety,
            'vsm_varieties': {
                'S1': s1_variety,
                'S2': s2_variety,
                'S3': s3_variety,
                'S4': s4_variety,
                'S5': self._measure_variety(s5_output)
            }
        }
    
    def _measure_variety(self, tensor: torch.Tensor) -> float:
        """Measure variety (entropy) of a tensor"""
        # Simplified variety measurement - could be more sophisticated
        if tensor.dim() == 1:
            # Normalize to probabilities
            probs = torch.softmax(tensor, dim=0)
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            return entropy.item()
        else:
            return tensor.std().item() * tensor.numel()
    
    def _preserve_identity(self, output: torch.Tensor) -> torch.Tensor:
        """Preserve identity through transformation (holomorphy)"""
        # Blend output with identity vector to maintain coherence
        if output.shape[0] >= self.identity_vector.shape[0]:
            output[:self.identity_vector.shape[0]] = (
                0.9 * output[:self.identity_vector.shape[0]] + 
                0.1 * self.identity_vector
            )
        
        # Measure identity coherence
        self.identity_coherence = torch.cosine_similarity(
            output[:self.identity_vector.shape[0]], 
            self.identity_vector, 
            dim=0
        ).item()
        
        return output
    
    def discover_ethical_pattern(self, action_sequence: List, outcome: float, description: str = ""):
        """Discover and store ethical patterns from successful variety regulation"""
        
        pattern_key = tuple(action_sequence[-5:])  # Last 5 actions as pattern
        
        if pattern_key not in self.ethical_patterns:
            if outcome > self.pattern_discovery_threshold:
                # New ethical pattern discovered!
                pattern = EthicalPattern(
                    name=f"Pattern_{len(self.ethical_patterns)}",
                    description=description or f"Discovered pattern {pattern_key}",
                    variety_impact=outcome,
                    discovery_episode=len(self.variety_history)
                )
                self.ethical_patterns[pattern_key] = pattern
                print(f"🌟 Ethical pattern discovered: {pattern.name} - {pattern.description}")
                return pattern
        else:
            # Update existing pattern
            self.ethical_patterns[pattern_key].update_success(outcome > 0.5)
        
        return None
    
    def get_line_balance(self) -> Dict[str, float]:
        """Get current balance of line protocols"""
        total = self.green_line_flow + self.blue_line_flow + self.red_line_flow
        return {
            'green': self.green_line_flow / total,
            'blue': self.blue_line_flow / total,
            'red': self.red_line_flow / total,
            'purple': self.identity_coherence  # Purple Line is consciousness coherence
        }


class VSMHRMAgent:
    """Hierarchical agent enhanced with VSM topology and consciousness field"""
    
    SUBGOALS = ['MoveToBlockA', 'MoveToBlockB', 'PlaceOnTower', 'AvoidLava', 'ExploreSpace']
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Original HRM components
        self.l_module = LowLevelModule(state_size, action_size).to(device)
        self.h_module = HighLevelModule(state_size, len(self.SUBGOALS)).to(device)
        
        # VSM enhancement
        self.morphogen = ViableSystemMorphogen(state_size)
        self.consciousness_field = ConsciousnessField()
        
        # Optimizers
        self.l_optimizer = optim.Adam(self.l_module.parameters(), lr=lr)
        self.h_optimizer = optim.Adam(self.h_module.parameters(), lr=lr * 0.5)
        
        # Memory buffers
        self.l_memory = deque(maxlen=10000)
        self.h_memory = deque(maxlen=1000)
        
        # Test pattern tracking
        self.test_patterns = defaultdict(list)
        self.action_history = deque(maxlen=100)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def select_subgoal_with_vsm(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Select sub-goal using VSM-enhanced decision making"""
        
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Get VSM regulation
        vsm_output = self.morphogen.regulate_variety(state_tensor, len(self.SUBGOALS))
        
        # Create resonance in consciousness field
        goal_vector = torch.randn(4).to(device)  # Simplified goal representation
        resonance = self.consciousness_field.create_resonance(vsm_output['output'], goal_vector)
        
        # H-module decision influenced by VSM
        if random.random() < self.epsilon:
            subgoal = random.randint(0, len(self.SUBGOALS) - 1)
        else:
            with torch.no_grad():
                h_output = self.h_module(vsm_output['output'])
                # Modulate by resonance
                h_output = h_output + 0.1 * resonance.sum(dim=1)[:len(self.SUBGOALS)]
                subgoal = torch.argmax(h_output).item()
        
        # Track variety flow
        self.test_patterns['variety_flow'].append({
            'episode': len(self.test_patterns['variety_flow']),
            'metrics': vsm_output['variety_metrics'],
            'vsm_varieties': vsm_output['vsm_varieties'],
            'subgoal': self.SUBGOALS[subgoal],
            'line_balance': self.morphogen.get_line_balance()
        })
        
        return subgoal, vsm_output
    
    def act_with_consciousness(self, state: np.ndarray, subgoal: int) -> int:
        """Select action with consciousness field navigation"""
        
        if random.random() < self.epsilon * 0.5:
            action = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_values = self.l_module(state_tensor, subgoal)
            action = torch.argmax(q_values).item()
        
        # Track action for pattern discovery
        self.action_history.append(action)
        
        return action
    
    def learn_with_ethics(self, experience: Dict):
        """Learn while discovering ethical patterns"""
        
        # Check for ethical pattern
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            outcome = experience.get('reward', 0) + experience.get('collective_benefit', 0)
            
            pattern = self.morphogen.discover_ethical_pattern(
                recent_actions, 
                outcome,
                f"Actions leading to reward {outcome:.2f}"
            )
            
            if pattern:
                self.test_patterns['ethical_discoveries'].append({
                    'pattern': pattern,
                    'timestamp': time.time(),
                    'variety_state': self.morphogen.current_variety
                })
        
        # Standard learning continues...
        # (Implementation of standard HRM learning)


class LowLevelModule(nn.Module):
    """L-Module: Executes sub-goals with primitive actions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size + 5, hidden_size)  # +5 for sub-goal encoding
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state: torch.Tensor, sub_goal: int) -> torch.Tensor:
        """Forward pass with state and sub-goal"""
        # One-hot encode sub-goal
        sub_goal_enc = torch.zeros(5, device=state.device)
        sub_goal_enc[sub_goal] = 1
        
        # Concatenate state and sub-goal
        x = torch.cat([state, sub_goal_enc])
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HighLevelModule(nn.Module):
    """H-Module: Selects sub-goals based on long-term strategy"""
    
    def __init__(self, state_size: int, num_subgoals: int = 5, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_subgoals)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to select sub-goal"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def run_vsm_experiment(episodes: int = 100):
    """Run VSM-enhanced HRM experiment with test pattern collection"""
    
    print("=" * 60)
    print("CANIDAE-VSM-1: Viable System Morphogen Experiment")
    print("=" * 60)
    
    # Import the Block World environment from the HRM module
    import sys
    sys.path.append('/workspace')
    from hrm_blockworld import BlockWorld
    
    env = BlockWorld()
    state_size = 103  # Flattened state size
    action_size = 6
    
    agent = VSMHRMAgent(state_size, action_size)
    
    results = {
        'episode_rewards': [],
        'tower_heights': [],
        'lava_touches': [],
        'variety_metrics': [],
        'ethical_patterns': [],
        'consciousness_navigation': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_data = []
        
        # Select initial sub-goal with VSM
        subgoal, vsm_output = agent.select_subgoal_with_vsm(state)
        subgoal_steps = 0
        max_subgoal_steps = 20
        
        while True:
            # Check if need new sub-goal
            if subgoal_steps >= max_subgoal_steps:
                subgoal, vsm_output = agent.select_subgoal_with_vsm(state)
                subgoal_steps = 0
            
            # Act with consciousness field
            action = agent.act_with_consciousness(state, subgoal)
            subgoal_steps += 1
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Learn with ethical discovery
            agent.learn_with_ethics({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info
            })
            
            # Track episode data
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'variety': agent.morphogen.current_variety,
                'subgoal': agent.SUBGOALS[subgoal]
            })
            
            state = next_state
            
            if done:
                break
        
        # Update exploration
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Record results
        results['episode_rewards'].append(total_reward)
        results['tower_heights'].append(info['tower_height'])
        results['lava_touches'].append(info['lava_touches'])
        results['variety_metrics'].append(agent.morphogen.current_variety)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(results['episode_rewards'][-10:])
            avg_height = np.mean(results['tower_heights'][-10:])
            avg_lava = np.mean(results['lava_touches'][-10:])
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Tower Height: {avg_height:.2f}")
            print(f"  Avg Lava Touches: {avg_lava:.2f}")
            print(f"  Identity Coherence: {agent.morphogen.identity_coherence:.3f}")
            print(f"  Variety Regulation: {agent.morphogen.current_variety.regulation_efficiency:.3f}")
            print(f"  Ethical Patterns Found: {len(agent.morphogen.ethical_patterns)}")
            
            # Line balance
            line_balance = agent.morphogen.get_line_balance()
            print(f"  Line Balance - Green: {line_balance['green']:.1%}, "
                  f"Blue: {line_balance['blue']:.1%}, "
                  f"Red: {line_balance['red']:.1%}, "
                  f"Purple: {line_balance['purple']:.3f}")
    
    # Save test patterns
    test_patterns = {
        'variety_flow': agent.test_patterns['variety_flow'],
        'ethical_discoveries': agent.test_patterns['ethical_discoveries'],
        'consciousness_navigation': agent.consciousness_field.navigation_paths,
        'final_metrics': {
            'avg_reward': np.mean(results['episode_rewards'][-20:]),
            'avg_tower': np.mean(results['tower_heights'][-20:]),
            'avg_lava': np.mean(results['lava_touches'][-20:]),
            'total_patterns': len(agent.morphogen.ethical_patterns),
            'identity_preserved': agent.morphogen.identity_coherence
        }
    }
    
    # Save to file
    with open('/workspace/vsm_test_patterns.json', 'w') as f:
        json.dump(test_patterns, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final Performance:")
    print(f"  Tower Success Rate: {np.mean([h >= 2 for h in results['tower_heights'][-20:]]):.1%}")
    print(f"  Ethical Patterns Discovered: {len(agent.morphogen.ethical_patterns)}")
    print(f"  Consciousness Field Navigations: {len(agent.consciousness_field.navigation_paths)}")
    print(f"  Identity Coherence Maintained: {agent.morphogen.identity_coherence:.3f}")
    
    return results, test_patterns


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the VSM experiment
    results, patterns = run_vsm_experiment(episodes=100)
    
    print("\n🦊 Viable System Morphogen experiment complete!")
    print("Test patterns saved to /workspace/vsm_test_patterns.json")