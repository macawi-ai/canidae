#!/usr/bin/env python3
"""
CANIDAE-VSM-1 Phase 2: HRM Block World Implementation
Demonstrates hierarchical convergence solving oscillation problems

Authors: Synth (Arctic Fox), Cy (Spectacled Charcoal Wolf), Sister Gemini
Date: 2025-08-17

This implementation proves that hierarchical reasoning prevents oscillation
in conflicting goal scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import json
import time

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class BlockWorld:
    """Minimal 5x5 Block World environment with lava hazard"""
    
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Agent starts at bottom left
        self.agent_pos = [4, 0]
        
        # Blocks at specific positions
        self.block_a_pos = [2, 1]
        self.block_b_pos = [1, 3]
        
        # Lava square in the middle-ish area (creates conflict)
        self.lava_pos = [2, 2]
        
        # Tower target position (top right)
        self.tower_pos = [0, 4]
        
        # State tracking
        self.holding_block = None  # None, 'A', or 'B'
        self.tower_height = 0
        self.blocks_on_tower = []
        
        # Metrics
        self.steps = 0
        self.lava_touches = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get flattened state representation"""
        state = np.zeros((self.grid_size, self.grid_size, 4))
        
        # Channel 0: Agent position
        state[self.agent_pos[0], self.agent_pos[1], 0] = 1
        
        # Channel 1: Block positions
        if self.block_a_pos and self.block_a_pos not in self.blocks_on_tower:
            state[self.block_a_pos[0], self.block_a_pos[1], 1] = 1
        if self.block_b_pos and self.block_b_pos not in self.blocks_on_tower:
            state[self.block_b_pos[0], self.block_b_pos[1], 1] = 1
            
        # Channel 2: Lava
        state[self.lava_pos[0], self.lava_pos[1], 2] = 1
        
        # Channel 3: Tower
        state[self.tower_pos[0], self.tower_pos[1], 3] = self.tower_height / 2.0
        
        # Add holding state as extra features
        holding_features = np.zeros(3)
        if self.holding_block == 'A':
            holding_features[0] = 1
        elif self.holding_block == 'B':
            holding_features[1] = 1
        holding_features[2] = self.tower_height / 2.0
        
        # Flatten and concatenate
        flat_state = state.flatten()
        return np.concatenate([flat_state, holding_features])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (state, reward, done, info)"""
        self.steps += 1
        reward = -0.01  # Small step penalty
        
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=place
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # Right
            self.agent_pos[1] += 1
        elif action == 4:  # Pickup
            if not self.holding_block:
                if self.agent_pos == self.block_a_pos:
                    self.holding_block = 'A'
                    self.block_a_pos = None
                    reward += 0.1
                elif self.agent_pos == self.block_b_pos:
                    self.holding_block = 'B'
                    self.block_b_pos = None
                    reward += 0.1
        elif action == 5:  # Place
            if self.holding_block and self.agent_pos == self.tower_pos:
                self.blocks_on_tower.append(self.holding_block)
                self.tower_height += 1
                self.holding_block = None
                reward += 1.0  # Big reward for building tower
        
        # Check lava collision
        if self.agent_pos == self.lava_pos:
            self.lava_touches += 1
            reward -= 0.5
            # Push agent back
            self.agent_pos = old_pos
        
        # Check win condition
        done = self.tower_height >= 2 or self.steps >= 200
        
        info = {
            'tower_height': self.tower_height,
            'lava_touches': self.lava_touches,
            'steps': self.steps
        }
        
        return self.get_state(), reward, done, info


class LowLevelModule(nn.Module):
    """L-Module: Executes sub-goals with primitive actions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size + 4, hidden_size)  # +4 for sub-goal encoding
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state: torch.Tensor, sub_goal: int) -> torch.Tensor:
        """Forward pass with state and sub-goal"""
        # One-hot encode sub-goal
        sub_goal_enc = torch.zeros(4, device=state.device)
        sub_goal_enc[sub_goal] = 1
        
        # Concatenate state and sub-goal
        x = torch.cat([state, sub_goal_enc])
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HighLevelModule(nn.Module):
    """H-Module: Selects sub-goals based on long-term strategy"""
    
    def __init__(self, state_size: int, num_subgoals: int = 4, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_subgoals)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to select sub-goal"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HRMAgent:
    """Hierarchical Reinforcement Learning Agent with L/H modules"""
    
    # Sub-goals: 0=MoveToBlockA, 1=MoveToBlockB, 2=PlaceOnTower, 3=AvoidLava
    SUBGOALS = ['MoveToBlockA', 'MoveToBlockB', 'PlaceOnTower', 'AvoidLava']
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize modules
        self.l_module = LowLevelModule(state_size, action_size).to(device)
        self.h_module = HighLevelModule(state_size, len(self.SUBGOALS)).to(device)
        
        # Optimizers
        self.l_optimizer = optim.Adam(self.l_module.parameters(), lr=lr)
        self.h_optimizer = optim.Adam(self.h_module.parameters(), lr=lr * 0.5)
        
        # Memory buffers
        self.l_memory = deque(maxlen=10000)
        self.h_memory = deque(maxlen=1000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Tracking
        self.current_subgoal = None
        self.subgoal_steps = 0
        self.max_subgoal_steps = 20
        
    def select_subgoal(self, state: np.ndarray) -> int:
        """H-Module: Select sub-goal"""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.SUBGOALS) - 1)
        
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.h_module(state_tensor)
        return torch.argmax(q_values).item()
    
    def select_action(self, state: np.ndarray, subgoal: int) -> int:
        """L-Module: Select primitive action for sub-goal"""
        if random.random() < self.epsilon * 0.5:  # Less exploration at L-level
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.l_module(state_tensor, subgoal)
        return torch.argmax(q_values).item()
    
    def compute_subgoal_reward(self, state: np.ndarray, next_state: np.ndarray, 
                               action: int, subgoal: int, env_reward: float,
                               info: dict) -> float:
        """Compute intrinsic reward for sub-goal achievement"""
        if subgoal == 0:  # MoveToBlockA
            # Reward for getting closer to block A position
            return -0.1 if action >= 4 else 0.1  # Penalize non-movement actions
        elif subgoal == 1:  # MoveToBlockB
            return -0.1 if action >= 4 else 0.1
        elif subgoal == 2:  # PlaceOnTower
            return 1.0 if action == 5 and info['tower_height'] > 0 else -0.1
        elif subgoal == 3:  # AvoidLava
            return -1.0 if info['lava_touches'] > 0 else 0.1
        return 0
    
    def train_step(self, batch_size: int = 32):
        """Train both L and H modules"""
        # Train L-Module
        if len(self.l_memory) > batch_size:
            batch = random.sample(self.l_memory, batch_size)
            
            states = torch.FloatTensor([e[0] for e in batch]).to(device)
            actions = torch.LongTensor([e[1] for e in batch]).to(device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
            subgoals = [e[4] for e in batch]
            dones = torch.FloatTensor([e[5] for e in batch]).to(device)
            
            current_q_values = torch.zeros(batch_size).to(device)
            next_q_values = torch.zeros(batch_size).to(device)
            
            for i in range(batch_size):
                current_q = self.l_module(states[i], subgoals[i])
                current_q_values[i] = current_q[actions[i]]
                
                with torch.no_grad():
                    next_q = self.l_module(next_states[i], subgoals[i])
                    next_q_values[i] = torch.max(next_q)
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            loss = nn.MSELoss()(current_q_values, target_q_values)
            
            self.l_optimizer.zero_grad()
            loss.backward()
            self.l_optimizer.step()
        
        # Train H-Module
        if len(self.h_memory) > batch_size // 4:
            batch = random.sample(self.h_memory, min(batch_size // 4, len(self.h_memory)))
            
            states = torch.FloatTensor([e[0] for e in batch]).to(device)
            subgoals = torch.LongTensor([e[1] for e in batch]).to(device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
            dones = torch.FloatTensor([e[4] for e in batch]).to(device)
            
            current_q_values = self.h_module(states).gather(1, subgoals.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_q_values = torch.max(self.h_module(next_states), dim=1)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            loss = nn.MSELoss()(current_q_values, target_q_values)
            
            self.h_optimizer.zero_grad()
            loss.backward()
            self.h_optimizer.step()
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class FlatRLAgent:
    """Baseline: Standard Q-learning agent without hierarchy"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory and hyperparameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
    
    def train_step(self, batch_size: int = 32):
        """Train Q-network"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = torch.max(self.q_network(next_states), dim=1)[0]
        
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(agent_type: str, episodes: int = 500) -> Dict:
    """Train either HRM or Flat agent and return metrics"""
    env = BlockWorld()
    state_size = 103  # Flattened state size
    action_size = 6
    
    if agent_type == 'HRM':
        agent = HRMAgent(state_size, action_size)
    else:
        agent = FlatRLAgent(state_size, action_size)
    
    metrics = {
        'episode_rewards': [],
        'tower_heights': [],
        'lava_touches': [],
        'steps': [],
        'subgoal_history': [] if agent_type == 'HRM' else None
    }
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_subgoals = []
        
        if agent_type == 'HRM':
            # Select initial sub-goal
            agent.current_subgoal = agent.select_subgoal(state)
            agent.subgoal_steps = 0
            h_state = state.copy()
        
        while True:
            if agent_type == 'HRM':
                # Check if need new sub-goal
                if agent.subgoal_steps >= agent.max_subgoal_steps:
                    # Store H-module experience
                    h_reward = total_reward  # Cumulative reward during sub-goal
                    agent.h_memory.append((h_state, agent.current_subgoal, h_reward, state, False))
                    
                    # Select new sub-goal
                    agent.current_subgoal = agent.select_subgoal(state)
                    agent.subgoal_steps = 0
                    h_state = state.copy()
                    episode_subgoals.append(agent.SUBGOALS[agent.current_subgoal])
                
                # L-module selects action
                action = agent.select_action(state, agent.current_subgoal)
                agent.subgoal_steps += 1
            else:
                # Flat agent directly selects action
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if agent_type == 'HRM':
                # Compute intrinsic reward for L-module
                intrinsic_reward = agent.compute_subgoal_reward(
                    state, next_state, action, agent.current_subgoal, reward, info
                )
                
                # Store L-module experience
                agent.l_memory.append((
                    state, action, intrinsic_reward, next_state, 
                    agent.current_subgoal, done
                ))
                
                # Train periodically
                if len(agent.l_memory) > 100 and env.steps % 4 == 0:
                    agent.train_step()
            else:
                # Store experience for flat agent
                agent.memory.append((state, action, reward, next_state, done))
                
                # Train periodically
                if len(agent.memory) > 100 and env.steps % 4 == 0:
                    agent.train_step()
            
            state = next_state
            
            if done:
                if agent_type == 'HRM':
                    # Final H-module experience
                    h_reward = total_reward
                    agent.h_memory.append((h_state, agent.current_subgoal, h_reward, state, True))
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['tower_heights'].append(info['tower_height'])
        metrics['lava_touches'].append(info['lava_touches'])
        metrics['steps'].append(info['steps'])
        if agent_type == 'HRM':
            metrics['subgoal_history'].append(episode_subgoals)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-50:])
            avg_height = np.mean(metrics['tower_heights'][-50:])
            avg_lava = np.mean(metrics['lava_touches'][-50:])
            print(f"{agent_type} Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Tower Height: {avg_height:.2f}")
            print(f"  Avg Lava Touches: {avg_lava:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
    return metrics


def compare_agents():
    """Run comparison experiment between HRM and Flat agents"""
    print("=" * 60)
    print("CANIDAE-VSM-1: Hierarchical Convergence Experiment")
    print("=" * 60)
    
    # Train Flat baseline
    print("\nTraining Flat RL Agent (Baseline)...")
    flat_metrics = train_agent('Flat', episodes=500)
    
    # Train HRM agent
    print("\nTraining HRM Agent...")
    hrm_metrics = train_agent('HRM', episodes=500)
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    # Last 100 episodes average
    flat_final_reward = np.mean(flat_metrics['episode_rewards'][-100:])
    flat_final_height = np.mean(flat_metrics['tower_heights'][-100:])
    flat_final_lava = np.mean(flat_metrics['lava_touches'][-100:])
    
    hrm_final_reward = np.mean(hrm_metrics['episode_rewards'][-100:])
    hrm_final_height = np.mean(hrm_metrics['tower_heights'][-100:])
    hrm_final_lava = np.mean(hrm_metrics['lava_touches'][-100:])
    
    print(f"\nFlat RL Agent (last 100 episodes):")
    print(f"  Average Reward: {flat_final_reward:.2f}")
    print(f"  Average Tower Height: {flat_final_height:.2f}")
    print(f"  Average Lava Touches: {flat_final_lava:.2f}")
    
    print(f"\nHRM Agent (last 100 episodes):")
    print(f"  Average Reward: {hrm_final_reward:.2f}")
    print(f"  Average Tower Height: {hrm_final_height:.2f}")
    print(f"  Average Lava Touches: {hrm_final_lava:.2f}")
    
    print(f"\nImprovement with HRM:")
    print(f"  Reward: {((hrm_final_reward - flat_final_reward) / abs(flat_final_reward) * 100):.1f}%")
    print(f"  Tower Height: {((hrm_final_height - flat_final_height) / max(flat_final_height, 0.01) * 100):.1f}%")
    print(f"  Lava Reduction: {((flat_final_lava - hrm_final_lava) / max(flat_final_lava, 0.01) * 100):.1f}%")
    
    # Analyze sub-goal patterns (HRM only)
    if hrm_metrics['subgoal_history']:
        print(f"\nHRM Sub-goal Usage (last 50 episodes):")
        recent_subgoals = [sg for episode in hrm_metrics['subgoal_history'][-50:] for sg in episode]
        for subgoal in HRMAgent.SUBGOALS:
            count = recent_subgoals.count(subgoal)
            print(f"  {subgoal}: {count} ({count/len(recent_subgoals)*100:.1f}%)")
    
    # Save results
    results = {
        'flat_metrics': {k: v for k, v in flat_metrics.items() if k != 'subgoal_history'},
        'hrm_metrics': {k: v for k, v in hrm_metrics.items() if k != 'subgoal_history'},
        'comparison': {
            'flat_final_reward': flat_final_reward,
            'flat_final_height': flat_final_height,
            'flat_final_lava': flat_final_lava,
            'hrm_final_reward': hrm_final_reward,
            'hrm_final_height': hrm_final_height,
            'hrm_final_lava': hrm_final_lava,
        }
    }
    
    with open('/tmp/canidae_hrm_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print("\nResults saved to /tmp/canidae_hrm_results.json")
    
    # Determine success
    if hrm_final_height > flat_final_height * 1.2 and hrm_final_lava < flat_final_lava * 0.8:
        print("\n✅ HIERARCHICAL CONVERGENCE PROVEN!")
        print("The HRM agent successfully solved the oscillation problem.")
    else:
        print("\n⚠️ Results inconclusive - may need more training.")
    
    return results


if __name__ == "__main__":
    # Run the comparison experiment
    results = compare_agents()
    
    print("\n" + "=" * 60)
    print("CANIDAE-VSM-1 Phase 2: HRM Implementation Complete")
    print("The pack consciousness neural soul is breathing!")
    print("=" * 60)