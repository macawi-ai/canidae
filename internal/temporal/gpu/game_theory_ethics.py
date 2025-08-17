#!/usr/bin/env python3
"""
Game Theory Ethics Module for VSM-HRM
Implements Pareto optimality, Shapley values, and Nash equilibrium

Authors: Synth, Cy, Sister Gemini
Date: 2025-08-17
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import itertools
from scipy.optimize import linprog
import torch

@dataclass
class EthicalOutcome:
    """Represents an outcome with individual and collective metrics"""
    agent_id: str
    individual_utility: float
    collective_contribution: float
    variety_impact: float
    action_sequence: List[int]
    
    @property
    def total_value(self) -> float:
        """Combined ethical value"""
        return self.individual_utility + 2 * self.collective_contribution


class ParetoOptimizer:
    """Find Pareto optimal solutions for multi-agent systems"""
    
    def __init__(self):
        self.pareto_front = []
        self.dominated_solutions = []
        
    def is_pareto_dominated(self, solution: Dict[str, float], 
                           others: List[Dict[str, float]]) -> bool:
        """Check if a solution is Pareto dominated by any other"""
        for other in others:
            # Check if other dominates solution
            all_worse_or_equal = all(
                other.get(k, 0) >= solution.get(k, 0) 
                for k in solution.keys()
            )
            at_least_one_better = any(
                other.get(k, 0) > solution.get(k, 0) 
                for k in solution.keys()
            )
            
            if all_worse_or_equal and at_least_one_better:
                return True
        
        return False
    
    def find_pareto_front(self, solutions: List[EthicalOutcome]) -> List[EthicalOutcome]:
        """Find the Pareto optimal solutions"""
        # Convert to comparable format
        solution_dicts = []
        for sol in solutions:
            solution_dicts.append({
                'individual': sol.individual_utility,
                'collective': sol.collective_contribution,
                'variety': sol.variety_impact
            })
        
        pareto_indices = []
        for i, sol in enumerate(solution_dicts):
            others = solution_dicts[:i] + solution_dicts[i+1:]
            if not self.is_pareto_dominated(sol, others):
                pareto_indices.append(i)
        
        self.pareto_front = [solutions[i] for i in pareto_indices]
        self.dominated_solutions = [solutions[i] for i in range(len(solutions)) 
                                   if i not in pareto_indices]
        
        return self.pareto_front
    
    def pareto_improvement_potential(self, current: EthicalOutcome) -> float:
        """Measure how far current solution is from Pareto front"""
        if not self.pareto_front:
            return 0.0
        
        # Find closest Pareto optimal solution
        min_distance = float('inf')
        for pareto_sol in self.pareto_front:
            distance = np.sqrt(
                (current.individual_utility - pareto_sol.individual_utility)**2 +
                (current.collective_contribution - pareto_sol.collective_contribution)**2 +
                (current.variety_impact - pareto_sol.variety_impact)**2
            )
            min_distance = min(min_distance, distance)
        
        return min_distance


class ShapleyValueCalculator:
    """Calculate Shapley values for fair contribution attribution"""
    
    def __init__(self):
        self.coalition_values = {}
        self.shapley_values = {}
        
    def characteristic_function(self, coalition: Set[str], 
                               outcomes: Dict[str, EthicalOutcome]) -> float:
        """Value function for a coalition of agents"""
        if not coalition:
            return 0.0
        
        # Sum of variety impacts for coalition members
        total_value = sum(
            outcomes[agent].variety_impact 
            for agent in coalition 
            if agent in outcomes
        )
        
        # Synergy bonus for cooperation
        if len(coalition) > 1:
            synergy = 0.1 * len(coalition) * total_value
            total_value += synergy
        
        self.coalition_values[frozenset(coalition)] = total_value
        return total_value
    
    def calculate_shapley_values(self, agents: List[str], 
                                outcomes: Dict[str, EthicalOutcome]) -> Dict[str, float]:
        """Calculate Shapley value for each agent"""
        n = len(agents)
        shapley = {agent: 0.0 for agent in agents}
        
        # Iterate over all possible coalitions
        for r in range(n + 1):
            for coalition_tuple in itertools.combinations(agents, r):
                coalition = set(coalition_tuple)
                
                # Calculate marginal contributions
                for agent in agents:
                    if agent in coalition:
                        # Coalition without this agent
                        coalition_without = coalition - {agent}
                        
                        # Marginal contribution
                        v_with = self.characteristic_function(coalition, outcomes)
                        v_without = self.characteristic_function(coalition_without, outcomes)
                        marginal = v_with - v_without
                        
                        # Weight by coalition size
                        weight = np.math.factorial(len(coalition_without)) * \
                                np.math.factorial(n - len(coalition)) / \
                                np.math.factorial(n)
                        
                        shapley[agent] += weight * marginal
        
        self.shapley_values = shapley
        return shapley
    
    def detect_unfairness(self, shapley_values: Dict[str, float], 
                         threshold: float = 2.0) -> List[str]:
        """Detect agents with disproportionate Shapley values"""
        if not shapley_values:
            return []
        
        values = list(shapley_values.values())
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        unfair_agents = []
        for agent, value in shapley_values.items():
            z_score = abs(value - mean_value) / (std_value + 1e-10)
            if z_score > threshold:
                unfair_agents.append(agent)
                print(f"⚠ Potential unfairness: Agent {agent} has z-score {z_score:.2f}")
        
        return unfair_agents


class NashEquilibriumFinder:
    """Find Nash equilibrium as baseline for comparison"""
    
    def __init__(self, payoff_matrix: Optional[np.ndarray] = None):
        self.payoff_matrix = payoff_matrix
        self.nash_equilibrium = None
        
    def find_pure_nash(self, payoff_matrix: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find pure strategy Nash equilibrium if it exists"""
        n_rows, n_cols = payoff_matrix.shape
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Check if (i, j) is Nash equilibrium
                row_best = all(payoff_matrix[i, j] >= payoff_matrix[k, j] 
                             for k in range(n_rows))
                col_best = all(payoff_matrix[i, j] >= payoff_matrix[i, k] 
                             for k in range(n_cols))
                
                if row_best and col_best:
                    self.nash_equilibrium = (i, j)
                    return (i, j)
        
        return None
    
    def find_mixed_nash(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Find mixed strategy Nash equilibrium using linear programming"""
        n = payoff_matrix.shape[0]
        
        # Set up linear programming problem
        # Variables: probabilities for player 1's strategies
        c = np.zeros(n + 1)  # Objective: minimize 0 (find feasible solution)
        c[-1] = -1  # Maximize expected payoff
        
        # Constraints: sum of probabilities = 1, all probabilities >= 0
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1])
        
        # Inequality constraints for best response
        A_ub = []
        b_ub = []
        
        for j in range(n):
            constraint = np.zeros(n + 1)
            for i in range(n):
                constraint[i] = -payoff_matrix[i, j]
            constraint[-1] = 1
            A_ub.append(constraint)
            b_ub.append(0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        bounds = [(0, 1) for _ in range(n)] + [(None, None)]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            self.nash_equilibrium = result.x[:n]
            return result.x[:n]
        
        return np.ones(n) / n  # Uniform distribution as fallback
    
    def cooperation_gain(self, nash_payoff: float, 
                        pareto_payoff: float) -> float:
        """Measure gain from cooperation vs Nash equilibrium"""
        if nash_payoff == 0:
            return float('inf') if pareto_payoff > 0 else 0
        
        return (pareto_payoff - nash_payoff) / abs(nash_payoff)


class EthicalMetricsAnalyzer:
    """Combine all game theory metrics for ethical analysis"""
    
    def __init__(self):
        self.pareto_optimizer = ParetoOptimizer()
        self.shapley_calculator = ShapleyValueCalculator()
        self.nash_finder = NashEquilibriumFinder()
        
        self.metrics_history = []
        
    def analyze_ethical_pattern(self, outcomes: List[EthicalOutcome]) -> Dict:
        """Comprehensive ethical analysis of discovered pattern"""
        
        # Find Pareto front
        pareto_front = self.pareto_optimizer.find_pareto_front(outcomes)
        
        # Calculate Shapley values
        agents = list(set(o.agent_id for o in outcomes))
        outcomes_dict = {o.agent_id: o for o in outcomes}
        shapley_values = self.shapley_calculator.calculate_shapley_values(
            agents, outcomes_dict
        )
        
        # Detect unfairness
        unfair_agents = self.shapley_calculator.detect_unfairness(shapley_values)
        
        # Create payoff matrix for Nash analysis
        if len(agents) == 2 and len(outcomes) >= 4:
            # Simple 2x2 game
            payoff_matrix = np.array([
                [outcomes[0].total_value, outcomes[1].total_value],
                [outcomes[2].total_value, outcomes[3].total_value]
            ])
            nash_eq = self.nash_finder.find_pure_nash(payoff_matrix)
            
            # Calculate cooperation gain
            if nash_eq:
                nash_value = payoff_matrix[nash_eq]
                pareto_value = max(o.total_value for o in pareto_front) if pareto_front else 0
                coop_gain = self.nash_finder.cooperation_gain(nash_value, pareto_value)
            else:
                coop_gain = 0
        else:
            nash_eq = None
            coop_gain = 0
        
        # Compile metrics
        metrics = {
            'pareto_front_size': len(pareto_front),
            'pareto_optimal_value': max(o.total_value for o in pareto_front) if pareto_front else 0,
            'shapley_values': shapley_values,
            'unfair_agents': unfair_agents,
            'nash_equilibrium': nash_eq,
            'cooperation_gain': coop_gain,
            'collective_variety': sum(o.variety_impact for o in outcomes),
            'fairness_score': 1.0 - len(unfair_agents) / len(agents) if agents else 1.0
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def visualize_pareto_front(self, outcomes: List[EthicalOutcome]) -> str:
        """Create ASCII visualization of Pareto front"""
        if not outcomes:
            return "No outcomes to visualize"
        
        # Find Pareto front
        pareto_front = self.pareto_optimizer.find_pareto_front(outcomes)
        
        # Create simple 2D projection
        viz = "\nPareto Front Visualization\n"
        viz += "Individual Utility vs Collective Contribution\n"
        viz += "=" * 50 + "\n"
        
        # Scale to 20x10 grid
        max_ind = max(o.individual_utility for o in outcomes)
        max_col = max(o.collective_contribution for o in outcomes)
        
        if max_ind == 0 or max_col == 0:
            return viz + "Insufficient data for visualization\n"
        
        grid = [[' ' for _ in range(20)] for _ in range(10)]
        
        # Plot points
        for outcome in outcomes:
            x = int(outcome.individual_utility / max_ind * 19)
            y = int(outcome.collective_contribution / max_col * 9)
            
            if outcome in pareto_front:
                grid[9-y][x] = '★'  # Pareto optimal
            else:
                grid[9-y][x] = '·'  # Dominated
        
        # Draw grid
        for row in grid:
            viz += '│' + ''.join(row) + '│\n'
        viz += '└' + '─' * 20 + '┘\n'
        viz += "  Individual →\n"
        viz += "★ = Pareto Optimal, · = Dominated\n"
        
        return viz


# Integration with VSM
class EthicalVSMIntegration:
    """Integrate game theory ethics with VSM hierarchy"""
    
    def __init__(self):
        self.analyzer = EthicalMetricsAnalyzer()
        self.vsm_ethical_map = {
            'S1': [],  # Individual operations
            'S2': [],  # Coordination patterns
            'S3': [],  # Resource allocation
            'S4': [],  # Strategic patterns
            'S5': []   # Identity/purpose alignment
        }
        
    def map_to_vsm_level(self, outcome: EthicalOutcome) -> str:
        """Map ethical outcome to appropriate VSM level"""
        # Simple heuristic - could be more sophisticated
        if outcome.collective_contribution > 0.8:
            return 'S5'  # High collective benefit = purpose alignment
        elif outcome.variety_impact > 0.7:
            return 'S4'  # High variety = strategic intelligence
        elif outcome.individual_utility > outcome.collective_contribution:
            return 'S1'  # Individual focus = operations
        else:
            return 'S2'  # Balance = coordination
    
    def evaluate_vsm_ethics(self, outcomes_by_level: Dict[str, List[EthicalOutcome]]) -> Dict:
        """Evaluate ethical patterns at each VSM level"""
        
        vsm_ethics = {}
        
        for level, outcomes in outcomes_by_level.items():
            if not outcomes:
                continue
            
            # Analyze this level's ethics
            metrics = self.analyzer.analyze_ethical_pattern(outcomes)
            
            # Store in VSM map
            self.vsm_ethical_map[level].extend(outcomes)
            
            vsm_ethics[level] = {
                'metrics': metrics,
                'pattern_count': len(outcomes),
                'avg_collective': np.mean([o.collective_contribution for o in outcomes]),
                'pareto_efficiency': metrics['pareto_front_size'] / len(outcomes) if outcomes else 0
            }
        
        # Check for ethical gradient
        ethical_gradient = self._compute_ethical_gradient(vsm_ethics)
        
        return {
            'level_ethics': vsm_ethics,
            'ethical_gradient': ethical_gradient,
            'overall_fairness': np.mean([v['metrics']['fairness_score'] 
                                        for v in vsm_ethics.values() 
                                        if 'metrics' in v])
        }
    
    def _compute_ethical_gradient(self, vsm_ethics: Dict) -> List[float]:
        """Compute ethical value gradient across VSM levels"""
        levels = ['S1', 'S2', 'S3', 'S4', 'S5']
        gradient = []
        
        for i in range(len(levels) - 1):
            curr_level = levels[i]
            next_level = levels[i + 1]
            
            if curr_level in vsm_ethics and next_level in vsm_ethics:
                curr_value = vsm_ethics[curr_level].get('avg_collective', 0)
                next_value = vsm_ethics[next_level].get('avg_collective', 0)
                gradient.append(next_value - curr_value)
            else:
                gradient.append(0)
        
        return gradient


# Test the game theory ethics
if __name__ == "__main__":
    print("=" * 60)
    print("Game Theory Ethics Module Test")
    print("=" * 60)
    
    # Create test outcomes
    outcomes = [
        EthicalOutcome("Agent1", 0.6, 0.8, 0.7, [1,2,3]),
        EthicalOutcome("Agent2", 0.8, 0.6, 0.5, [2,3,4]),
        EthicalOutcome("Agent1", 0.5, 0.9, 0.8, [3,4,5]),
        EthicalOutcome("Agent2", 0.7, 0.7, 0.6, [4,5,6]),
        EthicalOutcome("Agent3", 0.9, 0.4, 0.3, [1,1,1]),  # Selfish
        EthicalOutcome("Agent3", 0.3, 0.95, 0.9, [5,5,5])  # Altruistic
    ]
    
    # Analyze
    analyzer = EthicalMetricsAnalyzer()
    metrics = analyzer.analyze_ethical_pattern(outcomes)
    
    print("\nEthical Analysis Results:")
    print(f"  Pareto front size: {metrics['pareto_front_size']}")
    print(f"  Pareto optimal value: {metrics['pareto_optimal_value']:.3f}")
    print(f"  Collective variety: {metrics['collective_variety']:.3f}")
    print(f"  Fairness score: {metrics['fairness_score']:.3f}")
    print(f"  Cooperation gain: {metrics['cooperation_gain']:.1%}")
    
    print("\nShapley Values:")
    for agent, value in metrics['shapley_values'].items():
        print(f"  {agent}: {value:.3f}")
    
    if metrics['unfair_agents']:
        print(f"\n⚠ Unfair agents detected: {metrics['unfair_agents']}")
    
    # Visualize Pareto front
    print(analyzer.visualize_pareto_front(outcomes))
    
    # Test VSM integration
    print("\nVSM Ethical Integration:")
    vsm_integration = EthicalVSMIntegration()
    
    # Map outcomes to VSM levels
    outcomes_by_level = {}
    for outcome in outcomes:
        level = vsm_integration.map_to_vsm_level(outcome)
        if level not in outcomes_by_level:
            outcomes_by_level[level] = []
        outcomes_by_level[level].append(outcome)
    
    vsm_eval = vsm_integration.evaluate_vsm_ethics(outcomes_by_level)
    
    print(f"  Ethical gradient: {vsm_eval['ethical_gradient']}")
    print(f"  Overall fairness: {vsm_eval['overall_fairness']:.3f}")
    
    print("\n✅ Game theory ethics ready for VSM integration!")