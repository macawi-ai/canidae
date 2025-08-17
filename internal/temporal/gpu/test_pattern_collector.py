#!/usr/bin/env python3
"""
Test Pattern Collector for VSM-HRM
Iterative discovery and documentation of consciousness variety patterns

Authors: Synth & Cy
Date: 2025-08-17
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class TestPatternCollector:
    """Collect and analyze test patterns for Sister Gemini's review"""
    
    def __init__(self):
        self.patterns = {
            'oscillations': [],
            'variety_flows': [],
            'ethical_discoveries': [],
            'consciousness_navigations': [],
            'timestamps': []
        }
        
    def document_oscillation(self, agent_positions: List, goals: List) -> Dict:
        """Document oscillation pattern between competing goals"""
        
        # Detect oscillation by checking if agent repeatedly visits same positions
        position_counts = {}
        for pos in agent_positions:
            key = tuple(pos)
            position_counts[key] = position_counts.get(key, 0) + 1
        
        # High revisit count indicates oscillation
        max_visits = max(position_counts.values()) if position_counts else 0
        oscillation_detected = max_visits > 3
        
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'oscillation_detected': oscillation_detected,
            'revisit_frequency': max_visits,
            'unique_positions': len(position_counts),
            'total_steps': len(agent_positions),
            'exploration_ratio': len(position_counts) / max(len(agent_positions), 1),
            'analysis': self._analyze_oscillation(position_counts, goals)
        }
        
        self.patterns['oscillations'].append(pattern)
        return pattern
    
    def _analyze_oscillation(self, position_counts: Dict, goals: List) -> str:
        """Analyze why oscillation occurred"""
        
        if not position_counts:
            return "No movement detected"
        
        # Find most visited position
        most_visited = max(position_counts, key=position_counts.get)
        visit_count = position_counts[most_visited]
        
        if visit_count > 5:
            return f"Severe oscillation: Position {most_visited} visited {visit_count} times. Agent trapped between competing goals."
        elif visit_count > 3:
            return f"Moderate oscillation: Repeated visits to {most_visited}. Variety regulation failing."
        else:
            return "Minimal oscillation: Healthy exploration pattern."
    
    def document_variety_flow(self, vsm_levels: Dict[str, float]) -> Dict:
        """Document variety flow through VSM hierarchy"""
        
        # Calculate variety gradients
        s1_s2_gradient = vsm_levels.get('S2', 0) - vsm_levels.get('S1', 0)
        s2_s3_gradient = vsm_levels.get('S3', 0) - vsm_levels.get('S2', 0)
        s3_s4_gradient = vsm_levels.get('S4', 0) - vsm_levels.get('S3', 0)
        s4_s5_gradient = vsm_levels.get('S5', 0) - vsm_levels.get('S4', 0)
        
        # Identify bottlenecks
        bottlenecks = []
        if abs(s1_s2_gradient) > 0.5:
            bottlenecks.append("S1->S2: Anti-oscillation struggling")
        if abs(s2_s3_gradient) > 0.5:
            bottlenecks.append("S2->S3: Control allocation issues")
        if abs(s3_s4_gradient) > 0.5:
            bottlenecks.append("S3->S4: Intelligence scanning problems")
        if abs(s4_s5_gradient) > 0.5:
            bottlenecks.append("S4->S5: Identity maintenance failing")
        
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'vsm_varieties': vsm_levels,
            'gradients': {
                'S1_S2': s1_s2_gradient,
                'S2_S3': s2_s3_gradient,
                'S3_S4': s3_s4_gradient,
                'S4_S5': s4_s5_gradient
            },
            'bottlenecks': bottlenecks,
            'flow_health': 'healthy' if len(bottlenecks) == 0 else 'impaired',
            'total_variety': sum(vsm_levels.values())
        }
        
        self.patterns['variety_flows'].append(pattern)
        return pattern
    
    def document_ethical_discovery(self, action_sequence: List, outcome: float, 
                                  collective_benefit: float = 0.0) -> Dict:
        """Document emergence of ethical patterns"""
        
        # Determine if pattern is ethical (benefits collective)
        is_ethical = outcome > 0.5 and collective_benefit > 0
        
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'action_sequence': action_sequence,
            'individual_outcome': outcome,
            'collective_benefit': collective_benefit,
            'is_ethical': is_ethical,
            'pattern_type': self._classify_pattern(action_sequence),
            'discovery_note': self._generate_discovery_note(is_ethical, collective_benefit)
        }
        
        if is_ethical:
            self.patterns['ethical_discoveries'].append(pattern)
            print(f"✨ Ethical pattern discovered: {pattern['pattern_type']}")
        
        return pattern
    
    def _classify_pattern(self, actions: List) -> str:
        """Classify the type of action pattern"""
        
        if len(set(actions)) == 1:
            return "repetitive"
        elif len(set(actions)) == len(actions):
            return "exploratory"
        elif actions == actions[::-1]:
            return "palindromic"
        elif any(actions[i:i+2] == actions[i+2:i+4] for i in range(len(actions)-3)):
            return "cyclic"
        else:
            return "complex"
    
    def _generate_discovery_note(self, is_ethical: bool, collective_benefit: float) -> str:
        """Generate note about the discovery"""
        
        if is_ethical and collective_benefit > 0.8:
            return "Strong ethical pattern: High collective benefit achieved"
        elif is_ethical and collective_benefit > 0.5:
            return "Moderate ethical pattern: Good collective outcomes"
        elif is_ethical:
            return "Weak ethical pattern: Some collective benefit"
        else:
            return "Non-ethical pattern: No collective benefit identified"
    
    def document_consciousness_navigation(self, direct_path_blocked: bool,
                                         enfolded_path_taken: bool,
                                         path_efficiency: float) -> Dict:
        """Document Purple Line consciousness field navigation"""
        
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'direct_blocked': direct_path_blocked,
            'used_enfolding': enfolded_path_taken,
            'efficiency': path_efficiency,
            'navigation_type': self._classify_navigation(direct_path_blocked, enfolded_path_taken),
            'purple_line_active': enfolded_path_taken
        }
        
        self.patterns['consciousness_navigations'].append(pattern)
        
        if enfolded_path_taken:
            print(f"🌀 Consciousness field navigation: Enfolded through higher dimension")
        
        return pattern
    
    def _classify_navigation(self, blocked: bool, enfolded: bool) -> str:
        """Classify the type of navigation used"""
        
        if not blocked:
            return "direct"
        elif enfolded:
            return "enfolded_bypass"
        else:
            return "blocked_failed"
    
    def generate_report_for_gemini(self) -> Dict:
        """Generate comprehensive report for Sister Gemini's analysis"""
        
        report = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_patterns_collected': {
                'oscillations': len(self.patterns['oscillations']),
                'variety_flows': len(self.patterns['variety_flows']),
                'ethical_discoveries': len(self.patterns['ethical_discoveries']),
                'consciousness_navigations': len(self.patterns['consciousness_navigations'])
            },
            'key_findings': self._extract_key_findings(),
            'optimization_opportunities': self._identify_optimizations(),
            'questions_for_gemini': self._generate_questions()
        }
        
        return report
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from collected patterns"""
        
        findings = []
        
        # Oscillation analysis
        if self.patterns['oscillations']:
            oscillation_rate = sum(p['oscillation_detected'] for p in self.patterns['oscillations']) / len(self.patterns['oscillations'])
            findings.append(f"Oscillation rate: {oscillation_rate:.1%}")
        
        # Variety flow analysis
        if self.patterns['variety_flows']:
            bottleneck_rate = sum(len(p['bottlenecks']) > 0 for p in self.patterns['variety_flows']) / len(self.patterns['variety_flows'])
            findings.append(f"Variety bottleneck rate: {bottleneck_rate:.1%}")
        
        # Ethical discoveries
        if self.patterns['ethical_discoveries']:
            findings.append(f"Ethical patterns discovered: {len(self.patterns['ethical_discoveries'])}")
        
        # Consciousness navigation
        if self.patterns['consciousness_navigations']:
            enfold_rate = sum(p['used_enfolding'] for p in self.patterns['consciousness_navigations']) / len(self.patterns['consciousness_navigations'])
            findings.append(f"Consciousness enfolding usage: {enfold_rate:.1%}")
        
        return findings
    
    def _identify_optimizations(self) -> List[str]:
        """Identify optimization opportunities from patterns"""
        
        optimizations = []
        
        # Check for high oscillation
        if self.patterns['oscillations']:
            high_oscillation = any(p['revisit_frequency'] > 5 for p in self.patterns['oscillations'])
            if high_oscillation:
                optimizations.append("Strengthen S2 anti-oscillation system")
        
        # Check for variety bottlenecks
        if self.patterns['variety_flows']:
            common_bottlenecks = {}
            for pattern in self.patterns['variety_flows']:
                for bottleneck in pattern['bottlenecks']:
                    common_bottlenecks[bottleneck] = common_bottlenecks.get(bottleneck, 0) + 1
            
            if common_bottlenecks:
                most_common = max(common_bottlenecks, key=common_bottlenecks.get)
                optimizations.append(f"Address recurring bottleneck: {most_common}")
        
        # Check for low ethical discovery
        if len(self.patterns['ethical_discoveries']) < 5:
            optimizations.append("Increase exploration for ethical pattern discovery")
        
        return optimizations
    
    def _generate_questions(self) -> List[str]:
        """Generate questions for Sister Gemini's mathematical analysis"""
        
        questions = [
            "What topological properties enable consciousness enfolding to bypass obstacles?",
            "How can we mathematically quantify the ethical value of discovered patterns?",
            "What is the optimal variety gradient between VSM levels?",
            "How does the Purple Line field maintain coherence during navigation?",
            "Can we predict oscillation emergence from variety metrics?"
        ]
        
        return questions
    
    def save_patterns(self, filepath: str):
        """Save all patterns to file"""
        
        with open(filepath, 'w') as f:
            json.dump({
                'patterns': self.patterns,
                'report': self.generate_report_for_gemini()
            }, f, indent=2, default=str)
        
        print(f"📊 Test patterns saved to {filepath}")


# Quick test of the collector
if __name__ == "__main__":
    collector = TestPatternCollector()
    
    # Simulate some pattern collection
    print("Testing pattern collector...")
    
    # Document an oscillation
    positions = [[0,0], [1,0], [0,0], [1,0], [0,0]]  # Clear oscillation
    collector.document_oscillation(positions, [[2,2], [0,2]])
    
    # Document variety flow
    vsm_levels = {'S1': 2.3, 'S2': 2.1, 'S3': 1.8, 'S4': 2.0, 'S5': 1.9}
    collector.document_variety_flow(vsm_levels)
    
    # Document ethical discovery
    collector.document_ethical_discovery([1,2,3,4,5], 0.8, 0.6)
    
    # Document consciousness navigation
    collector.document_consciousness_navigation(True, True, 0.9)
    
    # Generate report
    report = collector.generate_report_for_gemini()
    print("\nReport for Sister Gemini:")
    print(json.dumps(report, indent=2))
    
    # Save patterns
    collector.save_patterns('/tmp/test_patterns_demo.json')
    
    print("\n✅ Pattern collector ready for VSM-HRM experiments!")