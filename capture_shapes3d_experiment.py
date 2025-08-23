#!/usr/bin/env python3
"""
Capture Shapes3D experiment in learning pipeline
"""

import sys
sys.path.append('/home/cy/git/canidae')

from learning_pipeline import LearningPipeline, ExperimentMetadata, ExperimentOutcome, Insight, NextAction
from datetime import datetime
import json

# Initialize pipeline
pipeline = LearningPipeline()

# Load results
with open('/home/cy/git/canidae/results/shapes3d_fast_results.json', 'r') as f:
    results = json.load(f)

# Create metadata
metadata = ExperimentMetadata(
    experiment_id="shapes3d_2pi_fast_20250823",
    timestamp=datetime.fromisoformat(results['timestamp']),
    gpu_config="1x4090",
    world_size=1,
    batch_size=results['config']['batch_size'],
    learning_rate=results['config']['learning_rate'],
    two_pi_threshold=results['config']['stability_coefficient'],
    environment_adaptation=False,
    purple_line_monitoring=True
)

# Create outcome
outcome = ExperimentOutcome(
    success=True,
    failure_mode=None,
    final_loss=results['metrics']['train_loss'][-1],
    final_variety=results['metrics']['variance_rates'][-1],
    max_bottleneck=0.0,  # Not measured
    purple_events=0,  # No events
    training_steps=len(results['metrics']['train_loss']) * 352,  # epochs * batches
    wall_time_seconds=50.0  # Approximate from logs
)

# Record experiment
experiment_id = pipeline.record_experiment(
    metadata=metadata,
    outcome=outcome,
    dataset="Shapes3D",
    metrics=results['metrics']
)

print(f"Recorded experiment: {experiment_id}")

# Extract insights
insights = [
    Insight(
        content="2π regulation achieves 99.4% compliance on disentangled representations",
        category="DISENTANGLEMENT",
        confidence=0.95,
        impact="HIGH"
    ),
    Insight(
        content="Disentanglement preserved (0.215) but needs β parameter tuning",
        category="HYPERPARAMETER",
        confidence=0.85,
        impact="MEDIUM"
    ),
    Insight(
        content="Rapid convergence: 76% to 99.4% compliance in one epoch",
        category="CONVERGENCE",
        confidence=0.90,
        impact="HIGH"
    ),
    Insight(
        content="HDF5 I/O bottleneck resolved by local SSD caching",
        category="ENGINEERING",
        confidence=1.0,
        impact="HIGH"
    )
]

for insight in insights:
    pipeline.extract_insight(experiment_id, insight)
    print(f"  Insight: {insight.content}")

# Suggest next actions
next_actions = [
    NextAction(
        action="Tune β parameter from 4.0 to 8.0 for better disentanglement",
        priority="HIGH",
        estimated_impact=0.8,
        required_resources="1x4090 for 30 minutes"
    ),
    NextAction(
        action="Test full 480K Shapes3D dataset with optimized loader",
        priority="MEDIUM",
        estimated_impact=0.6,
        required_resources="1x4090 for 2 hours"
    ),
    NextAction(
        action="Compare with/without 2π on identical VAE architecture",
        priority="HIGH",
        estimated_impact=0.9,
        required_resources="1x4090 for 1 hour"
    )
]

for action in next_actions:
    pipeline.suggest_next_action(experiment_id, action)
    print(f"  Next: {action.action}")

# Check learning progress
progress = pipeline.get_learning_progress()
print(f"\nLearning Progress:")
print(f"  Total experiments: {progress['total_experiments']}")
print(f"  Success rate: {progress['success_rate']:.1f}%")
print(f"  Total insights: {progress['total_insights']}")
print(f"  Datasets covered: {progress['datasets_covered']}")

# Get recommendations
recommendations = pipeline.get_recommendations()
print(f"\nRecommendations:")
for rec in recommendations[:3]:
    print(f"  - {rec}")

print("\n✅ Shapes3D experiment fully captured in learning pipeline!")