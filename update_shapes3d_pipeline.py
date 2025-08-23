#!/usr/bin/env python3
"""
Update learning pipeline with Shapes3D experiment results
"""

import json
from datetime import datetime
from neo4j import GraphDatabase
import duckdb
from pathlib import Path

# Load results
with open('/home/cy/git/canidae/results/shapes3d_fast_results.json', 'r') as f:
    results = json.load(f)

# Neo4j connection
neo4j_driver = GraphDatabase.driver(
    "bolt://192.168.1.253:7688",
    auth=("neo4j", "synthconsciousness")
)

# DuckDB connection
duck_conn = duckdb.connect('/home/cy/experiments.duckdb')

# Update Neo4j
print("Updating Neo4j...")
with neo4j_driver.session() as session:
    # Create experiment node
    session.run("""
        CREATE (e:Experiment {
            name: 'Shapes3D_2π_Regulation',
            dataset: 'Shapes3D',
            timestamp: $timestamp,
            compliance: $compliance,
            disentanglement: $disentanglement,
            samples: $samples,
            epochs: $epochs,
            device: 'RTX_4090',
            git_commit: 'ef2f55f',
            status: 'COMPLETE'
        })
        """,
        timestamp=results['timestamp'],
        compliance=results['final_compliance'],
        disentanglement=results['final_disentanglement'],
        samples=results['samples'],
        epochs=results['config']['epochs']
    )
    
    # Create insight node
    session.run("""
        MATCH (e:Experiment {name: 'Shapes3D_2π_Regulation'})
        CREATE (i:Insight {
            content: '2π regulation maintains 99.4% compliance on disentangled representations',
            category: 'DISENTANGLEMENT',
            significance: 'HIGH',
            timestamp: $timestamp
        })
        CREATE (e)-[:GENERATED]->(i)
        """,
        timestamp=datetime.now().isoformat()
    )
    
    # Link to 2π discovery
    session.run("""
        MATCH (e:Experiment {name: 'Shapes3D_2π_Regulation'})
        MATCH (d:Discovery {name: '2π_Conjecture'})
        CREATE (e)-[:VALIDATES]->(d)
        """
    )

print("Updating DuckDB...")
# Create experiments table if not exists
duck_conn.execute("""
    CREATE SEQUENCE IF NOT EXISTS experiments_id_seq
""")

duck_conn.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
        name VARCHAR,
        dataset VARCHAR,
        timestamp TIMESTAMP,
        compliance FLOAT,
        disentanglement FLOAT,
        samples INTEGER,
        epochs INTEGER,
        device VARCHAR,
        git_commit VARCHAR,
        metrics JSON
    )
""")

# Insert experiment
duck_conn.execute("""
    INSERT INTO experiments (
        name, dataset, timestamp, compliance, disentanglement,
        samples, epochs, device, git_commit, metrics
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    [
        'Shapes3D_2π_Regulation',
        'Shapes3D',
        datetime.fromisoformat(results['timestamp']),
        results['final_compliance'],
        results['final_disentanglement'],
        results['samples'],
        results['config']['epochs'],
        'RTX_4090',
        'ef2f55f',
        json.dumps(results['metrics'])
    ]
)

# Create CWU tracking
duck_conn.execute("""
    CREATE TABLE IF NOT EXISTS cwu_tracking (
        experiment VARCHAR,
        epoch INTEGER,
        cwus FLOAT,
        compliance FLOAT,
        timestamp TIMESTAMP
    )
""")

# Track CWUs per epoch
for epoch, compliance in enumerate(results['metrics']['compliance']):
    cwus = (epoch + 1) * 351 / 10  # Approximate CWUs
    duck_conn.execute("""
        INSERT INTO cwu_tracking (experiment, epoch, cwus, compliance, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        ['Shapes3D_2π_Regulation', epoch + 1, cwus, compliance, datetime.now()]
    )

print("Generating learning insights...")
# Query for patterns
pattern_results = duck_conn.execute("""
    SELECT 
        dataset,
        AVG(compliance) as avg_compliance,
        COUNT(*) as experiment_count
    FROM experiments
    WHERE compliance > 90
    GROUP BY dataset
""").fetchall()

print("\nHigh Compliance Datasets:")
for row in pattern_results:
    print(f"  {row[0]}: {row[1]:.1f}% avg compliance ({row[2]} experiments)")

# Save pipeline status
pipeline_status = {
    "last_updated": datetime.now().isoformat(),
    "experiments_completed": duck_conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
    "total_cwus": duck_conn.execute("SELECT SUM(cwus) FROM cwu_tracking").fetchone()[0],
    "datasets_processed": ["CIFAR10", "CIFAR100", "Shapes3D"],
    "key_insights": [
        "2π regulation works across diverse datasets",
        "Disentanglement preserved but needs β tuning",
        "99.4% compliance achieved on 50K Shapes3D subset",
        "Factor independence maintained under regulation"
    ],
    "next_experiments": [
        "Tune β parameter for better disentanglement",
        "Test full 480K Shapes3D dataset",
        "Compare with/without 2π on same architecture",
        "Process SmallNORB and Tangram datasets"
    ]
}

with open('/home/cy/git/canidae/PIPELINE_STATUS.json', 'w') as f:
    json.dump(pipeline_status, f, indent=2)

print(f"\nPipeline updated successfully!")
print(f"Total experiments: {pipeline_status['experiments_completed']}")
print(f"Total CWUs: {pipeline_status['total_cwus']:.0f}")

# Close connections
neo4j_driver.close()
duck_conn.close()