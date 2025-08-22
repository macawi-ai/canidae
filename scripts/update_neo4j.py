#!/usr/bin/env python3
"""
Update Neo4j Knowledge Graph with experiment results
Part of the CANIDAE MLOps pipeline
"""

import json
import argparse
import os
from datetime import datetime
from neo4j import GraphDatabase
import redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphUpdater:
    def __init__(self, neo4j_uri=None, neo4j_auth=None, redis_host=None):
        # Use environment variables or defaults
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://192.168.1.253:7688')
        auth_user = os.getenv('NEO4J_USER', 'neo4j')
        auth_pass = os.getenv('NEO4J_PASSWORD', 'synthconsciousness')
        self.neo4j_auth = neo4j_auth or (auth_user, auth_pass)
        self.redis_host = redis_host or os.getenv('REDIS_HOST', '192.168.1.253')
        
        # Connect to databases
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
            self.redis_client = redis.Redis(host=self.redis_host, port=6379, decode_responses=True)
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
            logger.info(f"Connected to Redis at {self.redis_host}")
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            self.driver = None
            self.redis_client = None

    def update_knowledge_graph(self, experiment_id, metadata, success, release_url):
        """Update Neo4j with experiment results"""
        
        if not self.driver:
            logger.warning("Neo4j not connected, skipping knowledge graph update")
            return
            
        with self.driver.session() as session:
            try:
                # Create experiment node
                result = session.run("""
                    CREATE (e:Experiment {
                        id: $id,
                        timestamp: datetime($timestamp),
                        git_sha: $git_sha,
                        git_branch: $git_branch,
                        gpu_config: $gpu_config,
                        dataset: $dataset,
                        experiment_name: $experiment_name,
                        success: $success,
                        release_url: $release_url,
                        github_run_id: $github_run_id,
                        github_run_url: $github_run_url,
                        two_pi_regulated: true
                    })
                    RETURN e
                """, 
                    id=experiment_id,
                    timestamp=metadata.get('timestamp', datetime.now().isoformat()),
                    git_sha=metadata.get('git_sha', 'unknown'),
                    git_branch=metadata.get('git_branch', 'main'),
                    gpu_config=metadata.get('gpu_config', 'unknown'),
                    dataset=metadata.get('dataset', 'unknown'),
                    experiment_name=metadata.get('experiment_name', 'unnamed'),
                    success=success,
                    release_url=release_url,
                    github_run_id=metadata.get('github_run_id', ''),
                    github_run_url=metadata.get('github_run_url', '')
                )
                
                logger.info(f"Created experiment node: {experiment_id}")
                
                # Link to previous experiments with same dataset
                session.run("""
                    MATCH (e1:Experiment {id: $id})
                    MATCH (e2:Experiment)
                    WHERE e2.dataset = e1.dataset 
                    AND e2.id <> e1.id
                    AND e2.timestamp < e1.timestamp
                    WITH e1, e2
                    ORDER BY e2.timestamp DESC
                    LIMIT 1
                    CREATE (e2)-[:PRECEDED]->(e1)
                """, id=experiment_id)
                
                # Create dataset node if it doesn't exist
                session.run("""
                    MATCH (e:Experiment {id: $id})
                    MERGE (d:Dataset {name: e.dataset})
                    CREATE (e)-[:TRAINED_ON]->(d)
                """, id=experiment_id)
                
                # Create GPU configuration node
                session.run("""
                    MATCH (e:Experiment {id: $id})
                    MERGE (g:GPUConfig {name: e.gpu_config})
                    CREATE (e)-[:USED_CONFIG]->(g)
                """, id=experiment_id)
                
                # Track 2π regulation
                session.run("""
                    MATCH (e:Experiment {id: $id})
                    MERGE (r:Regulation {type: '2pi', threshold: 0.06283185307})
                    CREATE (e)-[:REGULATED_BY]->(r)
                """, id=experiment_id)
                
                # If failed, create failure analysis node
                if not success:
                    session.run("""
                        MATCH (e:Experiment {id: $id})
                        CREATE (f:FailureAnalysis {
                            timestamp: datetime(),
                            needs_investigation: true
                        })
                        CREATE (e)-[:FAILED_WITH]->(f)
                    """, id=experiment_id)
                
                # Update statistics
                stats = session.run("""
                    MATCH (e:Experiment)
                    WHERE e.dataset = $dataset
                    RETURN 
                        count(e) as total_runs,
                        sum(CASE WHEN e.success = true THEN 1 ELSE 0 END) as successful_runs,
                        collect(DISTINCT e.gpu_config) as gpu_configs_used
                """, dataset=metadata.get('dataset', 'unknown')).single()
                
                logger.info(f"Dataset statistics for {metadata.get('dataset')}:")
                logger.info(f"  Total runs: {stats['total_runs']}")
                logger.info(f"  Successful: {stats['successful_runs']}")
                logger.info(f"  GPU configs: {stats['gpu_configs_used']}")
                
            except Exception as e:
                logger.error(f"Failed to update Neo4j: {e}")
    
    def cache_experiment(self, experiment_id, metadata, success):
        """Cache experiment in Redis for quick access"""
        
        if not self.redis_client:
            logger.warning("Redis not connected, skipping cache")
            return
            
        try:
            # Store experiment metadata
            self.redis_client.hset(f"experiment:{experiment_id}", mapping={
                "timestamp": metadata.get('timestamp', ''),
                "success": str(success),
                "gpu_config": metadata.get('gpu_config', ''),
                "dataset": metadata.get('dataset', ''),
                "experiment_name": metadata.get('experiment_name', '')
            })
            
            # Set expiration to 7 days
            self.redis_client.expire(f"experiment:{experiment_id}", 604800)
            
            # Update recent experiments list
            self.redis_client.lpush("recent_experiments", experiment_id)
            self.redis_client.ltrim("recent_experiments", 0, 99)  # Keep last 100
            
            logger.info(f"Cached experiment in Redis: {experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to update Redis: {e}")
    
    def close(self):
        """Clean up connections"""
        if self.driver:
            self.driver.close()

def main():
    parser = argparse.ArgumentParser(description='Update knowledge graph with experiment results')
    parser.add_argument('--experiment-id', required=True, help='Unique experiment ID')
    parser.add_argument('--metadata-file', required=True, help='Path to metadata JSON file')
    parser.add_argument('--success', type=lambda x: x.lower() == 'true', required=True, help='Training success status')
    parser.add_argument('--release-url', default='', help='GitHub release URL if created')
    
    args = parser.parse_args()
    
    # Load metadata
    try:
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        metadata = {"experiment_id": args.experiment_id}
    
    # Update knowledge graph
    updater = KnowledgeGraphUpdater()
    
    if updater.driver or updater.redis_client:
        updater.update_knowledge_graph(
            args.experiment_id,
            metadata,
            args.success,
            args.release_url
        )
        
        updater.cache_experiment(
            args.experiment_id,
            metadata,
            args.success
        )
        
        updater.close()
        logger.info("Knowledge graph update completed!")
    else:
        logger.warning("No database connections available, skipping updates")

if __name__ == "__main__":
    main()