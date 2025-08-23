#!/usr/bin/env python3
"""
Central Experiment Manager for 2π Validation Pipeline
Based on Sister Gemini's rigorous experimental framework

Orchestrates all components of the experimental pipeline:
- Configuration loading
- Data loading and preprocessing  
- Model training (2π regulated + baselines)
- Evaluation and metrics calculation
- Statistical significance testing
- Error analysis and ablation studies
- Robustness testing
- Visualization and reporting
- Legal documentation generation
"""

import argparse
import logging
import yaml
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add scripts directory to path
sys.path.append('/home/cy/git/canidae/scripts')

def setup_logging(experiment_name: str, output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging for the experiment"""
    
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('experiment_manager')
    logger.info(f"Experiment logging initialized: {log_file}")
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate experiment configuration from YAML"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Basic validation
    required_fields = ['experiment_name', 'dataset', 'model', 'hyperparameters']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    return config

def create_output_directory(config: Dict[str, Any]) -> Path:
    """Create timestamped output directory for experiment results"""
    
    base_path = Path(config['output']['base_path'])
    experiment_name = config['experiment_name']
    
    if config['output'].get('create_timestamp_dir', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_path / f"{experiment_name}_{timestamp}"
    else:
        output_dir = base_path / experiment_name
    
    # Create subdirectories
    subdirs = ['models', 'results', 'visualizations', 'logs', 'legal', 'reports']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return output_dir

def validate_environment() -> bool:
    """Validate that all required dependencies and resources are available"""
    
    try:
        import torch
        import numpy as np
        import matplotlib
        import seaborn
        import scipy
        import sklearn
        
        # Check CUDA availability if requested
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
        
        return True
        
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        return False

class ExperimentManager:
    """Central orchestrator for rigorous 2π validation experiments"""
    
    def __init__(self, config_path: str):
        """Initialize experiment manager with configuration"""
        
        self.config = load_config(config_path)
        self.output_dir = create_output_directory(self.config)
        self.logger = setup_logging(self.config['experiment_name'], self.output_dir)
        
        # Initialize results storage
        self.results = {
            'experiment_info': {
                'name': self.config['experiment_name'],
                'id': self.config.get('experiment_id', 'Unknown'),
                'start_time': datetime.now().isoformat(),
                'config_path': config_path,
                'output_dir': str(self.output_dir)
            },
            'training_results': {},
            'evaluation_results': {},
            'statistical_results': {},
            'ablation_results': {},
            'robustness_results': {},
            'error_analysis': {},
            'legal_documentation': {}
        }
        
        self.logger.info(f"Experiment Manager initialized: {self.config['experiment_name']}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_experiment(self) -> Dict[str, Any]:
        """Execute the complete experimental pipeline"""
        
        try:
            self.logger.info("="*60)
            self.logger.info(f"STARTING EXPERIMENT: {self.config['experiment_name']}")
            self.logger.info("="*60)
            
            # Phase 1: Environment validation
            self.logger.info("Phase 1: Environment Validation")
            if not validate_environment():
                raise RuntimeError("Environment validation failed")
            
            # Phase 2: Data loading
            self.logger.info("Phase 2: Data Loading and Preprocessing")
            train_data, val_data, test_data = self.load_data()
            
            # Phase 3: Model training (2π + baselines)
            self.logger.info("Phase 3: Model Training")
            trained_models = self.train_models(train_data, val_data)
            
            # Phase 4: Model evaluation
            self.logger.info("Phase 4: Model Evaluation")
            evaluation_results = self.evaluate_models(trained_models, test_data)
            
            # Phase 5: Statistical analysis
            if self.config.get('statistical_analysis', {}).get('enabled', False):
                self.logger.info("Phase 5: Statistical Significance Testing")
                statistical_results = self.run_statistical_analysis(evaluation_results)
            
            # Phase 6: Error analysis
            if self.config.get('error_analysis', {}).get('enabled', False):
                self.logger.info("Phase 6: Error Analysis")
                error_results = self.run_error_analysis(trained_models, test_data)
            
            # Phase 7: Ablation studies
            if self.config.get('ablation_study', {}).get('enabled', False):
                self.logger.info("Phase 7: Ablation Studies")
                ablation_results = self.run_ablation_study(train_data, val_data, test_data)
            
            # Phase 8: Robustness testing
            if self.config.get('robustness_testing', {}).get('enabled', False):
                self.logger.info("Phase 8: Robustness Testing")
                robustness_results = self.run_robustness_testing(trained_models, test_data)
            
            # Phase 9: Visualization and reporting
            if self.config.get('visualization', {}).get('enabled', False):
                self.logger.info("Phase 9: Visualization and Reporting")
                self.generate_visualizations()
                self.generate_report()
            
            # Phase 10: Legal documentation
            if self.config.get('legal', {}).get('enabled', False):
                self.logger.info("Phase 10: Legal Documentation")
                self.generate_legal_documentation()
            
            # Phase 11: Glossary updates
            if self.config.get('glossary', {}).get('enabled', False):
                self.logger.info("Phase 11: Glossary Updates")
                self.update_glossary()
            
            # Finalize results
            self.results['experiment_info']['end_time'] = datetime.now().isoformat()
            self.results['experiment_info']['duration_minutes'] = (
                datetime.fromisoformat(self.results['experiment_info']['end_time']) -
                datetime.fromisoformat(self.results['experiment_info']['start_time'])
            ).total_seconds() / 60
            
            # Save final results
            results_file = self.output_dir / 'results' / 'experiment_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info("="*60)
            self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results saved to: {results_file}")
            self.logger.info("="*60)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            self.results['experiment_info']['status'] = 'FAILED'
            self.results['experiment_info']['error'] = str(e)
            raise
    
    def load_data(self) -> tuple:
        """Load and preprocess data using Data Loader module"""
        try:
            from data_loader import load_data
            
            self.logger.info(f"Loading {self.config['dataset']['name']} dataset...")
            train_loader, val_loader, test_loader = load_data(self.config)
            
            self.logger.info(f"Data loading completed successfully")
            self.logger.info(f"Train batches: {len(train_loader) if train_loader else 0}")
            self.logger.info(f"Validation batches: {len(val_loader) if val_loader else 0}")  
            self.logger.info(f"Test batches: {len(test_loader) if test_loader else 0}")
            
            # Store data info in results
            self.results['data_info'] = {
                'dataset_name': self.config['dataset']['name'],
                'train_batches': len(train_loader) if train_loader else 0,
                'val_batches': len(val_loader) if val_loader else 0,
                'test_batches': len(test_loader) if test_loader else 0,
                'batch_size': self.config['hyperparameters']['batch_size']
            }
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def train_models(self, train_data, val_data) -> Dict[str, Any]:
        """Train 2π regulated model and baselines using Model Trainer"""
        try:
            from model_trainer import train_models
            
            self.logger.info("Starting model training (2π + baselines)...")
            
            # Train all models
            training_results = train_models(self.config, train_data, val_data)
            
            # Store results
            self.results['training_results'] = training_results
            
            # Log summary
            self.logger.info("Model training completed successfully:")
            for model_name, model_results in training_results.items():
                compliance = model_results.get('compliance', 0) * 100
                loss = model_results.get('final_loss', 0)
                training_time = model_results.get('training_time_min', 0)
                
                self.logger.info(f"  {model_name}: {compliance:.1f}% compliance, "
                               f"{loss:.2f} loss, {training_time:.1f}min")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_models(self, trained_models, test_data) -> Dict[str, Any]:
        """Evaluate all models using comprehensive metrics"""
        # This will call the Model Evaluator module when implemented
        self.logger.info("Model evaluation placeholder - module will be implemented next")
        return {}
    
    def run_statistical_analysis(self, evaluation_results) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        # This will call the Statistical Testing module when implemented
        self.logger.info("Statistical analysis placeholder - module will be implemented next")
        return {}
    
    def run_error_analysis(self, trained_models, test_data) -> Dict[str, Any]:
        """Analyze error patterns and failure modes"""
        # This will call the Error Analysis module when implemented
        self.logger.info("Error analysis placeholder - module will be implemented next")
        return {}
    
    def run_ablation_study(self, train_data, val_data, test_data) -> Dict[str, Any]:
        """Systematic ablation of 2π regulation components"""
        # This will call the Ablation Study module when implemented
        self.logger.info("Ablation study placeholder - module will be implemented next")
        return {}
    
    def run_robustness_testing(self, trained_models, test_data) -> Dict[str, Any]:
        """Test model robustness against various perturbations"""
        # This will call the Robustness Testing module when implemented
        self.logger.info("Robustness testing placeholder - module will be implemented next")
        return {}
    
    def generate_visualizations(self):
        """Generate all publication-ready visualizations"""
        # This will call visualization scripts when implemented
        self.logger.info("Visualization generation placeholder - scripts will be implemented next")
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        # This will call the Report Generator when implemented
        self.logger.info("Report generation placeholder - module will be implemented next")
    
    def generate_legal_documentation(self):
        """Generate legal evidence trail"""
        # This will integrate with existing legal evidence generator
        self.logger.info("Legal documentation placeholder - integration will be implemented next")
    
    def update_glossary(self):
        """Update technical glossary with new terms"""
        # This will call the Glossary Updater when implemented
        self.logger.info("Glossary update placeholder - module will be implemented next")

def main():
    """Main entry point for experiment execution"""
    
    parser = argparse.ArgumentParser(
        description='Run rigorous 2π validation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/fashion_mnist_controlled_comparison.yaml
  python run_experiment.py --config configs/2pi_validation_template.yaml --dry-run
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and setup without running experiment'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment manager
        manager = ExperimentManager(args.config)
        
        if args.dry_run:
            print("✅ Configuration validated successfully")
            print(f"✅ Output directory created: {manager.output_dir}")
            print("✅ Dry run completed - ready for actual experiment")
            return 0
        
        # Run the full experiment
        results = manager.run_experiment()
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Results available at: {manager.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())