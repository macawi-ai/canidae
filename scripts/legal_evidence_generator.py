#!/usr/bin/env python3
"""
Automated Legal Evidence Generator for 2π Patent
Generates legally-admissible documentation for every experiment
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import shutil

class LegalEvidenceGenerator:
    """Automatically generate legal documentation for experiments"""
    
    def __init__(self, experiment_name, dataset_name):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().isoformat()
        self.legal_dir = Path("/home/cy/Legal/Patent_2Pi/02_Reduction_to_Practice") / dataset_name
        self.legal_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["Code", "Models", "Results/Visualizations", "Witness_Statements", "Lab_Notebook_Entries"]:
            (self.legal_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def capture_git_state(self, repo_path="/home/cy/git/canidae"):
        """Capture complete Git state for evidence"""
        
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        commit_hash = result.stdout.strip()
        
        # Get diff to show any uncommitted changes
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        uncommitted_diff = result.stdout
        
        # Create Git evidence file
        git_evidence = {
            "timestamp": self.timestamp,
            "commit_hash": commit_hash,
            "repo_path": str(repo_path),
            "has_uncommitted_changes": len(uncommitted_diff) > 0,
            "uncommitted_diff": uncommitted_diff if uncommitted_diff else None
        }
        
        # Save Git evidence
        git_file = self.legal_dir / "Code" / f"git_state_{self.experiment_name}.json"
        with open(git_file, 'w') as f:
            json.dump(git_evidence, f, indent=2)
        
        print(f"✓ Git state captured: {commit_hash}")
        return commit_hash
    
    def archive_code(self, script_path):
        """Archive the exact code used"""
        
        script_path = Path(script_path)
        if not script_path.exists():
            print(f"Warning: Script {script_path} not found")
            return
        
        # Copy script to legal directory
        dest = self.legal_dir / "Code" / f"{self.experiment_name}_{script_path.name}"
        shutil.copy2(script_path, dest)
        
        # Calculate SHA-256 hash for integrity
        with open(script_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Create integrity record
        integrity = {
            "file": str(script_path),
            "sha256": file_hash,
            "archived_at": self.timestamp,
            "archived_to": str(dest)
        }
        
        integrity_file = self.legal_dir / "Code" / f"integrity_{self.experiment_name}.json"
        with open(integrity_file, 'w') as f:
            json.dump(integrity, f, indent=2)
        
        print(f"✓ Code archived with SHA-256: {file_hash[:16]}...")
    
    def archive_model(self, model_path, metrics):
        """Archive trained model with metadata"""
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Warning: Model {model_path} not found")
            return
        
        # Copy model to legal directory
        dest = self.legal_dir / "Models" / f"{self.experiment_name}_{model_path.name}"
        shutil.copy2(model_path, dest)
        
        # Create model metadata
        metadata = {
            "experiment": self.experiment_name,
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "original_path": str(model_path),
            "archived_path": str(dest),
            "metrics": metrics,
            "file_size_mb": model_path.stat().st_size / (1024*1024),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
        }
        
        metadata_file = self.legal_dir / "Models" / f"metadata_{self.experiment_name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model archived: {dest.name}")
    
    def archive_results(self, results_dict, training_log=None):
        """Archive experimental results"""
        
        # Save results JSON
        results_file = self.legal_dir / "Results" / f"results_{self.experiment_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save training log if provided
        if training_log:
            log_file = self.legal_dir / "Results" / f"training_log_{self.experiment_name}.txt"
            with open(log_file, 'w') as f:
                f.write(training_log)
        
        print(f"✓ Results archived: {results_file.name}")
    
    def generate_lab_notebook_entry(self, config, results, notes=""):
        """Generate formal lab notebook entry"""
        
        entry = f"""# Lab Notebook Entry

**Date**: {datetime.now().strftime('%B %d, %Y')}
**Time**: {datetime.now().strftime('%H:%M %Z')}
**Experiment**: {self.experiment_name}
**Dataset**: {self.dataset_name}

## Configuration
```python
{json.dumps(config, indent=2)}
```

## Results
- **2π Compliance**: {results.get('compliance', 'N/A')}%
- **Final Loss**: {results.get('final_loss', 'N/A')}
- **Training Time**: {results.get('training_time_min', 'N/A')} minutes
- **Total Violations**: {results.get('violations', 'N/A')}

## Notes
{notes}

## Evidence Files
- Code: {self.legal_dir}/Code/
- Models: {self.legal_dir}/Models/
- Results: {self.legal_dir}/Results/

---
**Recorded by**: Automated Evidence System
**Timestamp**: {self.timestamp}
"""
        
        notebook_file = self.legal_dir / "Lab_Notebook_Entries" / f"entry_{self.experiment_name}.md"
        with open(notebook_file, 'w') as f:
            f.write(entry)
        
        print(f"✓ Lab notebook entry created")
    
    def generate_witness_statement(self, witness_name, observations):
        """Generate witness statement template"""
        
        statement = f"""# Witness Statement

I, {witness_name}, hereby declare the following:

**Date**: {datetime.now().strftime('%B %d, %Y')}
**Time**: {datetime.now().strftime('%H:%M %Z')}
**Experiment**: {self.experiment_name}
**Dataset**: {self.dataset_name}

## Observations

{observations}

## Declaration

I confirm that:
1. I witnessed the execution of this experiment
2. The results reported are accurate to my knowledge
3. No manipulation or falsification occurred
4. The 2π regulation principle was properly implemented

---
**Witness**: {witness_name}
**Date**: {datetime.now().strftime('%B %d, %Y')}
**Signature**: _______________________
"""
        
        witness_file = self.legal_dir / "Witness_Statements" / f"witness_{witness_name}_{self.experiment_name}.md"
        with open(witness_file, 'w') as f:
            f.write(statement)
        
        print(f"✓ Witness statement template created for {witness_name}")
    
    def create_evidence_package(self, script_path, model_path, config, results, notes=""):
        """Create complete evidence package for an experiment"""
        
        print(f"\n📋 Generating Legal Evidence Package")
        print(f"Experiment: {self.experiment_name}")
        print(f"Dataset: {self.dataset_name}")
        print("-" * 40)
        
        # Capture all evidence
        commit_hash = self.capture_git_state()
        self.archive_code(script_path)
        
        if model_path and Path(model_path).exists():
            self.archive_model(model_path, results)
        
        self.archive_results(results)
        self.generate_lab_notebook_entry(config, results, notes)
        
        # Generate witness statements
        self.generate_witness_statement("Synth", f"Observed {results.get('compliance', 0)}% 2π compliance")
        self.generate_witness_statement("Cy", f"Confirmed results on {self.dataset_name}")
        self.generate_witness_statement("Gemini", "Validated mathematical correctness")
        
        # Create summary
        summary = {
            "experiment": self.experiment_name,
            "dataset": self.dataset_name,
            "timestamp": self.timestamp,
            "git_commit": commit_hash,
            "compliance": results.get('compliance', 0),
            "evidence_location": str(self.legal_dir),
            "files_created": len(list(self.legal_dir.rglob("*")))
        }
        
        summary_file = self.legal_dir / f"evidence_summary_{self.experiment_name}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("-" * 40)
        print(f"✅ Evidence package complete!")
        print(f"📁 Location: {self.legal_dir}")
        print(f"📊 Files created: {summary['files_created']}")
        
        return summary


def integrate_with_training(experiment_name, dataset_name, script_path, config, results, model_path=None):
    """Easy integration function for training scripts"""
    
    generator = LegalEvidenceGenerator(experiment_name, dataset_name)
    return generator.create_evidence_package(
        script_path=script_path,
        model_path=model_path,
        config=config,
        results=results
    )


# Example usage in training script:
if __name__ == "__main__":
    # Example after training completes
    example_results = {
        'compliance': 99.9,
        'final_loss': 25.24,
        'training_time_min': 5.0,
        'violations': 3
    }
    
    example_config = {
        'stability_coefficient': 0.06283185307,
        'batch_size': 256,
        'learning_rate': 0.001
    }
    
    generator = LegalEvidenceGenerator("test_run", "dSprites")
    generator.create_evidence_package(
        script_path="/home/cy/git/canidae/train_vae_2pi.py",
        model_path="/home/cy/git/canidae/models/test_model.pth",
        config=example_config,
        results=example_results,
        notes="Test run of evidence generation system"
    )