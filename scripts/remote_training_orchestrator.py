#!/usr/bin/env python3
"""
Remote Training Orchestrator for CANIDAE
Handles training on cloud GPUs and automatic artifact retrieval
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import paramiko
from scp import SCPClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemoteTrainingOrchestrator:
    def __init__(self, ssh_config):
        """Initialize with SSH configuration for remote GPU server"""
        self.ssh_config = ssh_config
        self.ssh_client = None
        self.scp_client = None
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.local_base = Path("/home/cy/git/canidae")
        
    def connect(self):
        """Establish SSH connection to remote GPU server"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Handle different authentication methods
            connect_params = {
                'hostname': self.ssh_config['host'],
                'port': self.ssh_config.get('port', 22),
                'username': self.ssh_config.get('username', 'root')
            }
            
            if 'key_file' in self.ssh_config:
                connect_params['key_filename'] = self.ssh_config['key_file']
            elif 'password' in self.ssh_config:
                connect_params['password'] = self.ssh_config['password']
                
            self.ssh_client.connect(**connect_params)
            self.scp_client = SCPClient(self.ssh_client.get_transport())
            logger.info(f"Connected to {self.ssh_config['host']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def prepare_remote_environment(self):
        """Setup remote environment for training"""
        commands = [
            # Create experiment directory
            f"mkdir -p /root/canidae/{self.experiment_id}",
            
            # Setup Python environment if needed
            "source /root/tensor-env/bin/activate 2>/dev/null || python3 -m venv /root/tensor-env",
            "source /root/tensor-env/bin/activate && pip install torch torchvision numpy tqdm",
            
            # Verify GPU access
            "nvidia-smi --query-gpu=name,memory.total --format=csv"
        ]
        
        for cmd in commands:
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            output = stdout.read().decode()
            error = stderr.read().decode()
            
            if error and "WARNING" not in error:
                logger.warning(f"Command '{cmd}' produced error: {error}")
            if output:
                logger.info(f"Command output: {output.strip()}")
    
    def upload_training_code(self, training_script, dataset_config):
        """Upload training code and configuration to remote server"""
        try:
            # Create remote directory for this experiment
            remote_dir = f"/root/canidae/{self.experiment_id}"
            
            # Upload main training script
            local_script = self.local_base / training_script
            if local_script.exists():
                self.scp_client.put(str(local_script), f"{remote_dir}/train.py")
                logger.info(f"Uploaded {training_script}")
            
            # Upload configuration
            config = {
                "experiment_id": self.experiment_id,
                "dataset": dataset_config.get('name', 'unknown'),
                "batch_size": dataset_config.get('batch_size', 32),
                "learning_rate": dataset_config.get('learning_rate', 0.001),
                "epochs": dataset_config.get('epochs', 10),
                "two_pi_threshold": 0.06283185307,
                "output_dir": f"{remote_dir}/outputs"
            }
            
            config_path = self.local_base / f"experiments/{self.experiment_id}_config.json"
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.scp_client.put(str(config_path), f"{remote_dir}/config.json")
            logger.info("Uploaded configuration")
            
            return remote_dir
            
        except Exception as e:
            logger.error(f"Failed to upload files: {e}")
            return None
    
    def run_training(self, remote_dir, gpu_count=1):
        """Execute training on remote GPUs"""
        
        # Construct training command based on GPU count
        if gpu_count > 1:
            train_cmd = f"""
                cd {remote_dir} && \
                source /root/tensor-env/bin/activate && \
                torchrun --nproc_per_node={gpu_count} \
                    --master_port=29500 \
                    train.py --config config.json \
                    --output-dir outputs \
                    --distributed
            """
        else:
            train_cmd = f"""
                cd {remote_dir} && \
                source /root/tensor-env/bin/activate && \
                python train.py --config config.json --output-dir outputs
            """
        
        logger.info(f"Starting training with command: {train_cmd}")
        
        # Start training (non-blocking)
        stdin, stdout, stderr = self.ssh_client.exec_command(train_cmd)
        
        # Monitor training progress
        start_time = time.time()
        while not stdout.channel.exit_status_ready():
            # Check for output
            if stdout.channel.recv_ready():
                output = stdout.channel.recv(1024).decode()
                print(output, end='')
            
            # Check for errors
            if stderr.channel.recv_ready():
                error = stderr.channel.recv(1024).decode()
                if error:
                    print(f"[ERROR] {error}", end='')
            
            # Timeout after 2 hours
            if time.time() - start_time > 7200:
                logger.warning("Training timeout reached (2 hours)")
                break
            
            time.sleep(1)
        
        # Get final output
        final_output = stdout.read().decode()
        final_error = stderr.read().decode()
        exit_status = stdout.channel.recv_exit_status()
        
        logger.info(f"Training completed with exit status: {exit_status}")
        
        return exit_status == 0
    
    def download_artifacts(self, remote_dir):
        """Download training artifacts from remote server"""
        try:
            local_model_dir = self.local_base / f"models/{self.experiment_id}"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # List remote files
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"find {remote_dir}/outputs -type f -name '*.pth' -o -name '*.pt' -o -name '*.json' -o -name '*.csv'"
            )
            remote_files = stdout.read().decode().strip().split('\n')
            
            # Download each file
            for remote_file in remote_files:
                if remote_file:
                    filename = Path(remote_file).name
                    local_path = local_model_dir / filename
                    self.scp_client.get(remote_file, str(local_path))
                    logger.info(f"Downloaded {filename}")
            
            # Download training logs
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"find {remote_dir} -name '*.log' -o -name 'training_metrics.json'"
            )
            log_files = stdout.read().decode().strip().split('\n')
            
            for log_file in log_files:
                if log_file:
                    filename = Path(log_file).name
                    local_path = self.local_base / f"experiments/{self.experiment_id}/{filename}"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.scp_client.get(log_file, str(local_path))
                    logger.info(f"Downloaded log: {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download artifacts: {e}")
            return False
    
    def cleanup_remote(self, remote_dir):
        """Clean up remote experiment directory"""
        # Optionally keep for debugging
        stdin, stdout, stderr = self.ssh_client.exec_command(
            f"du -sh {remote_dir}"
        )
        size = stdout.read().decode().strip()
        logger.info(f"Remote directory size: {size}")
        
        # Archive instead of deleting
        archive_cmd = f"tar -czf /root/canidae/archives/{self.experiment_id}.tar.gz -C {remote_dir} ."
        stdin, stdout, stderr = self.ssh_client.exec_command(
            f"mkdir -p /root/canidae/archives && {archive_cmd}"
        )
        
        if stdout.channel.recv_exit_status() == 0:
            logger.info("Remote artifacts archived")
            # Now safe to remove
            self.ssh_client.exec_command(f"rm -rf {remote_dir}")
    
    def close(self):
        """Close SSH connections"""
        if self.scp_client:
            self.scp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("Closed SSH connections")
    
    def run_complete_pipeline(self, training_script, dataset_config, gpu_count=1):
        """Run complete training pipeline"""
        
        success = False
        
        try:
            # 1. Connect to remote server
            if not self.connect():
                return False
            
            # 2. Prepare environment
            self.prepare_remote_environment()
            
            # 3. Upload code and config
            remote_dir = self.upload_training_code(training_script, dataset_config)
            if not remote_dir:
                return False
            
            # 4. Run training
            training_success = self.run_training(remote_dir, gpu_count)
            
            # 5. Download artifacts (even if training failed, get logs)
            self.download_artifacts(remote_dir)
            
            # 6. Clean up remote
            self.cleanup_remote(remote_dir)
            
            # 7. Trigger GitHub Actions for knowledge graph update
            self.trigger_github_workflow()
            
            success = training_success
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
        finally:
            self.close()
            
        return success
    
    def trigger_github_workflow(self):
        """Trigger GitHub Actions workflow to process artifacts"""
        try:
            # Create a marker file to trigger the workflow
            marker_file = self.local_base / f"experiments/{self.experiment_id}/ready_for_processing.json"
            marker_data = {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "trigger": "remote_training_complete"
            }
            
            with open(marker_file, 'w') as f:
                json.dump(marker_data, f, indent=2)
            
            # Commit and push to trigger workflow
            commands = [
                f"cd {self.local_base}",
                f"git add experiments/{self.experiment_id}",
                f"git add models/{self.experiment_id}",
                f'git commit -m "🤖 Remote training completed: {self.experiment_id}"',
                "git push"
            ]
            
            for cmd in commands:
                subprocess.run(cmd, shell=True, check=False)
            
            logger.info("Triggered GitHub workflow for artifact processing")
            
        except Exception as e:
            logger.error(f"Failed to trigger GitHub workflow: {e}")


def main():
    parser = argparse.ArgumentParser(description='Orchestrate remote GPU training')
    parser.add_argument('--host', required=True, help='Remote GPU server hostname/IP')
    parser.add_argument('--port', type=int, default=22, help='SSH port')
    parser.add_argument('--username', default='root', help='SSH username')
    parser.add_argument('--key-file', help='Path to SSH private key')
    parser.add_argument('--password', help='SSH password (not recommended)')
    parser.add_argument('--training-script', required=True, help='Training script to run')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # SSH configuration
    ssh_config = {
        'host': args.host,
        'port': args.port,
        'username': args.username
    }
    
    if args.key_file:
        ssh_config['key_file'] = args.key_file
    elif args.password:
        ssh_config['password'] = args.password
    else:
        # Try to find key in default location
        default_key = Path.home() / '.ssh/canidae_vast'
        if default_key.exists():
            ssh_config['key_file'] = str(default_key)
    
    # Dataset configuration
    dataset_config = {
        'name': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate
    }
    
    # Run orchestrator
    orchestrator = RemoteTrainingOrchestrator(ssh_config)
    success = orchestrator.run_complete_pipeline(
        args.training_script,
        dataset_config,
        args.gpu_count
    )
    
    if success:
        logger.info(f"✅ Training pipeline completed successfully!")
        logger.info(f"Artifacts saved to: models/{orchestrator.experiment_id}/")
    else:
        logger.error("❌ Training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()