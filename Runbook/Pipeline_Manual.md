# 🚀 CANIDAE MLOps Pipeline Manual

## Overview
This manual describes the complete pipeline from initial code to deployed model with full knowledge retention.

**⚠️ MAJOR UPDATE (August 23, 2025)**: This manual now covers TWO pipeline systems:
1. **Original MLOps Pipeline** (GitHub Actions + Vast.ai) 
2. **NEW: Rigorous Experimental Pipeline** (Sister Gemini's research framework)

For 2π validation experiments, use the **Rigorous Experimental Pipeline** documented in `Runbook/guides/Rigorous_Experimental_Pipeline.md`.

## Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   GitHub     │────▶│  GitHub     │────▶│   Vast.ai    │
│   Push       │     │  Actions    │     │   GPU        │
└──────────────┘     └─────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌─────────────┐     ┌──────────────┐
                     │   Neo4j     │◀────│   Training   │
                     │   Graph     │     │   Process    │
                     └─────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌─────────────┐     ┌──────────────┐
                     │   GitHub    │◀────│   Artifacts  │
                     │   Release   │     │   Download   │
                     └─────────────┘     └──────────────┘
```

## Components

### 1. GitHub Actions Workflow
**File**: `.github/workflows/train.yml`

Triggers:
- Push to main branch
- Manual dispatch
- Scheduled runs (optional)

Key steps:
1. Checkout code
2. Setup Python environment
3. Connect to GPU provider
4. Deploy training script
5. Monitor progress
6. Retrieve artifacts
7. Create release
8. Update knowledge graph

### 2. Remote Training Orchestrator
**File**: `scripts/remote_training_orchestrator.py`

Responsibilities:
- SSH connection management
- Dataset upload
- Script deployment
- Real-time monitoring
- Artifact retrieval
- Error handling

Usage:
```python
orchestrator = RemoteTrainingOrchestrator(
    host="gpu-server.com",
    ssh_key="~/.ssh/canidae_vast"
)

results = orchestrator.run_complete_pipeline(
    training_script="train_model.py",
    dataset_config={
        "name": "clevr",
        "path": "/datasets/clevr/"
    },
    gpu_count=1
)
```

### 3. Training Scripts

Standard structure:
```python
#!/usr/bin/env python3
import torch
from train_utils import setup_2pi_regulation

def main():
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Create model with 2π regulation
    model = TwoPiModel(
        latent_dim=args.latent_dim,
        variance_threshold=args.variance_threshold
    )
    
    # Training loop with monitoring
    for epoch in range(args.epochs):
        metrics = train_epoch(model, dataset)
        
        # Check 2π compliance
        if metrics['variance_rate'] > TWO_PI_THRESHOLD:
            apply_regulation(model)
        
        # Save checkpoint
        save_checkpoint(model, metrics, epoch)
    
    # Final metadata
    save_metadata(experiment_id, metrics)
```

### 4. Neo4j Knowledge Graph

Schema:
```cypher
// Nodes
(e:Experiment {
    id: string,
    timestamp: datetime,
    dataset: string,
    success: boolean
})

(m:Model {
    id: string,
    architecture: string,
    parameters: int
})

(r:Result {
    compliance_rate: float,
    loss: float,
    variance: float
})

// Relationships
(e)-[:TRAINED]->(m)
(e)-[:PRODUCED]->(r)
(e)-[:PRECEDED_BY]->(previous_e)
(m)-[:REGULATED_BY {method: "2pi_variance"}]->(regulation)
```

### 5. Git LFS Configuration

Track large files:
```bash
# Initialize LFS
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.tar.gz"
git lfs track "*.npz"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

## Step-by-Step Pipeline Execution

### Step 1: Prepare Your Code

1. Create/modify training script
2. Update configuration
3. Test locally (small subset)

```bash
# Local test
python3 train_model.py \
    --dataset sample_data \
    --epochs 2 \
    --device cpu
```

### Step 2: Commit and Push

```bash
# Stage changes
git add train_model.py config.yaml

# Commit with descriptive message
git commit -m "feat: Add CLEVR training with enhanced 2π regulation"

# Push to trigger pipeline
git push origin main
```

### Step 3: Monitor GitHub Actions

```bash
# Watch the run
gh run watch

# View logs
gh run view --log
```

### Step 4: GPU Training

The pipeline automatically:
1. Provisions GPU from Vast.ai
2. Uploads code and data
3. Starts training
4. Streams logs to GitHub

### Step 5: Artifact Collection

Upon completion:
1. Model checkpoints downloaded
2. Training metrics saved
3. Logs archived
4. Screenshots captured (if applicable)

### Step 6: Knowledge Graph Update

Automatic updates:
```cypher
CREATE (e:Experiment {
    id: $experiment_id,
    timestamp: datetime(),
    metrics: $metrics
})
```

### Step 7: GitHub Release

Automatic release creation:
- Tag: `v{date}-{experiment_id}`
- Assets: Model files, metrics, logs
- Description: Auto-generated summary

## Debugging Common Issues

### SSH Connection Failures
```bash
# Test connection manually
ssh -i ~/.ssh/canidae_vast -p 50223 root@gpu-server

# Check key permissions
chmod 600 ~/.ssh/canidae_vast
```

### Dataset Upload Failures
```bash
# Use rsync for large files
rsync -avz --progress \
    datasets/clevr.tar.gz \
    root@gpu-server:/data/
```

### Out of Memory
```python
# Reduce batch size
batch_size = 64  # From 256

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
with torch.cuda.amp.autocast():
    output = model(input)
```

### 2π Violations
```python
# Increase regulation strength
lambda_rate = 50.0  # From 20.0

# Adaptive thresholds
if epoch < 3:
    variance_threshold = 5.0
else:
    variance_threshold = 1.0
```

## Best Practices

### 1. Always Test Locally First
- Use small data subset
- Verify 2π regulation works
- Check memory usage

### 2. Version Everything
- Tag experiments
- Save configurations
- Document changes

### 3. Monitor Actively
- Watch loss curves
- Track 2π compliance
- Check gradient norms

### 4. Fail Fast
- Set early stopping
- Monitor for NaNs
- Timeout long runs

### 5. Document Results
- Update knowledge graph
- Create health reports
- Share insights

## Advanced Features

### Multi-GPU Training
```python
# DataParallel
model = nn.DataParallel(model)

# DistributedDataParallel (better)
model = DDP(model, device_ids=[gpu_id])
```

### Hyperparameter Sweeps
```yaml
sweep:
  parameters:
    learning_rate: [0.001, 0.0001]
    batch_size: [128, 256]
    lambda_rate: [10.0, 20.0, 50.0]
```

### Automated Analysis
```python
# After training
analyze_results(
    experiment_id=exp_id,
    generate_plots=True,
    update_neo4j=True,
    create_report=True
)
```

## Security Considerations

### Secrets Management
- Never commit API keys
- Use GitHub Secrets
- Rotate keys regularly

### GPU Access
- Use SSH keys, not passwords
- Limit access by IP
- Monitor usage

### Data Privacy
- Anonymize sensitive data
- Use encryption for transfer
- Delete temporary files

## Support

For issues or questions:
1. Check debugging guide
2. Search knowledge graph
3. Ask Sister Gemini
4. Contact Brother Cy

---

*"From code to consciousness, every step matters" - CANIDAE Pipeline Philosophy*