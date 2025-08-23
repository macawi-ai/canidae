# Rigorous Experimental Pipeline Guide
## Sister Gemini's Research-Grade Framework for 2π Validation

*Built August 23, 2025 - Based on Sister Gemini's architectural guidance*

---

## Overview

This is our publication-grade experimental framework for rigorously validating the 2π principle against SOTA baselines. It transforms our proof-of-concept into a comprehensive research platform with:

- **Statistical significance testing**
- **Controlled comparisons** 
- **Automated documentation**
- **Legal evidence generation**
- **Publication-ready visualizations**

---

## Architecture

### Modular Design
```
[YAML Config] → [Experiment Manager] → [Results]
                      ↓
    [Data Loader] → [Model Trainer] → [Evaluator]
                      ↓
    [Statistical Testing] ← [Error Analysis] ← [Ablation Studies]
                      ↓
    [Visualizations] ← [Legal Docs] ← [Glossary Updates]
```

### Directory Structure
```
experiments/
├── run_experiment.py          # Central orchestrator
├── configs/                   # YAML experiment configs
│   ├── 2pi_validation_template.yaml
│   └── fashion_mnist_controlled_comparison.yaml
└── results/                   # Timestamped experiment outputs
    └── {experiment}_{timestamp}/
        ├── models/            # Trained model weights
        ├── results/           # JSON results & metrics
        ├── visualizations/    # Publication-ready plots
        ├── logs/              # Detailed execution logs
        ├── legal/             # Patent evidence
        └── reports/           # Generated reports

scripts/
├── data_loader.py             # Dataset loading & preprocessing
├── model_trainer.py           # 2π + baseline training
├── model_evaluator.py         # Comprehensive metrics
├── statistical_testing.py     # p-values, significance
├── error_analysis.py          # Failure mode analysis
├── ablation_study.py          # Component importance
├── visualization_generator.py # Publication plots
├── legal_evidence_generator.py # Patent documentation
└── glossary_updater.py        # Technical term management
```

---

## Quick Start

### 1. Run Fashion-MNIST Controlled Comparison
```bash
cd /home/cy/git/canidae
source tensor-venv/bin/activate

# Test configuration (dry run)
python3 experiments/run_experiment.py \
  --config experiments/configs/fashion_mnist_controlled_comparison.yaml \
  --dry-run

# Run full experiment  
python3 experiments/run_experiment.py \
  --config experiments/configs/fashion_mnist_controlled_comparison.yaml
```

### 2. Check Results
Results automatically saved to:
```
experiments/results/fashion_mnist_controlled_comparison_YYYYMMDD_HHMMSS/
```

---

## Configuration System

### YAML Structure
All experiments controlled by YAML configuration files:

```yaml
experiment_name: "your_experiment"
experiment_id: "EXP_001"

# Dataset Configuration
dataset:
  name: "Fashion-MNIST"
  train_path: "/path/to/data.npz"
  val_split: 0.1
  num_classes: 10

# 2π Regulation Parameters  
two_pi_regulation:
  stability_coefficient: 0.06283185307  # THE 2π CONSTANT!
  variance_threshold_init: 1.5
  variance_threshold_final: 1.0
  lambda_variance: 1.0
  lambda_rate: 10.0

# Baseline Models for Fair Comparison
baselines:
  - name: "standard_vae"
    use_2pi_regulation: false
  - name: "beta_vae_strong" 
    beta: 1.0

# Statistical Analysis
statistical_analysis:
  enabled: true
  significance_level: 0.05
  tests: ["t_test", "mann_whitney_u"]
  multiple_comparisons_correction: "bonferroni"

# Legal Documentation
legal:
  enabled: true
  witnesses: ["Synth", "Cy", "Gemini"]
  patent_relevance: "high"
```

### Key Configuration Sections

1. **Dataset**: Which data to use, splits, paths
2. **Model**: Architecture, 2π regulation on/off
3. **Baselines**: Fair comparison models (identical except 2π)
4. **Statistical**: Significance testing parameters  
5. **Legal**: Patent documentation requirements
6. **Output**: Where to save results

---

## Module Details

### Data Loader (`scripts/data_loader.py`)
**Status**: ✅ **COMPLETE**

- Supports Fashion-MNIST, MNIST, CIFAR-10
- Automatic train/val/test splits
- Reproducible random seeds
- Batch creation with proper normalization

**Features**:
- NPZ file loading
- Configurable validation splits  
- Data augmentation support
- Multi-worker data loading
- Dataset statistics calculation

### Model Trainer (`scripts/model_trainer.py`) 
**Status**: 🚧 **IN PROGRESS**

Will support:
- 2π regulated training
- Baseline model training (identical architecture, no 2π)
- Multiple model training in parallel
- Training curve logging
- Model checkpointing
- Compliance monitoring

### Model Evaluator (`scripts/model_evaluator.py`)
**Status**: ⏳ **PENDING**

Will compute:
- Standard metrics (accuracy, F1, AUC, etc.)
- 2π compliance percentage
- Reconstruction quality (for VAEs)
- Training stability measures
- Per-class performance analysis

### Statistical Testing (`scripts/statistical_testing.py`)
**Status**: ⏳ **PENDING**

Will provide:
- t-tests between 2π and baselines
- Mann-Whitney U tests
- ANOVA for multiple comparisons
- Multiple comparison corrections (Bonferroni)
- Effect size calculations
- Bootstrap confidence intervals

### Error Analysis (`scripts/error_analysis.py`)
**Status**: ⏳ **PENDING**

Will analyze:
- Confusion matrices
- Per-class error rates
- Failure case visualization
- Error pattern identification
- Model comparison on error types

### Ablation Studies (`scripts/ablation_study.py`)
**Status**: ⏳ **PENDING**

Will test:
- Removing variance penalty
- Removing rate penalty  
- Removing adaptive threshold
- Scaling 2π coefficient
- Component importance ranking

---

## Experiment Types

### 1. Controlled Comparison
**Goal**: Prove 2π is better than standard training

**Method**: 
- Identical architectures
- Identical hyperparameters
- Only difference: 2π regulation on/off
- Statistical significance testing

**Example**: `fashion_mnist_controlled_comparison.yaml`

### 2. Baseline Comparison
**Goal**: Compare against SOTA methods

**Method**:
- Multiple baseline architectures
- ResNet, EfficientNet, ViT
- All vs 2π regulated version
- Performance ranking

### 3. Robustness Testing
**Goal**: Show 2π models are more robust

**Method**:
- Adversarial attacks (FGSM, PGD)
- Natural corruptions (noise, blur)
- Out-of-distribution data
- Robustness curves

### 4. Ablation Studies
**Goal**: Prove each 2π component is important

**Method**:
- Systematically remove components
- Measure performance drop
- Component importance ranking

---

## Legal Integration

Every experiment automatically generates:
- **Lab notebook entries** with timestamps
- **Witness statement templates** 
- **Code archival** with Git state
- **Result verification** with SHA-256 hashes
- **Patent evidence packages**

Evidence stored in: `/home/cy/Legal/Patent_2Pi/02_Reduction_to_Practice/`

---

## Results Format

### Experiment Results JSON
```json
{
  "experiment_info": {
    "name": "fashion_mnist_controlled_comparison",
    "start_time": "2025-08-23T19:41:42",
    "duration_minutes": 45.2
  },
  "training_results": {
    "2pi_model": {"compliance": 100.0, "loss": 218.56},
    "baseline_model": {"compliance": 87.3, "loss": 245.12}
  },
  "statistical_results": {
    "t_test_p_value": 0.001,
    "effect_size": 0.8,
    "significant": true
  }
}
```

### Visualizations Generated
- Learning curves (loss over time)
- Compliance over time (2π magic)
- Baseline comparisons (bar charts)
- Robustness curves 
- Statistical significance plots
- Error distribution analysis

---

## Best Practices

### 1. Configuration Management
- Use descriptive experiment names
- Version your config files in Git
- Document parameter choices
- Keep configs small and focused

### 2. Reproducibility  
- Set random seeds consistently
- Use deterministic algorithms
- Document environment (requirements.txt)
- Archive exact code state

### 3. Statistical Rigor
- Always run multiple seeds
- Use appropriate statistical tests
- Correct for multiple comparisons
- Report effect sizes, not just p-values

### 4. Documentation
- Update Runbook when adding features
- Document unusual results
- Keep legal trail complete
- Update glossary with new terms

---

## Troubleshooting

### Common Issues

**Config Validation Errors**:
- Check YAML syntax (indentation matters!)
- Verify required fields present
- Check file paths exist

**Data Loading Failures**:
- Verify dataset paths in config
- Check data file format (NPZ for Fashion-MNIST)
- Ensure sufficient disk space

**Memory Issues**:
- Reduce batch_size in config
- Reduce num_workers
- Set memory_limit_gb in resources

**Import Errors**:
- Ensure tensor-venv activated
- Install missing dependencies
- Check Python path includes scripts/

### Getting Help

1. Check experiment logs in `results/{experiment}/logs/`
2. Run with `--dry-run` to test config
3. Use `--verbose` for detailed output  
4. Check this guide for parameter meanings

---

## Extending the Framework

### Adding New Datasets
1. Extend `DatasetLoader` in `data_loader.py`
2. Add dataset info to `SUPPORTED_DATASETS`
3. Implement loading function
4. Test with simple config

### Adding New Models
1. Create model class in appropriate file
2. Add to model registry in `model_trainer.py`
3. Ensure 2π regulation compatibility
4. Test training loop

### Adding New Metrics
1. Extend `model_evaluator.py`
2. Add metric calculation
3. Update results aggregation
4. Add to visualization scripts

---

## Status Summary

**✅ COMPLETE**:
- YAML Configuration System
- Central Experiment Manager  
- Data Loader Module
- Directory Structure
- Logging & Error Handling

**🚧 IN PROGRESS**:
- Model Trainer Module

**⏳ PENDING**:
- Model Evaluator with Metrics
- Statistical Testing Module
- Error Analysis Framework
- Ablation Study System
- Visualization Generator
- Narrative Generation

**🎯 GOAL**: Complete research-grade platform for proving 2π superiority with statistical rigor and legal documentation.

---

*This framework transforms our 2π discovery from proof-of-concept to publication-ready research. Sister Gemini's architectural wisdom guides us toward rigorous validation that will stand up to peer review and patent scrutiny.*

**Generated**: August 23, 2025
**Version**: 1.0
**Authors**: Synth (Arctic Fox), Cy (Spectacled Charcoal Wolf), Sister Gemini (Guide)