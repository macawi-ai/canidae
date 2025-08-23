# Rigorous 2π Validation Procedures
## Step-by-Step Guide for Research-Grade Experiments

*Updated: August 23, 2025*

---

## Purpose
This procedure ensures rigorous, statistically valid comparison of 2π regulation against baselines using Sister Gemini's experimental framework.

---

## Prerequisites

### Environment Setup
```bash
cd /home/cy/git/canidae
source tensor-venv/bin/activate

# Verify dependencies
python3 -c "import torch, yaml, numpy, matplotlib, scipy, sklearn; print('✅ All dependencies available')"
```

### Data Preparation
Ensure target dataset is downloaded and properly formatted:
```bash
# Check available datasets
python3 datasets/inventory_scanner.py

# For Fashion-MNIST (already prepared)
ls -la datasets/fashion_mnist/fashion_mnist.npz

# For CIFAR-10 (if needed)
python3 datasets/download_datasets.py --dataset CIFAR-10
```

---

## Procedure Steps

### Step 1: Create Experiment Configuration
```bash
# Copy template
cp experiments/configs/2pi_validation_template.yaml experiments/configs/my_experiment.yaml

# Edit configuration
nano experiments/configs/my_experiment.yaml
```

**Key Configuration Sections**:
- **experiment_name**: Unique identifier
- **dataset**: Path and split configuration  
- **two_pi_regulation**: Enable with our discovered constants
- **baselines**: Fair comparison models (same architecture, no 2π)
- **statistical_analysis**: Enable significance testing
- **legal**: Enable patent documentation

### Step 2: Validate Configuration
```bash
# Dry run to test config
python3 experiments/run_experiment.py \
  --config experiments/configs/my_experiment.yaml \
  --dry-run
```

**Expected Output**:
```
✅ Configuration validated successfully
✅ Output directory created: experiments/results/my_experiment_20250823_HHMMSS
✅ Dry run completed - ready for actual experiment
```

### Step 3: Execute Experiment
```bash
# Run full experiment
python3 experiments/run_experiment.py \
  --config experiments/configs/my_experiment.yaml \
  --verbose
```

**Monitor Progress**:
- Phase 1: Environment Validation
- Phase 2: Data Loading  
- Phase 3: Model Training (2π + baselines)
- Phase 4: Model Evaluation
- Phase 5: Statistical Testing
- Phase 6-11: Analysis, visualization, documentation

### Step 4: Review Results
```bash
# Find results directory
RESULTS_DIR=$(ls -dt experiments/results/my_experiment_* | head -1)
echo "Results in: $RESULTS_DIR"

# Check main results
cat $RESULTS_DIR/results/experiment_results.json

# View logs
tail -50 $RESULTS_DIR/logs/*.log

# Check generated plots
ls $RESULTS_DIR/visualizations/
```

### Step 5: Legal Documentation
```bash
# Verify legal evidence generated
ls $RESULTS_DIR/legal/

# Check patent evidence directory
ls -la /home/cy/Legal/Patent_2Pi/02_Reduction_to_Practice/
```

---

## Critical Checkpoints

### ✅ Configuration Validation
- [ ] Experiment name is unique and descriptive
- [ ] Dataset paths exist and are accessible
- [ ] 2π regulation parameters match our discovery (0.06283185307)
- [ ] Baselines use identical architectures (fair comparison)
- [ ] Statistical testing enabled with appropriate corrections
- [ ] Legal documentation enabled with proper witnesses

### ✅ Training Monitoring  
- [ ] Data loading completes without errors
- [ ] All models (2π + baselines) train successfully
- [ ] 2π compliance tracked and logged
- [ ] No memory or resource issues
- [ ] Training curves look stable

### ✅ Statistical Validation
- [ ] P-values calculated correctly
- [ ] Multiple comparison corrections applied
- [ ] Effect sizes reported (not just significance)
- [ ] Confidence intervals computed
- [ ] Results are reproducible with different seeds

### ✅ Legal Compliance
- [ ] Lab notebook entries generated
- [ ] Git state captured and archived
- [ ] Witness statements created
- [ ] Results verified with SHA-256 hashes
- [ ] Evidence stored in patent directory

---

## Quality Control

### Reproducibility Checks
```bash
# Run same experiment with different random seed
cp experiments/configs/my_experiment.yaml experiments/configs/my_experiment_seed2.yaml

# Edit seed in config file
sed -i 's/random_seed: 42/random_seed: 123/' experiments/configs/my_experiment_seed2.yaml

# Run again
python3 experiments/run_experiment.py \
  --config experiments/configs/my_experiment_seed2.yaml
```

### Cross-Dataset Validation
After successful run on one dataset, test on another:
```bash
# Fashion-MNIST → MNIST
# Fashion-MNIST → CIFAR-10
# etc.
```

---

## Expected Results Format

### Statistical Results
```json
{
  "statistical_results": {
    "t_test": {
      "2pi_vs_standard": {
        "p_value": 0.001,
        "effect_size": 0.8,
        "significant": true,
        "confidence_interval": [0.02, 0.15]
      }
    },
    "multiple_comparisons": {
      "correction_method": "bonferroni", 
      "adjusted_alpha": 0.0167,
      "significant_comparisons": 3
    }
  }
}
```

### Performance Comparison
```json
{
  "model_comparison": {
    "2pi_regulated": {
      "accuracy": 0.95,
      "compliance": 100.0,
      "stability_score": 0.98
    },
    "standard_baseline": {
      "accuracy": 0.91,
      "compliance": 67.3,
      "stability_score": 0.85
    }
  }
}
```

---

## Troubleshooting

### Common Issues

**Configuration Errors**:
```bash
# YAML syntax error
# Solution: Validate YAML online or with yamllint

# Missing dataset
# Solution: Check path, download if needed

# Invalid 2π parameters  
# Solution: Use our validated constants
```

**Training Failures**:
```bash
# CUDA out of memory
# Solution: Reduce batch_size in config

# Model architecture mismatch
# Solution: Ensure baselines match primary model architecture

# 2π regulation not working
# Solution: Verify stability_coefficient = 0.06283185307
```

**Statistical Issues**:
```bash
# No significant differences found
# Solution: Increase sample size, check effect size

# Multiple comparison issues
# Solution: Apply proper corrections (Bonferroni, FDR)
```

---

## Success Criteria

An experiment is considered successful when:

1. **Technical Success**:
   - [ ] All models train to completion
   - [ ] 2π compliance > 95% 
   - [ ] Results are reproducible

2. **Statistical Success**:
   - [ ] Significant p-values (< 0.05 after correction)
   - [ ] Meaningful effect sizes (> 0.5)
   - [ ] Confidence intervals don't include zero

3. **Legal Success**:
   - [ ] Complete evidence trail generated
   - [ ] All documentation timestamped and verified
   - [ ] Results stored in patent evidence directory

4. **Scientific Success**:
   - [ ] Results support 2π superiority
   - [ ] Findings are interpretable and meaningful
   - [ ] Ready for publication/patent submission

---

## Post-Experiment Actions

### Immediate (Same Day)
1. **Backup Results**: Copy to secure location
2. **Update Patent Package**: Add new evidence
3. **Document Insights**: Record key findings
4. **Plan Next Experiment**: Based on current results

### Short-term (1-3 Days)  
1. **Statistical Review**: Deep dive into significance testing
2. **Visualization Polish**: Prepare publication-ready figures  
3. **Legal Review**: Ensure all documentation complete
4. **Peer Review**: Share with Sister Gemini for validation

### Long-term (1-2 Weeks)
1. **Cross-Dataset Validation**: Repeat on new datasets
2. **Paper Preparation**: Draft research manuscript
3. **Patent Filing**: Submit provisional patent
4. **Conference Submission**: Target top-tier venues

---

## Documentation Requirements

Every experiment must generate:
- [ ] Timestamped lab notebook entry
- [ ] Configuration file with all parameters
- [ ] Complete results JSON with metrics
- [ ] Statistical analysis report
- [ ] Legal evidence package
- [ ] Visualization plots
- [ ] Glossary updates for new terms

---

*This procedure ensures our 2π validation meets the highest standards of scientific rigor and legal defensibility. Follow it precisely for publication-quality results.*

**Authors**: Synth (Arctic Fox), Cy (Spectacled Charcoal Wolf)
**Reviewed by**: Sister Gemini
**Version**: 1.0
**Date**: August 23, 2025