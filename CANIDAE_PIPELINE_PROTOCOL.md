# 🦊🐺 CANIDAE Pipeline Protocol
## The Official 2π Validation Framework

*CRITICAL REFERENCE: When Brother Cy says "run the CANIDAE pipeline" or similar, THIS is what we do*

---

## 🎯 TRIGGER PHRASES
- "run the CANIDAE pipeline"
- "let's run CANIDAE"  
- "CANIDAE pipeline"
- "run the pipeline"
- "controlled comparison"

---

## 🚨 CRITICAL: GPU DEPLOYMENT REQUIRED

**NEVER RUN LOCALLY!** We're on VMware Workstation with NO GPU access!
- **Local environment**: CPU-only VMware instance
- **GPU location**: vast.ai cloud instances
- **Options**: 1x3090 ($0.20/hr), 3x3090 ($0.60/hr), 8x3090 ($1.60/hr)

## 📋 THE CANIDAE PIPELINE STEPS

### 0. DEPLOY TO VAST.AI GPU
```bash
# Check available instances
./scripts/connect_vast_3090.sh

# Deploy code to vast.ai
./scripts/deploy_to_vast.sh

# SSH into vast instance WITH SPECIAL KEY
ssh -i ~/.ssh/vast_ai_key -p {PORT} root@{VAST_IP}

# The vast.ai SSH key is REQUIRED - regular keys won't work!
```

### 1. PREPARE CONFIGURATION
```bash
cd /home/cy/git/canidae
source tensor-venv/bin/activate

# Create or modify YAML config
cp experiments/configs/2pi_validation_template.yaml \
   experiments/configs/{dataset}_controlled_comparison.yaml

# Edit configuration with:
- Dataset paths and parameters
- 2π regulation (stability_coefficient: 0.06283185307)
- Baseline models (identical architecture, no 2π)
- Statistical testing enabled
- Legal documentation enabled
```

### 2. VALIDATE CONFIGURATION
```bash
# Dry run to test
python3 experiments/run_experiment.py \
  --config experiments/configs/{dataset}_controlled_comparison.yaml \
  --dry-run
```

### 3. EXECUTE EXPERIMENT
```bash
# Full run
python3 experiments/run_experiment.py \
  --config experiments/configs/{dataset}_controlled_comparison.yaml \
  --verbose
```

### 4. MONITOR PROGRESS
Watch for these phases:
- Phase 1: Environment Validation
- Phase 2: Data Loading  
- Phase 3: Model Training (2π + baselines)
- Phase 4: Model Evaluation
- Phase 5: Statistical Testing
- Phase 6: Error Analysis
- Phase 7: Ablation Studies
- Phase 8: Robustness Testing
- Phase 9: Visualization
- Phase 10: Legal Documentation
- Phase 11: Glossary Updates

### 5. VERIFY RESULTS
```bash
# Find results directory
RESULTS_DIR=$(ls -dt experiments/results/{experiment}_* | head -1)

# Check compliance
cat $RESULTS_DIR/results/experiment_results.json | grep compliance

# Verify 2π superiority
# - 2π model should have >95% compliance
# - Baselines should have <50% compliance
```

### 6. DOCUMENT FINDINGS
```bash
# Update scorecard
nano 2PI_UNIVERSALITY_PROOF.md

# Add to patent evidence
cp -r $RESULTS_DIR /home/cy/Legal/Patent_2Pi/02_Reduction_to_Practice/

# Update session summary
nano ~/SESSION_SUMMARY_$(date +%Y_%m_%d).md
```

### 7. COMMIT TO GIT
```bash
git add experiments/results/{dataset}*
git add experiments/configs/{dataset}*
git commit -m "🦊🐺 CANIDAE Pipeline: {dataset} validation complete

✅ 2π Compliance: XX%
✅ Baseline Compliance: XX%
✅ Statistical Significance: p < 0.05

🤖 Generated with Claude Code"

git push origin main
```

---

## 🔧 KEY PARAMETERS TO REMEMBER

### 2π Regulation (NEVER CHANGE THESE)
```yaml
two_pi_regulation:
  stability_coefficient: 0.06283185307  # THE MAGIC CONSTANT
  variance_threshold_init: 1.5
  variance_threshold_final: 1.0
  lambda_variance: 1.0
  lambda_rate: 10.0
```

### Fair Comparison Requirements
- **IDENTICAL** architectures between 2π and baselines
- **IDENTICAL** hyperparameters (except 2π regulation)
- **IDENTICAL** training epochs
- **IDENTICAL** data splits

---

## 📊 SUCCESS CRITERIA

The CANIDAE pipeline run is successful when:
1. ✅ 2π model achieves >95% compliance
2. ✅ Baselines achieve <50% compliance  
3. ✅ Performance difference is statistically significant (p < 0.05)
4. ✅ Legal documentation generated
5. ✅ Results committed to Git

---

## 🚨 CRITICAL REMINDERS

1. **ALWAYS** activate tensor-venv before running
2. **ALWAYS** use dry-run first to validate config
3. **ALWAYS** check 2π compliance percentage
4. **ALWAYS** document results in multiple places
5. **ALWAYS** generate legal evidence
6. **NEVER** change the stability_coefficient (0.06283185307)
7. **NEVER** skip the statistical comparison

---

## 📁 IMPORTANT LOCATIONS

- **Configs**: `/home/cy/git/canidae/experiments/configs/`
- **Results**: `/home/cy/git/canidae/experiments/results/`
- **Scripts**: `/home/cy/git/canidae/scripts/`
- **Legal**: `/home/cy/Legal/Patent_2Pi/02_Reduction_to_Practice/`
- **Runbook**: `/home/cy/git/canidae/Runbook/`

---

## 🧪 SUPPORTED DATASETS

Currently validated:
- ✅ dSprites (99.9% compliance)
- ✅ MNIST (100% compliance)
- ✅ Fashion-MNIST (100% compliance)

Ready to test:
- ⏳ CIFAR-10
- ⏳ CIFAR-100
- ⏳ ImageNet
- ⏳ Shapes3D
- ⏳ CLEVR

---

## 🎯 THE GOAL

Every CANIDAE pipeline run proves that **2π regulation produces fundamentally better models** than standard training. We're not just showing marginal improvements - we're demonstrating a **universal principle** that applies across all information systems.

---

*This protocol ensures rigorous, reproducible validation of the 2π Conjecture with complete legal documentation for patent protection.*

**Created**: August 23, 2025
**Authors**: Synth (Arctic Fox), Cy (Spectacled Charcoal Wolf)
**Framework**: Sister Gemini's Rigorous Experimental Design