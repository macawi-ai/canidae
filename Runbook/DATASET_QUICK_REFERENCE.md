# 📋 Dataset Management Quick Reference Card
## Essential Commands for 2π Research

---

## 🚀 Daily Workflow

### Morning: Check Status
```bash
# See what datasets we have
python3 datasets/inventory_scanner.py

# Check recent results
python3 datasets/results_comparison.py

# View current priorities
grep -A5 "Testing Priority" Runbook/Dataset_Training_Curriculum.md
```

### Testing a Dataset
```bash
# 1. Check if dataset has metadata
cat datasets/phase2/DATASET_NAME/metadata.yaml

# 2. Run 2π experiment
python3 train_vae_2pi_universal.py --dataset DATASET_NAME

# 3. Update metadata with results
vim datasets/phase2/DATASET_NAME/metadata.yaml

# 4. Compare results
python3 datasets/results_comparison.py
```

### Adding New Dataset
```bash
# 1. Create directory
mkdir -p datasets/phase3/NEW_DATASET

# 2. Copy data files
cp /path/to/data.npz datasets/phase3/NEW_DATASET/

# 3. Create metadata
cat > datasets/phase3/NEW_DATASET/metadata.yaml << EOF
name: New Dataset
modality: image
format: npz
sparsity: medium
2pi_tested: false
priority: HIGH
EOF

# 4. Run scanner
python3 datasets/inventory_scanner.py
```

---

## 📊 Current Dataset Status

### ✅ Proven (2π Tested)
```
dSprites: 99.9% compliance ✅
```

### 🎯 High Priority Queue
```
1. Shapes3D    - Ready (255MB)
2. MNIST       - Need download
3. Fashion     - Need download  
4. Geometric   - Need generation
5. QuickDraw   - Need conversion
```

### 📁 Available Datasets
```
Total: 697 files
- QuickDraw: 346 categories (need conversion)
- Shapes3D: 1 file (ready)
- dSprites: ✅ Tested
```

---

## 🔧 2π Configurations

### Standard (Dense Images)
```python
{
    "stability_coefficient": 0.06283185307,  # 2π/100
    "variance_threshold_init": 1.5,
    "variance_threshold_final": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0
}
```

### Sparse (Sketches)
```python
{
    "stability_coefficient": 0.06283185307,  # Same!
    "variance_threshold_init": 3.0,  # Higher
    "variance_threshold_final": 1.5,  # Higher
    "lambda_variance": 1.5,
    "lambda_rate": 15.0
}
```

---

## 📈 Key Metrics to Track

1. **2π Compliance** - Must be >95%
2. **Reconstruction Loss** - Lower is better
3. **Purple Line Events** - Count violations
4. **Training Time** - For efficiency
5. **Final Variance** - Should stabilize

---

## 🔍 Useful Queries

### Find untested datasets
```bash
find datasets -name "metadata.yaml" -exec grep -l "2pi_tested: false" {} \;
```

### Check all compliance rates
```bash
grep -h "2pi_compliance" datasets/*/metadata.yaml | sort -n
```

### Find high priority datasets
```bash
grep -B2 "priority: HIGH" datasets/*/metadata.yaml
```

### Calculate total dataset size
```bash
du -sh datasets/phase*
```

---

## 📝 Metadata Template

```yaml
name: Dataset Name
modality: image|sketch|synthetic
format: npz|npy|h5
sparsity: low|medium|high
2pi_tested: false
2pi_compliance: null
reconstruction_loss: null
optimal_2pi_config:
  variance_threshold_init: 1.5
  variance_threshold_final: 1.0
  lambda_variance: 1.0
  lambda_rate: 10.0
priority: HIGH|MEDIUM|LOW
notes: "Ready for testing"
```

---

## 🚨 Common Issues

**Scanner not finding datasets**:
```bash
# Check permissions
ls -la datasets/

# Look for hidden files
find datasets -name ".*"
```

**Metadata parsing error**:
```bash
# Validate YAML
python3 -c "import yaml; yaml.safe_load(open('metadata.yaml'))"
```

**Low 2π compliance**:
- Increase lambda_rate
- Adjust variance thresholds for sparsity
- Check for Purple Line events in logs

---

## 📞 Quick Links

- [Full Documentation](guides/Dataset_Management_System.md)
- [Training Curriculum](Dataset_Training_Curriculum.md)
- [2π Paper](../paper/2pi_regulation_arxiv.md)
- [Pipeline Manual](Pipeline_Manual.md)

---

*Keep this card handy during experiments!*
*Last Updated: 2025-08-22*
*The Pack Discovers Together 🦊🐺✨*