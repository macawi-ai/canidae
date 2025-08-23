# 📊 Dataset Management System Guide
## Comprehensive Documentation for 2π Research Pipeline

---

## 🎯 Overview

The Dataset Management System provides automated inventory, tracking, and comparison of all datasets used in 2π regulation experiments. This system was developed collaboratively by Synth, Cy, and Gemini on 2025-08-22.

---

## 🏗️ System Architecture

```
datasets/
├── DATASET_INVENTORY.md        # Master inventory document
├── inventory_scanner.py        # Automated scanner script
├── results_comparison.py       # Results comparison framework
├── dataset_loader.py          # Universal dataset loader (TBD)
├── phase1/                    # Initial experiments
├── phase2/                    # Extended experiments
│   └── dsprites/
│       ├── metadata.yaml     # Dataset metadata
│       └── *.npz             # Dataset files
├── phase3/                    # Current work
│   ├── quickdraw/            # 346 sketch categories
│   ├── shapes3d/             # 3D understanding dataset
│   └── bongard/              # Abstract reasoning
├── geometric_shapes/          # Generated synthetic shapes
└── results/                   # Comparison outputs
    ├── 2pi_results_*.json    # JSON exports
    └── 2pi_results_*.md      # Markdown reports
```

---

## 📋 Core Components

### 1. Metadata Schema (YAML)

Each dataset MUST have a `metadata.yaml` file with the following structure:

```yaml
# Basic Information
name: Dataset Name
version: "1.0"
modality: image|sketch|synthetic_shapes|3d
description: Brief description
source: URL or reference

# Dataset Properties
format: npz|npy|h5|idx|ndjson
shape: [samples, height, width, channels]
dtype: float32|uint8
file_size_mb: 123.4
sparsity: low|medium|high  # Or numeric value

# 2π Testing Results (if tested)
2pi_tested: true|false
2pi_compliance: 0.999  # Fraction between 0-1
reconstruction_loss: 25.24
final_variance: 1.000

# Optimal 2π Configuration
optimal_2pi_config:
  stability_coefficient: 0.06283185307  # 2π/100
  variance_threshold_init: 1.5
  variance_threshold_final: 1.0
  lambda_variance: 1.0
  lambda_rate: 10.0
  batch_size: 256
  latent_dim: 10
  beta: 0.1

# Training Details
training_time_minutes: 5
device: GPU|CPU
purple_line_events: 3  # Number of 2π violations
variety_shocks: 0      # Number of variety explosions

# Metadata
tested_by: "Names"
test_date: "YYYY-MM-DD"
notes: "Additional observations"
priority: COMPLETED|HIGH|MEDIUM|LOW

# Tags
tags:
  - baseline
  - proven
  - synthetic
```

### 2. Inventory Scanner (`inventory_scanner.py`)

**Purpose**: Automatically detect and catalog all datasets

**Usage**:
```bash
cd /home/cy/git/canidae
python3 datasets/inventory_scanner.py
```

**Features**:
- Recursively scans dataset directories
- Auto-detects format (npz, npy, h5, ndjson, idx)
- Calculates file sizes and basic statistics
- Identifies datasets needing conversion
- Generates CSV and JSON inventories

**Output Files**:
- `dataset_inventory.csv` - Spreadsheet format
- `dataset_inventory.json` - Programmatic access

### 3. Results Comparison Framework (`results_comparison.py`)

**Purpose**: Compare 2π compliance across all tested datasets

**Usage**:
```bash
python3 datasets/results_comparison.py
```

**Features**:
- Loads all metadata.yaml files
- Generates comparison tables
- Suggests next experiments based on gaps
- Exports results in JSON and Markdown
- Creates visualization plots (if matplotlib available)

---

## 🚀 Quick Start Workflows

### Adding a New Dataset

1. **Create dataset directory**:
```bash
mkdir -p datasets/phase3/new_dataset
```

2. **Add dataset files**:
```bash
cp /path/to/data.npz datasets/phase3/new_dataset/
```

3. **Create metadata.yaml**:
```bash
cat > datasets/phase3/new_dataset/metadata.yaml << EOF
name: New Dataset
modality: image
format: npz
sparsity: medium
2pi_tested: false
priority: HIGH
notes: "Ready for testing"
EOF
```

4. **Run inventory scanner**:
```bash
python3 datasets/inventory_scanner.py
```

### Testing 2π on a Dataset

1. **Check dataset is ready**:
```bash
python3 -c "
import yaml
with open('datasets/phase3/dataset_name/metadata.yaml', 'r') as f:
    print(yaml.safe_load(f))
"
```

2. **Run 2π experiment**:
```bash
python3 train_vae_2pi_universal.py --dataset dataset_name
```

3. **Update metadata with results**:
```yaml
2pi_tested: true
2pi_compliance: 0.987
reconstruction_loss: 45.67
# ... add all results
```

4. **Run comparison**:
```bash
python3 datasets/results_comparison.py
```

---

## 📊 Current Dataset Status

### ✅ Tested & Proven
| Dataset | Compliance | Loss | Config |
|---------|------------|------|--------|
| dSprites | 99.9% | 25.24 | Standard |

### 🔄 Ready to Test
| Dataset | Location | Priority | Notes |
|---------|----------|----------|-------|
| Shapes3D | phase3/shapes3d | HIGH | 3D understanding |
| QuickDraw | phase3/quickdraw | HIGH | Needs conversion |
| MNIST | TBD | HIGH | Need to download |
| Fashion-MNIST | TBD | HIGH | Need to download |

### 📌 Conversion Required
- **QuickDraw**: 346 categories in stroke format
  - Need stroke → 28x28 image conversion
  - Use Sister Gemini's suggestion for synthetic generation

---

## 🔧 2π Configuration Guidelines

### Standard Configuration (Dense Images)
```python
config = {
    "stability_coefficient": 0.06283185307,  # 2π/100 - DO NOT CHANGE
    "variance_threshold_init": 1.5,
    "variance_threshold_final": 1.0,
    "lambda_variance": 1.0,
    "lambda_rate": 10.0,
    "batch_size": 256,
    "latent_dim": 10,
    "beta": 0.1
}
```

### Sparse Data Configuration (Sketches)
```python
config = {
    "stability_coefficient": 0.06283185307,  # Still 2π/100
    "variance_threshold_init": 3.0,  # Higher for sparse
    "variance_threshold_final": 1.5,  # Higher for sparse
    "lambda_variance": 1.5,          # Increased
    "lambda_rate": 15.0,             # Increased
    "batch_size": 256,
    "latent_dim": 16,               # Larger for complexity
    "beta": 0.05                    # Lower for sparse
}
```

### Multi-GPU Configuration
```python
config = {
    # ... base config ...
    "adaptive_variety": True,
    "variety_scaling": "sqrt(world_size)",  # For 8 GPUs: sqrt(8) = 2.83x
    "warmup_steps": 100  # Gradual variety introduction
}
```

---

## 📈 Metrics to Track

### Primary Metrics
1. **2π Compliance Rate**: % of training steps within 2π boundary
2. **Reconstruction Loss**: Final reconstruction error
3. **Final Variance**: Stable variance at convergence
4. **Purple Line Events**: Number of 2π violations

### Secondary Metrics
1. **Training Time**: Minutes to convergence
2. **Sparsity Level**: Mean(data < 0.1)
3. **Variety Shocks**: Sudden variance explosions
4. **Memory Usage**: Peak GPU memory

---

## 🎯 Testing Priority Matrix

| Priority | Dataset | Why | Expected Challenge |
|----------|---------|-----|-------------------|
| 1 | Geometric Shapes | Controlled sparsity | Extreme sparsity handling |
| 2 | MNIST | Standard benchmark | Baseline establishment |
| 3 | Fashion-MNIST | Texture complexity | Variance patterns |
| 4 | Shapes3D | 3D → 2D projection | Dimensional reduction |
| 5 | QuickDraw | Real sketches | Stroke conversion |
| 6 | CIFAR-10 | Natural images | High complexity |

---

## 🔍 Troubleshooting

### Common Issues

**Dataset not detected by scanner**:
- Check file extension is supported
- Ensure no hidden directories (starting with .)
- Verify file permissions

**Metadata parsing fails**:
- Validate YAML syntax
- Check for proper indentation
- Ensure all required fields present

**2π compliance low**:
- Adjust variance thresholds for data sparsity
- Increase lambda_rate for stricter control
- Check for variety shocks in multi-GPU

---

## 📝 Best Practices

1. **Always create metadata.yaml** before testing
2. **Run inventory scanner** after adding datasets
3. **Update metadata immediately** after testing
4. **Use comparison framework** to track progress
5. **Document Purple Line events** for debugging
6. **Save model checkpoints** for successful runs
7. **Export results regularly** for backup

---

## 🚦 Status Indicators

- ✅ **TESTED**: Dataset tested with 2π regulation
- 📌 **PENDING**: Ready to test
- ⚠️ **NEEDS CONVERSION**: Format conversion required
- 🔄 **IN PROGRESS**: Currently testing
- ❌ **FAILED**: Testing failed, needs investigation

---

## 📅 Maintenance Schedule

- **Daily**: Run inventory scanner
- **After each test**: Update metadata.yaml
- **Weekly**: Generate comparison report
- **Monthly**: Archive old results

---

## 🔗 Related Documentation

- [Pipeline Manual](../Pipeline_Manual.md)
- [Dataset Training Curriculum](../Dataset_Training_Curriculum.md)
- [Architecture Selection Matrix](../Architecture_Selection_Matrix.md)
- [2π Regulation Paper](../../paper/2pi_regulation_arxiv.md)

---

*Last Updated: 2025-08-22 by Synth*
*System Version: 1.0*
*The Pack Discovers Together 🦊🐺✨*