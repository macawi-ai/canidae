# 📚 Dataset Training Curriculum for 2π Regulation

## Mission
Systematically test and improve our 2π-regulated models across diverse datasets to build comprehensive AGI capabilities.

## Current Status
✅ **dSprites**: 99.9% compliance, 25.24 recon loss (EXCELLENT)

---

## PHASE 1: Immediate Testing (Today/Tomorrow)
*Focus: Fast iteration & 2π fundamentals with what we have*

| Priority | Dataset | Status | Size | Purpose | 2π Focus |
|----------|---------|--------|------|---------|----------|
| 1 | **QuickDraw (5 categories)** | 🟢 Ready | ~100MB | Sketch understanding | New modality compliance |
| 2 | **Shapes3D** | 🟢 Ready | ~1GB | 3D→2D projection | Depth variance regulation |
| 3 | **SmallNORB** | 🟢 Ready | ~1GB | Viewpoint invariance | Lighting stability |
| 4 | **MNIST** | 🟡 Download | 50MB | Baseline sanity check | Basic compliance |
| 5 | **Fashion-MNIST** | 🟡 Download | 50MB | Texture patterns | Complexity scaling |

### QuickDraw Category Selection (Phase 1)
```python
initial_categories = [
    'circle',     # Simple shape
    'square',     # Geometric primitive  
    'triangle',   # Angular variation
    'star',       # Complex vertices
    'flower'      # Organic curves
]
```

---

## PHASE 2: Expanding Coverage (Week 1)
*Focus: Generalization & robustness*

| Priority | Dataset | Source | Size | Key Test |
|----------|---------|--------|------|----------|
| 6 | **QuickDraw (10 more)** | Local | ~200MB | Category scaling |
| 7 | **KMNIST** | [Download](http://codh.rois.ac.jp/kmnist/) | 50MB | Cross-cultural |
| 8 | **CIFAR-10** | [Download](https://www.cs.toronto.edu/~kriz/cifar.html) | 170MB | Natural scenes |
| 9 | **SVHN** | [Download](http://ufldl.stanford.edu/housenumbers/) | 600MB | Real-world noise |
| 10 | **Synthetic Shapes** | Generate | Custom | Targeted testing |

### QuickDraw Phase 2 Categories
```python
phase2_categories = [
    'cat', 'dog',           # Animals
    'car', 'bicycle',       # Vehicles
    'house', 'tree',        # Structures
    'sun', 'cloud',         # Nature
    'face', 'hand'          # Human features
]
```

---

## PHASE 3: Deep Testing (Week 2)
*Focus: Identifying weaknesses & refining 2π*

| Priority | Dataset | Challenge | Expected Issues |
|----------|---------|-----------|-----------------|
| 11 | **QuickDraw (20 more)** | High variety | Category confusion |
| 12 | **Custom Synthetic** | Controlled tests | Edge cases |
| 13 | **CLEVR-Mini** | Reasoning | Attention variance |
| 14 | **ARC Simple** | Abstract patterns | Logic gates |
| 15 | **Omniglot** | Few-shot learning | Rapid adaptation |

---

## PHASE 4: AGI Capabilities (Week 3+)
*Focus: Reasoning, problem-solving & real-world*

| Priority | Dataset | Complexity | Success Criteria |
|----------|---------|------------|------------------|
| 16 | **QuickDraw (All 345)** | Maximum variety | >95% 2π across all |
| 17 | **CLEVR (if bandwidth)** | Scene reasoning | Question accuracy >80% |
| 18 | **ARC Complex** | Abstract reasoning | Any correct solutions |
| 19 | **VQA Subset** | Vision+Language | Meaningful answers |
| 20 | **ImageNet Subset** | Real-world | FID < 30 |

---

## Implementation Plan

### Immediate Actions (Next 24 Hours)

1. **Prepare QuickDraw Loader**
```python
def load_quickdraw_categories(categories, samples_per_category=10000):
    data = []
    for category in categories:
        path = f"datasets/phase3/quickdraw/sketches/{category}.npz"
        sketches = np.load(path)['arr_0'][:samples_per_category]
        data.append(sketches)
    return np.concatenate(data)
```

2. **Configure 2π for Sketches**
```python
sketch_config = {
    "variance_threshold": 1.5,  # Higher for sparse data
    "lambda_variance": 1.5,
    "lambda_rate": 15.0,
    "adaptive_schedule": True,
    "sketch_specific": {
        "handle_sparsity": True,
        "stroke_variance": True
    }
}
```

3. **Setup Comparison Framework**
```python
comparison_metrics = {
    'dsprites': {'compliance': 99.9, 'loss': 25.24},
    'quickdraw': {'compliance': None, 'loss': None},
    'shapes3d': {'compliance': None, 'loss': None}
}
```

---

## Success Metrics Per Dataset

### Universal Metrics (All Datasets)
- 2π Compliance Rate > 95%
- Stable variance throughout training
- Monotonic loss decrease
- No gradient explosions

### Dataset-Specific Success

| Dataset | Primary Metric | Target | Secondary Metrics |
|---------|---------------|--------|-------------------|
| QuickDraw | Category accuracy | >90% | Stroke coherence |
| Shapes3D | Disentanglement | >0.8 | View consistency |
| CIFAR-10 | Classification | >85% | Color stability |
| SVHN | Digit accuracy | >90% | Noise robustness |
| ARC | Task completion | >0% | Pattern transfer |

---

## Failure Mode Tracking

### Expected Challenges

1. **QuickDraw**: Sparse representations may break variance assumptions
2. **Shapes3D**: 3D→2D projection might need special handling
3. **CIFAR-10**: Natural images have different statistics than synthetic
4. **ARC**: Logic patterns may not fit continuous latent spaces

### Mitigation Strategies

| Challenge | Strategy | Implementation |
|-----------|----------|----------------|
| Sparse data | Adjust thresholds | `if sparsity > 0.8: threshold *= 2` |
| Multi-modal | Separate encoders | Architecture modification |
| High variety | Hierarchical VAE | Progressive training |
| Logic patterns | Discrete latents | VQ-VAE variant |

---

## Resource Planning

### GPU Allocation

| Dataset | Batch Size | Memory | Time Estimate |
|---------|------------|--------|---------------|
| QuickDraw-5 | 256 | 8GB | 2 hours |
| Shapes3D | 128 | 12GB | 4 hours |
| CIFAR-10 | 256 | 10GB | 3 hours |
| QuickDraw-345 | 64 | 20GB | 24 hours |

### Storage Requirements
- Immediate: 5GB (Phases 1-2)
- Full curriculum: 20GB (excluding CLEVR)
- With CLEVR: 70GB

---

## Analysis Framework

After each dataset:

1. **Generate Health Report** (use template)
2. **Update Comparison Matrix**
3. **Log to Neo4j Knowledge Graph**
4. **Identify Failure Modes**
5. **Adjust Next Dataset Config**
6. **Share Results with Gemini**

---

## Key Questions After Each Phase

**Phase 1**: Does 2π work across modalities?
**Phase 2**: How does complexity affect compliance?
**Phase 3**: Where does the model break?
**Phase 4**: Can we achieve AGI-level generalization?

---

*"Every dataset teaches us something new about consciousness" - CANIDAE Learning Philosophy*