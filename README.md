# RSNA 2024 Lumbar Spine Classification - v8

Production-ready notebook implementing competition-aligned architecture to solve the "dead class" problem in minority class prediction.

## 🎯 Problem Solved

Previous versions (v4-v7) suffered from:
- **Dead class phenomenon**: Moderate class had 0% accuracy for epochs 1-4
- **WeightedRandomSampler causing memorization**: Same 385 Severe samples drawn 18x per epoch
- **CORAL ordinal loss collapse**: Middle class threshold failure
- **Early overfitting**: Peaked at epoch 3, then declined

## ✅ v8 Solution

- **Competition-weighted CE loss** `[1, 2, 4]` instead of weighted sampler
- **3-window multi-view input** (soft tissue, bone, original DICOM windows)
- **Per-slice EfficientNet + learnable aggregator** instead of RNN/7-channel CNN
- **Class-prior bias initialization** `[2.0, -2.5, -3.0]`
- **No Mixup** (preserves ordinal boundaries)

## 📊 Expected Results

| Epoch | Normal | Moderate | Severe | Balanced Accuracy |
|-------|--------|----------|--------|-------------------|
| 3 | 75-80% | 50-60% | 45-55% | 60-65% |
| 10 | 82-88% | 65-75% | 60-70% | 70-77% |

**Target:** 75%+ BA (previous best: 74.9%)

## 📁 Files

### Notebooks
- **`le3ba_v8.ipynb`** - Production notebook (ready to upload to Kaggle)
- `le3ba_v7.ipynb` → `le3ba.ipynb` - Evolution history

### Documentation
- **`README_v8.md`** - Complete guide with architecture details
- **`RSNA_v8_Blueprint.md`** - Deep architectural analysis (253 lines)
- **`v8_Summary.md`** - Quick reference (126 lines)

## 🚀 Quick Start

1. Upload `le3ba_v8.ipynb` to Kaggle
2. Enable GPU (T4 or P100)
3. Run fold 0 (~15-20 min)
4. Check epoch 3 metrics for dead class warnings

## 🛠️ Architecture

```
Raw DICOM → 3 windows (RGB) → CLAHE → Crop → Normalize
    ↓
7 slices × EfficientNet-V2-S (shared weights)
    ↓
SliceAggregator (learnable, center-biased)
    ↓
Level embedding (additive, 256-dim)
    ↓
Classifier (bias = class priors)
```

## 📚 Research

Based on:
- RSNA 2024 1st place solution (Avengers team)
- Competition-weighted CE preferred over weighted sampling
- Multi-view input for medical imaging

## 📝 Version History

| Version | Best BA | Key Change |
|---------|---------|------------|
| v4 | 74.9% | Aux heads + progressive sampling |
| v5 | 74.5% | AttentionPool + FiLM |
| v6 | ~74% | CORAL ordinal (⚠️ dead class) |
| v7 | ? | 2.5D CNN 7-channel stem |
| **v8** | **?** | **Competition CE + 3-window + aggregator** |

## 📧 Contact

**Author:** Built with Antigravity AI Assistant  
**Date:** 2026-02-13  
**Competition:** RSNA 2024 Lumbar Spine Degenerative Classification
