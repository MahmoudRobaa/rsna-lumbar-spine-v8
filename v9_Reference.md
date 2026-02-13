# v9 Quick Reference & Deployment Guide

**Status:** Built and Ready to Deploy  
**Target:** 75%+ BA (exceed v4's 74.9%)

---

## What is v9?

**Foundation:** v4's proven architecture (BiGRU, no freeze) that achieved 74.9% BA  
**Improvements:** Competition-weighted CE + Attention MIL from 1st place  
**Fixes:** Removed v8's regressions (broken multi-window, per-slice processing, backbone freeze)

---

## Architecture at a Glance

```
Input: 7 slices × (3, 256, 256) RGB
  ↓
EfficientNet-V2-S (unfrozen from epoch 1)
  ↓
BiGRU 2-layer bidirectional
  ↓
Attention MIL (learns which slices matter)
  ↓
Level embedding (concatenation)
  ↓
Classifier (bias = [2.0, -2.5, -3.0])
  ↓
Competition-weighted CE [1, 4, 6]
```

**Parameters:** ~22M total
- Backbone: 21.4M
- GRU: 4.7M
- Attention: 265K
- Classifier: 656K

---

## Key Configuration

```python
# Loss
'class_weights': [1.0, 4.0, 6.0]  # Normal, Moderate, Severe

# Training (CRITICAL)
'freeze_backbone_epochs': 0  # NO FREEZE - prevents dead class

# Learning Rates
'learning_rate': 1e-4    # Head/GRU/Attention
'backbone_lr': 1e-5      # Backbone (10x slower)

# Architecture
'gru_hidden': 512
'gru_layers': 2
'attention_hidden': 256
```

---

## Deploy to Kaggle

### Step 1: Upload Notebook
- File: `le3ba_v9.ipynb`
- GPU: T4 or P100
- Internet: OFF

### Step 2: Add Dataset
1. Click "+ Add Data"
2. Search: "RSNA 2024 Lumbar Spine"
3. Add dataset

### Step 3: Fix Path (IMPORTANT)
Check the dataset path in your notebook:
```python
DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
```

If it fails, run the diagnostic cell from `KAGGLE_PATH_FIX.md`.

### Step 4: Run
- Expected time: ~3.5 hours (fold 0, 25 epochs)
- Watch for epoch 1 metrics!

---

## Critical Checkpoints

### ✅ Epoch 1 Success Criteria
**Must have:**
- All 3 classes predicted (no `[N, 0, 0]`)
- Moderate: 25-40%
- Severe: 25-40%
- BA: 50-55%

**If failed:**
- Any class = 0 → Increase that class weight to 8.0
- BA < 45% → Something is wrong, check loss values

---

### ✅ Epoch 3 Success Criteria
**Must have:**
- Moderate: 45-60%
- Severe: 50-65%
- BA: 65-70%

**If failed:**
- Moderate < 40% → Increase weight from 4.0 to 5.0
- Severe < 45% → Increase weight from 6.0 to 8.0

---

### ✅ Final Target (Epoch 10-15)
**Goal:**
- BA: 75%+ (exceeds v4)
- Moderate: 65%+
- Severe: 70%+

---

## What Changed from v8?

| Component | v8 (Failed) | v9 (Fixed) |
|-----------|-------------|------------|
| **Input** | 3-window multi-view | Standard RGB (back to v4) |
| **Sequence** | Per-slice EfficientNet | BiGRU (back to v4) |
| **Aggregation** | Simple weighted avg | **Attention MIL** (new) |
| **Freeze** | 2 epochs | **0 epochs** (no freeze) |
| **Weights** | [1, 2, 4] | **[1, 4, 6]** (higher Moderate) |

---

## Why v8 Failed (62.8% BA)

1. **Broken 3-window:** CLAHE converted RGB back to grayscale, destroyed the multi-view
2. **Per-slice processing:** 7x slower, lost sequence context
3. **Backbone freeze:** Caused dead class in epoch 1
4. **Weak Moderate weight:** 2x wasn't enough, oscillated 14-25%

---

## Why v9 Should Succeed

1. ✅ **v4's foundation worked** (74.9% BA with BiGRU, no freeze)
2. ✅ **Competition CE validated** (direct gradient to all classes)
3. ✅ **Attention MIL from 1st place** (proven in winning solution)
4. ✅ **Higher Moderate weight** (4x vs 2x addresses oscillation)
5. ✅ **No backbone freeze** (critical for minority classes)

---

## Expected Training Behavior

### Healthy Progress:
```
Epoch 1: BA=50-55%, All classes active
Epoch 3: BA=65-70%, Moderate climbing
Epoch 6: BA=70-74%, Approaching v4
Epoch 10: BA=75%+, Exceeding v4 ✅
```

### Warning Signs:
```
Epoch 1: [1930, 0, 0] → Dead class (shouldn't happen)
Epoch 3: Moderate <40% → Weights too weak
Epoch 6: BA plateau <70% → May need longer training
```

---

## Post-Training Analysis

### If BA ≥ 75%:
1. ✅ Run full 5-fold ensemble
2. Expected ensemble gain: +1-2% BA
3. Target: 76-78% final BA

### If 73% ≤ BA < 75%:
1. Tune weights to [1, 5, 7]
2. Extend training to 30 epochs
3. Re-run fold 0

### If BA < 73%:
1. Check attention weights visualization
2. Error analysis (which samples confused?)
3. May need different architecture changes

---

## Files in v9 Package

### Notebook:
- `le3ba_v9.ipynb` ⭐ — Main production notebook

### Documentation (Component-Based):
- `v9_Architecture.md` — Full component hierarchy, 800+ lines
- `v9_Implementation_Plan.md` — Build sequence, checkpoints, risks
- `v9_Reference.md` — This file (quick reference)

### Build Scripts:
- `build_v9.py` — Notebook builder
- `build_helpers.py` — Helper functions

---

## Troubleshooting

### Issue: Dataset not found
**Solution:** Check `KAGGLE_PATH_FIX.md`, likely needs `/competitions/` prefix

### Issue: Device mismatch error
**Solution:** Already fixed in v9's `CompetitionWeightedCE` (register_buffer)

### Issue: Dead class in epoch 1
**Solution:** Shouldn't happen (no freeze), but if it does:
- Increase affected class weight to 8.0
- Verify backbone is unfrozen

### Issue: Out of memory
**Solution:** Reduce batch size from 12 to 10

---

## Quick Commands

### Upload to Kaggle:
1. Download `le3ba_v9.ipynb` from GitHub
2. Upload to Kaggle
3. Add RSNA dataset
4. Enable GPU
5. Run all

### Commit to GitHub:
```bash
git add le3ba_v9.ipynb v9_*.md
git commit -m "Add v9: v4 foundation + attention MIL + higher weights"
git push origin main
```

---

## Success Metrics

### Minimum (Must Achieve):
- BA ≥ 74.9% (match v4)
- No dead class (all classes >20% by epoch 3)

### Target (Expected):
- BA ≥ 75.5% (exceed v4 by +0.6%)
- Moderate ≥ 65%
- Severe ≥ 70%

### Stretch (Aspirational):
- BA ≥ 77%
- All classes ≥ 75%
- Attention weights align with clinical expectations

---

**Last Updated:** 2026-02-13  
**Notebook Version:** v9  
**Confidence:** High (based on v4 proven foundation)
