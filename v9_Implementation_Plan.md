# RSNA v9 Implementation Plan

**Status:** Ready to Build  
**Confidence:** High (based on v4 proven foundation)

---

## Build Sequence

### Phase 1: Core Components (Independent Development)

#### Task 1.1: Dataset
**File:** Dataset cell in notebook  
**Dependencies:** None  
**Estimated Lines:** ~120

**Implementation:**
```python
class RSNADatasetV9(Dataset):
    # Standard RGB DICOM loading
    # Single window (WC=50, WW=350)
    # CLAHE → Crop → Resize
    # 7-slice stacking
```

**Validation:**
- [ ] Load 5 samples, verify shape `(7, 3, 256, 256)`
- [ ] Visualize crops for Normal/Moderate/Severe
- [ ] Check CLAHE is applied correctly

---

#### Task 1.2: Attention MIL Module
**File:** Model architecture cell  
**Dependencies:** None  
**Estimated Lines:** ~25

**Implementation:**
```python
class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256):
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features):
        attn_scores = self.attention(features)
        attn_weights = F.softmax(attn_scores, dim=1)
        attended = (features * attn_weights).sum(dim=1)
        return attended, attn_weights
```

**Validation:**
- [ ] Test with dummy input `(4, 7, 1024)`
- [ ] Verify attention weights sum to 1.0
- [ ] Check output shape `(4, 1024)`

---

#### Task 1.3: Competition-Weighted CE Loss
**File:** Loss function cell  
**Dependencies:** None  
**Estimated Lines:** ~15

**Implementation:**
```python
class CompetitionWeightedCE(nn.Module):
    def __init__(self, weights=[1.0, 4.0, 6.0]):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits, labels):
        w = self.weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=w)
```

**Validation:**
- [ ] Test on dummy logits + labels
- [ ] Verify weights auto-move to CUDA
- [ ] Check loss values reasonable (0-3 range)

---

### Phase 2: Model Integration

#### Task 2.1: Full Model Architecture
**File:** Model cell  
**Dependencies:** Task 1.2  
**Estimated Lines:** ~90

**Implementation:**
```python
class SpineModelV9(nn.Module):
    def __init__(self):
        # Backbone: EfficientNet-V2-S (unfrozen)
        # GRU: 2-layer bidirectional
        # Attention: AttentionMIL
        # Level: Embedding(5, 256)
        # Classifier: 1280 → 512 → 3
        # Bias init: [2.0, -2.5, -3.0]
```

**Validation:**
- [ ] Forward pass with dummy input
- [ ] Print parameter counts per component
- [ ] Verify output shape `(B, 3)`
- [ ] Check attention weights returned

---

### Phase 3: Training Pipeline

#### Task 3.1: Training Function
**File:** Training cell  
**Dependencies:** All Phase 1 + 2  
**Estimated Lines:** ~180

**Key Features:**
- Optimizer with 2 LR groups (backbone=1e-5, rest=1e-4)
- Cosine schedule with warmup
- Standard DataLoader (shuffle=True, **NO sampler**)
- Mixed precision (autocast + GradScaler)
- Gradient clipping (max_norm=1.0)
- SWA (epochs 18-25)
- Dead class monitoring
- Model saving with min_minority_recall gate

**Validation:**
- [ ] Run 1 epoch on small subset
- [ ] Verify all 3 classes predicted
- [ ] Check LR schedule curve
- [ ] Confirm gradient clipping active

---

### Phase 4: Evaluation & Analysis

#### Task 4.1: Metrics & Visualization
**File:** Evaluation cells  
**Dependencies:** Task 3.1  
**Estimated Lines:** ~80

**Components:**
- Per-class recall calculation
- Balanced accuracy
- Confusion matrix
- Training curves (loss, BA, per-class recall)
- **Attention weight visualization** (NEW)

**Validation:**
- [ ] Metrics match manual calculation
- [ ] Plots render correctly
- [ ] Attention weights visualized per slice

---

## Configuration

```python
CONFIG = {
    'seed': 42,
    'img_size': 256,
    'num_slices': 7,
    'batch_size': 12,
    'epochs': 25,
    
    # Learning rates
    'learning_rate': 1e-4,      # Head/GRU/Attention
    'backbone_lr': 1e-5,        # Backbone (10x slower)
    'weight_decay': 0.03,
    
    # Loss
    'class_weights': [1.0, 4.0, 6.0],  # Normal, Moderate, Severe
    
    # Training
    'freeze_backbone_epochs': 0,  # NO FREEZE
    'warmup_epochs': 2,
    'clip_grad_norm': 1.0,
    'patience': 12,
    
    # SWA
    'use_swa': True,
    'swa_start_epoch': 18,
    'swa_lr': 5e-6,
    
    # Architecture
    'gru_hidden': 512,
    'gru_layers': 2,
    'gru_dropout': 0.3,
    'attention_hidden': 256,
    'dropout': 0.35,
    
    # Data
    'num_folds': 5,
    'train_folds': [0],
}
```

---

## Critical Checkpoints

### Checkpoint 1: After Epoch 1
**Expected:**
- All 3 classes have predictions (no `[N, 0, 0]`)
- Normal: 75-85%
- Moderate: 25-40%
- Severe: 25-40%
- BA: 50-55%

**Action if Failed:**
- If any class has 0 predictions → Increase that class weight by 2x
- If BA <45% → Check loss values, may need higher overall LR

---

### Checkpoint 2: After Epoch 3
**Expected:**
- Moderate: 45-60%
- Severe: 50-65%
- BA: 65-70%

**Action if Failed:**
- If Moderate <40% → Increase weight from 4.0 to 5.0
- If Severe <45% → Increase weight from 6.0 to 8.0

---

### Checkpoint 3: After Epoch 10
**Expected:**
- BA: 72-76%
- All classes >60%

**Action if Failed:**
- If BA plateaued <72% → May need to extend training to epoch 30
- If overfitting (train-val gap >5%) → Increase dropout to 0.4

---

## Estimated Timeline

| Phase | Tasks | Time | Validation |
|-------|-------|------|------------|
| 1 | Core components | 15 min | Unit tests pass |
| 2 | Model integration | 10 min | Forward pass works |
| 3 | Training pipeline | 20 min | 1-epoch test run |
| 4 | Evaluation | 10 min | Plots render |
| **Total Build** | | **~55 min** | |
| **Training (fold 0)** | | **~3.5 hours** | 25 epochs × 8-9 min/epoch |

---

## Risk Mitigation

### Risk 1: Attention MIL Doesn't Improve Over v4
**Probability:** Medium  
**Impact:** Low (we'd still match v4's 74.9%)  
**Mitigation:** Attention is optional - can fall back to mean pooling

---

### Risk 2: Higher Class Weights Cause Instability
**Probability:** Low  
**Impact:** Medium (oscillating validation)  
**Mitigation:** Can reduce weights back to `[1, 2, 4]` mid-training

---

### Risk 3: No Backbone Freeze Causes Early Overfitting
**Probability:** Low (v4 validated this)  
**Impact:** Medium  
**Mitigation:** Lower backbone LR (1e-5) acts as regularization

---

## Success Criteria

### Minimum (Must Achieve):
- ✅ **No dead class** (all 3 classes >20% recall by epoch 3)
- ✅ **BA ≥ 74.9%** (match v4)

### Target (Expected):
- ✅ **BA ≥ 75.5%** (exceed v4 by 0.6%)
- ✅ **Moderate ≥ 65%** recall
- ✅ **Severe ≥ 70%** recall

### Stretch (Aspirational):
- ✅ **BA ≥ 77%**
- ✅ All classes ≥ 75%
- ✅ Attention weights interpretable (diagnostic slices have higher weights)

---

## Post-Training Analysis

### 1. Attention Weight Analysis
For each test sample, visualize:
- Which slices got highest attention
- Does attention correlate with severity?
- Are edge slices ignored (as expected)?

### 2. Error Analysis
Identify:
- Most confused classes (likely Moderate ↔ Normal)
- Failure modes (motion blur, poor crop, etc.)

### 3. Ensemble Potential
If BA ≥ 75%, plan 5-fold ensemble:
- Expected gain: +1-2% BA
- Target: 76-78% final BA

---

## Files to Generate

1. `le3ba_v9.ipynb` - Main notebook
2. `v9_Architecture.md` - This file (component docs)
3. `v9_Implementation_Plan.md` - This file
4. `v9_Results.md` - Post-training analysis (created after training)

---

**Ready to build?** All components are specified. Build sequence is clear. Validation criteria defined.
