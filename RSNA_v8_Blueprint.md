# RSNA v8 — Comprehensive Implementation Plan & Architectural Blueprint

## Part 1: Root Cause Analysis (Comparative)

### 1.1 Results Summary

| Version | Architecture | Loss | Best BA | Dead Class? |
|---------|-------------|------|---------|-------------|
| v4 (aux heads) | BiLSTM + Self-Attn + MeanPool | Focal + ClassWeights | 74.9% | No |
| v5 (redesign) | BiGRU + AttentionPool + FiLM | Focal (no weights) | 74.5% | No |
| v6 (ordinal) | BiGRU + AttentionPool + FiLM | CORAL + CE | ~74% | **Yes** (Moderate=0% epochs 1-4) |
| v7 (2.5D CNN) | EfficientNet 7ch stem, no RNN | Focal + Ordinal | ? | ? |

### 1.2 The Four Mutual Problems

**Problem 1: WeightedRandomSampler is NOT the right tool here.**

Why it failed:
- With `batch_size=8` and 3 classes, even with perfect sampling you get ~2-3 samples per class per batch
- Minority classes have only 581 (Moderate) and 385 (Severe) unique samples
- With replacement=True, the sampler redraws the SAME minority samples many times per epoch
- This is functionally equivalent to overfitting on a small memorized set
- **The sampler balances CLASS FREQUENCY but not DATA DIVERSITY** — the model sees the same 385 Severe images 18x each while Normal images are seen once

**What top RSNA solutions actually used instead:** Competition-metric-aligned **class-weighted CrossEntropy** with weights `[1, 2, 4]` (matching the official scoring weights). No sampler. This is simpler and proven.

**Problem 2: Ordinal loss (CORAL) has a "dead middle class" failure mode.**

The CORAL architecture outputs 2 logits: P(Y>0) and P(Y>1). To predict Moderate:
- P(Y>0) must be high (>0.5) — "this is worse than Normal"
- P(Y>1) must be low (<0.5) — "but not as bad as Severe"

This creates a *narrow band* that's hard to learn. Early in training, the two thresholds often collapse to the same value, making P(Moderate) ≈ 0. The biases we added [-2.0, -2.95] help, but the fundamental issue is that **CORAL optimizes two independent binary classifiers that don't directly receive gradient for the middle class**.

**Problem 3: The model overfits after ~3 epochs, hitting a "Normal ceiling".**

Every version shows the same pattern:
```
Epoch 3:  Normal=80%   Moderate=70%  Severe=73%  BA=74%  ← PEAK
Epoch 8:  Normal=93%   Moderate=59%  Severe=69%  BA=73%  ← Declining
Epoch 15: Normal=93%   Moderate=62%  Severe=64%  BA=73%  ← Plateau
```

The backbone (EfficientNet-V2-S, 21M params) has far more capacity than needed for ~9,700 training samples. It memorizes Normal quickly, then the loss gradients become dominated by Normal samples even with balanced batches, because the per-sample loss for hard Normal cases becomes the dominant signal.

**Problem 4: Augmentation doesn't create enough semantic diversity for minorities.**

Geometric transforms (rotate, shift, scale) and intensity transforms (brightness, gamma) change HOW an image looks but not WHAT it represents. With only 385 Severe samples, you're augmenting 385 unique anatomies. No augmentation pipeline can generate "new pathology" — only noise around existing pathology.

### 1.3 What Top RSNA Solutions Did Differently

The 1st place solution ("Avengers") used:
1. **Two-stage pipeline**: Stage 1 localizes keypoints, Stage 2 classifies ROIs
2. **Competition-weight CE loss**: `weights=[1, 2, 4]` — NOT weighted sampler, NOT focal loss
3. **Attention-based MIL (Multiple Instance Learning)**: NOT BiLSTM sequence modeling
4. **ConvNeXt + EfficientNet ensemble**: Multiple backbones, not one
5. **No ordinal loss**: Standard weighted CE was sufficient

---

## Part 2: Detailed Architectural Pipeline

### Module 1: Data Ingestion & Preprocessing

```
Raw DICOM → Window/Level → CLAHE → Crop around (x,y) → Resize → Normalize
```

**Changes from v6/v7:**
- **Larger crop**: Use `img_size * 0.75` crop radius instead of `img_size // 2` (128px). The current 128px crop is too tight — stenosis context extends beyond the immediate coordinate.
- **Multi-window stacking**: Instead of single Window/Level, use 3 different windows as RGB channels:
  - Soft tissue window (WC=50, WW=350)
  - Bone window (WC=400, WW=2000)  
  - Original DICOM window
  - This gives the CNN three different "views" of the same anatomy
- **Sequence construction**: Keep 7 slices, but verify instance numbers are sorted by ImagePositionPatient[2] (z-coordinate), not raw instance number. Some DICOM series have non-sequential instance numbering.

### Module 2: Imbalance Strategy

**Replace WeightedRandomSampler with competition-weighted CE loss.**

```python
# Competition scoring weights: Normal=1, Moderate=2, Severe=4
class CompetitionWeightedCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([1.0, 2.0, 4.0])
    
    def forward(self, logits, labels):
        w = self.weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=w)
```

**Why this works when the sampler didn't:**
1. Every sample is seen exactly once per epoch — no memorization from repeated draws
2. The gradient magnitude is proportional to both class weight AND prediction difficulty
3. The weights match the evaluation metric — optimizing what you're measured on
4. No interaction bugs with batch size, drop_last, or replacement sampling

**Optional: Add Focal modulation on top**
```python
# Focal + competition weights — downweights well-classified Normal samples
ce = F.cross_entropy(logits, labels, weight=w, reduction='none')
pt = torch.exp(-ce)
focal = ((1 - pt) ** 1.5) * ce  # gamma=1.5 (lighter than 2.0)
return focal.mean()
```

### Module 3: Model Architecture

**Use the v7 2.5D approach as the base, with these critical fixes:**

```
Input: (B, 3, 224, 224)  — 3-window stacked as RGB
  → EfficientNet-V2-S (standard pretrained, no stem modification)
  → GAP → 1280 features
  → Level embedding (additive, not FiLM)
  → Classifier head with proper initialization
```

**Key architecture decisions:**

1. **Keep pretrained RGB input (no 7-channel stem)**. The 7-channel stem in v7 throws away pretrained weights for 4 channels. Instead, use multi-window stacking to leverage full ImageNet pretraining.

2. **Process slices independently, then aggregate.** Process each of the 7 slices through the shared backbone, then use a simple weighted average (learnable) or max-pool to select the most informative slice:

```python
class SliceAggregator(nn.Module):
    def __init__(self, num_slices=7, feature_dim=1280):
        super().__init__()
        # Learnable importance weight per slice position
        self.slice_weights = nn.Parameter(torch.zeros(num_slices))
        # Initialize center slice higher
        nn.init.constant_(self.slice_weights[num_slices//2], 1.0)
    
    def forward(self, features):
        # features: (B, num_slices, feature_dim)
        weights = F.softmax(self.slice_weights, dim=0)  # (num_slices,)
        # Weighted sum
        return (features * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
```

3. **Level embedding: simple additive, not FiLM.** FiLM has 2×1280×5 = 12,800 parameters modulating a 1280-dim space with only 5 levels. This is heavily overparameterized. Use a simpler additive embedding:

```python
self.level_embed = nn.Embedding(5, 256)  # Project to head dim, not backbone dim
# Applied after backbone, before classifier
features = backbone_features  # 1280
level_feat = self.level_embed(level_idx)  # 256
combined = torch.cat([features, level_feat], dim=1)  # 1536
logits = classifier(combined)  # 1536 → 3
```

4. **Classifier head: initialize final bias with class priors.**
```python
# Initialize final linear bias to match class distribution
# P(Normal)=0.88 → logit ≈ 2.0, P(Moderate)=0.07 → logit ≈ -2.5, P(Severe)=0.05 → logit ≈ -3.0
classifier[-1].bias.data = torch.tensor([2.0, -2.5, -3.0])
```

### Module 4: Training Loop

```python
# Phase 1: Frozen backbone (epochs 1-2)
# - Only classifier + level embedding train
# - LR: 3e-4 for head
# - Purpose: classifier learns class boundaries before backbone shifts features

# Phase 2: Unfrozen (epochs 3-25)
# - Backbone: 1e-5, Head: 1e-4
# - Cosine decay to 1e-6
# - Gradient clipping: max_norm=1.0

# Phase 3: SWA (epochs 20-25)
# - Average model weights for smoother decision boundaries
```

**No Mixup.** Mixup creates interpolated labels like "0.7 Normal + 0.3 Moderate". With only 3 ordinal classes, this blurs the critical boundaries. Remove it.

**Scheduler:**
```python
# Simple cosine decay — no warm restarts (restarts destabilize minority learning)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup, eta_min=1e-6)
```

**Key monitoring per batch:**
```python
# Log per-class prediction counts every epoch
# If any class has 0 predictions for 2 consecutive epochs → emergency action
batch_pred_counts = torch.bincount(preds, minlength=3)
if batch_pred_counts.min() == 0:
    print(f"⚠️ Dead class detected: {batch_pred_counts.tolist()}")
```

---

## Part 3: Implementation Roadmap

### Step 1: Data Pipeline (Develop independently)
- [ ] Multi-window DICOM loading (3 windows → RGB channels)
- [ ] Verify instance ordering by z-position
- [ ] Larger crop radius (0.75x instead of 0.5x)
- [ ] Test: Visualize crops for 5 samples of each class to verify quality

### Step 2: Loss Function (Develop independently)
- [ ] Competition-weighted CE: weights=[1, 2, 4]
- [ ] Optional focal modulation (gamma=1.5)
- [ ] Test: Run 5 epochs with FROZEN backbone to verify all 3 classes get nonzero gradients

### Step 3: Model Architecture (Develop independently)  
- [ ] Standard EfficientNet-V2-S (RGB input, pretrained)
- [ ] Per-slice processing + SliceAggregator (learnable weights)
- [ ] Additive level embedding (256-dim, concatenated)
- [ ] Classifier bias initialized with class priors
- [ ] Test: Print prediction distribution after 1 epoch — all 3 classes should appear

### Step 4: Training Loop (Integrate)
- [ ] Phase 1/2/3 training with progressive unfreezing
- [ ] Remove Mixup
- [ ] Remove WeightedRandomSampler (use standard shuffle=True)
- [ ] Cosine decay (no restarts)
- [ ] Dead class monitoring in validation

### Step 5: Evaluation
- [ ] Per-class recall at every epoch
- [ ] Confusion matrix (watch Moderate↔Severe confusion)
- [ ] TTA: brightness + scale only (no hflip)

### Metrics to Watch

| Metric | Healthy Range | Alarm |
|--------|--------------|-------|
| Min class recall | >30% by epoch 3 | 0% for any class = dead class |
| BA variance (last 5 epochs) | <3% | >5% = unstable |
| Normal recall | 75-85% | >90% = overfitting on majority |
| Train-val loss gap | <0.05 | >0.1 = overfitting |
| Moderate recall | >50% | <30% = model collapsing to binary |
| Severe recall | >50% | <30% = model ignoring extreme cases |

### Summary: What Changes from v6/v7

| Component | v6/v7 (Failed) | v8 (Proposed) |
|-----------|----------------|---------------|
| Sampling | WeightedRandomSampler | **Standard shuffle** (no sampler) |
| Loss | CORAL / Focal | **Competition-weighted CE** [1,2,4] |
| Input | 7ch stack OR 3ch per frame | **3-window stack as RGB** per frame |
| Sequence | GRU / 7ch CNN | **Per-slice CNN + learnable aggregator** |
| Level embed | FiLM (12.8K params) | **Additive concat** (1.3K params) |
| Mixup | Yes | **No** (blurs ordinal boundaries) |
| Head init | Random / ordinal bias | **Class-prior bias** [2.0, -2.5, -3.0] |
| Backbone freeze | 0-3 epochs | **2 epochs** (fixed) |
| LR schedule | Cosine with restarts | **Simple cosine decay** |
