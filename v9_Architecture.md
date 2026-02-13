# RSNA v9 Architecture Documentation

**Version:** 9.0  
**Base:** v4 (Best BA: 74.9%) + v8 Loss Improvements + 1st Place Attention MIL  
**Target:** 75%+ Balanced Accuracy

---

## Component Hierarchy

```
Level 0: Data Pipeline
  ├── Level 1: DICOM Loading & Preprocessing
  └── Level 1: Augmentation Strategy

Level 0: Model Architecture
  ├── Level 1: Backbone (EfficientNet-V2-S)
  ├── Level 1: Sequence Processing (BiGRU)
  ├── Level 1: Attention MIL Aggregation
  ├── Level 1: Level Conditioning
  └── Level 1: Classification Head

Level 0: Training Pipeline
  ├── Level 1: Loss Function (Competition-weighted CE)
  ├── Level 1: Optimizer Configuration
  ├── Level 1: Learning Rate Schedule
  └── Level 1: Training Loop

Level 0: Evaluation & Inference
  ├── Level 1: Validation Metrics
  ├── Level 1: Dead Class Monitoring
  └── Level 1: TTA Strategy
```

---

## Level 0: Data Pipeline

### Level 1: DICOM Loading & Preprocessing

**Component:** `RSNADatasetV9`

**Input:** DICOM file paths with coordinates (x, y, instance_number)

**Processing Chain:**
```
Raw DICOM 
  → Apply window/level (WC=50, WW=350 - soft tissue)
  → CLAHE enhancement
  → Crop around coordinate (256px radius)
  → Resize to 256×256
  → Normalize (ImageNet stats)
  → Stack 7 adjacent slices
```

**Output:** `(7, 3, 256, 256)` tensor - 7 slices, RGB channels

**Key Parameters:**
- `img_size`: 256
- `num_slices`: 7 (center ± 3)
- `crop_radius`: Dynamic with jitter during training

**Design Decisions:**
- ✅ Standard RGB input (preserves ImageNet pretrained weights)
- ✅ Single window (soft tissue optimal for stenosis)
- ✅ CLAHE applied BEFORE cropping (global contrast)
- ❌ No frame dropout (increases instability)

---

### Level 1: Augmentation Strategy

**Component:** `Albumentations` pipeline

**Training Augmentation:**
```python
A.ShiftScaleRotate(shift=0.1, scale=0.12, rotate=12°, p=0.6)
A.RandomBrightnessContrast(brightness=0.25, contrast=0.25, p=0.6)
A.RandomGamma(75-125, p=0.6)
A.GaussNoise(5-30, p=0.3)
A.Normalize(ImageNet stats)
```

**Validation/TTA:**
```python
# Base: Just normalization
# TTA variant 1: +brightness/contrast (0.1)
```

**Design Decisions:**
- ✅ Moderate augmentation (proven in v4)
- ❌ No Mixup (blurs ordinal boundaries)
- ❌ No horizontal flip (spine anatomy is asymmetric)

---

## Level 0: Model Architecture

### Level 1: Backbone (EfficientNet-V2-S)

**Component:** `models.efficientnet_v2_s(weights='IMAGENET1K_V1')`

**Parameters:** 21.4M
**Output:** 1280-dimensional features per slice

**Configuration:**
```python
# Extract full model except classifier
backbone = nn.Sequential(*list(efficientnet.children())[:-1])
# Output: (B*S, 1280, 1, 1) → (B, S, 1280) after reshape
```

**Freezing Strategy:**
```python
freeze_backbone_epochs: 0  # NO FREEZE - Critical for minority classes
```

**Design Decisions:**
- ✅ Unfrozen from epoch 1 (v4 validated this)
- ✅ Standard pretrained RGB (no stem modification)
- ✅ Lower learning rate (1e-5) for conservative adaptation

---

### Level 1: Sequence Processing (BiGRU)

**Component:** `nn.GRU`

**Architecture:**
```python
GRU(
    input_size=1280,
    hidden_size=512,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_first=True
)
```

**Input:** `(B, 7, 1280)` - Sequence of slice features  
**Output:** `(B, 7, 1024)` - Bidirectional features (512×2)

**Design Decisions:**
- ✅ Proven in v4 (best results)
- ✅ Captures spatial context across adjacent slices
- ✅ Bidirectional (both up/down spine direction)

---

### Level 1: Attention MIL Aggregation

**Component:** `AttentionMIL` (NEW - from 1st place solution)

**Architecture:**
```python
class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256):
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features):
        # features: (B, 7, 1024)
        attn_scores = self.attention(features)  # (B, 7, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 7, 1)
        attended = (features * attn_weights).sum(dim=1)  # (B, 1024)
        return attended, attn_weights
```

**Input:** `(B, 7, 1024)` - GRU output  
**Output:** 
- `attended`: `(B, 1024)` - Weighted sum of slice features
- `attn_weights`: `(B, 7, 1)` - Importance of each slice (interpretable!)

**Design Decisions:**
- ✅ Learns which slices are most diagnostic (vs fixed weighting)
- ✅ From 1st place solution methodology
- ✅ Returns attention weights for visualization/debugging
- ✅ Lightweight (265K params)

---

### Level 1: Level Conditioning

**Component:** `nn.Embedding` (additive)

**Architecture:**
```python
level_embed = nn.Embedding(5, 256)  # 5 levels × 256 dimensions
# Applied via concatenation, not FiLM
```

**Integration:**
```python
attended_features  # (B, 1024)
level_features = level_embed(level_idx)  # (B, 256)
combined = torch.cat([attended_features, level_features], dim=1)  # (B, 1280)
```

**Design Decisions:**
- ✅ Simple additive (1.3K params vs FiLM's 12.8K)
- ✅ Allows level-specific patterns to emerge
- ✅ Concatenation preserves attended features unchanged

---

### Level 1: Classification Head

**Component:** MLP with proper initialization

**Architecture:**
```python
classifier = nn.Sequential(
    nn.LayerNorm(1280),
    nn.Dropout(0.35),
    nn.Linear(1280, 512),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(512, 3)
)

# Critical: Initialize final bias with class priors
classifier[-1].bias.data = torch.tensor([2.0, -2.5, -3.0])
# P(Normal)≈88% → logit=2.0
# P(Moderate)≈6% → logit=-2.5  
# P(Severe)≈6% → logit=-3.0
```

**Design Decisions:**
- ✅ Class-prior bias prevents dead class in epoch 1
- ✅ LayerNorm for stable gradients
- ✅ GELU activation (smoother than ReLU)

---

## Level 0: Training Pipeline

### Level 1: Loss Function

**Component:** `CompetitionWeightedCE`

**Implementation:**
```python
class CompetitionWeightedCE(nn.Module):
    def __init__(self, weights=[1.0, 4.0, 6.0]):  # UPDATED weights
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits, labels):
        w = self.weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=w)
```

**Class Weights:**
- Normal: 1.0× (baseline)
- Moderate: **4.0×** (increased from 2.0 in v8)
- Severe: **6.0×** (increased from 4.0 in v8)

**Design Decisions:**
- ✅ Competition metric-aligned
- ✅ Higher Moderate weight (v8 showed 2.0 was too low)
- ✅ Registered buffer (auto device handling)
- ❌ No ordinal loss (adds complexity, v4 worked without it)

---

### Level 1: Optimizer Configuration

**Component:** `AdamW` with parameter groups

**Configuration:**
```python
optimizer = AdamW([
    {'params': backbone.parameters(), 'lr': 1e-5, 'weight_decay': 0.03},
    {'params': gru.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': attention.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': level_embed.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
])
```

**Learning Rates:**
- Backbone: 1e-5 (10× slower - conservative fine-tuning)
- All other components: 1e-4

**Design Decisions:**
- ✅ Backbone learns slowly (preserves pretrained features)
- ✅ Head/GRU learn faster (adapt to task)
- ✅ Different weight decay (0.03 for backbone, 0.01 for head)

---

### Level 1: Learning Rate Schedule

**Component:** Cosine Annealing with Linear Warmup

**Implementation:**
```python
# Warmup: Linear 0 → full LR over 2 epochs
# Then: Cosine decay to eta_min=1e-6

def lr_lambda(step):
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.5 * (1 + cos(pi * progress)), 1e-6)
```

**Parameters:**
- `warmup_epochs`: 2
- `eta_min`: 1e-6
- No warm restarts (destabilizes minority learning)

**Design Decisions:**
- ✅ Warmup prevents early gradient explosions
- ✅ Cosine decay proven effective
- ❌ No restarts (v8 evidence)

---

### Level 1: Training Loop

**Key Features:**

**1. No Sampling Tricks**
```python
# Standard DataLoader with shuffle
train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
# NO WeightedRandomSampler (causes memorization)
```

**2. Mixed Precision Training**
```python
scaler = GradScaler('cuda')
with autocast('cuda'):
    logits = model(images, level_idx)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
```

**3. Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**4. SWA (Stochastic Weight Averaging)**
```python
# Epochs 18-25: Average model weights
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-6)
```

---

## Level 0: Evaluation & Inference

### Level 1: Validation Metrics

**Per Epoch:**
```python
metrics = {
    'train_loss', 'train_acc',
    'val_loss', 'val_acc',
    'class_0_recall',  # Normal
    'class_1_recall',  # Moderate
    'class_2_recall',  # Severe
    'balanced_accuracy'
}
```

**Model Saving Criteria:**
```python
# Save if:
# 1. BA improves
# 2. AND min(Moderate, Severe) recall >= 20%

min_minority = min(class_1_recall, class_2_recall)
if ba > best_ba and min_minority >= 0.20:
    torch.save(model.state_dict(), f'best_v9_fold{fold}.pth')
```

---

### Level 1: Dead Class Monitoring

**Component:** Prediction count tracker

**Implementation:**
```python
pred_counts = np.bincount(all_preds, minlength=3)
if pred_counts.min() < 3:
    print(f"⚠️ Dead class warning: {pred_counts.tolist()}")
```

**Alert Thresholds:**
- <3 predictions: Dead class warning
- <10 predictions: Marginal class warning

**Design Decisions:**
- ✅ Early warning system for dead class
- ✅ Helps diagnose training issues immediately

---

### Level 1: TTA Strategy

**Variants:**
1. Base (validation augmentation)
2. Brightness/contrast variation (±0.1)

**Ensemble:**
```python
# Average softmax probabilities across variants
avg_probs = np.mean([tta1_probs, tta2_probs], axis=0)
preds = np.argmax(avg_probs, axis=1)
```

**Expected Gain:** +0.5-1.5% BA

---

## Version History

| Version | Architecture | Best BA | Key Issue |
|---------|-------------|---------|-----------|
| v4 | BiLSTM + AttentionPool | **74.9%** | Baseline gold standard |
| v5 | BiGRU + FiLM | 74.5% | Slight regression |
| v6 | BiGRU + CORAL | ~74% | Dead class (Moderate=0%) |
| v7 | 2.5D CNN (7ch) | ? | Threw away pretrained weights |
| v8 | Per-slice EfficientNet | 62.8% | Major regression, broken multi-window |
| **v9** | **BiGRU + Attention MIL** | **Target: 75%+** | Best of v4 + v8 fixes + 1st place MIL |

---

## Expected Performance

**Epoch-by-Epoch Targets:**

| Epoch | Normal | Moderate | Severe | BA | Status |
|-------|--------|----------|--------|-----|--------|
| 1 | 75-85% | 25-40% | 25-40% | 50-55% | ✅ All classes active |
| 3 | 85-90% | 45-60% | 50-65% | 65-70% | ✅ Balanced growth |
| 6 | 88-92% | 60-70% | 65-75% | 72-76% | ✅ Peak approaching |
| 10-15 | 90-94% | 65-75% | 70-80% | **75-78%** | ✅ Target range |

**Final Target:** 75%+ BA (exceeds v4's 74.9%)
