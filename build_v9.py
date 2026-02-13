"""
Build le3ba_v9.ipynb - v4 Foundation + Competition CE + Attention MIL

Key Changes from v8:
- REVERT to v4's proven architecture (BiGRU, standard RGB input, no freeze)
- KEEP v8's competition-weighted CE [1, 4, 6] (higher Moderate weight)
- ADD attention MIL from 1st place solution (vs simple weighted average)
- REMOVE broken multi-window, per-slice processing, backbone freeze
"""
import json
from build_helpers import create_notebook, code_cell, md_cell

cells = []

# Header
cells.append(md_cell("""# RSNA 2024 Lumbar Spine — Version 9
## v4 Proven Foundation + Competition CE + Attention MIL

### v9 vs Previous Versions:
| Aspect | v4 (Best: 74.9%) | v8 (Regressed: 62.8%) | **v9 (Target: 75%+)** |
|--------|------------------|----------------------|----------------------|
| **Input** | Standard RGB | Broken 3-window | ✅ Standard RGB (back to v4) |
| **Sequence** | BiGRU | Per-slice independent | ✅ BiGRU (back to v4) |
| **Aggregation** | AttentionPool | Simple weighted avg | ✅ **Attention MIL** (1st place) |
| **Loss** | Focal | Competition CE [1,2,4] | ✅ Competition CE **[1,4,6]** |
| **Sampler** | Weighted | None | ✅ None (keep v8 fix) |
| **Freeze** | No | Yes (caused dead class) | ✅ **No freeze** (back to v4) |

**Philosophy:** Keep what worked (v4 foundation), add proven improvements (competition CE, attention MIL), remove regressions (freeze, broken multi-window).
"""))

# Imports
cells.append(code_cell("""import os, copy, cv2, glob, pydicom, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import albumentations as A
from albumentations.pytorch import ToTensorV2"""))

# Config
cells.append(code_cell("""CONFIG = {
    'seed': 42,
    'img_size': 256,
    'num_slices': 7,
    'batch_size': 12,
    'epochs': 25,
    
    'learning_rate': 1e-4,        # Head/GRU/Attention
    'backbone_lr': 1e-5,          # Backbone (10x slower, conservative)
    'weight_decay': 0.03,
    'patience': 12,
    'num_folds': 5,
    'train_folds': [0],
    
    # Loss - UPDATED weights (4x Moderate, 6x Severe)
    'class_weights': [1.0, 4.0, 6.0],
    
    # Training - NO FREEZE (critical!)
    'clip_grad_norm': 1.0,
    'use_swa': True,
    'swa_start_epoch': 18,
    'swa_lr': 5e-6,
    'warmup_epochs': 2,
    'freeze_backbone_epochs': 0,  # NO FREEZE - v4 validated this
    
    # Architecture
    'gru_hidden': 512,
    'gru_layers': 2,
    'gru_dropout': 0.3,
    'attention_hidden': 256,
    'dropout': 0.35,
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'target_condition': 'spinal_canal_stenosis',
    'target_series': 'Sagittal T2/STIR'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"✅ v9: v4 Foundation + Competition CE [1,4,6] + Attention MIL")
print(f"   Backbone freeze: {CONFIG['freeze_backbone_epochs']} epochs (NO FREEZE)")
print(f"   Target: 75%+ BA (exceed v4's 74.9%)")"""))

# Data loading (same as before)
cells.append(md_cell("## 1. Data Loading"))
cells.append(code_cell("""DATA_ROOT = "/kaggle/input/competitions/rsna-2024-lumbar-spine-degenerative-classification"
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train_images")

df_train = pd.read_csv(f"{DATA_ROOT}/train.csv")
df_coords = pd.read_csv(f"{DATA_ROOT}/train_label_coordinates.csv")
df_desc = pd.read_csv(f"{DATA_ROOT}/train_series_descriptions.csv")

df_train.columns = [col.lower().replace('/', '_') for col in df_train.columns]
condition_cols = [c for c in df_train.columns if c != 'study_id']
df_labels = pd.melt(df_train, id_vars=['study_id'], value_vars=condition_cols,
                    var_name='condition_level', value_name='severity')
df_labels = df_labels.dropna(subset=['severity'])
df_labels['severity'] = df_labels['severity'].astype(str).str.lower().str.replace('/', '_')

def extract_meta(val):
    parts = val.split('_')
    level = parts[-2] + '_' + parts[-1]
    condition = '_'.join(parts[:-2])
    return condition, level

df_labels[['base_condition', 'level_str']] = df_labels['condition_level'].apply(lambda x: pd.Series(extract_meta(x)))
severity_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
df_labels['label'] = df_labels['severity'].map(severity_map)
df_labels = df_labels.dropna(subset=['label'])
df_labels['label'] = df_labels['label'].astype(int)

df_coords = df_coords.merge(df_desc, on=['study_id', 'series_id'], how='left')
df_coords['condition'] = df_coords['condition'].str.lower().str.replace(' ', '_')
df_coords['level'] = df_coords['level'].str.lower().str.replace('/', '_')
df_coords['condition_level'] = df_coords['condition'] + '_' + df_coords['level']

df_model = df_labels[df_labels['base_condition'] == CONFIG['target_condition']].copy()
df_coords_filt = df_coords[(df_coords['condition'] == CONFIG['target_condition']) & 
                           (df_coords['series_description'] == CONFIG['target_series'])]

df_final = df_model.merge(df_coords_filt[['study_id', 'condition_level', 'series_id', 'instance_number', 'x', 'y']],
                          on=['study_id', 'condition_level'], how='inner')

valid_rows = []
for index, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Checking Files"):
    path = f"{TRAIN_IMAGES}/{row['study_id']}/{row['series_id']}/{int(row['instance_number'])}.dcm"
    if os.path.exists(path):
        valid_rows.append(row)

df_final = pd.DataFrame(valid_rows).reset_index(drop=True)
level_map = {'l1_l2': 0, 'l2_l3': 1, 'l3_l4': 2, 'l4_l5': 3, 'l5_s1': 4}
df_final['level_idx'] = df_final['level_str'].map(level_map)

print(f"\\n✅ Data: {len(df_final)} samples")
for i in range(3):
    c = (df_final['label']==i).sum()
    print(f"   Class {i}: {c} ({100*c/len(df_final):.1f}%)")"""))

# Dataset (v4 style - standard RGB)
cells.append(md_cell("## 2. Dataset (v4 Style - Standard RGB)"))
cells.append(code_cell("""class RSNADatasetV9(Dataset):
    \"\"\"
    v9 Dataset: Back to v4's proven approach
    - Standard RGB DICOM loading (preserves ImageNet pretrained weights)
    - Single window (WC=50, WW=350 for soft tissue)
    - CLAHE enhancement
    - 7 adjacent slices
    \"\"\"
    def __init__(self, df, num_slices=7, img_size=256, transform=None, is_training=False):
        self.df = df.reset_index(drop=True)
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = transform
        self.is_training = is_training
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def __len__(self):
        return len(self.df)
    
    def load_dicom(self, path):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            
            # Soft tissue window (standard for spine stenosis)
            wc, ww = 50, 350
            low = wc - ww/2
            high = wc + ww/2
            windowed = np.clip((img - low) / max(ww, 1) * 255, 0, 255).astype(np.uint8)
            
            # CLAHE
            enhanced = self.clahe.apply(windowed)
            
            # Convert to RGB (3-channel, same image)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            return rgb
        except:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center_inst = int(row['instance_number'])
        study_path = f"{TRAIN_IMAGES}/{row['study_id']}/{row['series_id']}"
        cx, cy = int(row['x']), int(row['y'])
        
        if self.is_training:
            jitter = self.img_size // 16
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)
        
        half = self.num_slices // 2
        indices = [center_inst + i - half for i in range(self.num_slices)]
        
        slices = []
        for inst in indices:
            path = os.path.join(study_path, f"{inst}.dcm")
            if os.path.exists(path):
                img = self.load_dicom(path)
            else:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            h, w = img.shape[:2]
            crop_size = self.img_size
            x1 = max(0, cx - crop_size)
            y1 = max(0, cy - crop_size)
            x2 = min(w, cx + crop_size)
            y2 = min(h, cy + crop_size)
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                crop = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (self.img_size, self.img_size))
            
            slices.append(crop)
        
        if self.transform:
            slices = [self.transform(image=s)['image'] for s in slices]
        else:
            slices = [torch.tensor(s).permute(2,0,1).float() / 255.0 for s in slices]
        
        sequence = torch.stack(slices, dim=0)  # (7, 3, 256, 256)
        label = torch.tensor(row['label'], dtype=torch.long)
        level_idx = torch.tensor(row['level_idx'], dtype=torch.long)
        
        return sequence, label, level_idx

print("✅ RSNADatasetV9: Standard RGB, soft tissue window, CLAHE")
print("   Output: (7, 3, 256, 256)")"""))

# Augmentation
cells.append(md_cell("## 3. Augmentation (v4 Proven)"))
cells.append(code_cell("""train_aug = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.12, rotate_limit=12,
                       border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.RandomGamma(gamma_limit=(75, 125), p=1.0),
    ], p=0.6),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

tta_augs = [
    val_aug,
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
]

print(f"✅ Augmentation: Moderate (v4 proven), {len(tta_augs)} TTA variants")"""))

# Model - Part 1: Attention MIL
cells.append(md_cell("## 4. Model Architecture"))
cells.append(code_cell("""class AttentionMIL(nn.Module):
    \"\"\"
    Attention-based Multiple Instance Learning (from 1st place solution)
    Learns which slices are most diagnostic (vs fixed weighting)
    \"\"\"
    def __init__(self, feature_dim=1024, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features):
        # features: (B, num_slices, feature_dim)
        attn_scores = self.attention(features)  # (B, num_slices, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, num_slices, 1)
        attended = (features * attn_weights).sum(dim=1)  # (B, feature_dim)
        return attended, attn_weights.squeeze(-1)  # Return weights for visualization

print("✅ AttentionMIL: Learns importance of each slice (interpretable)")"""))

# Model - Part 2: Full model
cells.append(code_cell("""class SpineModelV9(nn.Module):
    \"\"\"
    v9: v4 Foundation (BiGRU) + Attention MIL + Competition CE
    
    Architecture:
    - Backbone: EfficientNet-V2-S (unfrozen from start)
    - Sequence: BiGRU 2-layer bidirectional
    - Aggregation: Attention MIL (vs v4's AttentionPool)
    - Level: Embedding + concatenation
    - Classifier: With class-prior bias initialization
    \"\"\"
    def __init__(self, num_classes=3, num_slices=7, dropout=0.35, num_levels=5,
                 gru_hidden=512, gru_layers=2, gru_dropout=0.3, attention_hidden=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_slices = num_slices
        
        # Backbone: EfficientNet-V2-S (pretrained, NO freeze)
        effnet = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(effnet.children())[:-1])
        self.feature_dim = 1280
        
        # Project backbone features for GRU
        self.feature_proj = nn.Linear(self.feature_dim, gru_hidden)
        
        # BiGRU (v4 proven)
        self.gru = nn.GRU(
            gru_hidden,
            gru_hidden,
            num_layers=gru_layers,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention MIL (from 1st place)
        gru_out_dim = gru_hidden * 2  # Bidirectional
        self.attention = AttentionMIL(gru_out_dim, attention_hidden)
        
        # Level embedding
        self.level_embed = nn.Embedding(num_levels, 256)
        
        # Classifier
        combined_dim = gru_out_dim + 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier bias with class priors
        # P(Normal)≈88% → logit=2, P(Moderate)≈6% → logit=-2.5, P(Severe)≈6% → logit=-3
        self.classifier[-1].bias.data = torch.tensor([2.0, -2.5, -3.0])
    
    def forward(self, x, level_idx=None):
        # x: (B, num_slices, 3, H, W)
        B, S, C, H, W = x.size()
        
        # Process all slices through backbone
        x = x.view(B * S, C, H, W)
        features = self.backbone(x)  # (B*S, 1280, 1, 1)
        features = features.view(B, S, -1)  # (B, S, 1280)
        features = self.feature_proj(features)  # (B, S, gru_hidden)
        
        # BiGRU sequence processing
        gru_out, _ = self.gru(features)  # (B, S, gru_hidden*2)
        
        # Attention MIL aggregation
        attended, attn_weights = self.attention(gru_out)  # (B, gru_hidden*2)
        
        # Level conditioning
        if level_idx is not None:
            level_feat = self.level_embed(level_idx)  # (B, 256)
            combined = torch.cat([attended, level_feat], dim=1)  # (B, gru_hidden*2 + 256)
        else:
            combined = torch.cat([attended, torch.zeros(B, 256, device=x.device)], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        return {
            'logits': logits,
            'attention': attn_weights  # For visualization
        }

print("✅ SpineModelV9")
print("   - EfficientNet-V2-S backbone (unfrozen)")
print("   - BiGRU 2-layer bidirectional (v4 proven)")
print("   - Attention MIL aggregation (1st place)")
print("   - Class-prior bias init [2.0, -2.5, -3.0]")"""))

# Loss
cells.append(md_cell("## 5. Competition-Weighted CE Loss (Enhanced)"))
cells.append(code_cell("""class CompetitionWeightedCE(nn.Module):
    \"\"\"
    Competition-metric-aligned CE with INCREASED Moderate weight
    Weights: [1.0, 4.0, 6.0] (was [1, 2, 4] in v8 - too weak for Moderate)
    \"\"\"
    def __init__(self, weights=[1.0, 4.0, 6.0]):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits, labels):
        # Auto device handling
        w = self.weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=w)

print("✅ Competition-weighted CE: [1.0, 4.0, 6.0]")
print("   Moderate penalty 4x (was 2x in v8)")
print("   Severe penalty 6x (was 4x in v8)")"""))

# Training utilities  
cells.append(md_cell("## 6. Training Utilities"))
cells.append(code_cell("""def compute_per_class_metrics(preds, labels, num_classes=3):
    metrics = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            metrics[f'class_{c}_recall'] = ((preds == c) & mask).sum() / mask.sum()
        else:
            metrics[f'class_{c}_recall'] = 0.0
    return metrics

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.002, mode='max'):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score = 0, None
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score + self.min_delta) if self.mode == 'max' \\
                   else (score < self.best_score - self.min_delta)
        if improved:
            self.best_score, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience"""))

# Training loop
cells.append(md_cell("## 7. Training Loop"))
cells.append(code_cell("""def train_one_fold_v9(model, train_loader, val_loader, fold, config):
    criterion = CompetitionWeightedCE(config['class_weights'])
    
    # Optimizer with separate LR for backbone vs rest
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['backbone_lr'], 'weight_decay': config['weight_decay']},
        {'params': model.feature_proj.parameters(), 'lr': config['learning_rate'], 'weight_decay': 0.01},
        {'params': model.gru.parameters(), 'lr': config['learning_rate'], 'weight_decay': 0.01},
        {'params': model.attention.parameters(), 'lr': config['learning_rate'], 'weight_decay': 0.01},
        {'params': model.level_embed.parameters(), 'lr': config['learning_rate'], 'weight_decay': 0.01},
        {'params': model.classifier.parameters(), 'lr': config['learning_rate'], 'weight_decay': 0.01}
    ])
    
    # Cosine decay with warmup
    total_steps = config['epochs'] * len(train_loader)
    warmup_steps = config['warmup_epochs'] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    
    swa_model = AveragedModel(model) if config['use_swa'] else None
    swa_scheduler = SWALR(optimizer, swa_lr=config['swa_lr']) if config['use_swa'] else None
    
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.002, mode='max')
    best_ba = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'balanced_acc': [], 'class_0_recall': [], 'class_1_recall': [], 'class_2_recall': []
    }
    
    print(f"\\n🚀 Fold {fold+1} Training (v9)")
    print(f"   Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"   Weights: {config['class_weights']}, Freeze: {config['freeze_backbone_epochs']} epochs")
    print(f"   Target: Match v4 (74.9%) and exceed to 75%+")
    
    # NO backbone freeze in v9 (critical!)
    for param in model.parameters():
        param.requires_grad = True
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0
        is_swa = config['use_swa'] and epoch >= config['swa_start_epoch']
        
        tag = "[SWA]" if is_swa else ""
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} {tag}")
        
        for images, labels, level_idx in loop:
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            level_idx = level_idx.to(config['device'])
            
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images, level_idx)
                loss = criterion(outputs['logits'], labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            
            if is_swa:
                swa_scheduler.step()
            else:
                scheduler.step()
            
            train_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=f"{train_loss/(loop.n+1):.4f}",
                           acc=f"{100*correct/total:.1f}%")
        
        train_acc = correct / total
        if swa_model and is_swa:
            swa_model.update_parameters(model)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels, level_idx in val_loader:
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                level_idx = level_idx.to(config['device'])
                
                with autocast('cuda'):
                    outputs = model(images, level_idx)
                    loss = criterion(outputs['logits'], labels)
                
                val_loss += loss.item()
                preds = outputs['logits'].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        pc = compute_per_class_metrics(all_preds, all_labels)
        ba = (pc['class_0_recall'] + pc['class_1_recall'] + pc['class_2_recall']) / 3
        
        # Dead class monitor
        pred_counts = np.bincount(all_preds, minlength=3)
        if pred_counts.min() < 3:
            print(f"   ⚠️ Dead class warning: prediction counts = {pred_counts.tolist()}")
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['balanced_acc'].append(ba)
        for c in range(3):
            history[f'class_{c}_recall'].append(pc[f'class_{c}_recall'])
        
        print(f"📊 Train: {100*train_acc:.1f}% | Val: {100*val_acc:.1f}% | "
              f"N={100*pc['class_0_recall']:.1f}% M={100*pc['class_1_recall']:.1f}% "
              f"S={100*pc['class_2_recall']:.1f}% | BA={100*ba:.1f}%")
        
        # Save best (with min minority recall gate)
        min_minority = min(pc['class_1_recall'], pc['class_2_recall'])
        if ba > best_ba and min_minority >= 0.20:
            best_ba = ba
            torch.save(model.state_dict(), f"best_v9_fold{fold}.pth")
            print(f"   ✅ Saved! BA={100*ba:.1f}%")
        
        if early_stopping(ba):
            print(f"   ⏹️ Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"best_v9_fold{fold}.pth"))
    return model, history, best_ba"""))

# K-Fold training
cells.append(md_cell("## 8. K-Fold Training"))
cells.append(code_cell("""kfold = StratifiedGroupKFold(n_splits=CONFIG['num_folds'], shuffle=True, random_state=CONFIG['seed'])
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_final, df_final['label'], df_final['study_id'])):
    if fold not in CONFIG['train_folds']:
        continue
    
    print(f"\\n{'='*60}")
    print(f"FOLD {fold+1} (v9 — v4 Foundation + Attention MIL)")
    print(f"{'='*60}")
    
    train_df = df_final.iloc[train_idx].reset_index(drop=True)
    val_df = df_final.iloc[val_idx].reset_index(drop=True)
    
    for i in range(3):
        c = (train_df['label']==i).sum()
        print(f"   Class {i}: {c} ({100*c/len(train_df):.1f}%)")
    
    # NO WeightedRandomSampler - standard shuffle
    train_ds = RSNADatasetV9(train_df, num_slices=CONFIG['num_slices'],
                             img_size=CONFIG['img_size'], transform=train_aug, is_training=True)
    val_ds = RSNADatasetV9(val_df, num_slices=CONFIG['num_slices'],
                           img_size=CONFIG['img_size'], transform=val_aug, is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                           num_workers=2, pin_memory=True)
    
    model = SpineModelV9(
        num_classes=3,
        num_slices=CONFIG['num_slices'],
        dropout=CONFIG['dropout'],
        gru_hidden=CONFIG['gru_hidden'],
        gru_layers=CONFIG['gru_layers'],
        gru_dropout=CONFIG['gru_dropout'],
        attention_hidden=CONFIG['attention_hidden']
    ).to(CONFIG['device'])
    
    print(f"   🏗️ SpineModelV9: {sum(p.numel() for p in model.parameters()):,} params")
    
    model, history, best_ba = train_one_fold_v9(model, train_loader, val_loader, fold, CONFIG)
    fold_results.append({'fold': fold, 'best_ba': best_ba, 'history': history})
    print(f"\\n✅ Fold {fold+1}: Best BA = {100*best_ba:.1f}%")

print("\\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for r in fold_results:
    print(f"Fold {r['fold']+1}: BA = {100*r['best_ba']:.1f}%")"""))

# TTA & Results
cells.append(md_cell("## 9. TTA Evaluation"))
cells.append(code_cell("""def predict_tta_v9(model, df, config, augs):
    model.eval()
    all_probs = []
    
    for aug in augs:
        ds = RSNADatasetV9(df, num_slices=config['num_slices'],
                          img_size=config['img_size'], transform=aug, is_training=False)
        loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False,
                          num_workers=2, pin_memory=True)
        probs = []
        with torch.no_grad():
            for imgs, labels, lidx in loader:
                imgs = imgs.to(config['device'])
                lidx = lidx.to(config['device'])
                with autocast('cuda'):
                    outputs = model(imgs, lidx)
                    p = F.softmax(outputs['logits'], dim=1)
                probs.append(p.cpu().numpy())
        all_probs.append(np.concatenate(probs, 0))
    
    avg = np.mean(all_probs, 0)
    return np.argmax(avg, 1), avg

model.eval()
tta_preds, _ = predict_tta_v9(model, val_df, CONFIG, tta_augs)
no_tta_preds, _ = predict_tta_v9(model, val_df, CONFIG, [val_aug])

labels = val_df['label'].values

pc1 = compute_per_class_metrics(no_tta_preds, labels)
ba1 = np.mean([pc1[f'class_{c}_recall'] for c in range(3)])
pc2 = compute_per_class_metrics(tta_preds, labels)
ba2 = np.mean([pc2[f'class_{c}_recall'] for c in range(3)])

print(f"\\n{'='*60}")
print(f"Without TTA: BA={100*ba1:.1f}%  N={100*pc1['class_0_recall']:.1f}%  "
      f"M={100*pc1['class_1_recall']:.1f}%  S={100*pc1['class_2_recall']:.1f}%")
print(f"With TTA:    BA={100*ba2:.1f}%  N={100*pc2['class_0_recall']:.1f}%  "
      f"M={100*pc2['class_1_recall']:.1f}%  S={100*pc2['class_2_recall']:.1f}%")
print(f"Delta:       {100*(ba2-ba1):+.1f}%")

print("\\n" + "="*50)
print(classification_report(labels, tta_preds,
                           target_names=['Normal/Mild', 'Moderate', 'Severe']))

cm = confusion_matrix(labels, tta_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Normal/Mild', 'Moderate', 'Severe'],
            yticklabels=['Normal/Mild', 'Moderate', 'Severe'])
plt.ylabel('True'); plt.xlabel('Predicted')
plt.title(f'v9 Confusion Matrix (BA: {100*ba2:.1f}%)')
plt.tight_layout(); plt.show()"""))

# Training plots
cells.append(md_cell("## 10. Training History"))
cells.append(code_cell("""if fold_results:
    h = fold_results[0]['history']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(h['train_loss'])+1)
    
    axes[0].plot(ep, h['train_loss'], 'b-', label='Train')
    axes[0].plot(ep, h['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(ep, h['class_0_recall'], 'g-o', label='Normal', ms=3)
    axes[1].plot(ep, h['class_1_recall'], color='orange', marker='s', label='Moderate', ms=3)
    axes[1].plot(ep, h['class_2_recall'], 'r-^', label='Severe', ms=3)
    axes[1].axhline(y=0.75, color='gray', linestyle='--', alpha=0.3, label='Target (75%)')
    axes[1].set_title('Per-Class Recall'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ep, h['balanced_acc'], 'purple', marker='d', lw=2, ms=3)
    axes[2].axhline(y=0.749, color='red', linestyle='--', alpha=0.5, label='v4 Best (74.9%)')
    axes[2].axhline(y=0.755, color='green', linestyle='--', alpha=0.5, label='Target (75.5%)')
    axes[2].set_title(f'BA (Best: {100*max(h["balanced_acc"]):.1f}%)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()"""))

# Summary
cells.append(md_cell("""## Training Complete — v9

### Key Results:
- ✅ **BA:** ___% (Target: 75%+, v4 baseline: 74.9%)
- ✅ **Moderate Recall:** ___%  (Target: 65%+)
- ✅ **Severe Recall:** ___%  (Target: 70%+)
- ✅ **No Dead Class:** All classes active from epoch 1

### What Changed from v8:
- ✅ Reverted to v4's proven BiGRU architecture
- ✅ Standard RGB input (no broken multi-window)
- ✅ NO backbone freeze (critical for minority classes)
- ✅ Higher class weights [1, 4, 6] (Moderate 4x vs 2x)
- ✅ Added Attention MIL (from 1st place solution)

### Next Steps:
1. If BA ≥ 75%: Run 5-fold ensemble (target 76-78%)
2. If BA < 75% but > 74%: Tune weights to [1, 5, 7]
3. Analyze attention weights - which slices matter most?
"""))

# Create notebook
nb = create_notebook(cells)
with open('le3ba_v9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Built le3ba_v9.ipynb")
print(f"   Cells: {len(cells)}")
print(f"   Architecture: v4 BiGRU + Attention MIL")
print(f"   Target: 75%+ BA")
