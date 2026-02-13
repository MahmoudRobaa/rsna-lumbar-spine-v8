"""
Build le3ba_v10.ipynb — v4 Faithful Core + 3 Targeted Improvements

KEPT FROM v4 (proven at 74.9% BA):
  - LSTM bidirectional 2-layer (hidden=256)
  - nn.MultiheadAttention (4 heads) -> mean pooling
  - Progressive oversampling (data-level, stratified by level×label)
  - Dual augmentation (normal + strong for oversampled copies)
  - Auxiliary heads (binary: normal vs abnormal, severity: moderate vs severe)
  - FocalLoss with label_smoothing=0.1, gamma=2.5
  - Feature projection: Dropout→Linear→LayerNorm→GELU
  - Level embedding (dim=64)
  - LR: 3e-4 head / 3e-5 backbone
  - Crop: img_size // 2 from center
  - DICOM: min-max → CLAHE
  - Batch size 8

TARGETED IMPROVEMENTS (3 changes):
  1. Competition weights [1, 2, 4] as Focal alpha (instead of sqrt-inverse)
  2. Gradient clipping max_norm=1.0 (safety net)
  3. Fixed syntax error in v4 model init (missing comma)
"""
import json
from build_helpers import create_notebook, code_cell, md_cell

cells = []

# ─── Header ───
cells.append(md_cell("""# RSNA 2024 — Version 10
## v4 Faithful Core + Minimal Improvements

**Philosophy:** Keep v4's proven system (74.9% BA). Change only 3 things with strong evidence.

| Component | v4 (74.9%) | v10 | Changed? |
|-----------|-----------|-----|----------|
| RNN | LSTM bidir | LSTM bidir | ✅ Same |
| Attention | MultiheadAttention(4) | MultiheadAttention(4) | ✅ Same |
| Oversampling | Progressive stratified | Progressive stratified | ✅ Same |
| Aux heads | Binary + Severity | Binary + Severity | ✅ Same |
| Loss | FocalLoss(γ=2.5, LS=0.1) | FocalLoss(γ=2.5, LS=0.1) | ✅ Same |
| LR | 3e-4 / 3e-5 | 3e-4 / 3e-5 | ✅ Same |
| Crop | img_size//2 | img_size//2 | ✅ Same |
| **Weights** | **sqrt-inverse** | **[1, 2, 4] competition** | 🔄 Changed |
| **Grad clip** | **None** | **max_norm=1.0** | 🔄 Added |
| **Syntax bug** | **Missing comma** | **Fixed** | 🐛 Fixed |
"""))

# ─── Imports ───
cells.append(code_cell("""import os, copy, cv2, glob, pydicom, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2"""))

# ─── Config (v4 exact + 3 changes) ───
cells.append(code_cell("""CONFIG = {
    'seed': 42,
    'img_size': 256,
    'seq_length': 7,
    'batch_size': 8,           # v4 exact
    'epochs': 25,
    'learning_rate': 3e-4,     # v4 exact (NOT 1e-4)
    'backbone_lr': 3e-5,       # v4 exact (NOT 1e-5)
    'weight_decay': 0.05,      # v4 exact
    'patience': 10,
    'num_folds': 5,
    'train_folds': [0],
    'focal_gamma': 2.5,        # v4 exact
    'label_smoothing': 0.1,    # v4 exact
    'dropout': 0.4,            # v4 exact
    'num_attention_heads': 4,  # v4 exact
    'warmup_epochs': 2,        # v4 exact
    'oversample_strategy': 'progressive',  # v4 exact
    'min_minority_recall': 0.20,
    'aux_weight': 0.1,         # v4 exact
    
    # --- CHANGE 1: Competition weights instead of sqrt-inverse ---
    'class_weights': [1.0, 2.0, 4.0],
    
    # --- CHANGE 2: Gradient clipping (v4 had none) ---
    'clip_grad_norm': 1.0,
    
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
print(f"✅ v10: v4 Faithful Core + Competition Weights + Grad Clip")
print(f"   LR: {CONFIG['learning_rate']}/{CONFIG['backbone_lr']} (v4 exact)")
print(f"   Focal: γ={CONFIG['focal_gamma']}, LS={CONFIG['label_smoothing']} (v4 exact)")
print(f"   Weights: {CONFIG['class_weights']} (competition, was sqrt-inverse)")
print(f"   Grad clip: {CONFIG['clip_grad_norm']} (new safety net)")"""))

# ─── Data Loading (v4 exact) ───
cells.append(md_cell("## 1. Data Loading"))
cells.append(code_cell("""DATA_ROOT = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/"
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

# ─── Progressive Oversampling (v4 exact) ───
cells.append(md_cell("## 2. Progressive Oversampling (v4 — proven)"))
cells.append(code_cell("""def create_stratified_balanced_df(df, strategy='progressive', random_state=42):
    \"\"\"
    v4's progressive balancing: considers both label AND level together.
    Adds augmentation variant tracking for diversity.
    \"\"\"
    np.random.seed(random_state)
    grouped = df.groupby(['level_idx', 'label'])
    balanced_dfs = []
    
    print("\\n📊 Stratified Sampling Details:")
    
    for (level, label), group_df in grouped:
        group_df = group_df.copy()
        group_df['is_oversampled'] = False
        group_df['aug_variant'] = 0
        current_count = len(group_df)
        level_counts = df[df['level_idx'] == level]['label'].value_counts()
        
        if strategy == 'progressive':
            target_count = int(level_counts.median() * (1 + 0.3 * label))
        elif strategy == 'balanced':
            target_count = level_counts.max()
        else:
            target_count = current_count
        
        samples_needed = target_count - current_count
        
        if samples_needed > 0:
            oversample_indices = np.random.choice(group_df.index, size=samples_needed, replace=True)
            oversampled_df = df.loc[oversample_indices].copy()
            oversampled_df['is_oversampled'] = True
            oversampled_df['aug_variant'] = np.random.randint(0, 4, size=len(oversampled_df))
            print(f"   Level {level}, Label {label}: {current_count} → {target_count} (+{samples_needed})")
            balanced_dfs.append(group_df)
            balanced_dfs.append(oversampled_df)
        else:
            print(f"   Level {level}, Label {label}: {current_count} (no oversampling)")
            balanced_dfs.append(group_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

print("✅ Progressive oversampling (v4 exact)")
print("   Minority classes get MORE copies, with STRONGER augmentation")"""))

# ─── Dataset (v4 exact — with dual augmentation) ───
cells.append(md_cell("## 3. Dataset (v4 — with Adaptive Augmentation)"))
cells.append(code_cell("""class RSNASequenceDataset(Dataset):
    \"\"\"v4's exact dataset: min-max DICOM, CLAHE, crop_size=img_size//2, dual aug.\"\"\"
    def __init__(self, df, seq_length=7, img_size=256, transform=None, 
                 strong_transform=None, is_training=False):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.img_size = img_size
        self.transform = transform
        self.strong_transform = strong_transform
        self.is_training = is_training
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def __len__(self):
        return len(self.df)
    
    def load_dicom(self, path):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
            else:
                img = np.zeros_like(img)
            img = img.astype(np.uint8)
            img = self.clahe.apply(img)
            return img
        except:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center_inst = int(row['instance_number'])
        study_path = f"{TRAIN_IMAGES}/{row['study_id']}/{row['series_id']}"
        cx, cy = int(row['x']), int(row['y'])
        
        is_oversampled = row.get('is_oversampled', False)
        aug_variant = row.get('aug_variant', 0)
        
        if is_oversampled and self.is_training:
            np.random.seed(idx * 1000 + int(aug_variant))
            random.seed(idx * 1000 + int(aug_variant))
        
        start = center_inst - (self.seq_length // 2)
        indices = [start + i for i in range(self.seq_length)]
        
        images_list = []
        for inst in indices:
            path = os.path.join(study_path, f"{inst}.dcm")
            if os.path.exists(path):
                img = self.load_dicom(path)
            else:
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            h, w = img.shape
            crop_size = self.img_size // 2   # v4 exact: 128px from center
            x1 = max(0, cx - crop_size)
            y1 = max(0, cy - crop_size)
            x2 = min(w, cx + crop_size)
            y2 = min(h, cy + crop_size)
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                crop = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (self.img_size, self.img_size))
            
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            
            if self.is_training and is_oversampled and self.strong_transform:
                res = self.strong_transform(image=crop)
            elif self.transform:
                res = self.transform(image=crop)
            else:
                res = {'image': torch.tensor(crop).permute(2, 0, 1).float() / 255.0}
            
            images_list.append(res['image'])
            
        sequence = torch.stack(images_list, dim=0)
        label = torch.tensor(row['label'], dtype=torch.long)
        level_idx = torch.tensor(row['level_idx'], dtype=torch.long)
        
        return sequence, label, level_idx

print("✅ Dataset (v4 exact)")
print("   DICOM: min-max normalize → CLAHE")
print("   Crop: img_size//2 = 128px from center")
print("   Oversampled copies → strong augmentation")"""))

# ─── Augmentation (v4 exact) ───
cells.append(md_cell("## 4. Augmentation (v4 — Normal + Strong)"))
cells.append(code_cell("""# Normal augmentation (for original samples)
train_aug = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, 
                       border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Strong augmentation (for oversampled minority copies — prevents memorization)
strong_aug = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=8,
                       border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.CLAHE(clip_limit=4.0, p=1.0),
    ], p=0.8),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
    ], p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

print("✅ Dual augmentation (v4 exact)")
print("   Normal: conservative (original samples)")
print("   Strong: aggressive (oversampled copies — Elastic, Grid, CLAHE)")"""))

# ─── Model (v4 exact + syntax fix) ───
cells.append(md_cell("## 5. Model Architecture (v4 — with Aux Heads)"))
cells.append(code_cell("""class SpineModelV10(nn.Module):
    \"\"\"
    v4's exact architecture with fixed syntax.
    
    LSTM (not GRU) + MultiheadAttention + Aux Heads
    This is what achieved 74.9% BA.
    \"\"\"
    def __init__(self, num_classes=3, hidden_dim=256, lstm_layers=2, 
                 num_heads=4, dropout=0.4, num_levels=5):
        super().__init__()
        
        # Backbone: EfficientNet-V2-S (v4 exact)
        effnet = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(effnet.children())[:-1]) 
        self.feature_dim = 1280 
        
        # Feature projection (v4 exact: Dropout→Linear→LN→GELU)
        self.feature_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        # LSTM bidirectional (v4 exact — NOT GRU)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # MultiheadAttention (v4 exact — NOT custom MIL)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Level embedding dim=64 (v4 exact)
        self.level_embedding = nn.Embedding(num_levels, 64)
        
        context_dim = hidden_dim * 2 + 64  # 512 + 64 = 576
        
        # Main 3-class classifier (v4 exact)
        self.main_classifier = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Dropout(dropout),
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary binary classifier: normal vs abnormal (v4 exact)
        self.aux_binary = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Auxiliary severity classifier: moderate vs severe (v4 exact)
        self.aux_severity = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x, level_idx=None):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
        
        features = self.backbone(x)
        features = features.view(b, s, -1)
        features = self.feature_proj(features)
        
        lstm_out, _ = self.lstm(features)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out.mean(dim=1)  # v4 exact: mean pooling
        
        if level_idx is not None:
            level_feat = self.level_embedding(level_idx)
            context = torch.cat([context, level_feat], dim=-1)
        else:
            context = torch.cat([context, torch.zeros(b, 64, device=x.device)], dim=-1)
        
        main_out = self.main_classifier(context)
        binary_out = self.aux_binary(context)
        severity_out = self.aux_severity(context)
        
        return {
            'main': main_out,
            'binary': binary_out,
            'severity': severity_out,
            'attention': attn_weights
        }

print("✅ SpineModelV10 (v4 exact architecture)")
print("   - LSTM bidirectional 2-layer (hidden=256)")
print("   - MultiheadAttention (4 heads) → mean pooling")
print("   - Aux heads: binary (normal vs abnormal) + severity (mod vs sev)")
print("   - Level embedding (dim=64)")
print("   - Feature proj: Dropout→Linear→LN→GELU")"""))

# ─── Loss (v4 Focal + CHANGE 1: competition weights) ───
cells.append(md_cell("## 6. Loss Function (v4 Focal + Competition Weights)"))
cells.append(code_cell("""class FocalLoss(nn.Module):
    \"\"\"v4's exact Focal Loss with label smoothing.\"\"\"
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

def compute_loss_with_aux(outputs, labels, criterion, aux_weight=0.1):
    \"\"\"v4's exact auxiliary loss computation.\"\"\"
    main_loss = criterion(outputs['main'], labels)
    
    binary_labels = (labels > 0).long()
    binary_loss = F.cross_entropy(outputs['binary'], binary_labels)
    
    abnormal_mask = labels > 0
    if abnormal_mask.sum() > 0:
        severity_labels = (labels[abnormal_mask] - 1)
        severity_loss = F.cross_entropy(outputs['severity'][abnormal_mask], severity_labels)
    else:
        severity_loss = torch.tensor(0.0, device=labels.device)
    
    total_loss = main_loss + aux_weight * (binary_loss + severity_loss)
    
    return total_loss, {
        'total': total_loss.item(),
        'main': main_loss.item(),
        'binary': binary_loss.item(),
        'severity': severity_loss.item()
    }

print("✅ FocalLoss(γ=2.5, label_smoothing=0.1) + Auxiliary losses")
print("   Main: 3-class focal with competition weights [1, 2, 4]")
print("   Aux binary: normal vs abnormal (CE)")
print("   Aux severity: moderate vs severe (CE, abnormal samples only)")"""))

# ─── Training utilities ───
cells.append(md_cell("## 7. Training Utilities"))
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
    def __init__(self, patience=10, min_delta=0.005, mode='max'):
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

# ─── Training Loop ───
cells.append(md_cell("## 8. Training Loop"))
cells.append(code_cell("""def train_one_fold(model, train_loader, val_loader, fold, config):
    # --- CHANGE 1: Competition weights instead of sqrt-inverse ---
    loss_weights = torch.FloatTensor(config['class_weights']).to(config['device'])
    print(f"\\n   📊 Loss weights: {config['class_weights']} (competition metric aligned)")
    
    criterion = FocalLoss(
        alpha=loss_weights, 
        gamma=config['focal_gamma'], 
        label_smoothing=config['label_smoothing']
    )
    
    # v4 exact optimizer setup
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['backbone_lr']},
        {'params': model.feature_proj.parameters(), 'lr': config['learning_rate']},
        {'params': model.lstm.parameters(), 'lr': config['learning_rate']},
        {'params': model.attention.parameters(), 'lr': config['learning_rate']},
        {'params': model.level_embedding.parameters(), 'lr': config['learning_rate']},
        {'params': model.main_classifier.parameters(), 'lr': config['learning_rate']},
        {'params': model.aux_binary.parameters(), 'lr': config['learning_rate']},
        {'params': model.aux_severity.parameters(), 'lr': config['learning_rate']}
    ], weight_decay=config['weight_decay'])
    
    # v4 exact scheduler
    warmup_steps = config['warmup_epochs'] * len(train_loader)
    total_steps = config['epochs'] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + np.cos(np.pi * progress)), 1e-6)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.005, mode='max')
    best_ba = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'balanced_acc': [], 'class_0_recall': [], 'class_1_recall': [], 'class_2_recall': []
    }
    
    print(f"\\n🚀 Fold {fold+1} Training (v10 = v4 core + competition weights)")
    print(f"   Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"   Focal γ={config['focal_gamma']}, LS={config['label_smoothing']}")
    print(f"   LR: {config['learning_rate']}/{config['backbone_lr']} (v4 exact)")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for images, labels, level_idx in loop:
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            level_idx = level_idx.to(config['device'])
            
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images, level_idx)
                loss, loss_dict = compute_loss_with_aux(
                    outputs, labels, criterion, aux_weight=config['aux_weight']
                )
            
            scaler.scale(loss).backward()
            
            # --- CHANGE 2: Gradient clipping (v4 had none) ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            preds = outputs['main'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(
                loss=f"{train_loss/(loop.n+1):.4f}",
                acc=f"{100*correct/total:.1f}%"
            )
        
        train_acc = correct / total
        
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
                    loss, _ = compute_loss_with_aux(
                        outputs, labels, criterion, aux_weight=config['aux_weight']
                    )
                
                val_loss += loss.item()
                preds = outputs['main'].argmax(dim=1)
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
        
        min_minority = min(pc['class_1_recall'], pc['class_2_recall'])
        if ba > best_ba and min_minority >= config['min_minority_recall']:
            best_ba = ba
            torch.save(model.state_dict(), f"best_v10_fold{fold}.pth")
            print(f"   ✅ Saved! BA={100*ba:.1f}% (min minority={100*min_minority:.1f}%)")
        
        if early_stopping(ba):
            print(f"   ⏹️ Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"best_v10_fold{fold}.pth"))
    return model, history, best_ba"""))

# ─── K-Fold Training ───
cells.append(md_cell("## 9. Training"))
cells.append(code_cell("""kfold = StratifiedGroupKFold(n_splits=CONFIG['num_folds'], shuffle=True, random_state=CONFIG['seed'])
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_final, df_final['label'], df_final['study_id'])):
    if fold not in CONFIG['train_folds']:
        continue
    
    print(f"\\n{'='*60}")
    print(f"FOLD {fold+1} (v10 = v4 core + competition weights + grad clip)")
    print(f"{'='*60}")
    
    train_df_original = df_final.iloc[train_idx].reset_index(drop=True)
    val_df = df_final.iloc[val_idx].reset_index(drop=True)
    
    # Show original class distribution
    original_counts = np.bincount(train_df_original['label'].values, minlength=3)
    print(f"\\n📊 ORIGINAL class distribution:")
    for i, c in enumerate(original_counts):
        print(f"   Class {i}: {c} ({100*c/len(train_df_original):.1f}%)")
    
    # Apply v4's progressive oversampling
    train_df = create_stratified_balanced_df(
        train_df_original, 
        strategy=CONFIG['oversample_strategy'],
        random_state=CONFIG['seed'] + fold
    )
    
    print(f"\\n📊 After oversampling: {len(train_df_original)} → {len(train_df)} samples")
    print(f"   New distribution: {train_df['label'].value_counts().sort_index().to_dict()}")
    
    train_ds = RSNASequenceDataset(
        train_df, seq_length=CONFIG['seq_length'], img_size=CONFIG['img_size'],
        transform=train_aug, strong_transform=strong_aug, is_training=True
    )
    
    val_df['is_oversampled'] = False
    val_ds = RSNASequenceDataset(
        val_df, seq_length=CONFIG['seq_length'], img_size=CONFIG['img_size'],
        transform=val_aug, is_training=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                           num_workers=2, pin_memory=True)
    
    model = SpineModelV10(
        num_classes=3,
        num_heads=CONFIG['num_attention_heads'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   🏗️ SpineModelV10: {total_params:,} params")
    
    model, history, best_ba = train_one_fold(model, train_loader, val_loader, fold, CONFIG)
    fold_results.append({'fold': fold, 'best_ba': best_ba, 'history': history})
    print(f"\\n✅ Fold {fold+1}: Best BA = {100*best_ba:.1f}%")

print("\\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for r in fold_results:
    print(f"Fold {r['fold']+1}: BA = {100*r['best_ba']:.1f}%")"""))

# ─── Evaluation ───
cells.append(md_cell("## 10. Evaluation"))
cells.append(code_cell("""model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels, level_idx in val_loader:
        images = images.to(CONFIG['device'])
        level_idx = level_idx.to(CONFIG['device'])
        
        with autocast('cuda'):
            outputs = model(images, level_idx)
            probs = F.softmax(outputs['main'], dim=1)
        
        preds = outputs['main'].argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_labels, all_preds,
                           target_names=['Normal/Mild', 'Moderate', 'Severe']))

pc = compute_per_class_metrics(all_preds, all_labels)
ba = np.mean([pc[f'class_{c}_recall'] for c in range(3)])
print(f"\\n🎯 Final Balanced Accuracy: {100*ba:.1f}%")

cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Normal/Mild', 'Moderate', 'Severe'],
            yticklabels=['Normal/Mild', 'Moderate', 'Severe'])
plt.ylabel('True'); plt.xlabel('Predicted')
plt.title(f'v10 Confusion Matrix (BA: {100*ba:.1f}%)')
plt.tight_layout(); plt.show()"""))

# ─── Training Plots ───
cells.append(md_cell("## 11. Training History"))
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
    axes[1].axhline(y=0.75, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_title('Per-Class Recall'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ep, h['balanced_acc'], 'purple', marker='d', lw=2, ms=3)
    axes[2].axhline(y=0.749, color='red', linestyle='--', alpha=0.5, label='v4 Best (74.9%)')
    axes[2].set_title(f'BA (Best: {100*max(h["balanced_acc"]):.1f}%)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('v10: v4 Core + Competition Weights + Grad Clip')
    plt.tight_layout(); plt.show()"""))

# ─── Summary ───
cells.append(md_cell("""## Summary

### v10 = v4 Faithful Core + 3 Targeted Improvements

**Kept from v4 (every component that produced 74.9% BA):**
- ✅ LSTM bidirectional 2-layer
- ✅ MultiheadAttention (4 heads) → mean pooling
- ✅ Progressive oversampling (stratified by level × label)
- ✅ Dual augmentation (normal + strong for oversampled)
- ✅ Auxiliary heads (binary + severity)
- ✅ FocalLoss (γ=2.5, label_smoothing=0.1)
- ✅ LR: 3e-4 / 3e-5 (v4 exact)
- ✅ Crop: img_size//2 (v4 exact)
- ✅ DICOM: min-max → CLAHE (v4 exact)
- ✅ Batch size 8 (v4 exact)

**3 targeted changes:**
1. 🔄 Competition weights [1, 2, 4] (instead of sqrt-inverse)
2. 🔄 Gradient clipping max_norm=1.0 (safety net)
3. 🐛 Fixed syntax error in model init
"""))

# ─── Build ───
nb = create_notebook(cells)
with open('le3ba_v10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Built le3ba_v10.ipynb")
print(f"   Cells: {len(cells)}")
print(f"   Architecture: v4 EXACT (LSTM + MHA + Aux Heads)")
print(f"   Changes: competition weights [1,2,4] + grad clip + syntax fix")
