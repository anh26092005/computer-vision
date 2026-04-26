#!/usr/bin/env python
# coding: utf-8

# # Multi-modal AI System for Quality Inspection and VQA
# This notebook implements a complete multi-task learning architecture combining a Vision Pipeline and a VQA Pipeline for industrial defect detection, specifically using the MVTec AD `metal_nut` dataset.

# ## Step 1: Environment Setup & Imports
# Installing required libraries and setting up device configuration.

# In[1]:


#!pip install -q torch torchvision transformers opencv-python matplotlib albumentations numpy requests tqdm


# In[5]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tarfile
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from transformers import DistilBertModel, DistilBertTokenizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cpu':
    print('⚠️ WARNING: CPU environment detected. Training will be extremely slow!')
    print('💡 TIP: Please switch to a GPU environment (e.g., Google Colab T4 GPU) for practical training times.')
if torch.cuda.is_available():
    print(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')


# ## Step 2: Exploratory Data Analysis (EDA) & Preprocessing
# Downloading the dataset and preparing PyTorch Datasets.

# In[6]:


# Dataset directory
DATA_DIR = './dataset'
print(f"Using dataset from {DATA_DIR}/metal_nut")


# In[7]:


# EDA: Visualizing Normal vs Defect
def show_sample(img_path, mask_path=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Simple Digital Image Processing Baseline (Canny Edge)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Canny Edges Baseline')
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title('Normal Image')
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title('Canny Edges')
        plt.show()

# Show a normal sample
normal_dir = os.path.join(DATA_DIR, 'metal_nut/train/good')
normal_files = os.listdir(normal_dir)
if normal_files:
    show_sample(os.path.join(normal_dir, normal_files[0]))

# Show an anomalous sample
scratch_dir = os.path.join(DATA_DIR, 'metal_nut/test/scratch')
mask_dir = os.path.join(DATA_DIR, 'metal_nut/ground_truth/scratch')
if os.path.exists(scratch_dir):
    scratch_files = os.listdir(scratch_dir)
    if scratch_files:
        img_name = scratch_files[0]
        mask_name = img_name.replace('.png', '_mask.png')
        show_sample(os.path.join(scratch_dir, img_name), os.path.join(mask_dir, mask_name))


# In[8]:


# Dataset definition
import random
class DefectDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # Define some basic VQA questions
        self.vocab = {"yes": 0, "no": 1, "scratch": 2, "bent": 3, "color": 4, "flip": 5, "good": 6}
        self.idx2vocab = {v: k for k, v in self.vocab.items()}

        master_list = []
        base_dir = os.path.join(root_dir, 'metal_nut')

        # Traverse BOTH original train and test folders
        for orig_split in ['train', 'test']:
            split_dir = os.path.join(base_dir, orig_split)
            if not os.path.exists(split_dir): continue
            categories = os.listdir(split_dir)
            for cat in categories:
                cat_dir = os.path.join(split_dir, cat)
                if not os.path.isdir(cat_dir): continue
                for img_name in os.listdir(cat_dir):
                    if not img_name.endswith('.png'): continue
                    img_path = os.path.join(cat_dir, img_name)

                    has_defect = (cat != 'good')
                    mask_path = None
                    if has_defect:
                        mask_name = img_name.replace('.png', '_mask.png')
                        mask_path = os.path.join(base_dir, 'ground_truth', cat, mask_name)

                    # Create synthetic Q&A
                    questions = []
                    if has_defect:
                        questions.append(("Is there a defect?", "yes"))
                        questions.append(("What type of defect is this?", cat))
                    else:
                        questions.append(("Is there a defect?", "no"))
                        questions.append(("What type of defect is this?", "good"))

                    for q, a in questions:
                        if a in self.vocab:
                            master_list.append({
                                'image_path': img_path,
                                'mask_path': mask_path,
                                'question': q,
                                'answer_label': self.vocab[a],
                                'category': cat
                            })

        # Shuffle master list with a fixed seed
        random.seed(42)
        random.shuffle(master_list)

        # Custom 80/20 Split
        split_idx = int(0.8 * len(master_list))
        if self.split == 'train':
            self.samples = master_list[:split_idx]
        else:
            self.samples = master_list[split_idx:]

        if self.split == 'train':
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        orig_w, orig_h = img.size

        img_tensor = self.transform(img)

        # Load or generate mask
        mask_tensor = torch.zeros((1, self.img_size, self.img_size))
        bbox = torch.zeros(4) # [x_center, y_center, width, height]
        has_defect = 0.0

        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = Image.open(sample['mask_path']).convert('L')
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

            # Generate bbox from mask
            y_indices, x_indices = np.where(mask_np > 0.5)
            if len(y_indices) > 0:
                y1, y2 = y_indices.min(), y_indices.max()
                x1, x2 = x_indices.min(), x_indices.max()
                xc = (x1 + x2) / 2.0 / self.img_size
                yc = (y1 + y2) / 2.0 / self.img_size
                w = (x2 - x1) / self.img_size
                h = (y2 - y1) / self.img_size
                bbox = torch.tensor([xc, yc, w, h], dtype=torch.float32)
                has_defect = 1.0

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'bbox': bbox,
            'has_defect': torch.tensor([has_defect], dtype=torch.float32),
            'question': sample['question'],
            'answer': torch.tensor(sample['answer_label'], dtype=torch.long)
        }

# Batch size: 8 is very safe for 4GB-8GB VRAM, can be increased if you have Colab T4 15GB
BATCH_SIZE = 8
train_dataset = DefectDataset(DATA_DIR, split='train')
test_dataset = DefectDataset(DATA_DIR, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# In[9]:


import matplotlib.pyplot as plt
from PIL import Image

plt.figure(figsize=(15, 12))

for i in range(12):
    sample = train_dataset.samples[i] 
    img_path = sample['image_path']
    category = sample['category']

    img = Image.open(img_path)
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f"Defect: {category}", fontsize=14)
    plt.axis("off")

plt.tight_layout()
plt.show()


# ## Step 3: Vision Backbone & Heads
# Defining ResNet18 backbone, U-Net Segmentation Head, and Simplified Anchor-Free Detection Head.

# In[10]:


class VisionPipeline(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        self.img_size = img_size

        # Backbone (ResNet18)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # output shape: [B, 512, H/32, W/32]

        # Segmentation Head (U-Net style lightweight decoder)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # H/16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # H
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1) # Output mask [B, 1, H, W]
        )

        # Detection Head (Simplified anchor-free: CenterNet style)
        # Predicts 1 object per image for simplicity (Defect bounding box)
        self.det_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # [B, 256, 1, 1]
            nn.Flatten(),
            nn.Linear(256, 5) # [prob_defect, xc, yc, w, h]
        )

    def forward(self, x):
        features = self.backbone(x)

        # Segmentation Branch
        mask_logits = self.seg_decoder(features)

        # Detection Branch
        det_output = self.det_head(features)
        defect_logits = det_output[:, 0:1] # [B, 1]
        bbox_preds = torch.sigmoid(det_output[:, 1:5]) # [B, 4] (normalized 0-1)

        return features, mask_logits, defect_logits, bbox_preds


# ## Step 4: Language Model & Fusion
# Using DistilBERT for text encoding and a Cross-Attention mechanism to fuse visual and textual embeddings.

# In[11]:


class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, vision_pipeline):
        super().__init__()
        self.vision = vision_pipeline

        # Text Encoder
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False # Freeze BERT to save VRAM

        # Fusion Module (Cross-Attention)
        self.visual_proj = nn.Linear(512, 768) # ResNet output (512) to BERT hidden size (768)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # VQA Head
        self.vqa_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vocab_size)
        )

    def forward(self, images, questions):
        B = images.size(0)

        # 1. Vision Forward
        visual_features, mask_logits, defect_logits, bbox_preds = self.vision(images)

        # 2. Process Visual Features for Fusion
        # visual_features: [B, 512, H/32, W/32]
        H, W = visual_features.size(2), visual_features.size(3)
        visual_flat = visual_features.view(B, 512, -1).permute(0, 2, 1) # [B, H*W, 512]
        visual_emb = self.visual_proj(visual_flat) # [B, H*W, 768]

        # 3. Text Forward
        encoded_text = self.tokenizer(questions, padding=True, truncation=True, return_tensors='pt', max_length=32).to(images.device)
        text_emb = self.text_encoder(**encoded_text).last_hidden_state # [B, seq_len, 768]

        # 4. Fusion (Text attends to Visual)
        fused_emb, _ = self.cross_attention(query=text_emb, key=visual_emb, value=visual_emb)

        # Pool the fused embeddings (use the CLS token representation)
        pooled_fused_emb = fused_emb[:, 0, :] # [B, 768]

        # 5. VQA Output
        vqa_logits = self.vqa_classifier(pooled_fused_emb)

        return mask_logits, defect_logits, bbox_preds, vqa_logits


# ## Step 5: Multi-task Loss & Optimizer
# Defining a composite loss function for Segmentation, Detection, and VQA.

# In[12]:


# Loss weights
lambda_seg = 1.0
lambda_det_cls = 1.0
lambda_det_box = 5.0
lambda_vqa = 1.0

seg_criterion = nn.BCEWithLogitsLoss()
det_cls_criterion = nn.BCEWithLogitsLoss()
det_box_criterion = nn.MSELoss(reduction='none')
vqa_criterion = nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def compute_loss(mask_logits, defect_logits, bbox_preds, vqa_logits, batch):
    # 1. Segmentation Loss
    loss_seg_bce = seg_criterion(mask_logits, batch['mask'].to(device))
    loss_seg_dice = dice_loss(mask_logits, batch['mask'].to(device))
    loss_seg = loss_seg_bce + loss_seg_dice

    # 2. Detection Loss
    gt_has_defect = batch['has_defect'].to(device)
    gt_bbox = batch['bbox'].to(device)

    loss_det_cls = det_cls_criterion(defect_logits, gt_has_defect)

    # Box loss only computed if there is a defect in ground truth
    loss_box = det_box_criterion(bbox_preds, gt_bbox).sum(dim=1)
    loss_det_box = (loss_box * gt_has_defect.view(-1)).mean() # Average over batch

    # 3. VQA Loss
    loss_vqa = vqa_criterion(vqa_logits, batch['answer'].to(device))

    total_loss = (lambda_seg * loss_seg) + (lambda_det_cls * loss_det_cls) + (lambda_det_box * loss_det_box) + (lambda_vqa * loss_vqa)

    return total_loss, loss_seg, loss_det_cls, loss_det_box, loss_vqa


# ## Step 6: Training Loop
# Modular training and validation loop tracking all metrics.

# In[10]:


# Initialize Model
vision_net = VisionPipeline().to(device)
model = MultiModalModel(vocab_size=len(train_dataset.vocab), vision_pipeline=vision_net).to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)

# 1. Lịch trình học thuật toán theo đường cong nghệ thuật Cosine
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

# 2. Bộ Tăng Tốc Kép Dành Riêng Cho RTX GPUs (AMP)
scaler = torch.amp.GradScaler('cuda') 

EPOCHS = 60
best_loss = float('inf')

print(" train")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        images = batch['image'].to(device)
        questions = batch['question']

        optimizer.zero_grad()

        # 3. Ép xung Autocast Giúp RTX 3050 tính toán dấu phẩy động F32/F16 cực mượt
        with torch.amp.autocast('cuda'):
            mask_logits, defect_logits, bbox_preds, vqa_logits = model(images, questions)
            loss, l_seg, l_dcls, l_dbox, l_vqa = compute_loss(mask_logits, defect_logits, bbox_preds, vqa_logits, batch)

        # 4. Truyền đáp án ngược lại vào Model (Lõi AMP)
        scaler.scale(loss).backward()

        # 5. Khóa an toàn chống nghẽn mạch Gradient (Gradient Clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Ghi nhớ bài học
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'vqa_l': f"{l_vqa.item():.4f}"})

    scheduler.step()

    avg_loss = total_train_loss / len(train_loader)
    print(f"🌟 Epoch {epoch+1} - Mức Sai Số: {avg_loss:.4f} | Rê ga Tốc học: {scheduler.get_last_lr()[0]:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'   (Loss: {best_loss:.4f})')


# ## Step 7: Evaluation & Visualization
# Running inference on the test set and visualizing the multi-modal outputs.

# In[13]:


import os
# Khởi tạo mô hình
vision_net = VisionPipeline().to(device)
model = MultiModalModel(vocab_size=len(train_dataset.vocab), vision_pipeline=vision_net).to(device)

# Load pth thay vì phải train
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

print("Đã nạp thành công mô hình MultiModalModel v2!")

if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print('Loaded best model weights for evaluation.')
model.eval()

# Helper function to convert normalized bbox to image pixels
def unnormalize_bbox(bbox, img_size):
    xc, yc, w, h = bbox
    x1 = int((xc - w/2) * img_size)
    y1 = int((yc - h/2) * img_size)
    x2 = int((xc + w/2) * img_size)
    y2 = int((yc + h/2) * img_size)
    return x1, y1, x2, y2

# Take a batch from test set
batch = next(iter(test_loader))
images = batch['image'].to(device)
questions = batch['question']

with torch.no_grad():
    mask_logits, defect_logits, bbox_preds, vqa_logits = model(images, questions)

    pred_masks = torch.sigmoid(mask_logits).cpu().numpy()
    pred_defect_probs = torch.sigmoid(defect_logits).cpu().numpy()
    pred_bboxes = bbox_preds.cpu().numpy()
    pred_answers = torch.argmax(vqa_logits, dim=1).cpu().numpy()

# Denormalize image for display
def denormalize_img(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = tensor.cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# Visualize first 3 samples in the batch
for i in range(min(3, BATCH_SIZE)):
    img_disp = denormalize_img(batch['image'][i])
    gt_mask = batch['mask'][i].squeeze().numpy()

    # Predicted components
    pred_mask = pred_masks[i].squeeze()
    pred_answer_text = train_dataset.idx2vocab[pred_answers[i]]
    prob_defect = pred_defect_probs[i][0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original Image with Bounding Box
    ax = axes[0]
    ax.imshow(img_disp)
    if prob_defect > 0.5:
        x1, y1, x2, y2 = unnormalize_bbox(pred_bboxes[i], 256)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    ax.set_title(f"Q: {questions[i]}\nPred A: {pred_answer_text}")
    ax.axis('off')

    # 2. Ground Truth Mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # 3. Predicted Mask
    axes[2].imshow(pred_mask, cmap='viridis')
    axes[2].set_title('Predicted Segmentation Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ## Step 8: Build Giao diện Người dùng (GUI) & Tracking
# 

# In[14]:


import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import math
import csv
from datetime import datetime
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vocab = {"yes": 0, "no": 1, "scratch": 2, "bent": 3, "color": 4, "flip": 5, "good": 6}
idx2vocab = {v: k for k, v in vocab.items()}

class PhanMemKiemTraLoi:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.vid = None
        self.is_running = False
        self.is_paused = False 

        self.tracked_items = {} 
        self.next_id = 1
        self.report_data = [] 

        self.canvas = tk.Canvas(window, width=800, height=600, bg="black")
        self.canvas.pack()

        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X, pady=15)

        self.btn_file = tk.Button(btn_frame, text="Mở File Video", command=self.open_file, font=('Arial', 12, 'bold'), bg='lightblue', width=15)
        self.btn_file.pack(side=tk.LEFT, padx=30)
        self.btn_export = tk.Button(btn_frame, text="Xuất Báo Cáo (CSV)", command=self.export_report, font=('Arial', 12, 'bold'), bg='orange', width=20)
        self.btn_export.pack(side=tk.LEFT, padx=30)
        self.btn_pause = tk.Button(btn_frame, text="Tạm Dừng", command=self.toggle_pause, font=('Arial', 12, 'bold'), bg='#ff6666', width=15)
        self.btn_pause.pack(side=tk.RIGHT, padx=30)

        self.delay = 15
        self.update_video()
        self.window.mainloop()

    def export_report(self):
        if not self.report_data:
            return
        filename = f"BaoCao_KiemTra_OcVit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["ID Sản Phẩm", "Tình Trạng", "Tỷ Lệ Báo Lỗi"])
            for row in self.report_data:
                writer.writerow([row["ID"], row["Ket_Qua"], row["Ty_Le"]])
        self.canvas.create_text(400, 50, text=f"ĐÃ LƯU EXCEL THÀNH CÔNG!", fill="yellow", font=('Arial', 20, 'bold'))

    def open_file(self):
        video_source = filedialog.askopenfilename(title="Chọn Video MP4", filetypes=[("Video Files", "*.mp4 *.avi")])
        if video_source:
            self.vid = cv2.VideoCapture(video_source)
            self.is_running = True
            self.is_paused = False 
            self.btn_pause.config(text="Tạm Dừng", bg='#ff6666')

            self.tracked_items.clear()
            self.report_data.clear() 
            self.next_id = 1

    def toggle_pause(self):
        if self.vid and self.vid.isOpened():
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.btn_pause.config(text="Tiếp Tục", bg='lightgreen')
            else:
                self.btn_pause.config(text="Tạm Dừng", bg='#ff6666')

    def update_video(self):
        if self.is_running and self.vid.isOpened() and not self.is_paused:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (800, 600)) 
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

                # --- CROP ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(blur, 50, 150)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                dilated = cv2.dilate(edges, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                current_frame_objects = []
                for c in contours:

                    if cv2.contourArea(c) > 15000:
                        x, y, w, h = cv2.boundingRect(c)
                        if y < 140: continue 

                        pad = 15
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(800, x + w + pad), min(600, y + h + pad)
                        cx, cy = x + w/2, y + h/2

                        roi_img = rgb_frame[y1:y2, x1:x2]
                        if roi_img.shape[0] < 10 or roi_img.shape[1] < 10: continue

                        current_frame_objects.append((cx, cy, x1, y1, x2, y2, roi_img))

                # ---  TRACKING ĐÁNH ID ---
                new_tracked_items = {}
                for (cx, cy, x1, y1, x2, y2, roi_img) in current_frame_objects:
                    matched_id = None
                    min_dist = 200 

                    for obj_id, data in self.tracked_items.items():
                        old_cx, old_cy = data['ct']
                        dist = math.hypot(cx - old_cx, cy - old_cy)
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = obj_id
                            break 

                    if matched_id is not None:
                        obj_data = self.tracked_items.pop(matched_id)
                        obj_data['ct'] = (cx, cy)
                        obj_data['missed_frames'] = 0
                    else:
                        matched_id = self.next_id
                        self.next_id += 1
                        obj_data = {
                            'ct': (cx, cy), 
                            'label': 'DANG XET AI...', 
                            'color': (200, 200, 200), 
                            'locked': False, 
                            'missed_frames': 0,
                            'mask': None  
                        }


                    if not obj_data['locked'] and (200 < cx < 600):
                        pil_img = Image.fromarray(roi_img)
                        img_tensor = transform(pil_img).unsqueeze(0).to(device)
                        questions = ["What type of defect is this?"]

                        with torch.no_grad():
                            # SEGMENTATION 
                            mask_logits, defect_logits, _, vqa_logits = model(img_tensor, questions)
                            prob = torch.sigmoid(defect_logits).item()
                            ans_idx = torch.argmax(vqa_logits, dim=-1).item()
                            loai_loi = idx2vocab[ans_idx] 

                            if prob > 0.5 and loai_loi != "good":
                                obj_data['label'] = f"Loi: {loai_loi.upper()}"
                                obj_data['color'] = (255, 0, 0)
                                tyle_str = f"{prob*100:.1f}%"

                                # LẤY TOẠ ĐỘ VÙNG BỊ LỖI   
                                mask = torch.sigmoid(mask_logits[0, 0]).cpu().numpy() > 0.05                             
                                mask_resized = cv2.resize(mask.astype(np.uint8), (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                                obj_data['mask'] = mask_resized
                            else:
                                obj_data['label'] = "OK (GOOD)"
                                obj_data['color'] = (0, 255, 0)
                                tyle_str = "N/A"
                                obj_data['mask'] = None 

                            obj_data['locked'] = True 

                            self.report_data.append({
                                "ID": f"OC_VIT_SO_{matched_id}", 
                                "Ket_Qua": obj_data['label'], 
                                "Ty_Le": tyle_str
                            })


                    if obj_data.get('mask') is not None:
                        curr_w, curr_h = x2 - x1, y2 - y1
                        # Khớp giãn lớp Màng đỏ theo sự dịch chuyển của ốc
                        m = cv2.resize(obj_data['mask'], (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)

                        overlay = rgb_frame[y1:y2, x1:x2].copy()
                        overlay[m > 0] = (255, 0, 0) # Sơn màu Đỏ

                        rgb_frame[y1:y2, x1:x2] = cv2.addWeighted(rgb_frame[y1:y2, x1:x2], 0.5, overlay, 0.5, 0)

                    # VẼ KHUNG
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), obj_data['color'], 3)
                    cv2.putText(rgb_frame, f"ID {matched_id}: {obj_data['label']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obj_data['color'], 2)

                    new_tracked_items[matched_id] = obj_data

                for obj_id, data in self.tracked_items.items():
                    data['missed_frames'] += 1
                    if data['missed_frames'] < 10:
                        new_tracked_items[obj_id] = data
                self.tracked_items = new_tracked_items

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.is_running = False
                self.canvas.create_text(400, 300, text="ĐÃ PHÂN TÍCH HẾT VIDEO", fill="white", font=('Arial', 24, 'bold'))

        self.window.after(self.delay, self.update_video)

root = tk.Tk()
app = PhanMemKiemTraLoi(root, "kiem tra sam pham loi")