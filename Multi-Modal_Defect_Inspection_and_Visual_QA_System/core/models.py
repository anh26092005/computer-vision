import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel, DistilBertTokenizer

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
