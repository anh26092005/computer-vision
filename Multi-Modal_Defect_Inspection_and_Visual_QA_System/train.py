import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DEVICE, DATA_DIR, BATCH_SIZE, EPOCHS, BASE_DIR
from core.dataset import DefectDataset
from core.models import VisionPipeline, MultiModalModel
from core.losses import compute_loss

if __name__ == "__main__":
    train_dataset = DefectDataset(DATA_DIR, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Initialize Model
    vision_net = VisionPipeline().to(DEVICE)
    model = MultiModalModel(vocab_size=len(train_dataset.vocab), vision_pipeline=vision_net).to(DEVICE)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)

    # 1. Lịch trình học thuật toán theo đường cong nghệ thuật Cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    # 2. Bộ Tăng Tốc Kép Dành Riêng Cho RTX GPUs (AMP)
    scaler = torch.amp.GradScaler('cuda') 

    best_loss = float('inf')

    print(" train")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
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
            model_path = os.path.join(BASE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'   (Loss: {best_loss:.4f})')
