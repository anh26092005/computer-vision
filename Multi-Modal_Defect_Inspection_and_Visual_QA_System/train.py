import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.config import DEVICE, DATA_DIR, BATCH_SIZE, EPOCHS, BASE_DIR
from core.dataset import DefectDataset
from core.models import VisionPipeline, MultiModalModel
from core.losses import compute_loss

def calculate_val_metrics(model, val_loader):
    """Hàm đánh giá nhanh mô hình trên tập Validation/Test sau mỗi Epoch"""
    model.eval()
    total_val_loss = 0.0
    total_dice_num = 0
    total_dice_den = 0
    correct_vqa = 0
    total_vqa = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(DEVICE)
            questions = batch['question']
            gt_masks = batch['mask'].to(DEVICE)
            gt_answers = batch['answer'].to(DEVICE)

            # Dự đoán
            with torch.amp.autocast('cuda'):
                mask_logits, defect_logits, bbox_preds, vqa_logits = model(images, questions)
                loss, _, _, _, _ = compute_loss(mask_logits, defect_logits, bbox_preds, vqa_logits, batch)

            total_val_loss += loss.item()

            # Tính Dice Score (Segmentation)
            pred_masks = (torch.sigmoid(mask_logits) > 0.5).float()
            intersection = (pred_masks * gt_masks).sum(dim=(2, 3))
            total_dice_num += (2.0 * intersection).sum().item()
            total_dice_den += (pred_masks.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))).sum().item()

            # Tính VQA Accuracy
            pred_answers = torch.argmax(vqa_logits, dim=1)
            correct_vqa += (pred_answers == gt_answers).sum().item()
            total_vqa += gt_answers.size(0)

    avg_loss = total_val_loss / len(val_loader)
    dice_score = total_dice_num / (total_dice_den + 1e-6)
    vqa_acc = correct_vqa / (total_vqa + 1e-6)
    
    return avg_loss, dice_score, vqa_acc

if __name__ == "__main__":
    # 1. Khởi tạo dữ liệu Train và Validation
    train_dataset = DefectDataset(DATA_DIR, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    val_dataset = DefectDataset(DATA_DIR, split='test')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Khởi tạo Model
    vision_net = VisionPipeline().to(DEVICE)
    model = MultiModalModel(vocab_size=len(train_dataset.vocab), vision_pipeline=vision_net).to(DEVICE)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') 

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_vqa_acc': []
    }

    best_loss = float('inf')

    print(f"--- Bat dau huan luyen mo hinh trong {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        # --- GIAI ĐOẠN TRAIN ---
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            questions = batch['question']

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                mask_logits, defect_logits, bbox_preds, vqa_logits = model(images, questions)
                loss, l_seg, l_dcls, l_dbox, l_vqa = compute_loss(mask_logits, defect_logits, bbox_preds, vqa_logits, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- GIAI ĐOẠN VALIDATION ---
        print(f"--- Dang danh gia Epoch {epoch+1}...")
        avg_val_loss, dice_score, vqa_acc = calculate_val_metrics(model, val_loader)

        # Lưu lại lịch sử
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(dice_score)
        history['val_vqa_acc'].append(vqa_acc)

        print(f"*** Epoch {epoch+1} Ket qua:")
        print(f"   - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   - Val Dice (Seg): {dice_score*100:.2f}% | Val VQA Acc: {vqa_acc*100:.2f}%")

        # Lưu model tốt nhất
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = os.path.join(BASE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'   [OK] Da luu model tot nhat moi voi Val Loss: {best_loss:.4f}')

    # 3. VẼ BIỂU ĐỒ SAU KHI TRAIN XONG
    epochs_range = range(1, EPOCHS + 1)

    # 3. LƯU DỮ LIỆU RA CSV
    import pandas as pd
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(BASE_DIR, 'train_history.csv'), index=False)
    
    # Tách ra 2 file CSV 
    df_history[['train_loss', 'val_loss']].to_csv(os.path.join(BASE_DIR, 'train_losses.csv'), index=False)
    df_history[['val_dice', 'val_vqa_acc']].to_csv(os.path.join(BASE_DIR, 'train_metrics.csv'), index=False)

    # 4. VẼ BIỂU ĐỒ SAU KHI TRAIN XONG
    epochs_range = range(1, EPOCHS + 1)

    # Biểu đồ 1: train_losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], 'go-', markersize=3, linewidth=1.5, label='loss')
    plt.plot(epochs_range, history['val_loss'], 'ro--', markersize=3, linewidth=1.5, label='val_loss')
    plt.title('train_losses', fontsize=14, fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    loss_plot_path = os.path.join(BASE_DIR, 'train_losses_chart.png')
    plt.savefig(loss_plot_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ Loss tại: {loss_plot_path}")

    # Biểu đồ 2: train_metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['val_dice'], 'go-', markersize=3, linewidth=1.5, label='dice_score')
    plt.plot(epochs_range, history['val_vqa_acc'], 'ro--', markersize=3, linewidth=1.5, label='val_vqa_accuracy')
    plt.title('train_metrics', fontsize=14, fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    metrics_plot_path = os.path.join(BASE_DIR, 'train_metrics_chart.png')
    plt.savefig(metrics_plot_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ Metrics tại: {metrics_plot_path}")

    print("🎉 Hoàn thành toàn bộ quá trình!")
