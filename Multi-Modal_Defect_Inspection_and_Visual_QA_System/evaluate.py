import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision.ops import box_iou

from core.config import DEVICE, DATA_DIR, BATCH_SIZE, IDX2VOCAB, VOCAB, BASE_DIR
from core.dataset import DefectDataset
from core.models import VisionPipeline, MultiModalModel

def cxcywh_to_xyxy(boxes):
    """Chuyển đổi định dạng bounding box từ [center_x, center_y, w, h] sang [x1, y1, x2, y2]"""
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

if __name__ == "__main__":
    # 1. Khởi tạo dữ liệu và mô hình
    test_dataset = DefectDataset(DATA_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    vision_net = VisionPipeline().to(DEVICE)
    model = MultiModalModel(vocab_size=len(VOCAB), vision_pipeline=vision_net).to(DEVICE)

    model_path = os.path.join(BASE_DIR, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print('Da load best_model.pth thanh cong.')
    else:
        print('Khong tim thay best_model.pth. Dang dung mo hinh chua duoc huan luyen!')
    model.eval()

    # 2. Khởi tạo các biến lưu trữ kết quả để tính toán
    all_gt_defect, all_pred_defect = [], []
    all_gt_vqa, all_pred_vqa = [], []
    
    total_intersection = 0
    total_union = 0
    total_dice_num = 0
    total_dice_den = 0
    
    total_box_iou = 0
    valid_box_count = 0

    print("🚀 Đang chạy đánh giá mô hình trên toàn bộ tập Test. Vui lòng chờ...")
    
    # 3. Quét qua toàn bộ tập Test
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            questions = batch['question']

            # Lấy đáp án chuẩn (Ground Truth)
            gt_masks = batch['mask'].to(DEVICE)
            gt_bboxes = batch['bbox'].to(DEVICE)
            gt_has_defect = batch['has_defect'].to(DEVICE)
            gt_answers = batch['answer'].to(DEVICE)

            # Lấy dự đoán của mô hình (Predictions)
            mask_logits, defect_logits, bbox_preds, vqa_logits = model(images, questions)

            # --- NHÁNH 1: CLASSIFICATION (Có lỗi hay không?) ---
            pred_defect_probs = torch.sigmoid(defect_logits)
            pred_defect = (pred_defect_probs > 0.5).float()
            all_gt_defect.extend(gt_has_defect.cpu().numpy().flatten())
            all_pred_defect.extend(pred_defect.cpu().numpy().flatten())

            # --- NHÁNH 2: VQA (Đây là loại lỗi gì?) ---
            pred_answers = torch.argmax(vqa_logits, dim=1)
            all_gt_vqa.extend(gt_answers.cpu().numpy())
            all_pred_vqa.extend(pred_answers.cpu().numpy())

            # --- NHÁNH 3: SEGMENTATION (Màng đỏ phân vùng) ---
            pred_masks = (torch.sigmoid(mask_logits) > 0.5).float()
            # Tính Intersection và Union
            intersection = (pred_masks * gt_masks).sum(dim=(2, 3))
            union = pred_masks.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - intersection
            total_intersection += intersection.sum().item()
            total_union += union.sum().item()
            # Tính công thức Dice
            total_dice_num += (2.0 * intersection).sum().item()
            total_dice_den += (pred_masks.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))).sum().item()

            # --- NHÁNH 4: DETECTION (Bounding Box) ---
            pred_boxes_xyxy = cxcywh_to_xyxy(bbox_preds)
            gt_boxes_xyxy = cxcywh_to_xyxy(gt_bboxes)
            
            for i in range(len(gt_has_defect)):
                # Chỉ tính IoU cho bounding box nếu sản phẩm thực sự có lỗi
                if gt_has_defect[i].item() == 1.0: 
                    iou = box_iou(pred_boxes_xyxy[i:i+1], gt_boxes_xyxy[i:i+1])
                    total_box_iou += iou.item()
                    valid_box_count += 1

    # 4. TÍNH TOÁN & IN BÁO CÁO KẾT QUẢ TỔNG QUAN
    print("\n" + "="*55)
    print(" 📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH MULTI-TASK (EVALUATION REPORT) ")
    print("="*55)

    # Chỉ số Cảnh báo lỗi
    acc = accuracy_score(all_gt_defect, all_pred_defect)
    prec = precision_score(all_gt_defect, all_pred_defect, zero_division=0)
    rec = recall_score(all_gt_defect, all_pred_defect, zero_division=0)
    print(f"[1] PHÁT HIỆN SẢN PHẨM LỖI (Binary Classification):")
    print(f"    - Accuracy (Độ chính xác tổng): {acc*100:.2f}%")
    print(f"    - Precision (Báo lỗi không bị nhầm): {prec*100:.2f}%")
    print(f"    - Recall (Không bỏ lọt hàng lỗi): {rec*100:.2f}%")

    # Chỉ số Phân vùng
    mean_iou = total_intersection / (total_union + 1e-6)
    mean_dice = total_dice_num / (total_dice_den + 1e-6)
    print(f"\n[2] PHÂN VÙNG LỖI (Segmentation):")
    print(f"    - mIoU (Mức độ khớp vùng lỗi): {mean_iou*100:.2f}%")
    print(f"    - Dice Score: {mean_dice*100:.2f}%")

    # Chỉ số Bounding Box
    avg_box_iou = total_box_iou / (valid_box_count + 1e-6) if valid_box_count > 0 else 0
    print(f"\n[3] XÁC ĐỊNH VỊ TRÍ (Object Detection):")
    print(f"    - Average Box IoU: {avg_box_iou*100:.2f}%")

    # Chỉ số VQA
    vqa_acc = accuracy_score(all_gt_vqa, all_pred_vqa)
    print(f"\n[4] PHÂN LOẠI CHI TIẾT LỖI (VQA/Captioning):")
    print(f"    - Top-1 Accuracy: {vqa_acc*100:.2f}%")
    print("="*55)

    # 5. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_gt_vqa, all_pred_vqa)
    
    # Chỉ lấy các label có xuất hiện trong tập test để biểu đồ đẹp hơn
    unique_labels = sorted(list(set(all_gt_vqa) | set(all_pred_vqa)))
    target_names = [IDX2VOCAB[i].upper() for i in unique_labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, 
                annot_kws={"size": 12})
    plt.xlabel('AI Dự đoán (Predicted Label)', fontsize=12, fontweight='bold')
    plt.ylabel('Thực tế (Ground Truth Label)', fontsize=12, fontweight='bold')
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix) - Nhánh VQA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Lưu file ảnh và hiển thị
    plt.savefig('Multi-Modal_Defect_Inspection_and_Visual_QA_System/confusion_matrix_report.png', dpi=300)
    print("\n✅ Đã lưu biểu đồ phân tích thành file 'confusion_matrix_report.png'.")
    plt.show()