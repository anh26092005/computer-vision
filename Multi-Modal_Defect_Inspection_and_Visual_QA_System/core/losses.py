import torch
import torch.nn as nn
from core.config import DEVICE

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
    loss_seg_bce = seg_criterion(mask_logits, batch['mask'].to(DEVICE))
    loss_seg_dice = dice_loss(mask_logits, batch['mask'].to(DEVICE))
    loss_seg = loss_seg_bce + loss_seg_dice

    # 2. Detection Loss
    gt_has_defect = batch['has_defect'].to(DEVICE)
    gt_bbox = batch['bbox'].to(DEVICE)

    loss_det_cls = det_cls_criterion(defect_logits, gt_has_defect)

    # Box loss only computed if there is a defect in ground truth
    loss_box = det_box_criterion(bbox_preds, gt_bbox).sum(dim=1)
    loss_det_box = (loss_box * gt_has_defect.view(-1)).mean() # Average over batch

    # 3. VQA Loss
    loss_vqa = vqa_criterion(vqa_logits, batch['answer'].to(DEVICE))

    total_loss = (lambda_seg * loss_seg) + (lambda_det_cls * loss_det_cls) + (lambda_det_box * loss_det_box) + (lambda_vqa * loss_vqa)

    return total_loss, loss_seg, loss_det_cls, loss_det_box, loss_vqa
