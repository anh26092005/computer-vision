import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from core.config import VOCAB

class DefectDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # Define some basic VQA questions
        self.vocab = VOCAB
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
