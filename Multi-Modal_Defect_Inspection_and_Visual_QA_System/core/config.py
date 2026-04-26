import os
import torch

# Lấy đường dẫn gốc của thư mục project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
BATCH_SIZE = 8
EPOCHS = 60

# Từ điển VQA 
VOCAB = {"yes": 0, "no": 1, "scratch": 2, "bent": 3, "color": 4, "flip": 5, "good": 6}
IDX2VOCAB = {v: k for k, v in VOCAB.items()}

if DEVICE.type == 'cpu':
    print('WARNING: CPU environment detected. Training will be extremely slow!')
    print('TIP: Please switch to a GPU environment (e.g., Google Colab T4 GPU) for practical training times.')
if torch.cuda.is_available():
    print(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
