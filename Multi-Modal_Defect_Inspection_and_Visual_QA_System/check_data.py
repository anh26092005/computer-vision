from core.config import DATA_DIR
from core.dataset import DefectDataset

train_set = DefectDataset(DATA_DIR, split='train')
test_set = DefectDataset(DATA_DIR, split='test')

print("="*30)
print(f"DATA SUMMARY:")
print(f" - Train samples: {len(train_set)}")
print(f" - Test samples:  {len(test_set)}")
print(f" - Total:     {len(train_set) + len(test_set)}")
print("="*30)