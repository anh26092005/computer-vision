import os

base = os.path.join('dataset', 'metal_nut')

for split in ['train', 'test', 'ground_truth']:
    split_dir = os.path.join(base, split)
    if not os.path.exists(split_dir):
        continue
    print(f"\n=== {split.upper()} ===")
    total = 0
    for cat in sorted(os.listdir(split_dir)):
        cat_dir = os.path.join(split_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        count = len([f for f in os.listdir(cat_dir) if f.endswith('.png')])
        total += count
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {total}")
