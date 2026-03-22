import json
import os

files = ['lab4_phan2.ipynb', 'lab4_phan3.ipynb']

for f in files:
    if not os.path.exists(f): continue
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                new_line = line.replace('size=(128, 128)', 'size=(512, 512)')
                new_line = new_line.replace('cv2.resize(img, (128, 128))', 'cv2.resize(img, (512, 512))')
                new_source.append(new_line)
            cell['source'] = new_source
            
    with open(f, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=1)

print("Updated notebooks.")
