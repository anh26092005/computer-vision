import json

notebook_path = r'd:\Dev\Code\ComputerVision\computer-vision\lab4\src\lab4_phan3.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # We are looking for the display_results method in WaveletImageSearchEngine
        if 'def display_results(self, query_path, top_k=5):' in source:
            source = """class WaveletImageSearchEngine:
    \"\"\"
    Bài III.2: Ứng dụng tìm kiếm hình ảnh dựa trên hàm băm Wavelet
    
    - Xây dựng cơ sở dữ liệu (CSDL) các mã băm wavelet
    - Tìm kiếm ảnh tương tự theo khoảng cách Hamming
    \"\"\"

    def __init__(self, wavelet='db4', level=1, bits=64, threshold=0.35):
        self.wavelet = wavelet
        self.level = level
        self.bits = bits
        self.threshold = threshold
        self.database = {}   # {image_name: hash}
        self.image_arrays = {}  # {image_name: array} để hiển thị

    def add_image(self, image_path, image_name=None):
        \"\"\"Thêm một ảnh vào CSDL\"\"\"
        if image_name is None:
            image_name = os.path.basename(image_path)
        h = wavelet_hash(image_path, wavelet=self.wavelet,
                         level=self.level, bits=self.bits)
        self.database[image_name] = h
        self.image_arrays[image_name] = load_and_preprocess(image_path)
        print(f'  Đã thêm: {image_name} ({self.bits}-bit hash)')

    def build_database(self, image_paths):
        \"\"\"Xây dựng CSDL từ danh sách ảnh\"\"\"
        print(f'Đang xây dựng CSDL với {len(image_paths)} ảnh...')
        for path in image_paths:
            self.add_image(path)
        print(f'CSDL đã có {len(self.database)} ảnh.')

    def search(self, query_path, top_k=5):
        \"\"\"Tìm kiếm top-K ảnh tương tự nhất\"\"\"
        query_hash = wavelet_hash(query_path, wavelet=self.wavelet,
                                  level=self.level, bits=self.bits)
        query_array = load_and_preprocess(query_path)

        scored = []
        for name, db_hash in self.database.items():
            dist = hamming_distance(query_hash, db_hash)
            similarity = 1 - dist
            is_match = dist < self.threshold
            scored.append({'name': name, 'distance': dist,
                           'similarity': similarity, 'is_match': is_match})

        # Sắp xếp theo khoảng cách tăng dần (tương tự nhất)
        scored.sort(key=lambda x: x['distance'])
        return query_hash, query_array, scored[:top_k]

    def display_results(self, query_path, top_k=5):
        \"\"\"Tìm kiếm và hiển thị kết quả trực quan dạng lưới 3 ảnh 1 hàng\"\"\"
        query_hash, query_array, results = self.search(query_path, top_k)

        n_results = len(results)
        total_images = n_results + 1
        cols = 3
        rows = (total_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Ảnh truy vấn
        axes[0].imshow(query_array, cmap='gray')
        axes[0].set_title(f'ẢNH TRUY VẤN\\n{os.path.basename(query_path)}',
                          fontsize=11, fontweight='bold', color='blue')
        axes[0].axis('off')
        for spine in axes[0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Kết quả tìm kiếm
        for idx, res in enumerate(results):
            ax = axes[idx + 1]
            if res['name'] in self.image_arrays:
                ax.imshow(self.image_arrays[res['name']], cmap='gray')

            color = 'green' if res['is_match'] else 'red'
            status = '✓ TƯƠNG TỰ' if res['is_match'] else '✗ KHÁC BIỆT'
            ax.set_title(
                f'#{idx+1}: {res["name"]}\\n'
                f'Tương đồng: {res["similarity"]*100:.1f}%\\n'
                f'{status}',
                fontsize=11, color=color, fontweight='bold'
            )
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
                spine.set_visible(True)

        # Ẩn các ô trống nếu số ảnh không lấp đầy hàng cuối
        for idx in range(total_images, len(axes)):
            axes[idx].axis('off')
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Bài III.2: Kết quả tìm kiếm hình ảnh (Wavelet Hash - {self.wavelet})\\n'
            f'Top-{n_results} ảnh tương tự nhất trong CSDL',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # In bảng kết quả
        print(f'\\nKết quả tìm kiếm (ngưỡng={self.threshold}):')
        print(f'{"Hạng":>5} {"Tên ảnh":<25} {"Khoảng cách":>12} {"Tương đồng":>12} {"Kết quả":>12}')
        print('-' * 70)
        for idx, res in enumerate(results):
            status = 'Tương tự' if res['is_match'] else 'Khác biệt'
            print(f"{idx+1:>5} {res['name']:<25} {res['distance']:>12.4f} {res['similarity']*100:>11.1f}% {status:>12}")

print('Lớp WaveletImageSearchEngine đã được định nghĩa!')"""
            
            lines = source.splitlines()
            new_source = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines else [])
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated grid display in lab4_phan3.ipynb")
