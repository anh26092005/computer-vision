# 🚀 Hệ thống Giám định Sản phẩm Công nghiệp bằng Multi-modal AI

---

## 📖 Giới thiệu

Đây là hệ thống **kiểm định lỗi sản phẩm công nghiệp** theo thời gian thực, được xây dựng trên nền tảng **Multi-modal AI** — kết hợp đồng thời thị giác máy tính (Computer Vision) và xử lý ngôn ngữ tự nhiên (NLP) trong một pipeline duy nhất.

Mô hình sử dụng kiến trúc **đa nhiệm vụ (Multi-task)**:

| Module | Vai trò |
|---|---|
| 🏗️ **ResNet18** | Backbone trích xuất đặc trưng ảnh |
| 🎭 **U-Net Decoder** | Phân vùng lỗi (Segmentation Mask) |
| 🔍 **Detection Head** | Định vị vị trí lỗi (Bounding Box) |
| 🧠 **DistilBERT + Cross-Attention** | Trả lời câu hỏi thị giác (Visual QA) |

---

## ✨ Tính năng chính

### 🎬 Giám sát Băng chuyền (Video Tracking Mode)
- Tải lên file video (MP4, AVI) quay từ băng chuyền sản xuất thông qua giao diện **Tkinter** trực quan.
- Hệ thống **tự động phát hiện và tracking** từng sản phẩm đi qua bằng thuật toán Computer Vision (Canny + Contour + Tracking).
- **Trực quan hóa (Video Overlay):** Tự động vẽ Bounding Box, 🔴 Mask đỏ phân vùng lỗi và Nhãn trạng thái (Lỗi/Đạt) trực tiếp lên từng khung hình video.
- ⏸️ Hỗ trợ **Tạm dừng / Tiếp tục** quá trình phân tích video bất cứ lúc nào.
- 📥 Xuất báo cáo tự động ra file **CSV** chứa chi tiết trạng thái kiểm định và tỷ lệ độ tin cậy của từng sản phẩm.

---

## 📂 Cấu trúc Thư mục

```plaintext
📦 project_root
 ┣ 📂 core/
 ┃  ┣ 📜 __init__.py
 ┃  ┣ 📜 config.py       # Cấu hình: DEVICE, VOCAB, đường dẫn dữ liệu
 ┃  ┣ 📜 dataset.py      # Dataset loader (DefectDataset)
 ┃  ┗ 📜 models.py       # Kiến trúc AI: VisionPipeline & MultiModalModel
 ┣ 📂 dataset/           # Dữ liệu huấn luyện
 ┣ 📜 app.py             # 🖥️ Giao diện Desktop (Tkinter)
 ┣ 📜 train.py           # Script huấn luyện mô hình
 ┣ 📜 evaluate.py        # Script đánh giá hiệu năng mô hình
 ┣ 📜 best_model.pth     # ⚙️ Trọng số AI đã huấn luyện
 ┣ 📜 requirements.txt   # Danh sách thư viện môi trường
 ┗ 📜 README.md
```

---

## ⚙️ Hướng dẫn Cài đặt & Chạy Ứng dụng

### Bước 1 — Lấy source code về máy

```bash
git clone https://github.com/anh26092005/computer-vision.git
cd "computer-vision/Multi-Modal_Defect_Inspection_and_Visual_QA_System"
```

### Bước 2 — Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo môi trường ảo tên "venv"
python -m venv venv

# Kích hoạt môi trường — Windows
venv\Scripts\activate

# Kích hoạt môi trường — macOS / Linux
source venv/bin/activate
```

### Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> 💡 **Lưu ý GPU:** Nếu bạn có NVIDIA GPU, hãy cài PyTorch với CUDA để tăng tốc inference. Xem hướng dẫn tại [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Bước 4 — Chạy ứng dụng

```bash
python app.py
```

Giao diện cửa sổ ứng dụng (Tkinter) sẽ được tự động mở lên.

---

## 🛠️ Huấn luyện & Đánh giá Mô hình (Dành cho Kỹ sư)

Nếu bạn có bộ dữ liệu mới hoặc muốn tinh chỉnh lại các chỉ số của AI, hệ thống đã cung cấp sẵn các script độc lập. Đảm bảo bạn đã đặt ảnh và file annotation vào thư mục `dataset/` trước khi chạy.

### 1. Build & Huấn luyện lại mô hình (Training)

Lệnh này sẽ khởi tạo quá trình huấn luyện từ đầu (hoặc fine-tune). Quá trình này sẽ sử dụng GPU (nếu có) để tối ưu hóa trọng số.

```bash
python train.py
```

> 💡 Sau khi training hoàn tất, trọng số mô hình tốt nhất sẽ tự động được lưu đè vào file `best_model.pth`.

### 2. Xuất báo cáo đánh giá (Evaluation)

Để kiểm tra độ chính xác của mô hình trên tập Test và xuất ra các chỉ số chuyên sâu (mIoU, Box IoU, Top-1 Accuracy, Dice Score) cùng Ma trận nhầm lẫn (Confusion Matrix):

```bash
python evaluate.py
```

---

## 📊 Các tác vụ AI được hỗ trợ

| Tác vụ | Phương pháp | Đầu ra |
|---|---|---|
| Phát hiện lỗi | Binary Classification | Có lỗi / Bình thường (Confidence Score) |
| Phân vùng lỗi | Semantic Segmentation | Mask nhị phân bám sát vật thể |
| Định vị lỗi | Anchor-free Detection | Bounding Box [x, y, w, h] |
| Phân loại chi tiết | Visual QA (VQA) | Nhãn tự động: scratch, bent, flip, color... |
