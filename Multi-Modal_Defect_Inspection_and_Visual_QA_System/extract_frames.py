import cv2
import os

# Đường dẫn tới video của bạn
video_path = "conveyor_simulation (1).mp4"
output_dir = "extracted_data"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

print("🚀 Đang trích xuất ảnh từ video... Vui lòng chờ.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lấy 1 ảnh mỗi 20 frame để tránh trùng lặp quá nhiều
    if count % 20 == 0:
        file_name = f"{output_dir}/nut_frame_{saved_count:03d}.png"
        cv2.imwrite(file_name, frame)
        saved_count += 1
    
    count += 1

cap.release()
print(f"✅ Hoàn thành! Đã lưu {saved_count} ảnh vào thư mục: {output_dir}")
print("👉 Bây giờ bạn hãy vào thư mục này, chọn các ảnh đẹp và chép vào: dataset/metal_nut/train/good/")
print("👉 Sau đó chạy lệnh 'python train.py' để bắt đầu huấn luyện bổ sung.")
