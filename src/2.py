import os
from PIL import Image
import matplotlib.pyplot as plt

# Đường dẫn tuyệt đối dựa theo vị trí file này
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "..", "input", "lab_image.jpg")
output_dir = os.path.join(base_dir, "..", "output")
output_path = os.path.join(output_dir, "lab_image.png")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đọc ảnh
image = Image.open(input_path)

# Hiển thị ảnh
plt.imshow(image)
plt.title("Lab Image")
plt.axis("off")
plt.show()

# Lưu ảnh sang định dạng khác
image.save(output_path)
print(f"Đã lưu ảnh tại: {output_path}")