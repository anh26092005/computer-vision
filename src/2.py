from PIL import Image
import cv2

# Đọc ảnh
image = Image.open("./input/lab_image.jpg")

# Hiển thị ảnh
display(image)

# Lưu ảnh sang định dạng khác
image.save("./output/lab_image.png")