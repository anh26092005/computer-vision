from PIL import Image
import cv2
# Đọc ảnh
image = Image.open("./input./lab_image.jpg")
# Hiển thị ảnh
image.show()
image.save("anh_lab.png")