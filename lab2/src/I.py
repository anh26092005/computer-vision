import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread("lab2/input/lab_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1.Thay đổi độ sáng
brightness_value = -100  # tăng sáng (giảm thì dùng -)
bright_img = np.clip(image.astype(int) + brightness_value, 0, 255).astype(np.uint8)

# 2.Thay đổi độ tương phản
contrast_value = 1.5  # >1 tăng, <1 giảm
contrast_img = np.clip(image * contrast_value, 0, 255).astype(np.uint8)

# 3. Ảnh âm bản
negative_img = 255 - image

# 4. Cắt ngưỡng (Binary)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
threshold_value = 128
_, thresh_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Hiển thị kết quả
titles = ['Original', 'Brightness', 'Contrast', 'Negative', 'Threshold']
images = [image, bright_img, contrast_img, negative_img, thresh_img]

plt.figure(figsize=(12,6))
for i in range(5):
    plt.subplot(2,3,i+1)
    if i == 4:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.show()