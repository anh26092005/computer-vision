import cv2
from matplotlib import pyplot as plt

# đọc ảnh
img = cv2.imread('../input/lab_image.jpg')

# kiểm tra ảnh có đọc được không
if img is None:
    print("Không tìm thấy ảnh")
    exit()

# chuyển sang grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# chuyển sang HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# chuyển sang LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# hiển thị ảnh gốc
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

# grayscale
plt.subplot(2,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")

# HSV
plt.subplot(2,2,3)
plt.imshow(hsv)
plt.title("HSV")

# LAB
plt.subplot(2,2,4)
plt.imshow(lab)
plt.title("LAB")

plt.show()