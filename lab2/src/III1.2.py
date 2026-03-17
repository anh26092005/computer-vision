import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread('../input/lab_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Phát hiện cạnh bằng Sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobel_x, sobel_y)

# 2. Phát hiện cạnh bằng Prewitt
kernelx = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

kernely = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])

prewitt_x = cv2.filter2D(gray, -1, kernelx)
prewitt_y = cv2.filter2D(gray, -1, kernely)

prewitt = prewitt_x + prewitt_y

# 3. Kernel tự thiết kế
custom_kernel = np.array([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
])

custom_result = cv2.filter2D(gray, -1, custom_kernel)

# Hiển thị kết quả
cv2.imshow("Original Image", img)
cv2.imshow("Sobel Edge", sobel)
cv2.imshow("Prewitt Edge", prewitt)
cv2.imshow("Custom Kernel Result", custom_result)

cv2.waitKey(0)
cv2.destroyAllWindows()