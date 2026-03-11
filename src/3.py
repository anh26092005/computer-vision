import cv2
from matplotlib import pyplot as plt

# đọc ảnh
img = cv2.imread('../input/image3.jpg') 

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

# hiển thị grayscale
plt.subplot(2,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")

# hiển thị HSV
plt.subplot(2,2,3)
plt.imshow(hsv)
plt.title("HSV")

# hiển thị LAB
plt.subplot(2,2,4)
plt.imshow(lab)
plt.title("LAB")

plt.show()