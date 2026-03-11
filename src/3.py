import cv2

# đọc ảnh
img = cv2.imread('./input/lab_image.jpg')

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

# hiển thị các ảnh bằng cv2
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("HSV", hsv)
cv2.imshow("LAB", lab)

cv2.waitKey(0)
cv2.destroyAllWindows()