import cv2

# Đọc ảnh
img = cv2.imread("./input/lab_image.jpg")

# Kiểm tra đọc ảnh
if img is None:
    print("Không đọc được ảnh, kiểm tra đường dẫn")
    exit()
    
# Lấy kích thước ảnh
h, w, c = img.shape

print("Height:", h)
print("Width:", w)
print("Channels:", c)

# Crop vùng ảnh
crop_img = img[200:600, 300:900]

# Resize kích thước cố định
resize_fixed = cv2.resize(img, (800, 600))

# Resize theo tỉ lệ
resize_half = cv2.resize(img, None, fx=0.5, fy=0.5)

# Hiển thị
cv2.imshow("Anh goc", img)
cv2.imshow("Crop", crop_img)
cv2.imshow("Resize fixed", resize_fixed)
cv2.imshow("Resize 50%", resize_half)

cv2.waitKey(0)
cv2.destroyAllWindows()