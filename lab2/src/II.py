import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. ĐỌC ẢNH
def load_image(path):
    img = cv2.imread("lab2/input/lab_image.jpg")
    if img is None:
        raise ValueError(f"Không đọc được ảnh tại: {path}")
    return img

# 2. LỌC TRUNG BÌNH (MEAN FILTER)
def mean_filter(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

# 3. LỌC GAUSSIAN
def gaussian_filter(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# 4. LÀM SẮC NÉT (SHARPEN)
def sharpen_filter(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)

# 5. CHUYỂN BGR → RGB 
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 6. HIỂN THỊ
def show_with_plt(images, titles):
    plt.figure(figsize=(12,6))

    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 7. CHẠY TEST
if __name__ == "__main__":
    path = "input/lab_image.jpg"

    try:
        # Đọc ảnh
        img = load_image(path)

        # Xử lý
        mean_img = mean_filter(img)
        gaussian_img = gaussian_filter(img)
        sharpen_img = sharpen_filter(img)

        # CHUYỂN RGB
        img_rgb = to_rgb(img)
        mean_rgb = to_rgb(mean_img)
        gaussian_rgb = to_rgb(gaussian_img)
        sharpen_rgb = to_rgb(sharpen_img)

        # Danh sách ảnh
        images = [img_rgb, mean_rgb, gaussian_rgb, sharpen_rgb]
        titles = ["Original", "Mean Filter", "Gaussian Filter", "Sharpen"]

        # Hiển thị
        show_with_plt(images, titles)

        print("Đã xử lý ảnh thành công!")

    except Exception as e:
        print("Lỗi:", e)