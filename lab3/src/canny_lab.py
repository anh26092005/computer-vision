"""
Bài Lab Thực Hành Canny Edge Detection (Gộp)
Bao gồm:
- Phần I:   Thực hiện thuật toán Canny bằng thư viện (OpenCV, skimage)
- Phần II:  Thay đổi tham số, Gaussian kernel, so sánh mặc định vs tùy chỉnh
- Phần III: Ảnh hưởng của nhiễu, tương phản thấp, ảnh nhiều chi tiết, so sánh các thuật toán
- Phần IV:  Kết hợp Canny với Contour, Hough Transform, Watershed Segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings

from skimage.feature import canny as skimage_canny
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

warnings.filterwarnings("ignore")

# Cấu hình Matplotlib và Đường dẫn
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(BASE_DIR, "input", "lab_image.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.titlesize":9,"axes.titleweight":"bold",
    "figure.facecolor":"#1e1e2e","axes.facecolor":"#1e1e2e",
    "text.color":"white","axes.labelcolor":"white","xtick.color":"white","ytick.color":"white",
})

# Hàm Tiện ích
def show(ax, img, title, cmap=None):
    if cmap is None: cmap = "gray" if img.ndim==2 else None
    ax.imshow(img, cmap=cmap); ax.set_title(title, pad=4); ax.axis("off")

def save_fig(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [Đã lưu] {p}")

def timer(fn, *a, **kw):
    t0 = time.perf_counter(); r = fn(*a, **kw)
    return r, (time.perf_counter()-t0)*1000

def psnr_ssim(ref, test):
    r=ref.astype(np.float64); t=test.astype(np.float64)
    return psnr(r,t,data_range=1.0), ssim(r,t,data_range=1.0)

def sobel_mag(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    m  = np.sqrt(sx**2 + sy**2)
    return np.uint8(np.clip(m/m.max()*255, 0, 255))

# Đọc ảnh
print("1. ĐỌC VÀ CHUẨN BỊ ẢNH")
img_bgr = cv2.imread(INPUT_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh tại {INPUT_PATH}")
img_bgr  = cv2.resize(img_bgr, (512,512), interpolation=cv2.INTER_AREA)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_f   = img_gray.astype(np.float64) / 255.0

# ---------------------------------------------------------
# PHẦN I: CANNY BẰNG CÁC THƯ VIỆN
# ---------------------------------------------------------
print("\n--- PHẦN I: THỰC HIỆN CANNY BẰNG THƯ VIỆN ---")
e_cv2, t_cv2 = timer(cv2.Canny, img_gray, 100, 200)
e_ski, t_ski = timer(skimage_canny, gray_f, sigma=2)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("I. So sánh Thư viện (OpenCV vs skimage)", color="cyan", fontsize=12, fontweight="bold")
show(axes[0], img_gray, "Grayscale gốc")
show(axes[1], e_cv2, f"cv2.Canny(100/200)\\n{np.sum(e_cv2>0):,} điểm | {t_cv2:.1f}ms")
show(axes[2], e_ski.astype(np.uint8)*255, f"skimage_canny(sigma=2)\\n{np.sum(e_ski):,} điểm | {t_ski:.1f}ms")
plt.tight_layout(); save_fig(fig, "full_I_compare_libs.png")

# ---------------------------------------------------------
# PHẦN II: THAY ĐỔI THAM SỐ VÀ KERNEL
# ---------------------------------------------------------
print("\n--- PHẦN II: THAY ĐỔI THAM SỐ VÀ GAUSSIAN KERNEL ---")

# OpenCV: low/high thresholds
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("II. Ảnh hưởng của Tham số OpenCV và skimage", color="cyan", fontsize=12, fontweight="bold")
for col, (lo, hi) in enumerate([(20,200), (50,200), (100,200), (50,300)]):
    e, t = timer(cv2.Canny, img_gray, lo, hi)
    show(axes[0][col], e, f"cv2 low={lo}, high={hi}\\n{np.sum(e>0):,} điểm | {t:.1f}ms")

# skimage: sigma
for col, sigma in enumerate([0.5, 1, 2, 4]):
    e, t = timer(skimage_canny, gray_f, sigma=sigma)
    show(axes[1][col], e.astype(np.uint8)*255, f"skimage sigma={sigma}\\n{np.sum(e):,} điểm | {t:.1f}ms")
plt.tight_layout(); save_fig(fig, "full_II_parameters.png")

# So sánh giá trị mặc định vs tùy chỉnh
e_cv2_def, _ = timer(cv2.Canny, img_gray, 100, 200)
e_cv2_mod, _ = timer(cv2.Canny, img_gray, 50, 100)
e_ski_def, _ = timer(skimage_canny, gray_f) # sigma=1.0
e_ski_mod, _ = timer(skimage_canny, gray_f, sigma=3.0)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("II. So sánh Mặc định vs Tùy chỉnh", color="cyan", fontweight="bold")
show(axes[0][0], e_cv2_def, "cv2 Mặc định (100,200)")
show(axes[0][1], e_cv2_mod, "cv2 Tùy chỉnh (50,100)")
show(axes[1][0], e_ski_def.astype(np.uint8)*255, "skimage Mặc định (sigma=1.0)")
show(axes[1][1], e_ski_mod.astype(np.uint8)*255, "skimage Tùy chỉnh (sigma=3.0)")
plt.tight_layout(); save_fig(fig, "full_II_defaults.png")

# ---------------------------------------------------------
# PHẦN III: ẢNH HƯỞNG CỦA NHIỄU, TƯƠNG PHẢN, CHI TIẾT
# ---------------------------------------------------------
print("\n--- PHẦN III: ẢNH HƯỞNG LOẠI ẢNH VÀ SO SÁNH THUẬT TOÁN ---")

# Tăng cường chi tiết (Sharpen)
sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_det = cv2.filter2D(img_gray, -1, sharpen)

# Tương phản thấp
img_low = np.uint8(img_gray.astype(np.float32)*0.4 + 100)
img_eq  = cv2.equalizeHist(img_low)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("III. Tương phản thấp & Cấu trúc chi tiết", color="cyan", fontsize=12, fontweight="bold")
show(axes[0][0], img_low, "Tương phản thấp")
show(axes[0][1], img_eq, "Sau EqualizeHist")
show(axes[0][2], img_det, "Nhiều chi tiết (Sharpen)")
show(axes[1][0], cv2.Canny(img_low, 30, 80), "Canny Tương phản thấp")
show(axes[1][1], cv2.Canny(img_eq, 100, 200), "Canny EqualizeHist")
show(axes[1][2], cv2.Canny(img_det, 150, 250), "Canny Ảnh nhiều chi tiết")
plt.tight_layout(); save_fig(fig, "full_III_contrast_details.png")

# Nhiễu & Thuật toán khác
g_noise = np.uint8(np.clip(random_noise(gray_f, mode='gaussian', var=0.01)*255, 0, 255))
s_mag   = sobel_mag(img_gray)
lap     = np.uint8(np.clip(np.abs(cv2.Laplacian(img_gray, cv2.CV_64F)), 0, 255))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("III. So sánh Nhiễu và Các thuật toán khác", color="cyan", fontsize=12, fontweight="bold")
show(axes[0][0], cv2.Canny(g_noise, 100, 200), "Canny trên Nhiễu (Trực tiếp)")
show(axes[0][1], cv2.Canny(cv2.GaussianBlur(g_noise,(5,5),0), 100, 200), "Blur 5x5 -> Canny trên Nhiễu")
show(axes[0][2], e_cv2_def, "Canny Chuẩn (100, 200)")
show(axes[1][0], s_mag, "Sobel Magnitude")
show(axes[1][1], np.uint8(np.clip(np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)), 0, 255)), "Sobel X")
show(axes[1][2], lap, "Laplacian (Đạo hàm bậc 2)")
plt.tight_layout(); save_fig(fig, "full_III_noise_algos.png")

# ---------------------------------------------------------
# PHẦN IV: KẾT HỢP VỚI KỸ THUẬT KHÁC (Hough, Watershed, Contour)
# ---------------------------------------------------------
print("\n--- PHẦN IV: KẾT HỢP KỸ THUẬT KHÁC ---")

# 1. Watershed Segmentation
edge = cv2.Canny(cv2.GaussianBlur(img_gray, (5,5), 0), 50, 150)
contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
markers = np.zeros(img_gray.shape, dtype=np.int32)
for i, c in enumerate(contours):
    cv2.drawContours(markers, [c], 0, i+1, -1)
img_ws = img_bgr.copy()
cv2.watershed(img_ws, markers)
img_ws[markers == -1] = [0, 0, 255]
img_ws_rgb = cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB)

# 2. Hough Lines
lines = cv2.HoughLinesP(edge, 1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=10)
img_lines = img_rgb.copy()
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]; cv2.line(img_lines, (x1,y1), (x2,y2), (255,50,50), 2)

# 3. Hough Circles
circles = cv2.HoughCircles(cv2.GaussianBlur(img_gray, (9,9), 2), cv2.HOUGH_GRADIENT,
                           dp=1.2, minDist=30, param1=100, param2=30, minRadius=10, maxRadius=100)
img_circles = img_rgb.copy()
if circles is not None:
    for c in np.uint16(np.around(circles))[0]:
        cv2.circle(img_circles, (c[0],c[1]), c[2], (50,255,50), 2)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle("IV. Kết hợp Kỹ thuật", color="cyan", fontsize=12, fontweight="bold")
show(axes[0], edge, "Canny Edge")
show(axes[1], img_ws_rgb, "Watershed Segmentation")
show(axes[2], img_lines, f"Hough Lines ({len(lines) if lines is not None else 0} đoạn)")
show(axes[3], img_circles, f"Hough Circles")
plt.tight_layout(); save_fig(fig, "full_IV_combined.png")

print("\n✅ Hoàn thành tất cả các phần! Kết quả đã được lưu.")
