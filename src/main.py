import cv2
import os

IMAGE_PATH = "./input/lab_image.jpg"
OUTPUT_PATH = "./output/lab_image.png"


def doc_anh():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Không tìm thấy ảnh:", IMAGE_PATH)
        return None
    return img


# -------------------------------------------------------
# Bài 2: Đọc và lưu ảnh
# -------------------------------------------------------
def bai_2():
    print("\n[Bài 2] Đọc và lưu ảnh")
    img = doc_anh()
    if img is None:
        return

    os.makedirs("./output", exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Đã lưu ảnh sang: {OUTPUT_PATH}")

    cv2.imshow("Bai 2 - Anh goc", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------------
# Bài 3: Chuyển đổi không gian màu
# -------------------------------------------------------
def bai_3():
    print("\n[Bài 3] Chuyển đổi không gian màu (Grayscale / HSV / LAB)")
    img = doc_anh()
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    cv2.imshow("Bai 3 - Original",  img)
    cv2.imshow("Bai 3 - Grayscale", gray)
    cv2.imshow("Bai 3 - HSV",       hsv)
    cv2.imshow("Bai 3 - LAB",       lab)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------------
# Bài 4: Crop và Resize ảnh
# -------------------------------------------------------
def bai_4():
    print("\n[Bài 4] Crop và Resize ảnh")
    img = doc_anh()
    if img is None:
        return

    h, w, c = img.shape
    print(f"  Height: {h}, Width: {w}, Channels: {c}")

    crop_img     = img[200:600, 300:900]
    resize_fixed = cv2.resize(img, (800, 600))
    resize_half  = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow("Bai 4 - Anh goc",     img)
    cv2.imshow("Bai 4 - Crop",        crop_img)
    cv2.imshow("Bai 4 - Resize 800x600", resize_fixed)
    cv2.imshow("Bai 4 - Resize 50%",  resize_half)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------------
# Bài 5: Vẽ hình và thêm chữ
# -------------------------------------------------------
def bai_5():
    print("\n[Bài 5] Vẽ hình và thêm chữ")
    img = doc_anh()
    if img is None:
        return

    # Vẽ đường thẳng
    cv2.line(img, (100, 100), (400, 100), (255, 0, 0), 3)

    # Vẽ hình chữ nhật
    cv2.rectangle(img, (200, 200), (400, 400), (0, 255, 0), 3)

    # Vẽ hình tròn
    cv2.circle(img, (300, 500), 60, (0, 0, 255), 3)

    # Thêm chữ
    cv2.putText(img, "Hello", (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Bai 5 - Ve hinh va chu", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------------
# Menu chính
# -------------------------------------------------------
def menu():
    options = {
        "1": ("Bài 2 - Đọc và lưu ảnh",                bai_2),
        "2": ("Bài 3 - Chuyển đổi không gian màu",      bai_3),
        "3": ("Bài 4 - Crop và Resize ảnh",              bai_4),
        "4": ("Bài 5 - Vẽ hình và thêm chữ",            bai_5),
        "0": ("Thoát",                                   None),
    }

    while True:
        print("\n========== MENU ==========")
        for key, (label, _) in options.items():
            print(f"  [{key}] {label}")
        print("==========================")

        choice = input("Nhập lựa chọn: ").strip()

        if choice == "0":
            print("Thoát chương trình.")
            break
        elif choice in options:
            _, func = options[choice]
            func()
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")


if __name__ == "__main__":
    menu()
