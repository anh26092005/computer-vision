import importlib.util
import os
import sys


def run_module(filepath):
    """Chạy một file Python theo đường dẫn tuyệt đối."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main():
    # Đường dẫn thư mục src (tương đối với vị trí main.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")

    menu = {
        "2": ("Bài 2 - Đọc và lưu ảnh bằng PIL", os.path.join(src_dir, "2.py")),
        "3": ("Bài 3 - Chuyển đổi không gian màu (BGR, Grayscale, HSV, LAB)", os.path.join(src_dir, "3.py")),
        "4": ("Bài 4 - Kích thước ảnh, Crop và Resize", os.path.join(src_dir, "4.py")),
        "5": ("Bài 5 - Vẽ hình và thêm văn bản lên ảnh", os.path.join(src_dir, "5.py")),
    }

    while True:
        print("\n" + "=" * 50)
        print("       MENU THỰC THI BÀI TẬP COMPUTER VISION")
        print("=" * 50)
        for key, (desc, _) in menu.items():
            print(f"  [{key}] {desc}")
        print("  [0] Thoát")
        print("=" * 50)

        choice = input("Nhập lựa chọn của bạn: ").strip()

        if choice == "0":
            print("Thoát chương trình.")
            break
        elif choice in menu:
            desc, filepath = menu[choice]
            print(f"\n>>> Đang chạy: {desc} <<<\n")
            try:
                run_module(filepath)
            except SystemExit:
                pass
            except Exception as e:
                print(f"Lỗi khi chạy bài {choice}: {e}")
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập lại.")


if __name__ == "__main__":
    main()
