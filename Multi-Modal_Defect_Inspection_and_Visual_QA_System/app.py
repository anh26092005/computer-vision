import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import math
import csv
import os
import subprocess
from datetime import datetime
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T
from core.config import BASE_DIR 
from core.models import VisionPipeline, MultiModalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform cho VIDEO
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform cho ẢNH ĐƠN 224x224
transform_image = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vocab = {"yes": 0, "no": 1, "scratch": 2, "bent": 3, "color": 4, "flip": 5, "good": 6}
idx2vocab = {v: k for k, v in vocab.items()}

# resnet18
vision_net = VisionPipeline().to(device)
model = MultiModalModel(vocab_size=len(vocab), vision_pipeline=vision_net).to(device)


model_path = os.path.join(BASE_DIR, 'best_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Đã load thành công model từ: {model_path}")
else:
    print(f"CẢNH BÁO: Không tìm thấy file {model_path}! AI sẽ đoán bừa.")
model.eval()
# ==========================================


class PhanMemKiemTraLoi:
    def __init__(self, window, window_title, model):
        self.window = window
        self.window.title(window_title)
        self.model = model
        self.vid = None
        self.is_running = False
        self.is_paused = False 
        
        self.tracked_items = {}
        self.next_id = 1
        self.report_data = []

        # --- CANVAS CHÍNH ---
        self.canvas = tk.Canvas(window, width=800, height=600, bg="black")
        self.canvas.pack()

        # --- NHÃN KẾT QUẢ ẢNH ĐƠN ---
        self.result_label = tk.Label(
            window, text="", font=('Arial', 11), bg='#1e1e1e', fg='white',
            justify=tk.LEFT, anchor='w', wraplength=780
        )
        self.result_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # --- THANH NÚT ---
        btn_frame = tk.Frame(window, bg='#2d2d2d')
        btn_frame.pack(fill=tk.X, pady=8)

        self.btn_file = tk.Button(
            btn_frame, text="🎬  Mở File Video", command=self.open_file,
            font=('Arial', 11, 'bold'), bg='#4a90d9', fg='white',
            relief='flat', padx=10, pady=6, cursor='hand2'
        )
        self.btn_file.pack(side=tk.LEFT, padx=12)

        self.btn_image = tk.Button(
            btn_frame, text="  Mở Ảnh", command=self.open_image,
            font=('Arial', 11, 'bold'), bg='#7b68ee', fg='white',
            relief='flat', padx=10, pady=6, cursor='hand2'
        )
        self.btn_image.pack(side=tk.LEFT, padx=12)

        self.btn_report = tk.Button(
            btn_frame, text="📂  Xem Báo Cáo", command=self.open_report_folder,
            font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
            relief='flat', padx=10, pady=6, cursor='hand2'
        )
        self.btn_report.pack(side=tk.LEFT, padx=12)

        self.btn_pause = tk.Button(
            btn_frame, text="⏸  Tạm Dừng", command=self.toggle_pause,
            font=('Arial', 11, 'bold'), bg='#e74c3c', fg='white',
            relief='flat', padx=10, pady=6, cursor='hand2'
        )
        self.btn_pause.pack(side=tk.RIGHT, padx=12)

        self.delay = 15
        self.update_video()
        self.window.mainloop()

    def open_report_folder(self):
        """Mở thư mục chứa các file báo cáo CSV."""
        path = os.path.abspath(os.path.dirname(__file__))
        os.startfile(path)

    def auto_save_report(self):
        """Tự động lưu file CSV khi video hoặc slideshow kết thúc."""
        if not self.report_data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"KetQua_KiemTra_AI_{timestamp}.csv"
        
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["ID Sản Phẩm/Ảnh", "Kết Quả", "Độ Tin Cậy (%)"])
            for row in self.report_data:
                writer.writerow([row["ID"], row["Ket_Qua"], row["Ty_Le"]])
        
        # Thông báo trên thanh trạng thái
        self.result_label.config(text=f"✅ ĐÃ LƯU BÁO CÁO: {filename}", fg='#2ecc71')
        
        # Thông báo lớn trên màn hình
        self.canvas.create_rectangle(200, 260, 600, 340, fill="#2c3e50", outline="white", width=2)
        self.canvas.create_text(400, 300, text=f"LƯU BÁO CÁO:\n{filename}", 
                                fill="#2ecc71", font=('Arial', 12, 'bold'), justify=tk.CENTER)
        print(f"--- Đã lưu báo cáo tự động: {filename}")

    # PHÂN TÍCH ẢNH
  
    def open_image(self):
        """Mở ảnh"""
        paths = filedialog.askopenfilenames(
            title="Chọn các ảnh sản phẩm",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not paths:
            return

        self.is_running = False
        self.is_paused = False
        self.report_data.clear()
        
        # Chạy slideshow bắt đầu từ ảnh đầu tiên
        self.run_slideshow(list(paths), 0)

    def run_slideshow(self, path_list, index):
        if index >= len(path_list):
            # Kết thúc slideshow - hiện thông báo và tự động lưu báo cáo
            self.canvas.delete('all')
            self.canvas.create_text(400, 280, text="ĐÃ PHÂN TÍCH HẾT ẢNH", fill="white", font=('Arial', 24, 'bold'))
            self.canvas.create_text(400, 320, text=f"Tổng: {len(path_list)} ảnh", fill="#aaa", font=('Arial', 14))
            self.auto_save_report()
            return

        path = path_list[index]
        pil_orig = Image.open(path).convert('RGB')
        display_img = pil_orig.resize((800, 600))
        display_arr = np.array(display_img)

        img_tensor = transform_image(pil_orig).unsqueeze(0).to(device)
        questions_list = ["Is there a defect?", "What type of defect is this?"]
        results = {}
        
        with torch.no_grad():
            for q in questions_list:
                mask_logits, defect_logits, bbox_preds, vqa_logits = self.model(img_tensor, [q])
                prob = torch.sigmoid(defect_logits).item()
                ans_idx = torch.argmax(vqa_logits, dim=-1).item()
                results[q] = {
                    'prob': prob, 'answer': idx2vocab[ans_idx],
                    'mask_logits': mask_logits, 'bbox': bbox_preds[0].cpu().numpy()
                }

        r = results["What type of defect is this?"]
        defect_prob = results["Is there a defect?"]['prob']
        has_defect = defect_prob > 0.5

        if has_defect:
            mask_np = torch.sigmoid(r['mask_logits'][0, 0]).cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            mask_display = cv2.resize(mask_bin, (800, 600), interpolation=cv2.INTER_NEAREST)
            overlay = display_arr.copy()
            overlay[mask_display > 0] = [255, 60, 60]
            display_arr = cv2.addWeighted(display_arr, 0.55, overlay, 0.45, 0)

            xc, yc, bw, bh = r['bbox']
            x1, y1 = int((xc - bw/2)*800), int((yc - bh/2)*600)
            x2, y2 = int((xc + bw/2)*800), int((yc + bh/2)*600)
            cv2.rectangle(display_arr, (x1, y1), (x2, y2), (255, 80, 80), 3)
            cv2.putText(display_arr, f"LOI: {r['answer'].upper()} ({defect_prob*100:.1f}%)", 
                        (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 80), 2)
        else:
            cv2.putText(display_arr, "OK - GOOD (Khong co loi)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 60), 3)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_arr))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Hiện số thứ tự ảnh
        self.canvas.create_text(700, 30, text=f"Ảnh {index+1}/{len(path_list)}", fill="white", font=('Arial', 12, 'bold'))

        q1_ans, q2_ans = results["Is there a defect?"], results["What type of defect is this?"]
        status = "🔴 CÓ LỖI" if has_defect else "🟢 BÌNH THƯỜNG"
        info = f"{status}  │  Q1: {q1_ans['answer'].upper()} ({q1_ans['prob']*100:.1f}%)  │  Q2: {q2_ans['answer'].upper()}  │  {os.path.basename(path)}"
        self.result_label.config(text=info, fg='#ff6b6b' if has_defect else '#2ecc71')

        self.report_data.append({
            "ID": f"ANH_{os.path.basename(path)}",
            "Ket_Qua": f"{'LOI: ' + q2_ans['answer'].upper() if has_defect else 'OK (GOOD)'}",
            "Ty_Le": f"{defect_prob*100:.1f}%"
        })

        # Chuyển ảnh tiếp theo sau 1.5 giây
        self.window.after(1500, lambda: self.run_slideshow(path_list, index + 1))

    def open_file(self):
        video_source = filedialog.askopenfilename(title="Chọn Video MP4", filetypes=[("Video Files", "*.mp4 *.avi")])
        if video_source:
            self.vid = cv2.VideoCapture(video_source)
            self.is_running = True
            self.is_paused = False 
            self.btn_pause.config(text="Tạm Dừng", bg='#ff6666')
            
            self.tracked_items.clear()
            self.report_data.clear() 
            self.next_id = 1
            
    def toggle_pause(self):
        if self.vid and self.vid.isOpened():
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.btn_pause.config(text="Tiếp Tục", bg='lightgreen')
            else:
                self.btn_pause.config(text="Tạm Dừng", bg='#ff6666')

    def update_video(self):
        if self.is_running and self.vid.isOpened() and not self.is_paused:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (800, 600)) 
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                
                # --- CROP ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(blur, 50, 150)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                dilated = cv2.dilate(edges, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                current_frame_objects = []
                for c in contours:
                    
                    if cv2.contourArea(c) > 10000: # Ngưỡng vừa phải
                        x, y, w, h = cv2.boundingRect(c)
                        if y < 140: continue 
                        
                        pad = 15
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(800, x + w + pad), min(600, y + h + pad)
                        cx, cy = x + w/2, y + h/2
                        
                        roi_img = rgb_frame[y1:y2, x1:x2].copy()
                        if roi_img.shape[0] < 10 or roi_img.shape[1] < 10: continue

                        roi_img = rgb_frame[y1:y2, x1:x2].copy()
                        if roi_img.shape[0] < 10 or roi_img.shape[1] < 10: continue
                            
                        current_frame_objects.append((cx, cy, x1, y1, x2, y2, roi_img))
                
                # ---  TRACKING ĐÁNH ID ---
                new_tracked_items = {}
                for (cx, cy, x1, y1, x2, y2, roi_img) in current_frame_objects:
                    matched_id = None
                    min_dist = 200 
                    
                    for obj_id, data in self.tracked_items.items():
                        old_cx, old_cy = data['ct']
                        dist = math.hypot(cx - old_cx, cy - old_cy)
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = obj_id
                            break 
                            
                    if matched_id is not None:
                        obj_data = self.tracked_items.pop(matched_id)
                        obj_data['ct'] = (cx, cy)
                        obj_data['missed_frames'] = 0
                    else:
                        matched_id = self.next_id
                        self.next_id += 1
                        obj_data = {
                            'ct': (cx, cy), 
                            'label': 'DANG XET AI...', 
                            'color': (200, 200, 200), 
                            'locked': False, 
                            'missed_frames': 0,
                            'mask': None  
                        }
                    
                  
                    if not obj_data['locked'] and (200 < cx < 600):
                        pil_img = Image.fromarray(roi_img)
                        img_tensor = transform(pil_img).unsqueeze(0).to(device)
                        questions = ["What type of defect is this?"]
                        
                        with torch.no_grad():
                            # SEGMENTATION 
                            mask_logits, defect_logits, _, vqa_logits = self.model(img_tensor, questions)
                            prob = torch.sigmoid(defect_logits).item()
                            ans_idx = torch.argmax(vqa_logits, dim=-1).item()
                            loai_loi = idx2vocab[ans_idx] 
                            
                            # Nâng ngưỡng lên 0.7 để tránh nhầm lẫn các sản phẩm sắc nét
                            if prob > 0.7 and loai_loi != "good":
                                obj_data['label'] = f"Loi: {loai_loi.upper()}"
                                obj_data['color'] = (255, 0, 0)
                                tyle_str = f"{prob*100:.1f}%"
                                
                                # LẤY TOẠ ĐỘ VÙNG BỊ LỖI
                                mask = torch.sigmoid(mask_logits[0, 0]).cpu().numpy() > 0.1                             
                                mask_resized = cv2.resize(mask.astype(np.uint8), (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                                obj_data['mask'] = mask_resized
                            else:
                                obj_data['label'] = "OK (GOOD)"
                                obj_data['color'] = (0, 255, 0)
                                tyle_str = "N/A"
                                obj_data['mask'] = None 
                                
                            obj_data['locked'] = True 
                            
                            self.report_data.append({
                                "ID": f"OC_VIT_SO_{matched_id}", 
                                "Ket_Qua": obj_data['label'], 
                                "Ty_Le": tyle_str
                            })
                    
                 
                    if obj_data.get('mask') is not None:
                        curr_w, curr_h = x2 - x1, y2 - y1
                        # Khớp giãn lớp Màng đỏ theo sự dịch chuyển của ốc
                        m = cv2.resize(obj_data['mask'], (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)
                        
                        overlay = rgb_frame[y1:y2, x1:x2].copy()
                        overlay[m > 0] = (255, 0, 0) # Sơn màu Đỏ
                      
                        rgb_frame[y1:y2, x1:x2] = cv2.addWeighted(rgb_frame[y1:y2, x1:x2], 0.5, overlay, 0.5, 0)

                    # VẼ KHUNG
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), obj_data['color'], 3)
                    cv2.putText(rgb_frame, f"ID {matched_id}: {obj_data['label']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obj_data['color'], 2)
                    
                    new_tracked_items[matched_id] = obj_data
                
                for obj_id, data in self.tracked_items.items():
                    data['missed_frames'] += 1
                    if data['missed_frames'] < 10:
                        new_tracked_items[obj_id] = data
                self.tracked_items = new_tracked_items
                
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.is_running = False
                self.canvas.create_text(400, 300, text="ĐÀ PHÂN TÍCH HẾT VIDEO", fill="white", font=('Arial', 24, 'bold'))
                # TỰ ĐỘNG LƯU BÁO CÁO KHI HẾT VIDEO
                self.auto_save_report()
        
        self.window.after(self.delay, self.update_video)

root = tk.Tk()
app = PhanMemKiemTraLoi(root, "Phần mềm kiểm tra sản phẩm lỗi", model)