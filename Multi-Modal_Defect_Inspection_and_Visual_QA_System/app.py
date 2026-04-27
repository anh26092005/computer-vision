import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import math
import csv
import os
from datetime import datetime
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T

# IMPORT THÊM BASE_DIR TỪ CONFIG
from core.config import BASE_DIR 
from core.models import VisionPipeline, MultiModalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vocab = {"yes": 0, "no": 1, "scratch": 2, "bent": 3, "color": 4, "flip": 5, "good": 6}
idx2vocab = {v: k for k, v in vocab.items()}

# ==========================================
# KHU VỰC SỬA LỖI LOAD MODEL
# ==========================================
vision_net = VisionPipeline().to(device)
model = MultiModalModel(vocab_size=len(vocab), vision_pipeline=vision_net).to(device)

# Sử dụng BASE_DIR giống như file evaluate.py
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
        
        self.canvas = tk.Canvas(window, width=800, height=600, bg="black")
        self.canvas.pack()
        
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X, pady=15)
        
        self.btn_file = tk.Button(btn_frame, text="Mở File Video", command=self.open_file, font=('Arial', 12, 'bold'), bg='lightblue', width=15)
        self.btn_file.pack(side=tk.LEFT, padx=30)
        self.btn_export = tk.Button(btn_frame, text="Xuất Báo Cáo (CSV)", command=self.export_report, font=('Arial', 12, 'bold'), bg='orange', width=20)
        self.btn_export.pack(side=tk.LEFT, padx=30)
        self.btn_pause = tk.Button(btn_frame, text="Tạm Dừng", command=self.toggle_pause, font=('Arial', 12, 'bold'), bg='#ff6666', width=15)
        self.btn_pause.pack(side=tk.RIGHT, padx=30)

        self.delay = 15
        self.update_video()
        self.window.mainloop()
        
    def export_report(self):
        if not self.report_data:
            return
        filename = f"BaoCao_KiemTra_OcVit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["ID Sản Phẩm", "Tình Trạng", "Tỷ Lệ Báo Lỗi"])
            for row in self.report_data:
                writer.writerow([row["ID"], row["Ket_Qua"], row["Ty_Le"]])
        self.canvas.create_text(400, 50, text=f"ĐÃ LƯU EXCEL THÀNH CÔNG!", fill="yellow", font=('Arial', 20, 'bold'))

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
                    
                    if cv2.contourArea(c) > 15000:
                        x, y, w, h = cv2.boundingRect(c)
                        if y < 140: continue 
                        
                        pad = 15
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(800, x + w + pad), min(600, y + h + pad)
                        cx, cy = x + w/2, y + h/2
                        
                        roi_img = rgb_frame[y1:y2, x1:x2]
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
                            
                            if prob > 0.5 and loai_loi != "good":
                                obj_data['label'] = f"Loi: {loai_loi.upper()}"
                                obj_data['color'] = (255, 0, 0)
                                tyle_str = f"{prob*100:.1f}%"
                                
                                # LẤY TOẠ ĐỘ VÙNG BỊ LỖI   
                                mask = torch.sigmoid(mask_logits[0, 0]).cpu().numpy() > 0.05                             
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
                self.canvas.create_text(400, 300, text="ĐÃ PHÂN TÍCH HẾT VIDEO", fill="white", font=('Arial', 24, 'bold'))
        
        self.window.after(self.delay, self.update_video)

root = tk.Tk()
app = PhanMemKiemTraLoi(root, "kiem tra sam pham loi", model)