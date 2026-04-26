import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import tempfile
import shutil
import csv
import math
from datetime import datetime

from core.config import DEVICE, VOCAB, IDX2VOCAB, BASE_DIR
from core.models import VisionPipeline, MultiModalModel

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Kiểm định AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ════════════════════════════════════════════
   FONT IMPORTS
════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500;600&display=swap');

/* ════════════════════════════════════════════
   DESIGN TOKENS
════════════════════════════════════════════ */
:root {
    --ink:           #0f172a;   /* slate-900 — primary body text */
    --ink-soft:      #1e293b;   /* slate-800 — secondary headings */
    --ink-muted:     #334155;   /* slate-700 — descriptions, labels */
    --ink-faint:     #475569;   /* slate-600 — meta / mono labels (NOT grey-400!) */

    --paper:         #faf7f2;
    --paper-warm:    #f3ede3;
    --paper-tan:     #e8dfd1;
    --paper-deep:    #ddd3c5;

    --amber:         #b45309;   /* amber-700 — darker for contrast on light bg */
    --amber-light:   #d97706;   /* amber-600 */
    --amber-dim:     rgba(180,83,9,0.10);
    --amber-glow:    rgba(180,83,9,0.05);

    --pass:          #166534;   /* green-800 — high contrast on light */
    --pass-bg:       rgba(22,101,52,0.08);
    --pass-border:   rgba(22,101,52,0.30);

    --fail:          #991b1b;   /* red-800 — high contrast on light */
    --fail-bg:       rgba(153,27,27,0.07);
    --fail-border:   rgba(153,27,27,0.28);

    --border:        rgba(15,23,42,0.12);
    --border-strong: rgba(15,23,42,0.22);

    --mono:   'DM Mono', 'Courier New', monospace;
    --sans:   'Inter', system-ui, -apple-system, sans-serif;
}

/* ════════════════════════════════════════════
   BASE
════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--paper);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════════
   MASTHEAD
════════════════════════════════════════════ */
.masthead {
    background: var(--ink);
    position: relative;
    overflow: hidden;
    padding: 2.8rem 4rem 2.4rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
}
.masthead::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(250,247,242,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(250,247,242,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
}
.masthead::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, var(--amber-light), var(--amber), transparent);
}
.masthead-left { position: relative; z-index: 1; }
.masthead-eyebrow {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--amber-light);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.masthead-title {
    font-family: var(--sans);
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--paper);
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0;
}
.masthead-title em {
    font-style: normal;
    font-weight: 800;
    color: var(--amber-light);
}
.masthead-sub {
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 0.6rem;
    letter-spacing: 0.02em;
}
.masthead-right {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
}
.status-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(45,106,79,0.15);
    border: 1px solid var(--pass-border);
    color: #5aad82;
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
}
.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #5aad82;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}
.model-tag {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    color: #94a3b8;
    letter-spacing: 0.06em;
}

/* ════════════════════════════════════════════
   TABS
════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--paper-warm) !important;
    border-bottom: 1px solid var(--border-strong) !important;
    border-radius: 0 !important;
    padding: 0 4rem !important;
    justify-content: flex-start !important;
    display: flex !important;
    margin-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--ink-muted) !important;
    padding: 1rem 2rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--ink) !important;
    background: var(--paper-tan) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 3rem 4rem 4rem !important;
    background: var(--paper) !important;
}

/* ════════════════════════════════════════════
   SECTION HEADER
════════════════════════════════════════════ */
.section-rule {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.section-num {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 500;
    color: var(--amber);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    background: var(--amber-dim);
    border: 1px solid rgba(200,146,42,0.28);
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    white-space: nowrap;
    display: inline-block;
    width: fit-content;
}
.section-title {
    font-family: var(--sans) !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    margin: 0 !important;
    line-height: 1.3 !important;
    letter-spacing: -0.02em !important;
}
.section-title em {
    font-style: normal !important;
    font-weight: 600 !important;
    color: var(--ink-muted) !important;
}

/* ════════════════════════════════════════════
   UPLOAD ZONE
════════════════════════════════════════════ */
.hint-text {
    font-family: var(--sans);
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--ink-muted);
    margin-bottom: 1rem;
}
[data-testid="stFileUploader"] {
    background: var(--paper-warm) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--amber) !important;
    background: var(--amber-glow) !important;
}
[data-testid="stFileUploader"] section {
    padding: 2rem !important;
    text-align: center !important;
}

/* ════════════════════════════════════════════
   VERDICT CARD
════════════════════════════════════════════ */
.verdict-card {
    border-radius: 12px;
    padding: 1.8rem 2.4rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 3.5rem;
    flex-wrap: wrap;
    position: relative;
    overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
.verdict-card.pass {
    background: var(--pass-bg);
    border: 1px solid var(--pass-border);
}
.verdict-card.pass::before { background: var(--pass); }
.verdict-card.fail {
    background: var(--fail-bg);
    border: 1px solid var(--fail-border);
}
.verdict-card.fail::before { background: var(--fail); }

.verdict-stamp {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 150px;
}
.verdict-label {
    font-family: var(--sans);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.verdict-label.pass { color: var(--pass); }
.verdict-label.fail { color: var(--fail); }
.verdict-status {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.verdict-status.pass { color: var(--pass); }
.verdict-status.fail { color: var(--fail); }

.verdict-divider {
    width: 1px;
    height: 48px;
    background: var(--border);
    flex-shrink: 0;
}
.verdict-field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.verdict-key {
    font-family: var(--sans);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--ink-faint);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.verdict-val {
    font-family: var(--sans);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
    letter-spacing: -0.01em;
}
.verdict-val.pass { color: var(--pass); }
.verdict-val.fail { color: var(--fail); }
.conf-track {
    background: var(--border);
    border-radius: 2px;
    height: 3px;
    width: 120px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
}
.conf-fill.pass { background: var(--pass); }
.conf-fill.fail { background: var(--fail); }

/* ════════════════════════════════════════════
   IMAGE PANEL
════════════════════════════════════════════ */
.img-panel {
    background: var(--paper-warm);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.img-caption {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--paper-tan);
}
.img-caption-text {
    font-family: var(--sans);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--ink-soft);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.img-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}

/* ════════════════════════════════════════════
   BUTTONS & METRICS
════════════════════════════════════════════ */
.stButton > button[kind="primary"] {
    background: var(--ink) !important;
    border: none !important;
    color: var(--paper) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    padding: 0.75rem 2rem !important;
    border-radius: 4px !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--ink-soft) !important;
    box-shadow: 0 4px 20px rgba(26,22,20,0.2) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: var(--paper-warm) !important;
    border: 1.5px solid var(--ink) !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    padding: 0.75rem 2rem !important;
    border-radius: 4px !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    background: var(--ink) !important;
    color: var(--paper) !important;
}
[data-testid="metric-container"] {
    background: var(--paper-warm) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem 1.75rem !important;
    text-align: center !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--sans) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--ink-muted) !important; 
}
[data-testid="stMetricValue"] {
    font-family: var(--sans) !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    color: var(--ink) !important;
    line-height: 1.1 !important;
    letter-spacing: -0.03em !important;
}

/* ════════════════════════════════════════════
   PROGRESS & UTILS
════════════════════════════════════════════ */
.stProgress > div {
    background: var(--paper-tan) !important;
    border-radius: 2px !important;
    height: 3px !important;
}
.stProgress > div > div {
    background: var(--amber) !important;
    border-radius: 2px !important;
}
div[data-testid="stSuccess"] {
    background: var(--pass-bg) !important;
    border: 1.5px solid var(--pass-border) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--pass) !important;
}
div[data-testid="stInfo"] {
    background: var(--amber-glow) !important;
    border: 1.5px solid rgba(180,83,9,0.25) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--ink-soft) !important;
}
.stSpinner > div { border-color: var(--amber) !important; }

.prod-sep {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin: 2.5rem 0 1.75rem;
}
.prod-sep-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.prod-sep-badge {
    font-family: var(--sans);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--ink-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
    background: var(--paper-tan);
    border: 1px solid var(--border-strong);
    padding: 0.2rem 0.65rem;
    border-radius: 2px;
}

[data-testid="stImage"] img { border-radius: 0 !important; }

.frame-counter {
    font-family: var(--sans);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--ink-muted);
    letter-spacing: 0.03em;
}
.frame-counter span { color: var(--amber); font-weight: 700; }

.analysis-text {
    font-family: var(--sans);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--amber);
    letter-spacing: 0.03em;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--paper-warm); }
::-webkit-scrollbar-thumb { background: var(--paper-deep); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }
</style>

<div class="masthead">
    <div class="masthead-left">
        <div class="masthead-eyebrow">⬡ Hệ thống Kiểm định Công nghiệp</div>
        <h1 class="masthead-title">Giám định Sản phẩm<br><em>bằng Trí tuệ Nhân tạo</em></h1>
        <p class="masthead-sub">Phát hiện lỗi đa phương thức · Phân tích hình ảnh thời gian thực · Báo cáo tự động</p>
    </div>
    <div class="masthead-right">
        <div class="status-badge">
            <div class="status-dot"></div>
            Hệ thống hoạt động
        </div>
        <div class="model-tag">Vision AI Pipeline · MultiModal Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TRANSFORM & MODEL
# ─────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    vision_net = VisionPipeline().to(DEVICE)
    model = MultiModalModel(vocab_size=len(VOCAB), vision_pipeline=vision_net).to(DEVICE)
    model_path = os.path.join(BASE_DIR, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ─────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────
def process_image(img_array, model):
    pil_img = Image.fromarray(img_array)
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    questions = ["What type of defect is this?"]
    with torch.no_grad():
        mask_logits, defect_logits, bbox_preds, vqa_logits = model(img_tensor, questions)
        prob = torch.sigmoid(defect_logits).item()
        ans_idx = torch.argmax(vqa_logits, dim=-1).item()
        defect_type = IDX2VOCAB[ans_idx]
        cx, cy, w, h = bbox_preds[0].cpu().numpy()
        H_img, W_img = img_array.shape[:2]
        x_center, y_center = int(cx * W_img), int(cy * H_img)
        box_w, box_h = int(w * W_img), int(h * H_img)
        x1 = max(0, int(x_center - box_w / 2))
        y1 = max(0, int(y_center - box_h / 2))
        x2 = min(W_img, int(x_center + box_w / 2))
        y2 = min(H_img, int(y_center + box_h / 2))
        mask = (torch.sigmoid(mask_logits[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    return prob, defect_type, mask_resized, (x1, y1, x2, y2)

def render_verdict(prob, defect_type, has_defect, vqa_label):
    cls      = "fail" if has_defect else "pass"
    lbl      = "Phát hiện lỗi" if has_defect else "Đạt kiểm định"
    status   = "Có lỗi" if has_defect else "Sản phẩm tốt"
    
    if has_defect:
        conf_pct = prob * 100
    else:
        conf_pct = (1.0 - prob) * 100
        
    st.markdown(f"""
    <div class="verdict-card {cls}">
        <div class="verdict-stamp">
            <span class="verdict-label {cls}">{lbl}</span>
            <span class="verdict-status {cls}">{status}</span>
        </div>
        <div class="verdict-divider"></div>
        <div class="verdict-field">
            <span class="verdict-key">Loại lỗi</span>
            <span class="verdict-val {cls}">{defect_type.upper()}</span>
        </div>
        <div class="verdict-divider"></div>
        <div class="verdict-field">
            <span class="verdict-key">Độ tin cậy</span>
            <span class="verdict-val">{conf_pct:.1f}%</span>
            <div class="conf-track">
                <div class="conf-fill {cls}" style="width:{min(conf_pct,100):.1f}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_images(orig, result=None):
    if result is not None:
        _, c1, c2, _ = st.columns([1, 2, 2, 1], gap="medium")
        with c1:
            st.markdown('<div class="img-panel"><div class="img-caption"><span class="img-caption-text">Ảnh gốc</span><div class="img-dot" style="background:var(--border-strong)"></div></div>', unsafe_allow_html=True)
            st.image(orig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="img-panel"><div class="img-caption"><span class="img-caption-text">Kết quả phân tích AI</span><div class="img-dot" style="background:var(--fail)"></div></div>', unsafe_allow_html=True)
            st.image(result, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        _, c_mid, _ = st.columns([3, 2, 3])
        with c_mid:
            st.markdown('<div class="img-panel"><div class="img-caption"><span class="img-caption-text">Ảnh gốc</span><div class="img-dot" style="background:var(--pass)"></div></div>', unsafe_allow_html=True)
            st.image(orig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["  01 · Giám định Ảnh đơn  ", "  02 · Giám sát Băng chuyền  "])

# ══════════════════ TAB 1 ══════════════════
with tab1:
    st.markdown("""
    <div class="section-rule">
        <span class="section-num">Mô-đun 01</span>
        <h2 class="section-title">Giám định Chi tiết — <em>Ảnh đơn</em></h2>
    </div>
    <p class="hint-text">Tải lên một ảnh sản phẩm để phân tích lỗi tức thì.</p>
    """, unsafe_allow_html=True)

    uploaded_imgs = st.file_uploader(
        "Kéo thả ảnh vào đây hoặc nhấn để duyệt file",
        type=['jpg', 'jpeg', 'png'],
        help="Hỗ trợ: JPG · JPEG · PNG",
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    if uploaded_imgs:
        for idx, uploaded_img in enumerate(uploaded_imgs):
            if idx > 0:
                st.markdown("---")
            st.markdown(f"**Tệp:** `{uploaded_img.name}`")
            file_bytes   = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            orig_img     = cv2.imdecode(file_bytes, 1)
            orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            with st.spinner(f"Đang phân tích ảnh {uploaded_img.name}…"):
                prob, defect_type, mask, bbox = process_image(orig_img_rgb, model)

            result_img = orig_img_rgb.copy()
            if prob > 0.5 and defect_type != "good":
                # Giữ nguyên logic vẽ màng đỏ cho mọi loại lỗi (Bao gồm cả FLIP)
                overlay = result_img.copy()
                overlay[mask > 0] = (220, 60, 50)
                cv2.addWeighted(overlay, 0.45, result_img, 0.55, 0, result_img)
                
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (220, 60, 50), 2)
                has_defect, vqa_label = True, "Có"
            else:
                has_defect, vqa_label = False, "Không"
                defect_type = "good"
                result_img  = None

            render_verdict(prob, defect_type, has_defect, vqa_label)
            st.markdown("<br>", unsafe_allow_html=True)
            render_images(orig_img_rgb, result_img)

# ══════════════════ TAB 2 ══════════════════
with tab2:
    st.markdown("""
    <div class="section-rule">
        <span class="section-num">Mô-đun 02</span>
        <h2 class="section-title">Giám sát Băng chuyền — <em>Xử lý Video hàng loạt</em></h2>
    </div>
    <p class="hint-text">Tải lên video băng chuyền để quét và phân loại tự động từng sản phẩm.</p>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Tải lên file video",
        type=['mp4', 'avi'],
        help="Hỗ trợ: MP4 · AVI",
        label_visibility="collapsed"
    )

    if uploaded_video is not None:
        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
            run_btn = st.button("▶  Bắt đầu xử lý", type="primary", use_container_width=True)

        if run_btn:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            vid          = cv2.VideoCapture(tfile.name)
            total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = vid.get(cv2.CAP_PROP_FPS)

            st.markdown("<br>", unsafe_allow_html=True)
            prog_label   = st.empty()
            progress_bar = st.progress(0)
            status_text  = st.empty()

            report_dir     = tempfile.mkdtemp()
            csv_path       = os.path.join(report_dir, "report.csv")
            video_out_path = os.path.join(report_dir, "output.mp4")

            fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(video_out_path, fourcc, fps, (800, 600))

            report_data     = []
            total_products  = 0
            passed_products = 0
            failed_products = 0
            tracked_items   = {}
            next_id         = 1
            frame_idx       = 0

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                frame_idx += 1
                progress = min(1.0, frame_idx / total_frames) if total_frames > 0 else 1.0
                progress_bar.progress(progress)

                if frame_idx % 10 == 0:
                    prog_label.markdown(
                        f'<p class="frame-counter">Đang đọc video — Khung hình <span>{frame_idx}</span> / <span>{total_frames}</span></p>',
                        unsafe_allow_html=True
                    )

                frame     = cv2.resize(frame, (800, 600))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur    = cv2.GaussianBlur(gray, (7, 7), 0)
                edges   = cv2.Canny(blur, 50, 150)
                kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                dilated = cv2.dilate(edges, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                current_frame_objects = []
                for c in contours:
                    if cv2.contourArea(c) > 15000:
                        x, y, w, h = cv2.boundingRect(c)
                        if y < 140:
                            continue
                        pad = 15
                        x1 = max(0, x - pad);  y1 = max(0, y - pad)
                        x2 = min(800, x+w+pad); y2 = min(600, y+h+pad)
                        cx, cy = x + w/2, y + h/2
                        roi_img = rgb_frame[y1:y2, x1:x2]
                        if roi_img.shape[0] < 10 or roi_img.shape[1] < 10:
                            continue
                        current_frame_objects.append((cx, cy, x1, y1, x2, y2, roi_img))

                new_tracked_items = {}
                out_frame_rgb = rgb_frame.copy()

                for (cx, cy, x1, y1, x2, y2, roi_img) in current_frame_objects:
                    matched_id = None
                    min_dist   = 200
                    for obj_id, data in tracked_items.items():
                        old_cx, old_cy = data['ct']
                        dist = math.hypot(cx - old_cx, cy - old_cy)
                        if dist < min_dist:
                            min_dist = dist; matched_id = obj_id
                            break

                    if matched_id is not None:
                        obj_data = tracked_items.pop(matched_id)
                        obj_data['ct'] = (cx, cy); obj_data['missed_frames'] = 0
                    else:
                        matched_id = next_id; next_id += 1; total_products += 1
                        obj_data = {
                            'ct': (cx, cy),
                            'label': 'DANG XET AI...',
                            'color_bgr': (200, 200, 200),
                            'locked': False,
                            'missed_frames': 0,
                            'analyzed': False,
                            'mask': None,
                            'roi_size': None,
                        }

                    # ── Phân tích AI (chỉ một lần, khi sản phẩm vào vùng giữa) ──
                    if not obj_data['locked'] and (200 < cx < 600):
                        status_text.markdown(
                            f'<p class="analysis-text">→ Đang phân tích sản phẩm #{matched_id}…</p>',
                            unsafe_allow_html=True
                        )
                        prob, defect_type, mask, bbox = process_image(roi_img, model)
                        obj_data['analyzed'] = True
                        obj_data['prob']        = prob
                        obj_data['defect_type'] = defect_type
                        obj_data['roi_size']    = (roi_img.shape[0], roi_img.shape[1])

                        has_defect = prob > 0.5 and defect_type != "good"

                        if has_defect:
                            failed_products += 1
                            obj_data['label']     = f"Loi: {defect_type.upper()}"
                            obj_data['color_bgr'] = (255, 0, 0)   # đỏ — BGR
                            obj_data['mask']      = mask
                            report_data.append({
                                "ID": matched_id,
                                "Trang thai": f"Lỗi: {defect_type.upper()}",
                                "Xac suat": f"{prob*100:.2f}%",
                            })
                        else:
                            passed_products += 1
                            obj_data['label']     = "OK (GOOD)"
                            obj_data['color_bgr'] = (0, 255, 0)   # xanh — BGR
                            obj_data['mask']      = None
                            report_data.append({
                                "ID": matched_id,
                                "Trang thai": "Đạt",
                                "Xac suat": f"{prob*100:.2f}%",
                            })

                        obj_data['locked'] = True

                    # ── Vẽ overlay lên out_frame_rgb ──
                    label = obj_data.get('label', '')
                    if label.startswith('Loi'):
                        color_rgb = (220, 60, 50)    # đỏ — có lỗi
                    elif label == 'OK (GOOD)':
                        color_rgb = (50, 205, 80)    # xanh lá — sản phẩm tốt
                    else:
                        color_rgb = (220, 220, 220)  # trắng xám — đang xét AI

                    # Màng bán trong suốt (chỉ khi có lỗi)
                    if obj_data.get('mask') is not None:
                        curr_w, curr_h = x2 - x1, y2 - y1
                        m = cv2.resize(
                            obj_data['mask'].astype(np.uint8),
                            (curr_w, curr_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                        roi_out = out_frame_rgb[y1:y2, x1:x2].copy()
                        overlay = roi_out.copy()
                        overlay[m > 0] = (220, 60, 50)   # đỏ — RGB
                        cv2.addWeighted(overlay, 0.45, roi_out, 0.55, 0, roi_out)
                        out_frame_rgb[y1:y2, x1:x2] = roi_out

                    # Khung bao + nhãn (giống tkinter app)
                    cv2.rectangle(out_frame_rgb, (x1, y1), (x2, y2), color_rgb, 3)
                    label_text = f"ID {matched_id}: {obj_data['label']}"
                    cv2.putText(
                        out_frame_rgb, label_text,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_rgb, 2, cv2.LINE_AA
                    )

                    new_tracked_items[matched_id] = obj_data

                for obj_id, data in tracked_items.items():
                    data['missed_frames'] += 1
                    if data['missed_frames'] < 10:
                        new_tracked_items[obj_id] = data
                tracked_items = new_tracked_items

                out_vid.write(cv2.cvtColor(out_frame_rgb, cv2.COLOR_RGB2BGR))

            vid.release()
            out_vid.release()
            os.remove(tfile.name)

            # ── Ghi CSV ──
            with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=["ID", "Trang thai", "Xac suat"])
                writer.writeheader()
                writer.writerows(report_data)

            # ── Re-encode sang H.264 để trình duyệt phát được ──
            h264_path = os.path.join(report_dir, "output_h264.mp4")
            os.system(
                f'ffmpeg -y -i "{video_out_path}" '
                f'-vcodec libx264 -crf 23 -preset fast '
                f'-acodec aac "{h264_path}" -loglevel error'
            )
            final_video_path = h264_path if os.path.exists(h264_path) else video_out_path

            # ── ZIP (video H.264 + CSV) ──
            zip_path = os.path.join(
                tempfile.gettempdir(),
                f"HoSoKiemDinh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', report_dir)

            prog_label.empty(); progress_bar.empty(); status_text.empty()

            st.success("✓ Xử lý video hoàn tất. Hồ sơ kiểm định đã sẵn sàng.")
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics ──
            m1, m2, m3 = st.columns(3, gap="medium")
            m1.metric("Tổng sản phẩm", total_products)
            m2.metric("Đạt  ✓", passed_products)
            m3.metric("Lỗi  ✗", failed_products)

            st.markdown("<br>", unsafe_allow_html=True)

            # ════ XEM VIDEO TRỰC TIẾP TRÊN WEB ════
            st.markdown("""
            <div class="section-rule" style="margin-top:1rem;">
                <span class="section-num">Kết quả Video</span>
                <h2 class="section-title">Xem lại — <em>Video đã phân tích</em></h2>
            </div>
            """, unsafe_allow_html=True)

            with open(final_video_path, "rb") as vf:
                video_bytes = vf.read()

            _, col_vid, _ = st.columns([1, 3, 1])
            with col_vid:
                st.video(video_bytes)

            # ── Download ZIP ──
            st.markdown("<br>", unsafe_allow_html=True)
            _, col_dl, _ = st.columns([2, 1, 2])
            with col_dl:
                with open(zip_path, "rb") as fp:
                    st.download_button(
                        label="⬇  Tải xuống Hồ sơ Kiểm định (.ZIP)",
                        data=fp,
                        file_name=os.path.basename(zip_path),
                        mime="application/zip",
                        use_container_width=True,
                    )

            # ── Dọn dẹp ──
            try:
                shutil.rmtree(report_dir)
                os.remove(zip_path)
            except Exception:
                pass