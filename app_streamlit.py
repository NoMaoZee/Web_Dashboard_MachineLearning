
import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import zipfile

# Import modul inference
import inference as inf

# ========================================
# KONFIGURASI
# ========================================

# Dapatkan direktori saat ini secara dinamis (untuk Windows/Linux/Mac)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models/cnn_classic_lora_best.h5")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
EDA_DIR = os.path. join(BASE_DIR, "eda")
XAI_DIR = os.path.join(BASE_DIR, "xai")
TEST_RESULTS_DIR = os.path.join(BASE_DIR, "test_results")

CLASS_NAMES = {0: "INDOLOGO", 1: "INTERLOGO", 2: "NOHALAL"}
IMG_SIZE = (128, 128)
LAST_CONV_LAYER = "conv3_1"

# ========================================
# LOAD MODEL (Cached)
# ========================================

@st.cache_resource
def load_model():
    """Load model CNN_Klasik_LoRA (cached)."""
    return inf.load_cnn_lora_model(MODEL_PATH)

model = load_model()

# ========================================
# PREMIUM CSS DESIGN (Hitam & Ungu Tua - Halal Indonesia Theme)
# ========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Sidebar - Gradient Ungu Tua & Hitam */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0033 0%, #2d0052 50%, #000000 100%);
        border-right: 2px solid #6a0dad;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff ! important;
    }
    
    [data-testid="stSidebar"] . css-1d391kg, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(90deg, #9d4edd 0%, #c77dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 28px ! important;
        margin-bottom:  1rem;
        text-align: center;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 600;
    }
    
    h1 {
        background: linear-gradient(90deg, #9d4edd 0%, #c77dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color:  transparent;
        font-size:  42px;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #c77dff;
        font-size: 28px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #6a0dad;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #e0aaff;
        font-size:  22px;
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #9d4edd 0%, #c77dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color:  transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8b8 !important;
        font-weight: 500;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background:  rgba(157, 78, 221, 0.1);
        border: 1px solid #6a0dad;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow:  0 8px 25px rgba(157, 78, 221, 0.4);
        border-color: #9d4edd;
    }
    
    /* Buttons */
    . stButton>button {
        background:  linear-gradient(135deg, #6a0dad 0%, #9d4edd 100%);
        color: white;
        border-radius: 10px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 16px;
        border: none;
        box-shadow: 0 4px 15px rgba(106, 13, 173, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #9d4edd 0%, #c77dff 100%);
        transform: translateY(-3px);
        box-shadow:  0 6px 20px rgba(157, 78, 221, 0.5);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(157, 78, 221, 0.05);
        border: 2px dashed #6a0dad;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Tabs */
    . stTabs [data-baseweb="tab-list"] {
        background: rgba(26, 0, 51, 0.8);
        border-radius: 10px;
        padding: 0.5rem;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #b8b8b8;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(157, 78, 221, 0.2);
        color: #c77dff;
    }
    
    . stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a0dad 0%, #9d4edd 100%);
        color: white ! important;
    }
    
    /* Info/Success/Warning boxes */
    . stAlert {
        border-radius: 10px;
        border-left: 4px solid #9d4edd;
        background: rgba(157, 78, 221, 0.1);
        color: #ffffff;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #6a0dad;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #6a0dad 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Radio buttons */
    [data-testid="stSidebar"] .row-widget. stRadio > div {
        background: rgba(157, 78, 221, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .row-widget.stRadio > div[role="radiogroup"] > label {
        background: transparent;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin:  0.3rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .row-widget. stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(157, 78, 221, 0.3);
    }
    
    /* Spinner */
    . stSpinner > div {
        border-top-color: #9d4edd ! important;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #e0e0e0;
    }
    
    /* Image captions */
    .image-caption {
        color: #b8b8b8 !important;
        font-size: 14px;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Footer */
    footer {
        color: #888;
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid #6a0dad;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================

st.sidebar.markdown('<h1><i class="fa-solid fa-certificate"></i> Halal AI Detection</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown('<p style="text-align: center; font-size: 16px; font-weight: 600;">CNN Klasik LoRA</p>', unsafe_allow_html=True)
st.sidebar.info("Best Deep Learning Model for Halal Logo Classification")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Halal AI Detection"],
    format_func=lambda x: f"{'📊' if x == 'Dashboard' else '🔍'} {x}"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size: 14px; font-weight: 600; color: #c77dff;">QUICK STATS</p>', unsafe_allow_html=True)

# Load test summary untuk quick stats
test_summary = inf.load_test_summary(os.path.join(TEST_RESULTS_DIR, "test_summary.json"))
if test_summary:
    st. sidebar.metric("Test Accuracy", f"{test_summary['test_accuracy']*100:.2f}%")
    st.sidebar.metric("F1-Score", f"{test_summary['f1_weighted']:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size: 14px; font-weight: 600; color: #c77dff;">DEVELOPED BY</p>', unsafe_allow_html=True)
st.sidebar.markdown("**Zeedan Mustami Argani**")
st.sidebar.markdown("University of Muhammadiyah Malang")

# ========================================
# PAGE 1: DASHBOARD
# ========================================

if page == "Dashboard": 
    st.markdown('<h1><i class="fa-solid fa-chart-line"></i> Dashboard - CNN Klasik LoRA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 18px; color: #b8b8b8; margin-bottom: 2rem;">Model Performance & Analysis</p>', unsafe_allow_html=True)
    
    if test_summary:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{test_summary['test_accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{test_summary['precision_weighted']:.4f}")
        with col3:
            st. metric("Recall", f"{test_summary['recall_weighted']:.4f}")
        with col4:
            st.metric("F1-Score", f"{test_summary['f1_weighted']:.4f}")
        
        st.markdown("---")
        
        # Training curves & Confusion matrix
        st.markdown('<h2><i class="fa-solid fa-chart-area"></i> Training Performance</h2>', unsafe_allow_html=True)
        col1, col2 = st. columns(2)
        
        with col1:
            training_curves_path = os.path.join(PLOTS_DIR, "training_curves. png")
            if os.path.exists(training_curves_path):
                st. image(training_curves_path, caption="Training & Validation Curves", use_container_width=True)
        
        with col2:
            cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix (Test Set)", use_container_width=True)
        
        # Per-class metrics
        st.markdown("---")
        st.markdown('<h2><i class="fa-solid fa-bars-progress"></i> Per-Class Metrics</h2>', unsafe_allow_html=True)
        
        per_class_img = os.path.join(PLOTS_DIR, "per_class_metrics.png")
        if os.path.exists(per_class_img):
            st.image(per_class_img, use_container_width=True)
        
        # Classification report
        st.markdown("---")
        st.markdown('<h2><i class="fa-solid fa-file-lines"></i> Classification Report</h2>', unsafe_allow_html=True)
        
        cr_path = os.path.join(HISTORY_DIR, "cnn_lora_classification_report.txt")
        if os.path. exists(cr_path):
            with open(cr_path, 'r') as f:
                st.code(f.read(), language="text")
        
        # XAI Gallery
        st.markdown("---")
        st.markdown('<h2><i class="fa-solid fa-eye"></i> XAI Gallery - Grad-CAM Examples</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            xai_indo = os.path.join(XAI_DIR, "gradcam_INDOLOGO_sample.png")
            if os.path.exists(xai_indo):
                st.image(xai_indo, caption="INDOLOGO", use_container_width=True)
        
        with col2:
            xai_inter = os.path.join(XAI_DIR, "gradcam_INTERLOGO_sample. png")
            if os.path.exists(xai_inter):
                st.image(xai_inter, caption="INTERLOGO", use_container_width=True)
        
        with col3:
            xai_no = os.path.join(XAI_DIR, "gradcam_NOHALAL_sample. png")
            if os.path.exists(xai_no):
                st.image(xai_no, caption="NOHALAL", use_container_width=True)
        
        # Random grid
        xai_grid = os.path.join(XAI_DIR, "xai_random_grid.png")
        if os.path.exists(xai_grid):
            st.image(xai_grid, caption="Grad-CAM Random Grid (Test Set)", use_container_width=True)
    
    else:
        st.warning("Test summary tidak ditemukan. Pastikan training sudah selesai.")

# ========================================
# PAGE 2: HALAL AI DETECTION
# ========================================

elif page == "Halal AI Detection":
    st.markdown('<h1><i class="fa-solid fa-microscope"></i> Halal AI Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 18px; color:  #b8b8b8; margin-bottom: 2rem;">Upload gambar untuk klasifikasi logo halal</p>', unsafe_allow_html=True)
    
    # Tabs untuk 3 mode input
    tab1, tab2, tab3 = st.tabs([
        "Real-Time Camera",
        "Upload Single Image", 
        "Upload Multiple (ZIP)"
    ])
    
    # ========================================
    # TAB 1: REAL-TIME CAMERA
    # ========================================
    
    with tab1:
        st.markdown('<h3><i class="fa-solid fa-camera"></i> Capture gambar dari webcam</h3>', unsafe_allow_html=True)
        
        camera_image = st.camera_input("Ambil foto produk:")
        
        if camera_image is not None:
            pil_image = Image.open(camera_image)
            
            with st.spinner("Memproses gambar..."):
                # Step 1: EDA
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-chart-pie"></i> Step 1: EDA Gambar</h3>', unsafe_allow_html=True)
                eda_data = inf.compute_image_eda(pil_image)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Width", f"{eda_data['width']}px")
                col2.metric("Height", f"{eda_data['height']}px")
                col3.metric("Brightness", f"{eda_data['brightness']}")
                col4.metric("Format", eda_data['format'])
                
                # Step 2: Prediksi
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-bullseye"></i> Step 2: Prediksi</h3>', unsafe_allow_html=True)
                
                pred_label, pred_conf, preds, heatmap, overlay = inf.predict_with_gradcam(
                    model, pil_image, CLASS_NAMES, last_conv_layer=LAST_CONV_LAYER, target_size=IMG_SIZE
                )
                
                st.success(f"**Prediksi:  {pred_label}** | Confidence: **{pred_conf:.2%}**")
                
                # Bar chart probabilitas
                prob_df = pd.DataFrame({
                    "Class": list(CLASS_NAMES.values()),
                    "Probability": preds
                })
                
                fig, ax = plt.subplots(figsize=(8, 3), facecolor='#1a1a2e')
                ax.set_facecolor('#1a1a2e')
                colors = ['#9d4edd' if c == pred_label else '#4a4a4a' for c in prob_df["Class"]]
                bars = ax.barh(prob_df["Class"], prob_df["Probability"], color=colors)
                ax.set_xlabel("Probability", fontweight='bold', color='white')
                ax. set_title("Class Probabilities", fontweight='bold', color='white')
                ax.set_xlim(0, 1)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top']. set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                plt.close()
                
                # Step 3: XAI (Grad-CAM)
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-layer-group"></i> Step 3: XAI (Grad-CAM)</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(pil_image, caption="Original", use_container_width=True)
                
                with col2:
                    fig_heat, ax_heat = plt.subplots(facecolor='#1a1a2e')
                    ax_heat.imshow(heatmap, cmap='jet')
                    ax_heat.axis('off')
                    st. pyplot(fig_heat)
                    plt.close()
                    st.markdown('<p class="image-caption">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
                
                with col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)
                
                # Step 4: Interpretasi
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-comments"></i> Step 4: Interpretasi XAI</h3>', unsafe_allow_html=True)
                
                xai_text = inf.generate_xai_text(pred_label, pred_conf)
                st.info(xai_text)
    
    # ========================================
    # TAB 2: UPLOAD SINGLE IMAGE
    # ========================================
    
    with tab2:
        st.markdown('<h3><i class="fa-solid fa-upload"></i> Upload satu gambar untuk klasifikasi</h3>', unsafe_allow_html=True)
        
        uploaded_file = st. file_uploader("Pilih gambar (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None: 
            pil_image = Image.open(uploaded_file)
            
            with st.spinner("Memproses gambar..."):
                # Step 1: EDA
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-chart-pie"></i> Step 1: EDA Gambar</h3>', unsafe_allow_html=True)
                eda_data = inf.compute_image_eda(pil_image)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Width", f"{eda_data['width']}px")
                col2.metric("Height", f"{eda_data['height']}px")
                col3.metric("Brightness", f"{eda_data['brightness']}")
                col4.metric("Format", eda_data['format'])
                
                # Step 2: Prediksi
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-bullseye"></i> Step 2: Prediksi</h3>', unsafe_allow_html=True)
                
                pred_label, pred_conf, preds, heatmap, overlay = inf.predict_with_gradcam(
                    model, pil_image, CLASS_NAMES, last_conv_layer=LAST_CONV_LAYER, target_size=IMG_SIZE
                )
                
                st.success(f"**Prediksi: {pred_label}** | Confidence: **{pred_conf:.2%}**")
                
                # Bar chart
                prob_df = pd.DataFrame({
                    "Class": list(CLASS_NAMES.values()),
                    "Probability": preds
                })
                
                fig, ax = plt.subplots(figsize=(8, 3), facecolor='#1a1a2e')
                ax.set_facecolor('#1a1a2e')
                colors = ['#9d4edd' if c == pred_label else '#4a4a4a' for c in prob_df["Class"]]
                ax.barh(prob_df["Class"], prob_df["Probability"], color=colors)
                ax.set_xlabel("Probability", fontweight='bold', color='white')
                ax.set_title("Class Probabilities", fontweight='bold', color='white')
                ax.set_xlim(0, 1)
                ax.tick_params(colors='white')
                ax.spines['bottom']. set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax. spines['right'].set_visible(False)
                st.pyplot(fig)
                plt.close()
                
                # Step 3: XAI
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-layer-group"></i> Step 3: XAI (Grad-CAM)</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st. image(pil_image, caption="Original", use_container_width=True)
                
                with col2:
                    fig_heat, ax_heat = plt. subplots(facecolor='#1a1a2e')
                    ax_heat.imshow(heatmap, cmap='jet')
                    ax_heat.axis('off')
                    st.pyplot(fig_heat)
                    plt.close()
                    st.markdown('<p class="image-caption">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
                
                with col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)
                
                # Step 4: Interpretasi
                st. markdown("---")
                st. markdown('<h3><i class="fa-solid fa-comments"></i> Step 4: Interpretasi XAI</h3>', unsafe_allow_html=True)
                
                xai_text = inf.generate_xai_text(pred_label, pred_conf)
                st.info(xai_text)
    
    # ========================================
    # TAB 3: UPLOAD MULTIPLE (ZIP)
    # ========================================
    
    with tab3:
        st.markdown('<h3><i class="fa-solid fa-file-zipper"></i> Upload file ZIP berisi banyak gambar</h3>', unsafe_allow_html=True)
        
        uploaded_zip = st.file_uploader("Upload ZIP file:", type=["zip"])
        
        if uploaded_zip is not None:
            images = []
            image_names = []
            
            with zipfile.ZipFile(uploaded_zip, 'r') as z:
                for fname in z.namelist():
                    if fname.lower().endswith(('.jpg', '.png', '. jpeg')):
                        with z.open(fname) as f:
                            images.append(Image.open(BytesIO(f.read())))
                            image_names.append(os.path.basename(fname))
            
            if len(images) > 0:
                st.success(f"{len(images)} gambar berhasil di-upload!")
                
                with st.spinner("Memproses batch... "):
                    results = []
                    for pil_img in images:
                        pred_label, pred_conf, preds = inf.predict_image(model, pil_img, CLASS_NAMES, target_size=IMG_SIZE)
                        results.append({
                            "pred_label": pred_label,
                            "confidence": pred_conf,
                            "probs": preds
                        })
                    
                    df_results = pd.DataFrame(results)
                    df_results["image_name"] = image_names
                
                # Tabel hasil
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-table"></i> Hasil Prediksi Batch</h3>', unsafe_allow_html=True)
                st.dataframe(df_results[["image_name", "pred_label", "confidence"]], use_container_width=True)
                
                # EDA Batch
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-chart-column"></i> Analisis Batch</h3>', unsafe_allow_html=True)
                
                col1, col2 = st. columns(2)
                
                with col1:
                    st.markdown("**Distribusi Prediksi per Kelas**")
                    class_counts = df_results["pred_label"].value_counts()
                    
                    fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor='#1a1a2e')
                    ax1.set_facecolor('#1a1a2e')
                    ax1.bar(class_counts. index, class_counts.values, color=['#9d4edd', '#6a0dad', '#c77dff'])
                    ax1.set_ylabel("Count", fontweight='bold', color='white')
                    ax1.set_title("Prediction Distribution", fontweight='bold', color='white')
                    ax1.tick_params(colors='white')
                    ax1.spines['bottom'].set_color('white')
                    ax1.spines['left'].set_color('white')
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    st. pyplot(fig1)
                    plt.close()
                
                with col2:
                    st.markdown("**Distribusi Confidence**")
                    fig2, ax2 = plt. subplots(figsize=(6, 4), facecolor='#1a1a2e')
                    ax2.set_facecolor('#1a1a2e')
                    ax2.hist(df_results["confidence"], bins=10, color='#9d4edd', edgecolor='white')
                    ax2.set_xlabel("Confidence", fontweight='bold', color='white')
                    ax2.set_ylabel("Frequency", fontweight='bold', color='white')
                    ax2.set_title("Confidence Distribution", fontweight='bold', color='white')
                    ax2.tick_params(colors='white')
                    ax2.spines['bottom'].set_color('white')
                    ax2.spines['left'].set_color('white')
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right']. set_visible(False)
                    st.pyplot(fig2)
                    plt.close()
                
                # Detail per gambar
                st.markdown("---")
                st.markdown('<h3><i class="fa-solid fa-magnifying-glass-chart"></i> Analisis Detail per Gambar</h3>', unsafe_allow_html=True)
                
                selected_idx = st.selectbox(
                    "Pilih gambar untuk analisis XAI:",
                    range(len(images)),
                    format_func=lambda x: image_names[x]
                )
                
                selected_img = images[selected_idx]
                
                with st.spinner("Generating Grad-CAM..."):
                    pred_label_sel, pred_conf_sel, preds_sel, heatmap_sel, overlay_sel = inf.predict_with_gradcam(
                        model, selected_img, CLASS_NAMES, last_conv_layer=LAST_CONV_LAYER, target_size=IMG_SIZE
                    )
                
                st.info(f"**Prediksi:** {pred_label_sel} | **Confidence:** {pred_conf_sel:.2%}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(selected_img, caption="Original", use_container_width=True)
                
                with col2:
                    fig_h, ax_h = plt.subplots(facecolor='#1a1a2e')
                    ax_h.imshow(heatmap_sel, cmap='jet')
                    ax_h.axis('off')
                    st.pyplot(fig_h)
                    plt.close()
                    st.markdown('<p class="image-caption">Heatmap</p>', unsafe_allow_html=True)
                
                with col3:
                    st.image(overlay_sel, caption="Overlay", use_container_width=True)
                
                xai_text_sel = inf.generate_xai_text(pred_label_sel, pred_conf_sel)
                st.info(xai_text_sel)
            
            else:
                st.warning("Tidak ada gambar ditemukan dalam ZIP file.")

# ========================================
# FOOTER
# ========================================

st. markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p style="margin: 0; font-size: 14px;">
        © 2025 Zeedan Mustami Argani | University of Muhammadiyah Malang
    </p>
    <p style="margin: 0. 5rem 0 0 0; font-size: 12px; color: #666;">
        CNN Klasik LoRA Dashboard - Halal Logo Detection System
    </p>
</div>
""", unsafe_allow_html=True)