
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras. models import load_model
from tensorflow.keras import Model
import json

# ========================================
# 1. Load Model (Cached)
# ========================================

_MODEL_CACHE = None

def load_cnn_lora_model(model_path):
    """Load model CNN_Klasik_LoRA (cached)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None: 
        _MODEL_CACHE = load_model(model_path)
        print(f"Model loaded from:  {model_path}")
    return _MODEL_CACHE

# ========================================
# 2. Preprocessing & Prediction
# ========================================

def preprocess_image(pil_image, target_size=(128, 128)):
    """Preprocess PIL Image untuk inference."""
    img = pil_image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, pil_image, class_names, target_size=(128, 128)):
    """Prediksi satu gambar."""
    img_array = preprocess_image(pil_image, target_size)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    # Convert NumPy int64 to Python int for compatibility
    pred_idx = int(pred_idx)
    pred_label = class_names[pred_idx]
    pred_conf = preds[pred_idx]
    return pred_label, pred_conf, preds

# ========================================
# 3. Grad-CAM
# ========================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap."""
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except: 
        raise ValueError(f"Layer '{last_conv_layer_name}' tidak ditemukan!")
    
    grad_model = Model(
        inputs=[model.input],
        outputs=[conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Ensure predictions is a tensor (not a list)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = tf.convert_to_tensor(predictions)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # Convert pred_index to Python int for proper indexing
        pred_index = int(pred_index.numpy()) if hasattr(pred_index, 'numpy') else int(pred_index)
        
        # Use tf.gather for safer indexing
        class_channel = tf.gather(predictions, pred_index, axis=1)
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf. reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap. numpy()

def generate_gradcam_overlay(pil_image, heatmap, alpha=0.4):
    """Generate overlay Grad-CAM pada gambar asli."""
    img = np.array(pil_image. convert("RGB"))
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return heatmap_resized, overlay

def predict_with_gradcam(model, pil_image, class_names, last_conv_layer="conv3_1", 
                         target_size=(128, 128)):
    """Prediksi + Grad-CAM sekaligus."""
    img_array = preprocess_image(pil_image, target_size)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    # Convert NumPy int64 to Python int for compatibility
    pred_idx = int(pred_idx)
    pred_label = class_names[pred_idx]
    pred_conf = preds[pred_idx]
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=pred_idx)
    heatmap_img, overlay = generate_gradcam_overlay(pil_image, heatmap)
    
    return pred_label, pred_conf, preds, heatmap, overlay

# ========================================
# 4. EDA per Gambar
# ========================================

def compute_image_eda(pil_image):
    """Compute EDA untuk satu gambar."""
    img_array = np.array(pil_image. convert("RGB"))
    
    width, height = pil_image.size
    brightness = np.mean(img_array)
    file_format = pil_image.format if pil_image.format else "Unknown"
    
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "format": file_format
    }

# ========================================
# 5. Generate Interpretasi Teks XAI
# ========================================

def generate_xai_text(pred_label, pred_conf):
    """Generate interpretasi teks untuk XAI."""
    explanations = {
        "INDOLOGO":  (
            f"Model memprediksi gambar ini sebagai **INDOLOGO** dengan confidence **{pred_conf:.1%}**.\n\n"
            "Grad-CAM menunjukkan model fokus pada area **lingkaran logo** dan **teks 'HALAL'** di tengah gambar.\n\n"
            "Ini adalah pola khas **logo halal resmi Indonesia** (MUI/BPJPH)."
        ),
        "INTERLOGO": (
            f"Model memprediksi gambar ini sebagai **INTERLOGO** dengan confidence **{pred_conf:.1%}**.\n\n"
            "Grad-CAM menunjukkan model fokus pada **simbol halal internasional** dan **teks sertifikasi**.\n\n"
            "Logo ini kemungkinan berasal dari **badan sertifikasi halal luar negeri**."
        ),
        "NOHALAL": (
            f"Model memprediksi gambar ini sebagai **NOHALAL** dengan confidence **{pred_conf:.1%}**.\n\n"
            "Grad-CAM menunjukkan model **tidak menemukan pola logo halal** yang konsisten.\n\n"
            "Gambar ini kemungkinan **tidak mengandung logo halal**, atau logo yang mirip namun bukan logo halal resmi."
        )
    }
    
    base_text = explanations.get(pred_label, "")
    
    # Tambahkan warning jika confidence rendah
    if pred_conf < 0.7:
        base_text += f"\n\n**Perhatian**:  Confidence relatif rendah ({pred_conf:.1%}). Model mungkin ragu atau gambar tidak jelas."
    
    return base_text

# ========================================
# 6. Load Test Summary
# ========================================

def load_test_summary(json_path):
    """Load test summary JSON."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None
