# 🎯 Halal AI Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Dashboard interaktif untuk deteksi logo halal menggunakan **CNN Klasik LoRA** (Low-Rank Adaptation).

## 🌟 Features

### 📊 Dashboard Analytics
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Training Visualization**: Training & Validation Curves
- **Confusion Matrix**: Model performance analysis
- **Per-Class Metrics**: Detailed metrics untuk setiap kelas
- **XAI Gallery**: Grad-CAM visualization examples

### 🔍 Halal AI Detection
- **Real-Time Camera**: Capture dan prediksi langsung dari webcam
- **Single Image Upload**: Upload dan analisis 1 gambar
- **Batch Processing**: Upload ZIP file untuk analisis multiple images
- **Grad-CAM Visualization**: Explainable AI dengan heatmap
- **Confidence Score**: Tingkat kepercayaan prediksi

## 🎯 Model Information

- **Architecture**: CNN Klasik dengan LoRA adaptation
- **Input Size**: 128x128 pixels
- **Classes**: 
  - 🇮🇩 **INDOLOGO**: Logo Halal Indonesia (MUI/BPJPH)
  - 🌍 **INTERLOGO**: Logo Halal Internasional
  - ❌ **NOHALAL**: Tidak ada logo halal
- **XAI Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/NoMaoZee/Web_Dashboard_MachineLearning.git
cd Web_Dashboard_MachineLearning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Dashboard
```bash
streamlit run app_streamlit.py
```

Dashboard akan terbuka di: **http://localhost:8501**

## 📦 Dependencies

- Python 3.8+
- Streamlit
- TensorFlow
- OpenCV
- Pillow
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

Lihat `requirements.txt` untuk versi lengkap.

## 📁 Project Structure

```
Web_Dashboard_MachineLearning/
├── app_streamlit.py          # Main Streamlit application
├── inference.py              # Inference & XAI backend
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/
│   └── cnn_classic_lora_best.h5    # Trained model (96.69 MB)
│
├── history/
│   └── cnn_lora_classification_report.txt
│
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── per_class_metrics.png
│
├── xai/
│   ├── gradcam_INDOLOGO_sample.png
│   ├── gradcam_INTERLOGO_sample.png
│   ├── gradcam_NOHALAL_sample.png
│   └── xai_random_grid.png
│
└── test_results/
    └── test_summary.json
```

## 🎨 Screenshots

### Dashboard Analytics
![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Analytics)

### Halal AI Detection
![Detection](https://via.placeholder.com/800x400?text=Halal+AI+Detection)

### Grad-CAM Visualization
![Grad-CAM](https://via.placeholder.com/800x400?text=Grad-CAM+Visualization)

## 🌐 Deploy to Streamlit Cloud

### Method 1: Via Streamlit Cloud Dashboard
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set main file: `app_streamlit.py`
6. Click "Deploy"

### Method 2: Via GitHub
1. Push to GitHub
2. Connect Streamlit Cloud to your GitHub account
3. Select repository and branch
4. Deploy automatically

## ⚙️ Configuration

Main configuration di `app_streamlit.py`:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/cnn_classic_lora_best.h5")
CLASS_NAMES = {0: "INDOLOGO", 1: "INTERLOGO", 2: "NOHALAL"}
IMG_SIZE = (128, 128)
LAST_CONV_LAYER = "conv3_1"
```

## 🧪 Testing

### Test Single Image
1. Buka tab "Upload Single Image"
2. Upload gambar logo halal (JPG/PNG/JPEG)
3. Lihat hasil prediksi + Grad-CAM

### Test Batch Processing
1. Buat ZIP file berisi beberapa gambar
2. Upload di tab "Upload Multiple (ZIP)"
3. Lihat analisis batch

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | Check dashboard |
| Precision | Check dashboard |
| Recall | Check dashboard |
| F1-Score | Check dashboard |

## 🛠️ Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: Model file not found
Pastikan file `models/cnn_classic_lora_best.h5` ada di repository.

### Camera tidak berfungsi
Gunakan tab "Upload Single Image" sebagai alternatif.

## 👨‍💻 Developer

**Zeedan Mustami Argani**  
University of Muhammadiyah Malang

## 📄 License

© 2025 - CNN Klasik LoRA Dashboard - Halal Logo Detection System

## 🙏 Acknowledgments

- TensorFlow & Keras team
- Streamlit team
- Grad-CAM paper authors

---

**Made with ❤️ for Halal Product Detection**
