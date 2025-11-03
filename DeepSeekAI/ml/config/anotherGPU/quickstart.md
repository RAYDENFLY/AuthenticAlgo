# 🚀 Quick Start Guide - Medium to High-End GPUs

> **Untuk GPU**: GTX 1660 (6GB), RTX 2060 Super (8GB), RX 6600 (8GB), atau lebih tinggi

## 📋 Prerequisites

### 1. Check Your GPU
```bash
# Untuk NVIDIA GPU
nvidia-smi

# Cek VRAM yang tersedia
```

### 2. Install Dependencies
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
# atau
source venv/bin/activate      # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

## 🎯 Quick Start - 3 Cara

### Option 1: Manual Selection (Paling Aman)
Pilih config sesuai GPU Anda:

```bash
# GTX 1660 6GB (Medium Performance)
python main.py --config config/gtx_1660_6gb.yaml

# RTX 2060 Super 8GB (High Performance)
python main.py --config config/rtx_2060s_8gb.yaml

# AMD RX 6600 8GB (Good Performance)
python main.py --config config/rx_6600_8gb.yaml
```

### Option 2: Auto Detection (Recommended)
Biarkan sistem detect GPU otomatis:

```bash
# Auto-detect GPU dan pilih config optimal
python main_auto_gpu.py
```

### Option 3: Benchmark First (Paling Akurat)
Test performa GPU Anda dulu:

```bash
# Run benchmark untuk test performance
python scripts/gpu_benchmark.py

# Hasil benchmark akan suggest config terbaik
```

## 📊 Performance Expectations

| GPU | Memory | Features | Training Time | LSTM | Best Use |
|-----|--------|----------|---------------|------|----------|
| **GTX 1660 6GB** | 6GB | 40-50 | 3-8 min | ✅ Basic | Medium complexity strategies |
| **RTX 2060S 8GB** | 8GB | 70-80 | 2-5 min | ✅ Advanced | High performance trading |
| **RX 6600 8GB** | 8GB | 60-70 | 4-10 min | ⚠️ ROCm | Medium-high (CPU fallback) |

### Apa Artinya?
- **Features**: Jumlah fitur ML yang bisa diproses (lebih banyak = lebih akurat)
- **Training Time**: Waktu untuk train model dengan 3-6 bulan data
- **LSTM**: Support untuk deep learning models (lebih kompleks)

## 🎮 Step-by-Step Usage

### Step 1: Test Setup
```bash
# Test apakah GPU terdeteksi
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Test XGBoost GPU support
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
```

### Step 2: Run Demo
```bash
# Test dengan demo ML
python demo_ml.py

# Output yang diharapkan:
# ✅ GPU detected: NVIDIA GeForce GTX 1660
# ✅ Features extracted: 45
# ✅ XGBoost training: 3.2 minutes
# ✅ Prediction accuracy: 62%
```

### Step 3: Run Full Bot
```bash
# Gunakan config sesuai GPU
python main.py --config config/gtx_1660_6gb.yaml

# Atau auto-detect
python main_auto_gpu.py
```

### Step 4: Monitor Performance
```bash
# Buka terminal baru, monitor GPU
watch -n 1 nvidia-smi  # Linux
# atau
nvidia-smi -l 1        # Windows

# Cek dashboard
streamlit run monitoring/dashboard.py
```

## 🛠️ Troubleshooting

### ❌ GPU Not Detected

**NVIDIA GPUs:**
```bash
# 1. Cek CUDA installation
nvidia-smi

# 2. Install CUDA toolkit (jika belum)
# Download dari: https://developer.nvidia.com/cuda-downloads
# Recommended: CUDA 11.8 + cuDNN 8.6

# 3. Reinstall TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow==2.15.0

# 4. Test lagi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**AMD RX 6600:**
```bash
# Linux: Install ROCm
# https://rocm.docs.amd.com/en/latest/

# Atau gunakan CPU fallback (lebih stable)
# Edit config file:
use_cpu_for_tensorflow: true
tree_method: "hist"          # Ganti dari "gpu_hist"
predictor: "cpu_predictor"    # Ganti dari "gpu_predictor"
```

### ⚠️ Out of Memory (OOM)

**Error: CUDA Out of Memory**
```bash
# 1. Kurangi fitur di config
n_features: 40  # Ganti dari 50

# 2. Kurangi batch size (untuk LSTM)
batch_size: 16  # Ganti dari 32

# 3. Set memory limit
gpu_memory_limit_mb: 4096  # Adjust sesuai VRAM
```

### 🐌 Training Too Slow

```yaml
# Edit config file:

# 1. Disable hyperparameter optimization
hyperparameter_optimization: false

# 2. Kurangi estimators
xgboost_params:
  n_estimators: 100  # Ganti dari 200

random_forest_params:
  n_estimators: 50   # Ganti dari 100

# 3. Skip LSTM jika tidak perlu
model_types: ["xgb", "rf"]  # Hapus "lstm"
```

### 🔥 GPU Overheating

```bash
# Monitor temperature
nvidia-smi -l 1

# Jika > 80°C:
# 1. Bersihkan cooling system
# 2. Improve airflow
# 3. Kurangi workload:

# Edit config:
n_estimators: 50      # Kurangi dari 100
max_depth: 6          # Kurangi dari 8
```

## 🎯 Optimization Tips

### Untuk GTX 1660 6GB:
```yaml
# Balanced config
feature_engineering:
  n_features: 45
  lookback_periods: [10, 20]

model_training:
  model_types: ["xgb", "rf"]  # Skip LSTM
  hyperparameter_optimization: false
```

### Untuk RTX 2060S 8GB:
```yaml
# High performance config
feature_engineering:
  n_features: 75
  lookback_periods: [10, 20, 50]

model_training:
  model_types: ["xgb", "rf", "lstm"]
  hyperparameter_optimization: true
```

### Untuk RX 6600 8GB:
```yaml
# Stable config (CPU fallback)
feature_engineering:
  n_features: 60

model_training:
  use_cpu_for_tensorflow: true
  xgboost_params:
    tree_method: "hist"
```

## 📚 Next Steps

1. **Baca dokumentasi lengkap**: `README.md`
2. **Test dengan backtesting**: `python demo_backtesting.py`
3. **Setup monitoring**: `PHASE8_COMPLETE.md`
4. **Join komunitas**: [Discord/Telegram link]

## 🆘 Need Help?

- **Quick fix**: Lihat troubleshooting di atas
- **GPU issues**: Check `GTX1050TI_ML_GUIDE.md` (berlaku untuk GPU lain)
- **Performance tuning**: Baca config file comments
- **Bugs**: Create issue di GitHub

---

**Last Updated**: November 2025  
**Tested On**: GTX 1660, RTX 2060S, RX 6600  
**Status**: Production Ready ✅