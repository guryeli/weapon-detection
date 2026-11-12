# 📋 Weapon Detection Project - Setup and Requirements

This document contains every setup step and system requirement needed to run the weapon detection project.

---

## 🖥️ System Requirements

### Minimum Requirements
- **Operating system:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python:** 3.8 or newer
- **RAM:** 8 GB
- **Disk space:** 20 GB
- **GPU:** CUDA-capable NVIDIA GPU (GTX 1060 or newer)

### Recommended Requirements
- **RAM:** 16 GB or more
- **GPU:** NVIDIA RTX 2070 or newer
- **VRAM:** 8 GB or more
- **CUDA:** 11.8 or 12.1

---

## 🔧 Setup Steps

### 1. Prepare the Python Environment

```bash
# Check the Python version
python --version  # must be >= 3.8

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 2. Install the Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install the core requirements
pip install -r requirements.txt
```

### 3. Install PyTorch (with GPU Support)

Check your CUDA version and install the matching PyTorch build:

```bash
# Check the CUDA version
nvidia-smi

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU (if there is no GPU)
pip install torch torchvision torchaudio
```

**Verify the installation:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## 📦 Per-Model Installation

### YOLOv8 (Ultralytics)

```bash
# Install Ultralytics
pip install ultralytics

# Verify the installation
yolo version
```

**Additional notes:**
- YOLOv8 has the simplest install
- Pretrained models are downloaded automatically
- GPU detection is automatic

---

### Faster R-CNN (Detectron2)

Detectron2 installation depends on the platform:

#### Linux / macOS

```bash
# Install Detectron2 (from source)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Or a specific release
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

#### Windows

Windows needs extra steps:

```bash
# 1. Visual Studio Build Tools must be installed
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. Install pycocotools
pip install pycocotools-windows

# 3. Detectron2 pre-built wheel
# For CUDA 11.8, PyTorch 2.0:
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

**Verify the installation:**
```python
import detectron2
from detectron2 import model_zoo
print(f"Detectron2: {detectron2.__version__}")
```

**Possible problems and fixes:**
- `ModuleNotFoundError: No module named 'cv2'` → `pip install opencv-python`
- `ModuleNotFoundError: No module named 'pycocotools'` → `pip install pycocotools`
- Windows build error → install Visual Studio Build Tools

---

## 📂 Dataset Layout

The project expects the following dataset layout:

```
SilahTespiti/
├── archive/
│   ├── metadata.csv
│   ├── test/
│   │   └── test/
│   │       ├── army.jpg
│   │       ├── terrorists.jpg
│   │       └── ...
│   └── weapon_detection/
│       ├── train/
│       │   ├── images/
│       │   │   ├── image1.jpg
│       │   │   └── ...
│       │   └── labels/
│       │       ├── image1.txt
│       │       └── ...
│       └── val/
│           ├── images/
│           │   └── ...
│           └── labels/
│               └── ...
```

**Label format (YOLO):**
```
class_id x_center y_center width height
```
- All values are normalized to 0-1
- class_id = 0 (weapon)

---

## ✅ Installation Check Script

To verify the whole installation in one pass:

```python
#!/usr/bin/env python3
# check_installation.py

def check_installation():
    """Check all dependencies"""
    
    results = {}
    
    # Python
    import sys
    results["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # PyTorch
    try:
        import torch
        results["PyTorch"] = torch.__version__
        results["CUDA Available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            results["GPU"] = torch.cuda.get_device_name(0)
    except ImportError:
        results["PyTorch"] = "NOT INSTALLED"
    
    # Ultralytics (YOLOv8)
    try:
        import ultralytics
        results["Ultralytics"] = ultralytics.__version__
    except ImportError:
        results["Ultralytics"] = "NOT INSTALLED"
    
    # Detectron2
    try:
        import detectron2
        results["Detectron2"] = detectron2.__version__
    except ImportError:
        results["Detectron2"] = "NOT INSTALLED"
    
    
    # OpenCV
    try:
        import cv2
        results["OpenCV"] = cv2.__version__
    except ImportError:
        results["OpenCV"] = "NOT INSTALLED"
    
    # Print the results
    print("\n" + "="*50)
    print("INSTALLATION CHECK")
    print("="*50)
    for name, version in results.items():
        status = "✓" if version != "NOT INSTALLED" else "✗"
        print(f"{status} {name}: {version}")
    print("="*50)
    
    return results

if __name__ == "__main__":
    check_installation()
```

---

## 🐛 Common Problems

### 1. CUDA Out of Memory
```bash
# Fix: reduce the batch size
python train_yolov8.py --batch-size 8  # 8 instead of 16
```

### 2. GPU Not Found
```python
# PyTorch CUDA check
import torch
print(torch.cuda.is_available())  # If False:
# 1. Update the NVIDIA driver
# 2. Reinstall PyTorch with CUDA
```

### 3. Import Errors
```bash
# Make sure the virtual environment is active
which python  # should show the virtual environment path

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### 4. Build Errors on Windows
```bash
# Install Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select the "C++ build tools" workload
```

---

## 📞 Support

For setup problems:
1. Copy the error message
2. Check your Python and CUDA versions
3. Make sure the virtual environment is active
4. Read the documentation:
   - [Ultralytics Docs](https://docs.ultralytics.com/)
   - [Detectron2 Docs](https://detectron2.readthedocs.io/)

---

**Last update:** January 2026
