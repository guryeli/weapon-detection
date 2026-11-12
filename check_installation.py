#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation check script
Verifies all dependencies and the system status

Usage: python check_installation.py
"""

import sys
import os
from pathlib import Path


def print_header(title: str):
    """Print a heading"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def check_python():
    """Check the Python version"""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 8)
    return "Python", version, ok


def check_pytorch():
    """Check the PyTorch install (including CUDA and Apple MPS)."""
    try:
        import torch
        version = torch.__version__
        cuda = torch.cuda.is_available()
        mps = (hasattr(torch.backends, "mps")
               and torch.backends.mps.is_available())

        info = f"{version}"
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            info += f" (CUDA {torch.version.cuda}, GPU: {gpu})"
        elif mps:
            info += " (Apple Silicon MPS)"
        else:
            info += " (CPU only)"

        return "PyTorch", info, True
    except ImportError:
        return "PyTorch", "NOT INSTALLED", False


def check_pycocotools():
    try:
        from pycocotools import __version__ as v
        return "pycocotools", str(v), True
    except Exception:
        try:
            import pycocotools
            return "pycocotools", "installed", True
        except ImportError:
            return "pycocotools", "NOT INSTALLED", False


def check_matplotlib():
    try:
        import matplotlib
        return "matplotlib", matplotlib.__version__, True
    except ImportError:
        return "matplotlib", "NOT INSTALLED", False


def check_ultralytics():
    """Check the Ultralytics (YOLOv8) install"""
    try:
        import ultralytics
        return "Ultralytics (YOLOv8)", ultralytics.__version__, True
    except ImportError:
        return "Ultralytics (YOLOv8)", "NOT INSTALLED", False


def check_detectron2():
    """Check the Detectron2 install"""
    try:
        import detectron2
        return "Detectron2", detectron2.__version__, True
    except ImportError:
        return "Detectron2", "NOT INSTALLED (optional)", False


def check_mmdetection():
    """Check the MMDetection install"""
    try:
        import mmdet
        return "MMDetection", mmdet.__version__, True
    except ImportError:
        return "MMDetection", "NOT INSTALLED (optional)", False


def check_opencv():
    """Check the OpenCV install"""
    try:
        import cv2
        return "OpenCV", cv2.__version__, True
    except ImportError:
        return "OpenCV", "NOT INSTALLED", False


def check_numpy():
    """Check the NumPy install"""
    try:
        import numpy
        return "NumPy", numpy.__version__, True
    except ImportError:
        return "NumPy", "NOT INSTALLED", False


def check_dataset():
    """Check the dataset layout"""
    project_root = Path(__file__).parent.resolve()
    dataset_dir = project_root / "archive" / "weapon_detection"
    
    checks = []
    
    # Train folder
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    
    if train_images.exists():
        img_count = len(list(train_images.glob("*")))
        checks.append(f"Train images: {img_count}")
    else:
        checks.append("Train images: NOT FOUND")
    
    if train_labels.exists():
        label_count = len(list(train_labels.glob("*.txt")))
        checks.append(f"Train labels: {label_count}")
    else:
        checks.append("Train labels: NOT FOUND")
    
    # Val folder
    val_images = dataset_dir / "val" / "images"
    val_labels = dataset_dir / "val" / "labels"
    
    if val_images.exists():
        img_count = len(list(val_images.glob("*")))
        checks.append(f"Val images: {img_count}")
    else:
        checks.append("Val images: NOT FOUND")
    
    # Test folder
    test_dir = project_root / "archive" / "test" / "test"
    if test_dir.exists():
        test_count = len(list(test_dir.glob("*.jpg")))
        checks.append(f"Test images: {test_count}")
    else:
        checks.append("Test images: NOT FOUND")
    
    return checks


def check_gpu():
    """Check GPU status"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{gpu_name} ({gpu_memory:.1f} GB VRAM)"
        else:
            return "No CUDA GPU found - CPU will be used"
    except:
        return "Could not read GPU information"


def main():
    """Main function"""
    print_header("WEAPON DETECTION PROJECT - INSTALLATION CHECK")
    
    # System information
    print_header("SYSTEM INFORMATION")
    print(f"  Operating system: {sys.platform}")
    print(f"  Python path: {sys.executable}")
    print(f"  GPU: {check_gpu()}")
    
    # Dependencies
    print_header("DEPENDENCIES")
    
    checks = [
        check_python(),
        check_pytorch(),
        check_ultralytics(),
        check_detectron2(),
        check_mmdetection(),
        check_opencv(),
        check_numpy(),
        check_pycocotools(),
        check_matplotlib(),
    ]
    
    all_ok = True
    for name, version, ok in checks:
        status = "✓" if ok else "✗"
        color = "" if ok else ""
        print(f"  {status} {name}: {version}")
        if not ok and "optional" not in version.lower():
            all_ok = False
    
    # Dataset
    print_header("DATASET")
    for info in check_dataset():
        print(f"  • {info}")
    
    # Project files
    print_header("PROJECT FILES")
    project_root = Path(__file__).parent.resolve()
    
    required_files = [
        "train_yolov8.py",
        "train_faster_rcnn.py",
        "benchmark_comparison.py",
        "eval/recompute_benchmarks.py",
        "eval/compare.py",
        "eval/metrics.py",
        "requirements.txt",
        "README.md",
    ]
    
    for filename in required_files:
        filepath = project_root / filename
        status = "✓" if filepath.exists() else "✗"
        print(f"  {status} {filename}")
    
    # Folders
    print_header("FOLDERS")
    required_dirs = [
        "archive",
        "benchmark",
        "models",
        "outputs",
        "results"
    ]
    
    for dirname in required_dirs:
        dirpath = project_root / dirname
        status = "✓" if dirpath.exists() else "✗"
        print(f"  {status} {dirname}/")
    
    # Summary
    print_header("SUMMARY")
    
    if all_ok:
        print("  ✓ All core dependencies are installed!")
        print("  ✓ The project is ready to run")
    else:
        print("  ✗ Some dependencies are missing")
        print("  → Install them with requirements.txt:")
        print("    pip install -r requirements.txt")
    
    print("\n" + "="*60)
    print(" USAGE")
    print("="*60)
    print("""
  # Train and test with YOLOv8:
  python train_yolov8.py --mode all
  
  # Benchmark comparison only:
  python benchmark_comparison.py
  
  # Predict on a custom image:
  python train_yolov8.py --mode predict --source image.jpg
""")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
