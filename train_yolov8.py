#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Weapon Detection - Training, Test, and Benchmark
Weapon detection model using Ultralytics YOLOv8

Author: Weapon Detection Research Project
Date: 2026
"""

import os
import sys
import json
import time
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import random
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def set_global_seed(seed: int = 42) -> None:
    """Fix every source of randomness (for reproducibility in the thesis)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    # Deterministic behavior (can slow things down; optional during training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATASET_DIR = ARCHIVE_DIR / "weapon_detection"
TEST_IMAGES_DIR = ARCHIVE_DIR / "test" / "test"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create folders
for dir_path in [BENCHMARK_DIR, MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class YOLOv8WeaponDetector:
    """YOLOv8-based weapon detection class"""
    
    def __init__(self, model_size: str = "n", pretrained: bool = True):
        """
        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            pretrained: Use pretrained weights
        """
        self.model_size = model_size
        self.pretrained = pretrained
        self.model: Optional[YOLO] = None
        self.training_time: float = 0.0
        self.inference_times: List[float] = []
        self.device = self._select_device()

        print(f"[INFO] Device: {self.device}")
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        elif self.device == "mps":
            print("[INFO] Apple Silicon MPS is active")

    @staticmethod
    def _select_device() -> str:
        """Order: Apple MPS > CUDA > CPU."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def fix_labels_to_single_class(self) -> None:
        """
        Set the class ID in every label file to 0.
        Fixes inconsistent labels in the dataset.
        """
        print(f"[INFO] Fixing label files (all classes -> 0)...")
        
        fixed_count = 0
        for split in ["train", "val"]:
            labels_dir = DATASET_DIR / split / "labels"
            if not labels_dir.exists():
                continue
            
            for label_file in labels_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    modified = False
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Set the class ID to 0
                            if parts[0] != '0':
                                parts[0] = '0'
                                modified = True
                            new_lines.append(' '.join(parts) + '\n')
                        else:
                            new_lines.append(line)
                    
                    if modified:
                        with open(label_file, 'w') as f:
                            f.writelines(new_lines)
                        fixed_count += 1
                except Exception as e:
                    print(f"  [WARN] {label_file.name}: {e}")
        
        print(f"[INFO] {fixed_count} label files fixed")
        
        # Delete cache files (so they are rebuilt)
        for split in ["train", "val"]:
            cache_file = DATASET_DIR / split / "labels.cache"
            if cache_file.exists():
                cache_file.unlink()
                print(f"[INFO] Cache deleted: {cache_file}")
    
    def create_dataset_yaml(self) -> str:
        """Create a dataset.yaml file in YOLO format"""
        # Fix labels first
        self.fix_labels_to_single_class()
        
        yaml_path = PROJECT_ROOT / "dataset_yolo.yaml"
        
        dataset_config = {
            "path": str(DATASET_DIR),
            "train": "train/images",
            "val": "val/images",
            "names": {
                0: "weapon"
            },
            "nc": 1
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"[INFO] Dataset YAML created: {yaml_path}")
        return str(yaml_path)
    
    def load_model(self, weights_path: Optional[str] = None):
        """Load the model"""
        if weights_path and Path(weights_path).exists():
            print(f"[INFO] Loading custom weights: {weights_path}")
            self.model = YOLO(weights_path)
        elif self.pretrained:
            model_name = f"yolov8{self.model_size}.pt"
            print(f"[INFO] Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            model_name = f"yolov8{self.model_size}.yaml"
            print(f"[INFO] Creating a model from scratch: {model_name}")
            self.model = YOLO(model_name)
        
        return self.model
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        patience: int = 20,
        save_period: int = 10,
        resume: bool = False,
        augment: bool = True,
        optimizer: str = "auto",   # Ultralytics chooses: SGD + lr=0.01 or
                                    # AdamW + lr=0.001 (depending on model size)
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        cos_lr: bool = True,
        workers: int = 8,
        project: str = None,
        name: str = None
    ) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary with training results and metrics
        """
        if self.model is None:
            self.load_model()
        
        # Create the dataset yaml
        data_yaml = self.create_dataset_yaml()
        
        # Set project and name
        if project is None:
            project = str(OUTPUTS_DIR / "yolov8")
        if name is None:
            name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"[TRAIN] YOLOv8{self.model_size.upper()} training is starting")
        print(f"{'='*60}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Image Size: {imgsz}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Learning Rate: {lr0}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Training start time
        start_time = time.time()
        
        train_kwargs = dict(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=patience,
            save_period=save_period,
            resume=resume,
            augment=augment,
            optimizer=optimizer,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            cos_lr=cos_lr,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
            device=self.device,
        )
        # When optimizer="auto", Ultralytics picks lr0 itself
        # (0.01 for SGD, 0.001 for AdamW). Passing lr0 from outside
        # breaks the auto choice — so we add it only for a manual optimizer.
        if optimizer != "auto":
            train_kwargs["lr0"] = lr0

        results = self.model.train(**train_kwargs)
        
        # Training duration
        self.training_time = time.time() - start_time
        
        # Save the best model
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            save_path = MODELS_DIR / f"yolov8{self.model_size}_weapon_best.pt"
            shutil.copy(best_model_path, save_path)
            print(f"[INFO] Best model saved: {save_path}")
        
        # Collect the results
        training_results = {
            "model": f"YOLOv8{self.model_size.upper()}",
            "framework": "Ultralytics",
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": imgsz,
            "optimizer": optimizer,
            "learning_rate": lr0,
            "training_time_seconds": self.training_time,
            "training_time_formatted": self._format_time(self.training_time),
            "device": self.device,
            "best_model_path": str(save_path) if best_model_path.exists() else None,
            "results_dir": str(Path(project) / name)
        }
        
        print(f"\n[TRAIN] Training finished!")
        print(f"[TRAIN] Duration: {training_results['training_time_formatted']}")
        
        return training_results
    
    def validate(self, weights_path: Optional[str] = None) -> Dict:
        """
        Evaluate on the validation set
        
        Returns:
            Validation metrics
        """
        if weights_path:
            self.load_model(weights_path)
        elif self.model is None:
            # Load the most recently trained model
            best_model = MODELS_DIR / f"yolov8{self.model_size}_weapon_best.pt"
            if best_model.exists():
                self.load_model(str(best_model))
            else:
                raise ValueError("Model not found. Train first or provide a model path.")
        
        data_yaml = self.create_dataset_yaml()
        
        print(f"\n{'='*60}")
        print(f"[VAL] Validation is starting")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Validation
        results = self.model.val(
            data=data_yaml,
            split="val",
            device=self.device,
            verbose=True
        )
        
        val_time = time.time() - start_time
        
        # Extract metrics
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "f1_score": 2 * (float(results.box.mp) * float(results.box.mr)) / 
                       (float(results.box.mp) + float(results.box.mr) + 1e-6),
            "validation_time_seconds": val_time,
            "validation_time_formatted": self._format_time(val_time)
        }
        
        print(f"\n[VAL] Validation results:")
        print(f"  mAP@50: {metrics['mAP50']:.4f}")
        print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def test_real_world(self, save_visualizations: bool = True) -> Dict:
        """
        Test on real-world test images
        
        Returns:
            Test results and predictions
        """
        if self.model is None:
            best_model = MODELS_DIR / f"yolov8{self.model_size}_weapon_best.pt"
            if best_model.exists():
                self.load_model(str(best_model))
            else:
                raise ValueError("Model not found.")
        
        print(f"\n{'='*60}")
        print(f"[TEST] Real-world test is starting")
        print(f"{'='*60}\n")
        
        # Find test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
            test_images.extend(TEST_IMAGES_DIR.glob(ext))
        
        if not test_images:
            print(f"[WARN] No test image found: {TEST_IMAGES_DIR}")
            return {}
        
        print(f"[INFO] Found {len(test_images)} test images")
        
        results_list = []
        total_inference_time = 0
        
        # Results folder
        vis_dir = RESULTS_DIR / "yolov8_predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in test_images:
            print(f"\n[TEST] Processing: {img_path.name}")
            
            # Skip GIF files (static images only)
            if img_path.suffix.lower() == '.gif':
                print(f"  [SKIP] GIF file skipped")
                continue
            
            # Read the image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [ERROR] Could not read the image")
                continue
            
            # Inference
            start_time = time.time()
            predictions = self.model.predict(
                source=img,
                conf=0.25,
                iou=0.45,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            self.inference_times.append(inference_time)
            
            # Process the results
            pred = predictions[0]
            detections = []
            
            for box in pred.boxes:
                det = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": "weapon"
                }
                detections.append(det)
            
            result = {
                "image": img_path.name,
                "image_path": str(img_path),
                "inference_time_ms": inference_time * 1000,
                "num_detections": len(detections),
                "detections": detections
            }
            results_list.append(result)
            
            print(f"  Detections: {len(detections)} weapons")
            print(f"  Inference: {inference_time*1000:.2f} ms")
            
            # Visualization
            if save_visualizations and detections:
                vis_img = img.copy()
                for det in detections:
                    x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                    conf = det["confidence"]
                    
                    # Draw the box
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label
                    label = f"Weapon: {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_img, (x1, y1-25), (x1+label_w, y1), (0, 255, 0), -1)
                    cv2.putText(vis_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Save
                save_path = vis_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(save_path), vis_img)
        
        # Overall statistics
        avg_inference = (total_inference_time / len(results_list)) * 1000 if results_list else 0
        fps = 1000 / avg_inference if avg_inference > 0 else 0
        
        test_summary = {
            "model": f"YOLOv8{self.model_size.upper()}",
            "total_images": len(results_list),
            "total_detections": sum(r["num_detections"] for r in results_list),
            "avg_inference_time_ms": avg_inference,
            "fps": fps,
            "predictions": results_list,
            "visualization_dir": str(vis_dir)
        }
        
        print(f"\n[TEST] Test summary:")
        print(f"  Total images: {test_summary['total_images']}")
        print(f"  Total detections: {test_summary['total_detections']}")
        print(f"  Average inference: {avg_inference:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        return test_summary
    
    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = True
    ) -> List[Dict]:
        """
        Run prediction on a single image or a folder
        
        Args:
            source: Image path or folder
            conf: Confidence threshold
            iou: IoU threshold
            save: Save the results
        
        Returns:
            Prediction results
        """
        if self.model is None:
            best_model = MODELS_DIR / f"yolov8{self.model_size}_weapon_best.pt"
            if best_model.exists():
                self.load_model(str(best_model))
            else:
                raise ValueError("Model not found.")
        
        predictions = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            project=str(RESULTS_DIR / "yolov8_predictions"),
            name="custom",
            device=self.device
        )
        
        results = []
        for pred in predictions:
            detections = []
            for box in pred.boxes:
                det = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_name": "weapon"
                }
                detections.append(det)
            results.append({
                "path": str(pred.path),
                "detections": detections
            })
        
        return results
    
    def generate_benchmark(self) -> Dict:
        """
        Build a full benchmark report
        
        Returns:
            Benchmark results
        """
        print(f"\n{'='*60}")
        print(f"[BENCHMARK] YOLOv8{self.model_size.upper()} Benchmark")
        print(f"{'='*60}\n")
        
        # Validation metrics
        try:
            val_metrics = self.validate()
        except Exception as e:
            print(f"[WARN] Validation error: {e}")
            val_metrics = {}
        
        # Real-world test
        try:
            test_results = self.test_real_world(save_visualizations=True)
        except Exception as e:
            print(f"[WARN] Test error: {e}")
            test_results = {}
        
        # Model information
        model_info = {}
        if self.model:
            model_info = {
                "model_name": f"YOLOv8{self.model_size.upper()}",
                "framework": "Ultralytics",
                "framework_version": "8.x",
                "parameters": self._count_parameters(),
                "input_size": 640,
                "device": self.device
            }
        
        # Benchmark report (v2 schema: scale=0-1)
        benchmark = {
            "schema_version": 2,
            "scale": "0-1",
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
            "training": {
                "training_time_seconds": self.training_time,
                "training_time_formatted": self._format_time(self.training_time)
            },
            "validation_metrics": val_metrics,
            "real_world_test": {
                "total_images": test_results.get("total_images", 0),
                "total_detections": test_results.get("total_detections", 0),
                "avg_inference_time_ms": test_results.get("avg_inference_time_ms", 0),
                "fps": test_results.get("fps", 0)
            },
            "performance_summary": {
                "mAP50": val_metrics.get("mAP50", 0),
                "mAP50_95": val_metrics.get("mAP50-95", val_metrics.get("mAP50_95", 0)),
                "precision": val_metrics.get("precision", 0),
                "recall": val_metrics.get("recall", 0),
                "f1_score": val_metrics.get("f1_score", 0),
                "avg_inference_ms": test_results.get("avg_inference_time_ms", 0),
                "fps": test_results.get("fps", 0)
            }
        }
        
        # Save the benchmark
        benchmark_path = BENCHMARK_DIR / "yolov8_benchmark.json"
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=4, ensure_ascii=False)
        
        print(f"\n[BENCHMARK] Report saved: {benchmark_path}")
        
        return benchmark
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        if self.model is None:
            return 0
        try:
            return sum(p.numel() for p in self.model.model.parameters())
        except:
            return 0
    
    def _format_time(self, seconds: float) -> str:
        """Format a duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv8 Weapon Detection")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test", "predict", "benchmark", "all"],
                       default="all", help="Run mode")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--weights", type=str, default=None, help="Custom model weights")
    parser.add_argument("--source", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (reproducibility)")
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["auto", "SGD", "AdamW", "Adam", "RMSProp"],
                        help="Optimizer (auto = Ultralytics chooses; 'auto' is recommended for large models)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (used only when optimizer != auto; "
                             "0.001 is recommended for AdamW, 0.01 for SGD)")

    args = parser.parse_args()

    set_global_seed(args.seed)
    print(f"[INFO] Global seed: {args.seed}")

    # Create the detector
    detector = YOLOv8WeaponDetector(model_size=args.model_size)
    
    if args.mode == "train" or args.mode == "all":
        print("\n" + "="*60)
        print("YOLOv8 TRAINING")
        print("="*60)
        
        detector.load_model(args.weights)
        training_results = detector.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            optimizer=args.optimizer,
            lr0=args.lr,
        )
        
        # Save the training results
        results_path = BENCHMARK_DIR / "yolov8_training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, indent=4, ensure_ascii=False)
    
    if args.mode == "val" or args.mode == "all":
        print("\n" + "="*60)
        print("YOLOv8 VALIDATION")
        print("="*60)
        
        val_metrics = detector.validate(args.weights)
    
    if args.mode == "test" or args.mode == "all":
        print("\n" + "="*60)
        print("YOLOv8 REAL-WORLD TEST")
        print("="*60)
        
        test_results = detector.test_real_world(save_visualizations=True)
    
    if args.mode == "predict" and args.source:
        print("\n" + "="*60)
        print("YOLOv8 PREDICTION")
        print("="*60)
        
        predictions = detector.predict(args.source, conf=args.conf)
        for pred in predictions:
            print(f"\n{pred['path']}:")
            for det in pred['detections']:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    
    if args.mode == "benchmark" or args.mode == "all":
        print("\n" + "="*60)
        print("YOLOv8 BENCHMARK")
        print("="*60)
        
        benchmark = detector.generate_benchmark()
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"  Model: {benchmark['model_info'].get('model_name', 'N/A')}")
        print(f"  mAP@50: {benchmark['performance_summary']['mAP50']:.4f}")
        print(f"  mAP@50-95: {benchmark['performance_summary']['mAP50_95']:.4f}")
        print(f"  Precision: {benchmark['performance_summary']['precision']:.4f}")
        print(f"  Recall: {benchmark['performance_summary']['recall']:.4f}")
        print(f"  F1 Score: {benchmark['performance_summary']['f1_score']:.4f}")
        print(f"  Inference: {benchmark['performance_summary']['avg_inference_ms']:.2f} ms")
        print(f"  FPS: {benchmark['performance_summary']['fps']:.2f}")
    
    print("\n" + "="*60)
    print("FINISHED")
    print("="*60)


if __name__ == "__main__":
    main()
