#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN (Detectron2) Weapon Detection - Training, Test, and Benchmark
Weapon detection model using the Meta/Facebook Detectron2 library

Author: Weapon Detection Research Project
Date: 2026
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Detectron2 imports
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor, DefaultTrainer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.data.datasets import register_coco_instances
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.structures import BoxMode
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("[WARNING] Detectron2 is not installed. Please read the setup instructions.")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATASET_DIR = ARCHIVE_DIR / "weapon_detection"
TEST_IMAGES_DIR = ARCHIVE_DIR / "test" / "test"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"

# Create folders
for dir_path in [BENCHMARK_DIR, MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def fix_labels_to_single_class() -> None:
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
    
    if fixed_count > 0:
        print(f"[INFO] {fixed_count} label files fixed")
    else:
        print(f"[INFO] All label files were already fixed")


class CustomTrainer(DefaultTrainer):
    """Customized Detectron2 trainer"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert from YOLO format to COCO format
    YOLO: [x_center, y_center, width, height] (normalized)
    COCO: [x_min, y_min, width, height] (absolute)
    """
    x_center, y_center, w, h = yolo_bbox
    
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    
    x_min = x_center_abs - w_abs / 2
    y_min = y_center_abs - h_abs / 2
    
    return [x_min, y_min, w_abs, h_abs]


def load_weapon_dataset(split: str = "train") -> List[Dict]:
    """
    Convert the YOLO-format dataset to Detectron2 format
    
    Args:
        split: 'train' or 'val'
    
    Returns:
        Dataset in Detectron2 format
    """
    split_dir = DATASET_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    dataset_dicts = []
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    for idx, img_path in enumerate(sorted(image_files)):
        record = {}
        
        # Image information
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        height, width = img.shape[:2]
        
        record["file_name"] = str(img_path)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        # Find the label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        yolo_bbox = [float(x) for x in parts[1:5]]
                        
                        # Convert to COCO format
                        coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)
                        
                        ann = {
                            "bbox": coco_bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": 0,  # weapon class
                            "iscrowd": 0
                        }
                        annotations.append(ann)
        
        record["annotations"] = annotations
        dataset_dicts.append(record)
    
    return dataset_dicts


class FasterRCNNWeaponDetector:
    """Faster R-CNN-based weapon detection class"""
    
    def __init__(self, backbone: str = "R50-FPN"):
        """
        Args:
            backbone: Backbone network - 'R50-FPN', 'R101-FPN', 'X101-FPN'
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 is not installed!")
        
        self.backbone = backbone
        self.cfg = None
        self.predictor = None
        self.training_time: float = 0.0
        self.inference_times: List[float] = []
        # NOTE: Detectron2's Faster R-CNN ops are not fully compatible
        # on MPS; if MPS is present we tell the user, but train/val
        # prefers CPU/CUDA.
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("[INFO] Apple MPS detected; using CPU for Faster R-CNN "
                      "because of Detectron2 op incompatibilities.")
            self.device = "cpu"

        # Model configuration map
        self.model_configs = {
            "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "X101-FPN": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        }

        print(f"[INFO] Device: {self.device}")
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        
        # Register the datasets
        self._register_datasets()
    
    def _register_datasets(self):
        """Register the datasets with Detectron2"""
        # Fix labels first
        fix_labels_to_single_class()
        
        for split in ["train", "val"]:
            dataset_name = f"weapon_{split}"
            
            # Clear any existing registration first
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
            
            # New registration
            DatasetCatalog.register(
                dataset_name,
                lambda s=split: load_weapon_dataset(s)
            )
            MetadataCatalog.get(dataset_name).set(thing_classes=["weapon"])
        
        print("[INFO] Datasets registered: weapon_train, weapon_val")
    
    def setup_config(
        self,
        max_iter: int = 3000,
        batch_size: int = 4,
        base_lr: float = 0.001,
        num_workers: int = 4,
        checkpoint_period: int = 500,
        eval_period: int = 500
    ):
        """
        Set up the model configuration
        """
        self.cfg = get_cfg()
        
        # Model configuration
        config_file = self.model_configs.get(self.backbone, self.model_configs["R50-FPN"])
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        
        # Pretrained weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        
        # Dataset
        self.cfg.DATASETS.TRAIN = ("weapon_train",)
        self.cfg.DATASETS.TEST = ("weapon_val",)
        
        # Training parameters
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
        self.cfg.SOLVER.WARMUP_ITERS = int(max_iter * 0.1)
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        self.cfg.SOLVER.WARMUP_FACTOR = 0.001
        
        # Model parameters
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # weapon only
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Test
        self.cfg.TEST.EVAL_PERIOD = eval_period
        
        # Output
        self.cfg.OUTPUT_DIR = str(OUTPUTS_DIR / "faster_rcnn")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
        # Device
        self.cfg.MODEL.DEVICE = self.device
        
        print(f"[INFO] Configuration complete:")
        print(f"  Backbone: {self.backbone}")
        print(f"  Max Iterations: {max_iter}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {base_lr}")
        
        return self.cfg
    
    def train(
        self,
        max_iter: int = 3000,
        batch_size: int = 4,
        base_lr: float = 0.001,
        resume: bool = False
    ) -> Dict:
        """
        Train the model
        
        Returns:
            Training results
        """
        if self.cfg is None:
            self.setup_config(max_iter=max_iter, batch_size=batch_size, base_lr=base_lr)
        
        print(f"\n{'='*60}")
        print(f"[TRAIN] Faster R-CNN ({self.backbone}) training is starting")
        print(f"{'='*60}\n")
        
        # Resume check
        if resume:
            last_checkpoint = Path(self.cfg.OUTPUT_DIR) / "last_checkpoint"
            if last_checkpoint.exists():
                with open(last_checkpoint) as f:
                    self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, f.read().strip())
                print(f"[INFO] Resuming training: {self.cfg.MODEL.WEIGHTS}")
        
        # Training start time
        start_time = time.time()
        
        # Trainer
        trainer = CustomTrainer(self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()
        
        # Training duration
        self.training_time = time.time() - start_time
        
        # Save the best model
        final_model = Path(self.cfg.OUTPUT_DIR) / "model_final.pth"
        if final_model.exists():
            save_path = MODELS_DIR / f"faster_rcnn_{self.backbone.lower()}_weapon.pth"
            import shutil
            shutil.copy(final_model, save_path)
            print(f"[INFO] Model saved: {save_path}")
        
        # Collect the results
        training_results = {
            "model": f"Faster R-CNN ({self.backbone})",
            "framework": "Detectron2",
            "max_iterations": max_iter,
            "batch_size": batch_size,
            "learning_rate": base_lr,
            "training_time_seconds": self.training_time,
            "training_time_formatted": self._format_time(self.training_time),
            "device": self.device,
            "model_path": str(save_path) if final_model.exists() else None,
            "output_dir": str(self.cfg.OUTPUT_DIR)
        }
        
        print(f"\n[TRAIN] Training finished!")
        print(f"[TRAIN] Duration: {training_results['training_time_formatted']}")
        
        return training_results
    
    def load_model(self, weights_path: Optional[str] = None):
        """Load the model"""
        if self.cfg is None:
            self.setup_config()
        
        if weights_path and Path(weights_path).exists():
            self.cfg.MODEL.WEIGHTS = weights_path
        else:
            # Load the most recently trained model
            best_model = MODELS_DIR / f"faster_rcnn_{self.backbone.lower()}_weapon.pth"
            if best_model.exists():
                self.cfg.MODEL.WEIGHTS = str(best_model)
            else:
                final_model = Path(self.cfg.OUTPUT_DIR) / "model_final.pth"
                if final_model.exists():
                    self.cfg.MODEL.WEIGHTS = str(final_model)
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(self.cfg)
        
        print(f"[INFO] Model loaded: {self.cfg.MODEL.WEIGHTS}")
        
        return self.predictor
    
    def validate(self, weights_path: Optional[str] = None) -> Dict:
        """
        Evaluate on the validation set
        
        Returns:
            Validation metrics
        """
        if self.cfg is None:
            self.setup_config()
        
        if weights_path:
            self.cfg.MODEL.WEIGHTS = weights_path
        else:
            best_model = MODELS_DIR / f"faster_rcnn_{self.backbone.lower()}_weapon.pth"
            if best_model.exists():
                self.cfg.MODEL.WEIGHTS = str(best_model)
            else:
                final_model = Path(self.cfg.OUTPUT_DIR) / "model_final.pth"
                if final_model.exists():
                    self.cfg.MODEL.WEIGHTS = str(final_model)
        
        print(f"\n{'='*60}")
        print(f"[VAL] Validation is starting")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Evaluator
        evaluator = COCOEvaluator("weapon_val", output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, "weapon_val")
        
        # Create the predictor
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(self.cfg)
        
        # Evaluation
        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        val_time = time.time() - start_time
        
        # COCO bbox metrics (Detectron2 default scale: 0-100)
        # NOTE: we scale these metrics down to 0-1 so they stay consistent
        # with the other pipelines (avoids the ×100 mix-up in the previous version).
        bbox_metrics = results.get("bbox", {})

        def _to_unit(v: float) -> float:
            """Detectron2 values are on a 0-100 scale; convert to 0-1."""
            try:
                return float(v) / 100.0 if v is not None else 0.0
            except Exception:
                return 0.0

        # Real precision/recall/F1 from a confidence sweep
        # (the CORRECT fix for the old version's precision=AP, recall=0 bug).
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from eval.metrics import confidence_sweep, evaluate_coco
            from pycocotools.coco import COCO

            preds_path = Path(self.cfg.OUTPUT_DIR) / "coco_instances_results.json"
            val_json_path = ANNOTATIONS_DIR / "val.json" if (PROJECT_ROOT / "annotations" / "val.json").exists() else None
            if preds_path.exists() and val_json_path and val_json_path.exists():
                preds = json.loads(preds_path.read_text())
                coco_gt = COCO(str(val_json_path))
                gt_per_image: Dict[int, list] = {img_id: [] for img_id in coco_gt.imgs}
                for ann in coco_gt.anns.values():
                    gt_per_image[ann["image_id"]].append(ann["bbox"])
                sweep = confidence_sweep(gt_per_image, preds, iou_thr=0.5)
                precision = sweep["best_precision"]
                recall = sweep["best_recall"]
                f1_score = sweep["best_f1"]
                best_thr = sweep["best_threshold"]
            else:
                print("[WARN] coco_instances_results.json or val.json is missing; "
                      "P/R/F1 could not be computed.")
                precision = recall = f1_score = best_thr = 0.0
        except Exception as e:
            print(f"[WARN] P/R/F1 calculation failed: {e}")
            precision = recall = f1_score = best_thr = 0.0

        metrics = {
            "scale": "0-1",
            "mAP50": _to_unit(bbox_metrics.get("AP50", 0.0)),
            "mAP50_95": _to_unit(bbox_metrics.get("AP", 0.0)),
            "mAP75": _to_unit(bbox_metrics.get("AP75", 0.0)),
            "mAP_small": _to_unit(bbox_metrics.get("APs", 0.0)),
            "mAP_medium": _to_unit(bbox_metrics.get("APm", 0.0)),
            "mAP_large": _to_unit(bbox_metrics.get("APl", 0.0)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "best_confidence_threshold": best_thr,
            "validation_time_seconds": val_time,
            "validation_time_formatted": self._format_time(val_time),
        }

        print(f"\n[VAL] Validation results (0-1 scale):")
        print(f"  mAP@50:        {metrics['mAP50']:.4f}")
        print(f"  mAP@[.5:.95]:  {metrics['mAP50_95']:.4f}")
        print(f"  mAP@75:        {metrics['mAP75']:.4f}")
        print(f"  Precision*:    {metrics['precision']:.4f}")
        print(f"  Recall*:       {metrics['recall']:.4f}")
        print(f"  F1 score*:     {metrics['f1_score']:.4f}  "
              f"(best thr={metrics['best_confidence_threshold']:.2f})")
        print("  *at the F1-optimal threshold (confidence sweep)")

        return metrics
    
    def test_real_world(self, save_visualizations: bool = True) -> Dict:
        """
        Test on real-world test images
        
        Returns:
            Test results
        """
        if self.predictor is None:
            self.load_model()
        
        print(f"\n{'='*60}")
        print(f"[TEST] Real-world test is starting")
        print(f"{'='*60}\n")
        
        # Find test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(TEST_IMAGES_DIR.glob(ext))
        
        if not test_images:
            print(f"[WARN] No test image found: {TEST_IMAGES_DIR}")
            return {}
        
        print(f"[INFO] Found {len(test_images)} test images")
        
        results_list = []
        total_inference_time = 0
        
        # Results folder
        vis_dir = RESULTS_DIR / "faster_rcnn_predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        metadata = MetadataCatalog.get("weapon_train")
        
        for img_path in test_images:
            print(f"\n[TEST] Processing: {img_path.name}")
            
            # Read the image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [ERROR] Could not read the image")
                continue
            
            # Inference
            start_time = time.time()
            outputs = self.predictor(img)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            self.inference_times.append(inference_time)
            
            # Process the results
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            
            detections = []
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(scores[i]),
                    "class_id": int(classes[i]),
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
            if save_visualizations:
                v = Visualizer(
                    img[:, :, ::-1],
                    metadata=metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE
                )
                out = v.draw_instance_predictions(instances)
                
                # Save
                save_path = vis_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1])
        
        # Overall statistics
        avg_inference = (total_inference_time / len(results_list)) * 1000 if results_list else 0
        fps = 1000 / avg_inference if avg_inference > 0 else 0
        
        test_summary = {
            "model": f"Faster R-CNN ({self.backbone})",
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
        conf: float = 0.5,
        save: bool = True
    ) -> List[Dict]:
        """
        Run prediction on a single image
        
        Args:
            source: Image path
            conf: Confidence threshold
            save: Save the results
        
        Returns:
            Prediction results
        """
        if self.predictor is None:
            self.load_model()
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
        self.predictor = DefaultPredictor(self.cfg)
        
        img = cv2.imread(source)
        if img is None:
            return []
        
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        
        detections = []
        for i in range(len(boxes)):
            det = {
                "bbox": boxes[i].tolist(),
                "confidence": float(scores[i]),
                "class_name": "weapon"
            }
            detections.append(det)
        
        if save:
            metadata = MetadataCatalog.get("weapon_train")
            v = Visualizer(
                img[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            out = v.draw_instance_predictions(instances)
            
            save_dir = RESULTS_DIR / "faster_rcnn_predictions" / "custom"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"pred_{Path(source).name}"
            cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1])
        
        return [{"path": source, "detections": detections}]
    
    def generate_benchmark(self) -> Dict:
        """
        Build a full benchmark report
        
        Returns:
            Benchmark results
        """
        print(f"\n{'='*60}")
        print(f"[BENCHMARK] Faster R-CNN ({self.backbone}) Benchmark")
        print(f"{'='*60}\n")
        
        # Validation metrikleri
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
        model_info = {
            "model_name": f"Faster R-CNN ({self.backbone})",
            "framework": "Detectron2",
            "framework_version": detectron2.__version__ if DETECTRON2_AVAILABLE else "N/A",
            "backbone": self.backbone,
            "input_size": 800,
            "device": self.device
        }
        
        # Benchmark report (v2 schema: scale=0-1, keys are snake_case)
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
                "mAP50_95": val_metrics.get("mAP50_95", 0),
                "precision": val_metrics.get("precision", 0),
                "recall": val_metrics.get("recall", 0),
                "f1_score": val_metrics.get("f1_score", 0),
                "avg_inference_ms": test_results.get("avg_inference_time_ms", 0),
                "fps": test_results.get("fps", 0)
            }
        }
        
        # Save the benchmark
        benchmark_path = BENCHMARK_DIR / "faster_rcnn_benchmark.json"
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=4, ensure_ascii=False)
        
        print(f"\n[BENCHMARK] Report saved: {benchmark_path}")
        
        return benchmark
    
    def _format_time(self, seconds: float) -> str:
        """Format a duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Faster R-CNN Weapon Detection")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test", "predict", "benchmark", "all"],
                       default="all", help="Run mode")
    parser.add_argument("--backbone", type=str, default="R50-FPN", 
                       choices=["R50-FPN", "R101-FPN", "X101-FPN"],
                       help="Backbone network")
    parser.add_argument("--max-iter", type=int, default=3000, help="Maximum iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weights", type=str, default=None, help="Custom model weights")
    parser.add_argument("--source", type=str, default=None, help="Image path for prediction")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (reproducibility)")

    args = parser.parse_args()

    if not DETECTRON2_AVAILABLE:
        print("[ERROR] Detectron2 is not installed! Please install it.")
        print("Install: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        sys.exit(1)

    set_global_seed(args.seed)
    print(f"[INFO] Global seed: {args.seed}")

    # Create the detector
    detector = FasterRCNNWeaponDetector(backbone=args.backbone)
    
    if args.mode == "train" or args.mode == "all":
        print("\n" + "="*60)
        print("FASTER R-CNN TRAINING")
        print("="*60)
        
        training_results = detector.train(
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            base_lr=args.lr,
            resume=args.resume
        )
        
        # Save the training results
        results_path = BENCHMARK_DIR / "faster_rcnn_training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, indent=4, ensure_ascii=False)
    
    if args.mode == "val" or args.mode == "all":
        print("\n" + "="*60)
        print("FASTER R-CNN VALIDATION")
        print("="*60)
        
        val_metrics = detector.validate(args.weights)
    
    if args.mode == "test" or args.mode == "all":
        print("\n" + "="*60)
        print("FASTER R-CNN REAL-WORLD TEST")
        print("="*60)
        
        detector.load_model(args.weights)
        test_results = detector.test_real_world(save_visualizations=True)
    
    if args.mode == "predict" and args.source:
        print("\n" + "="*60)
        print("FASTER R-CNN PREDICTION")
        print("="*60)
        
        predictions = detector.predict(args.source, conf=args.conf)
        for pred in predictions:
            print(f"\n{pred['path']}:")
            for det in pred['detections']:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    
    if args.mode == "benchmark" or args.mode == "all":
        print("\n" + "="*60)
        print("FASTER R-CNN BENCHMARK")
        print("="*60)
        
        benchmark = detector.generate_benchmark()
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"  Model: {benchmark['model_info'].get('model_name', 'N/A')}")
        print(f"  mAP@50: {benchmark['performance_summary']['mAP50']:.4f}")
        print(f"  mAP@50-95: {benchmark['performance_summary']['mAP50_95']:.4f}")
        print(f"  Inference: {benchmark['performance_summary']['avg_inference_ms']:.2f} ms")
        print(f"  FPS: {benchmark['performance_summary']['fps']:.2f}")
    
    print("\n" + "="*60)
    print("FINISHED")
    print("="*60)


if __name__ == "__main__":
    main()
