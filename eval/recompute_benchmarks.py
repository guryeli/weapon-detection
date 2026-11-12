"""
Using the existing model checkpoints and Faster R-CNN's saved predictions,
produce **fair**, **correct**, and **comparable** benchmark JSON files
for both models.

Problem spots (in the previous version):
- Faster R-CNN: precision=AP, recall=0, F1=2P²/2P → completely meaningless.
- benchmark_comparison.py applied the same *100 to both models;
  because of the Detectron2 (0-100) vs Ultralytics (0-1) scale difference,
  Faster R-CNN values looked absurd, such as 9501%.

This script recomputes the results:
1. On the same val set (143 images, 219 annotations, 1 class=weapon)
2. With the same protocol (pycocotools mAP@[0.5:0.95], confidence sweep
   for F1-optimal precision/recall/F1)
3. With the same speed protocol (warmup + N runs + median + IQR)

Outputs (all on a 0-1 scale; converted to percent in the reporting layer):
- benchmark/yolov8_benchmark_v2.json
- benchmark/faster_rcnn_benchmark_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    confidence_sweep,
    evaluate_coco,
    measure_speed,
)


def set_seed(seed: int = 42) -> None:
    """Global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def infer_yolo_model_name(weights_path: Path | str, n_parameters: int | None = None) -> str:
    """
    Infer the model size from the YOLO weight filename (or the parameter count).
    Example: 'yolov8l_weapon_best.pt' → 'YOLOv8L'.
    """
    name = Path(weights_path).stem.lower()

    for size in ("x", "l", "m", "s", "n"):
        if f"yolov8{size}" in name:
            return f"YOLOv8{size.upper()}"

    if n_parameters is not None:
        millions = n_parameters / 1_000_000
        if millions < 5:
            return "YOLOv8N"
        if millions < 15:
            return "YOLOv8S"
        if millions < 30:
            return "YOLOv8M"
        if millions < 55:
            return "YOLOv8L"
        return "YOLOv8X"

    return "YOLOv8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def select_device() -> str:
    """Order: Apple MPS > CUDA > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# -------------------------------------------------------------------------
# Load GT + normalize paths
# -------------------------------------------------------------------------

def load_val_groundtruth(val_json_path: Path, val_images_dir: Path) -> Dict:
    """Read val.json; rewrite file_name values to the local basename;
    return image_id -> bbox list.

    Returns:
        {
          "json_path": <path of the normalized copy>,
          "gt_per_image": {image_id: [[x,y,w,h], ...]},
          "id_to_path": {image_id: Path},
          "n_images": int, "n_annotations": int,
        }
    """
    raw = json.loads(val_json_path.read_text())
    fixed_images = []
    id_to_path: Dict[int, Path] = {}
    for img in raw["images"]:
        basename = Path(img["file_name"]).name
        local_path = val_images_dir / basename
        if not local_path.exists():
            # .jpeg / .jpg mismatches can happen
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                cand = val_images_dir / (Path(basename).stem + ext)
                if cand.exists():
                    local_path = cand
                    break
        new = dict(img)
        new["file_name"] = basename
        fixed_images.append(new)
        id_to_path[img["id"]] = local_path

    fixed = dict(raw)
    fixed["images"] = fixed_images

    out_path = PROJECT_ROOT / "annotations" / "val_local.json"
    out_path.write_text(json.dumps(fixed))

    gt_per_image: Dict[int, List[Sequence[float]]] = {img["id"]: [] for img in fixed_images}
    for ann in raw["annotations"]:
        gt_per_image[ann["image_id"]].append(ann["bbox"])

    return {
        "json_path": str(out_path),
        "gt_per_image": gt_per_image,
        "id_to_path": id_to_path,
        "n_images": len(fixed_images),
        "n_annotations": len(raw["annotations"]),
    }


# -------------------------------------------------------------------------
# YOLOv8 inference -> COCO predictions
# -------------------------------------------------------------------------

def run_yolov8_inference(
    weights_path: Path,
    id_to_path: Dict[int, Path],
    device: str,
    conf_min: float = 0.01,
    iou_nms: float = 0.6,
    imgsz: int = 640,
) -> Dict:
    """Run the YOLOv8 model on every val image;
    collect all candidate detections at a very low conf threshold (0.01)
    (so a sweep is possible).
    """
    from ultralytics import YOLO

    print(f"[YOLO] Loading model: {weights_path.name} (device={device})")
    model = YOLO(str(weights_path))
    # parameter count
    n_params = 0
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        pass

    predictions: List[dict] = []
    n_images = len(id_to_path)
    print(f"[YOLO] Inference on {n_images} val images...")

    # Pass a list for faster inference, but keep the image_id mapping
    sorted_ids = sorted(id_to_path.keys())
    paths = [str(id_to_path[i]) for i in sorted_ids]

    # batched prediction
    results_iter = model.predict(
        source=paths,
        conf=conf_min,
        iou=iou_nms,
        imgsz=imgsz,
        device=device,
        verbose=False,
        stream=True,
    )

    for img_id, res in zip(sorted_ids, results_iter):
        if res.boxes is None:
            continue
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            score = float(b.conf[0])
            cls = int(b.cls[0])
            x1, y1, x2, y2 = xyxy
            predictions.append({
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })

    print(f"[YOLO] Collected {len(predictions)} predictions in total (conf>={conf_min})")

    return {
        "predictions": predictions,
        "n_parameters": n_params,
        "imgsz": imgsz,
        "weights_path": str(weights_path),
    }


# -------------------------------------------------------------------------
# Speed: YOLO
# -------------------------------------------------------------------------

def measure_yolov8_speed(
    weights_path: Path,
    id_to_path: Dict[int, Path],
    device: str,
    imgsz: int = 640,
    n_warmup: int = 5,
    n_runs: int = 30,
) -> Dict[str, float]:
    """Measure single-image inference time."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    sample_ids = sorted(id_to_path.keys())[:max(n_warmup, n_runs)]
    images = [cv2.imread(str(id_to_path[i])) for i in sample_ids]
    images = [im for im in images if im is not None]

    def predict_fn(img):
        model.predict(source=img, imgsz=imgsz, device=device,
                      conf=0.25, iou=0.45, verbose=False)

    return measure_speed(predict_fn, images, warmup=n_warmup, runs=n_runs)


# -------------------------------------------------------------------------
# Faster R-CNN: predictions are ready; only path/preds check + speed
# -------------------------------------------------------------------------

def load_frcnn_predictions(coco_results_path: Path) -> List[dict]:
    """Read the coco_instances_results.json written by Detectron2.
    Each item: {image_id, category_id, bbox(xywh), score}
    """
    if not coco_results_path.exists():
        return []
    return json.loads(coco_results_path.read_text())


def measure_frcnn_speed(
    weights_path: Path,
    id_to_path: Dict[int, Path],
    device: str,
    n_warmup: int = 5,
    n_runs: int = 30,
) -> Dict[str, float]:
    try:
        import cv2
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
    except ImportError:
        print("[FRCNN] detectron2 could not be imported; skipping the speed measurement.")
        return {}

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device if device != "mps" else "cpu"
    predictor = DefaultPredictor(cfg)

    sample_ids = sorted(id_to_path.keys())[:max(n_warmup, n_runs)]
    images = [cv2.imread(str(id_to_path[i])) for i in sample_ids]
    images = [im for im in images if im is not None]

    def predict_fn(img):
        predictor(img)

    return measure_speed(predict_fn, images, warmup=n_warmup, runs=n_runs)


def count_frcnn_parameters(weights_path: Path) -> int:
    try:
        import torch
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        if isinstance(sd, dict):
            return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    except Exception as e:
        print(f"[FRCNN] skipping parameter count: {e}")
    return 0


# -------------------------------------------------------------------------
# Benchmark JSON builder
# -------------------------------------------------------------------------

def build_benchmark_record(
    *,
    model_name: str,
    framework: str,
    framework_version: str,
    device: str,
    weights_path: str,
    n_parameters: int,
    input_size: int,
    coco_metrics: Dict[str, float],
    sweep: Dict[str, object],
    speed: Dict[str, float],
    val_meta: Dict,
    extra: Dict,
) -> Dict:
    """Benchmark dict in one standard format.
    All AP/precision/recall/F1 values are stored on a **0-1** scale.
    """
    return {
        "schema_version": 2,
        "scale": "0-1",  # **critical**: the consumer (comparison) reads this
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_info": {
            "model_name": model_name,
            "framework": framework,
            "framework_version": framework_version,
            "device": device,
            "weights_path": weights_path,
            "parameters": n_parameters,
            "input_size": input_size,
        },
        "validation": {
            "n_images": val_meta["n_images"],
            "n_annotations": val_meta["n_annotations"],
            "iou_threshold_for_pr": 0.5,
            # COCO-style mAP (0-1 scale)
            "mAP50": coco_metrics.get("AP50", 0.0),
            "mAP50_95": coco_metrics.get("AP", 0.0),
            "mAP75": coco_metrics.get("AP75", 0.0),
            "mAP_small": coco_metrics.get("APs", 0.0),
            "mAP_medium": coco_metrics.get("APm", 0.0),
            "mAP_large": coco_metrics.get("APl", 0.0),
            # Confidence sweep / F1-optimal point
            "best_confidence_threshold": sweep["best_threshold"],
            "precision_at_best_f1": sweep["best_precision"],
            "recall_at_best_f1": sweep["best_recall"],
            "f1_at_best_f1": sweep["best_f1"],
            "n_predictions_total": sweep["n_pred"],
            "n_ground_truth": sweep["n_gt"],
            "sweep": {
                "thresholds": sweep["thresholds"],
                "precision": sweep["precision"],
                "recall": sweep["recall"],
                "f1": sweep["f1"],
                "edge_warning": sweep.get("edge_warning", ""),
                "f1_plateau_min_thr": sweep.get("f1_plateau_min_thr", sweep["best_threshold"]),
                "f1_plateau_max_thr": sweep.get("f1_plateau_max_thr", sweep["best_threshold"]),
                "operating_points": sweep.get("operating_points", {}),
            },
        },
        "speed": speed or {},
        "performance_summary": {
            "mAP50": coco_metrics.get("AP50", 0.0),
            "mAP50_95": coco_metrics.get("AP", 0.0),
            "precision": sweep["best_precision"],
            "recall": sweep["best_recall"],
            "f1_score": sweep["best_f1"],
            "fps_median": speed.get("fps_median", 0.0) if speed else 0.0,
            "median_inference_ms": speed.get("median_ms", 0.0) if speed else 0.0,
        },
        "extra": extra,
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair benchmark recomputation for YOLOv8 and Faster R-CNN")
    parser.add_argument("--yolo-weights",
                        default=str(PROJECT_ROOT / "models" / "yolov8n_weapon_best.pt"))
    parser.add_argument("--frcnn-weights",
                        default=str(PROJECT_ROOT / "models" / "faster_rcnn_r50-fpn_weapon.pth"))
    parser.add_argument("--frcnn-preds",
                        default=str(PROJECT_ROOT / "outputs" / "faster_rcnn" / "coco_instances_results.json"))
    parser.add_argument("--val-json",
                        default=str(PROJECT_ROOT / "annotations" / "val.json"))
    parser.add_argument("--val-images",
                        default=str(PROJECT_ROOT / "archive" / "weapon_detection" / "val" / "images"))
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | mps | cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-frcnn", action="store_true")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip the speed measurement (only accuracy is refreshed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device() if args.device == "auto" else args.device
    print(f"[INFO] Device: {device}, seed={args.seed}")

    # Load the shared ground truth
    val_json = Path(args.val_json)
    val_images = Path(args.val_images)
    print(f"[INFO] Loading GT: {val_json}")
    val_meta = load_val_groundtruth(val_json, val_images)
    print(f"[INFO] val: {val_meta['n_images']} images, "
          f"{val_meta['n_annotations']} annotations")

    benchmarks_dir = PROJECT_ROOT / "benchmark"
    benchmarks_dir.mkdir(exist_ok=True)

    # ---- YOLOv8 ----
    if not args.skip_yolo:
        yolo_w = Path(args.yolo_weights)
        if not yolo_w.exists():
            print(f"[YOLO][SKIP] Weights not found: {yolo_w}")
        else:
            print(f"\n{'='*60}\n[YOLO] benchmark starting\n{'='*60}")
            t0 = time.perf_counter()
            yolo_inf = run_yolov8_inference(
                weights_path=yolo_w,
                id_to_path=val_meta["id_to_path"],
                device=device, imgsz=args.imgsz,
            )
            print(f"[YOLO] inference time: {time.perf_counter() - t0:.1f} s")

            print("[YOLO] Computing COCO mAP...")
            yolo_coco = evaluate_coco(val_meta["json_path"], yolo_inf["predictions"])
            print(f"[YOLO] mAP50={yolo_coco['AP50']:.4f}  "
                  f"mAP=[0.5:0.95]={yolo_coco['AP']:.4f}")

            print("[YOLO] confidence sweep (P/R/F1)...")
            yolo_sweep = confidence_sweep(
                gt_per_image=val_meta["gt_per_image"],
                predictions=yolo_inf["predictions"],
                iou_thr=0.5,
            )
            print(f"[YOLO] best F1={yolo_sweep['best_f1']:.4f} @ "
                  f"thr={yolo_sweep['best_threshold']:.2f}  "
                  f"(P={yolo_sweep['best_precision']:.4f}, "
                  f"R={yolo_sweep['best_recall']:.4f})")

            yolo_speed = {}
            if not args.skip_speed:
                print("[YOLO] measuring speed...")
                yolo_speed = measure_yolov8_speed(
                    weights_path=yolo_w,
                    id_to_path=val_meta["id_to_path"],
                    device=device, imgsz=args.imgsz,
                    n_warmup=args.n_warmup, n_runs=args.n_runs,
                )
                print(f"[YOLO] median={yolo_speed['median_ms']:.2f} ms "
                      f"({yolo_speed['fps_median']:.1f} FPS)")

            yolo_model_name = infer_yolo_model_name(yolo_w, yolo_inf["n_parameters"])
            print(f"[YOLO] detected model: {yolo_model_name}")
            yolo_record = build_benchmark_record(
                model_name=yolo_model_name,
                framework="Ultralytics",
                framework_version=_get_pkg_version("ultralytics"),
                device=device,
                weights_path=str(yolo_w),
                n_parameters=yolo_inf["n_parameters"],
                input_size=args.imgsz,
                coco_metrics=yolo_coco,
                sweep=yolo_sweep,
                speed=yolo_speed,
                val_meta=val_meta,
                extra={"nms_iou": 0.6, "min_conf_for_sweep": 0.01},
            )
            out_path = benchmarks_dir / "yolov8_benchmark_v2.json"
            out_path.write_text(json.dumps(yolo_record, indent=2))
            print(f"[YOLO] wrote: {out_path}")

    # ---- Faster R-CNN ----
    if not args.skip_frcnn:
        frcnn_preds_path = Path(args.frcnn_preds)
        frcnn_w = Path(args.frcnn_weights)
        if not frcnn_w.exists():
            # fallback: outputs/faster_rcnn/model_final.pth
            alt = PROJECT_ROOT / "outputs" / "faster_rcnn" / "model_final.pth"
            if alt.exists():
                frcnn_w = alt

        if not frcnn_preds_path.exists():
            print(f"[FRCNN][SKIP] predictions missing: {frcnn_preds_path}")
        else:
            print(f"\n{'='*60}\n[FRCNN] benchmark starting\n{'='*60}")
            frcnn_preds = load_frcnn_predictions(frcnn_preds_path)
            print(f"[FRCNN] ready prediction count: {len(frcnn_preds)}")

            print("[FRCNN] Computing COCO mAP...")
            frcnn_coco = evaluate_coco(val_meta["json_path"], frcnn_preds)
            print(f"[FRCNN] mAP50={frcnn_coco['AP50']:.4f}  "
                  f"mAP=[0.5:0.95]={frcnn_coco['AP']:.4f}")

            print("[FRCNN] confidence sweep (P/R/F1)...")
            frcnn_sweep = confidence_sweep(
                gt_per_image=val_meta["gt_per_image"],
                predictions=frcnn_preds,
                iou_thr=0.5,
            )
            print(f"[FRCNN] best F1={frcnn_sweep['best_f1']:.4f} @ "
                  f"thr={frcnn_sweep['best_threshold']:.2f}  "
                  f"(P={frcnn_sweep['best_precision']:.4f}, "
                  f"R={frcnn_sweep['best_recall']:.4f})")

            frcnn_speed = {}
            if not args.skip_speed and frcnn_w.exists():
                print("[FRCNN] measuring speed (this can take a while)...")
                frcnn_speed = measure_frcnn_speed(
                    weights_path=frcnn_w,
                    id_to_path=val_meta["id_to_path"],
                    device=device,
                    n_warmup=args.n_warmup, n_runs=args.n_runs,
                )
                if frcnn_speed:
                    print(f"[FRCNN] median={frcnn_speed['median_ms']:.2f} ms "
                          f"({frcnn_speed['fps_median']:.1f} FPS)")

            n_params_frcnn = count_frcnn_parameters(frcnn_w) if frcnn_w.exists() else 0

            frcnn_record = build_benchmark_record(
                model_name="Faster R-CNN (R50-FPN)",
                framework="Detectron2",
                framework_version=_get_pkg_version("detectron2"),
                device=device,
                weights_path=str(frcnn_w),
                n_parameters=n_params_frcnn,
                input_size=800,
                coco_metrics=frcnn_coco,
                sweep=frcnn_sweep,
                speed=frcnn_speed,
                val_meta=val_meta,
                extra={"score_thresh_test": 0.05},
            )
            out_path = benchmarks_dir / "faster_rcnn_benchmark_v2.json"
            out_path.write_text(json.dumps(frcnn_record, indent=2))
            print(f"[FRCNN] wrote: {out_path}")

    print("\n[OK] Benchmark refreshed.")
    print("For the comparison:")
    print("  python benchmark_comparison.py --schema 2")


def _get_pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
