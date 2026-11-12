"""
Shared evaluation metrics — a fair comparison of YOLOv8 and Faster R-CNN.

Design choices (academic thesis context):
- COCO mAP/AP metrics via pycocotools (the literature standard)
- Precision/recall/F1 from a confidence-threshold sweep (for a single-number
  report the F1-optimal threshold is selected, so both models are compared
  at their best operating point)
- Speed: warmup + N runs + median/IQR (the mean alone is misleading)
- All values are returned on a **0-1 scale** (percent conversion happens in the reporting layer)
"""
from __future__ import annotations

import json
import time
import statistics
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# COCO mAP (pycocotools)
# --------------------------------------------------------------------------

def evaluate_coco(gt_json: str, predictions: List[dict]) -> Dict[str, float]:
    """Compute AP metrics in COCO format.

    Args:
        gt_json: COCO ground-truth JSON path
        predictions: COCO results list (image_id, category_id, bbox[xywh], score)

    Returns:
        {AP, AP50, AP75, APs, APm, APl} — all on a 0-1 scale
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json)
    if not predictions:
        return {k: 0.0 for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")}
    coco_dt = coco_gt.loadRes(predictions)

    # Swallow pycocotools verbose output
    buf = StringIO()
    with redirect_stdout(buf):
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # 12 values (COCO order)
    return {
        "AP": float(stats[0]),       # mAP @ [0.5:0.95]
        "AP50": float(stats[1]),     # mAP @ 0.50
        "AP75": float(stats[2]),     # mAP @ 0.75
        "APs": float(stats[3]),      # small (area<32^2)
        "APm": float(stats[4]),      # medium
        "APl": float(stats[5]),      # large
    }


# --------------------------------------------------------------------------
# Detection-level Precision/Recall/F1 (confidence sweep)
# --------------------------------------------------------------------------

def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def confidence_sweep(
    gt_per_image: Dict[int, List[Sequence[float]]],
    predictions: List[dict],
    iou_thr: float = 0.5,
    n_thresholds: int = 75,
    conf_min: float = 0.05,
    conf_max: float = 0.99,
) -> Dict[str, object]:
    """Sweep every prediction by confidence threshold and run
    per-image greedy IoU matching; emit precision/recall/F1 at each threshold.

    Args:
        gt_per_image: {image_id: [[x,y,w,h], ...]}
        predictions: COCO format [{image_id, bbox[xywh], score, ...}]
        iou_thr: TP threshold
        n_thresholds: number of thresholds to sweep (default 75)
        conf_min: lower bound of the sweep (default 0.05)
        conf_max: upper bound of the sweep (default 0.99 — wider than the old 0.95)

    Returns:
        {
          "thresholds": [...], "precision": [...], "recall": [...], "f1": [...],
          "best_threshold", "best_f1", "best_precision", "best_recall",
          "n_gt", "n_pred"
        }
    """
    if n_thresholds < 2:
        n_thresholds = 2
    thresholds = np.linspace(conf_min, conf_max, n_thresholds).tolist()
    n_gt = sum(len(v) for v in gt_per_image.values())

    out = {"thresholds": thresholds, "precision": [], "recall": [], "f1": [],
           "n_gt": n_gt, "n_pred": len(predictions)}

    # image_id -> sorted prediction list (highest score first)
    preds_by_img: Dict[int, List[dict]] = {}
    for p in predictions:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    for v in preds_by_img.values():
        v.sort(key=lambda d: d["score"], reverse=True)

    for thr in thresholds:
        tp = fp = 0
        # greedy match for each image
        for img_id, gts in gt_per_image.items():
            preds = [p for p in preds_by_img.get(img_id, []) if p["score"] >= thr]
            matched = [False] * len(gts)
            for p in preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gts):
                    if matched[j]:
                        continue
                    iou = _iou_xywh(p["bbox"], gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= iou_thr:
                    matched[best_j] = True
                    tp += 1
                else:
                    fp += 1
        # images with no predictions were already left out of the count
        # Ground truths whose image_id has no prediction are still counted as FN
        # via the n_gt - tp = fn formula
        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out["precision"].append(precision)
        out["recall"].append(recall)
        out["f1"].append(f1)

    # best F1
    f1_arr = np.array(out["f1"])
    p_arr = np.array(out["precision"])
    r_arr = np.array(out["recall"])
    thr_arr = np.array(thresholds)
    best_idx = int(np.argmax(f1_arr))
    out["best_threshold"] = float(thr_arr[best_idx])
    out["best_f1"] = float(f1_arr[best_idx])
    out["best_precision"] = float(p_arr[best_idx])
    out["best_recall"] = float(r_arr[best_idx])

    # Edge warning: report when best_thr sits at either end of the scanned range
    edge_warning = ""
    if best_idx == 0:
        edge_warning = (f"best_threshold ({thr_arr[best_idx]:.3f}) is at the LOWER bound of the sweep; "
                        f"consider lowering conf_min")
    elif best_idx == len(thresholds) - 1:
        edge_warning = (f"best_threshold ({thr_arr[best_idx]:.3f}) is at the UPPER bound of the sweep; "
                        f"the true optimum may be higher")
    out["edge_warning"] = edge_warning

    # F1 plateau: every threshold within 1% of the best F1; a reference for deployment
    plateau_mask = f1_arr >= (out["best_f1"] - 0.01)
    if plateau_mask.any():
        out["f1_plateau_min_thr"] = float(thr_arr[plateau_mask].min())
        out["f1_plateau_max_thr"] = float(thr_arr[plateau_mask].max())
    else:
        out["f1_plateau_min_thr"] = out["best_threshold"]
        out["f1_plateau_max_thr"] = out["best_threshold"]

    # Operating reference points (typical thresholds for deployment)
    out["operating_points"] = {}
    for op_thr in (0.25, 0.50, 0.75):
        # Find the nearest scanned threshold
        idx = int(np.argmin(np.abs(thr_arr - op_thr)))
        out["operating_points"][f"thr_{op_thr:.2f}"] = {
            "threshold_used": float(thr_arr[idx]),
            "precision": float(p_arr[idx]),
            "recall": float(r_arr[idx]),
            "f1": float(f1_arr[idx]),
        }

    return out


# --------------------------------------------------------------------------
# Speed measurement
# --------------------------------------------------------------------------

def measure_speed(
    predict_fn: Callable[[object], None],
    samples: Sequence[object],
    warmup: int = 5,
    runs: int = 30,
) -> Dict[str, float]:
    """Measure inference time.
    The first `warmup` calls are discarded (lazy init/caching). Median + IQR
    are taken from the remaining `runs` calls.

    Args:
        predict_fn: function that takes one input and runs it
        samples: list of input samples (used in rotation)

    Returns:
        statistics in milliseconds, plus fps
    """
    if not samples:
        return {"runs": 0, "warmup": 0,
                "median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0,
                "iqr_ms": 0.0, "fps_median": 0.0}

    # Warmup
    for i in range(warmup):
        predict_fn(samples[i % len(samples)])

    times_ms: List[float] = []
    for i in range(runs):
        s = samples[i % len(samples)]
        t0 = time.perf_counter()
        predict_fn(s)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_sorted = sorted(times_ms)
    median = statistics.median(times_ms)
    mean = statistics.fmean(times_ms)
    q1 = times_sorted[int(0.25 * len(times_sorted))]
    q3 = times_sorted[int(0.75 * len(times_sorted))]
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]

    return {
        "runs": runs,
        "warmup": warmup,
        "median_ms": float(median),
        "mean_ms": float(mean),
        "p95_ms": float(p95),
        "iqr_ms": float(q3 - q1),
        "fps_median": float(1000.0 / median) if median > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# YOLO label loader (xywh-normalized -> abs xywh)
# --------------------------------------------------------------------------

def load_yolo_gt(images_dir: Path, labels_dir: Path) -> Tuple[List[dict], Dict[int, List[Sequence[float]]], List[Path]]:
    """Read YOLO-format labels and return both COCO-style images/annotations
    and an image_id -> bbox list.
    """
    import cv2
    images_meta: List[dict] = []
    annotations: List[dict] = []
    gt_per_image: Dict[int, List[Sequence[float]]] = {}
    image_paths: List[Path] = []

    img_files = sorted([p for p in images_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    ann_id = 0
    for img_id, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        images_meta.append({"id": img_id, "file_name": img_path.name,
                            "width": w, "height": h})
        image_paths.append(img_path)
        gt_per_image[img_id] = []

        label_file = labels_dir / (img_path.stem + ".txt")
        if not label_file.exists():
            continue
        for line in label_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
            # YOLO normalized cxcywh -> abs xywh
            ax = (cx - bw / 2) * w
            ay = (cy - bh / 2) * h
            aw = bw * w
            ah = bh * h
            bbox = [ax, ay, aw, ah]
            annotations.append({"id": ann_id, "image_id": img_id,
                                "category_id": cls,
                                "bbox": bbox,
                                "area": aw * ah, "iscrowd": 0})
            gt_per_image[img_id].append(bbox)
            ann_id += 1
    return images_meta, gt_per_image, image_paths
