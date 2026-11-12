#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Benchmark Comparison Script
Comparative analysis of 2 models (YOLOv8, Faster R-CNN)

This script:
1. Reads each model's benchmark results
2. Runs a comparative analysis
3. Determines the best model
4. Builds a detailed report

Author: Weapon Detection Research Project
Date: 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# Weight factors (sum = 1.0)
METRIC_WEIGHTS = {
    "mAP50": 0.25,        # Highly important - primary accuracy metric
    "mAP50-95": 0.20,     # Important - overall accuracy
    "precision": 0.15,    # Moderately important
    "recall": 0.15,       # Moderately important
    "f1_score": 0.10,     # Low-to-moderate importance
    "fps": 0.10,          # Performance
    "inference_time": 0.05  # Speed (lower is better)
}


class BenchmarkComparator:
    """Class that compares benchmark results"""
    
    def __init__(self):
        self.benchmarks: Dict[str, Dict] = {}
        self.model_names = ["yolov8", "faster_rcnn"]
    
    def load_benchmarks(self) -> bool:
        """Load every benchmark file"""
        print(f"\n{'='*60}")
        print("LOADING BENCHMARK FILES")
        print(f"{'='*60}\n")
        
        loaded = 0

        for model_name in self.model_names:
            # Prefer the v2 file when it exists (correct scale + correct P/R/F1)
            v2_file = BENCHMARK_DIR / f"{model_name}_benchmark_v2.json"
            v1_file = BENCHMARK_DIR / f"{model_name}_benchmark.json"
            benchmark_file = v2_file if v2_file.exists() else v1_file

            if benchmark_file.exists():
                try:
                    with open(benchmark_file, 'r', encoding='utf-8') as f:
                        self.benchmarks[model_name] = json.load(f)
                    print(f"[OK] {model_name.upper()} benchmark loaded")
                    loaded += 1
                except Exception as e:
                    print(f"[ERROR] {model_name.upper()} could not be loaded: {e}")
            else:
                print(f"[WARN] {model_name.upper()} benchmark file not found: {benchmark_file}")
                # Create an empty placeholder
                self.benchmarks[model_name] = self._create_empty_benchmark(model_name)
        
        print(f"\n[INFO] {loaded}/{len(self.model_names)} benchmarks loaded")
        
        return loaded > 0
    
    def _create_empty_benchmark(self, model_name: str) -> Dict:
        """Create an empty benchmark placeholder"""
        return {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_name": model_name.upper(),
                "framework": "N/A",
                "device": "N/A"
            },
            "training": {
                "training_time_seconds": 0,
                "training_time_formatted": "00:00:00"
            },
            "validation_metrics": {
                "mAP50": 0,
                "mAP50-95": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            },
            "real_world_test": {
                "total_images": 0,
                "total_detections": 0,
                "avg_inference_time_ms": 0,
                "fps": 0
            },
            "performance_summary": {
                "mAP50": 0,
                "mAP50-95": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "avg_inference_ms": 0,
                "fps": 0
            }
        }
    
    def extract_metrics(self, benchmark: Dict) -> Dict[str, float]:
        """Extract comparison metrics from a benchmark.

        The v2 schema (`scale: "0-1"`) is converted to percent by ×100 directly.
        The v1 schema had an inconsistent scale (Detectron2 0-100, Ultralytics 0-1),
        and values already above 1.0 are assumed to be percent.
        """
        summary = benchmark.get("performance_summary", {})
        scale = benchmark.get("scale")

        def pct(key, alt_key=None):
            v = summary.get(key)
            if v is None and alt_key:
                v = summary.get(alt_key)
            if v is None:
                return 0.0
            try:
                v = float(v)
            except Exception:
                return 0.0
            if scale == "0-1":
                return v * 100.0
            # v1 / unknown: if the value is already above 1.0, assume percent
            return v if abs(v) > 1.0 else v * 100.0

        metrics = {
            "mAP50": pct("mAP50"),
            "mAP50-95": pct("mAP50_95", alt_key="mAP50-95"),
            "precision": pct("precision"),
            "recall": pct("recall"),
            "f1_score": pct("f1_score"),
            "fps": float(summary.get("fps", summary.get("fps_median", 0)) or 0),
            "inference_time": float(summary.get("avg_inference_ms",
                                                summary.get("median_inference_ms", 0)) or 0),
        }

        if scale != "0-1":
            print(f"  [WARN] {benchmark.get('model_info', {}).get('model_name', '?')}"
                  f" benchmark file has no 'scale' field; the old v1 schema was assumed."
                  f" For correct results: python eval/recompute_benchmarks.py")

        return metrics
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute a weighted score
        
        Higher score = better model
        """
        score = 0.0
        
        # Normalized values
        max_values = {
            "mAP50": 100,
            "mAP50-95": 100,
            "precision": 100,
            "recall": 100,
            "f1_score": 100,
            "fps": 200,  # Reasonable maximum FPS
            "inference_time": 100  # ms (will be normalized in reverse)
        }
        
        for metric, weight in METRIC_WEIGHTS.items():
            value = metrics.get(metric, 0)
            max_val = max_values.get(metric, 100)
            
            if metric == "inference_time":
                # Lower inference time is better
                if value > 0:
                    normalized = (max_val - min(value, max_val)) / max_val
                else:
                    normalized = 0
            else:
                # Higher value is better
                normalized = min(value, max_val) / max_val
            
            score += normalized * weight
        
        return score * 100  # score in the 0-100 range
    
    def compare_models(self) -> Dict:
        """
        Compare every model and analyze them
        
        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "metrics_comparison": {},
            "ranking": [],
            "best_model": None,
            "analysis": {}
        }
        
        all_metrics = {}
        scores = {}
        
        # Extract metrics for each model
        for model_name, benchmark in self.benchmarks.items():
            metrics = self.extract_metrics(benchmark)
            score = self.calculate_score(metrics)
            
            all_metrics[model_name] = metrics
            scores[model_name] = score
            
            comparison["models"][model_name] = {
                "model_info": benchmark.get("model_info", {}),
                "training": benchmark.get("training", {}),
                "metrics": metrics,
                "score": score
            }
        
        # Per-metric comparison
        metric_names = ["mAP50", "mAP50-95", "precision", "recall", "f1_score", "fps", "inference_time"]
        
        for metric in metric_names:
            values = {m: all_metrics[m].get(metric, 0) for m in self.model_names}
            
            if metric == "inference_time":
                # Lower is better
                best = min(values, key=values.get) if any(v > 0 for v in values.values()) else None
            else:
                # Higher is better
                best = max(values, key=values.get) if any(v > 0 for v in values.values()) else None
            
            comparison["metrics_comparison"][metric] = {
                "values": values,
                "best_model": best,
                "unit": "ms" if metric == "inference_time" else ("FPS" if metric == "fps" else "%")
            }
        
        # Ranking
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        comparison["ranking"] = [
            {"rank": i+1, "model": model, "score": score}
            for i, (model, score) in enumerate(sorted_scores)
        ]
        
        # Best model
        if sorted_scores:
            best_model_name = sorted_scores[0][0]
            comparison["best_model"] = {
                "name": best_model_name,
                "score": sorted_scores[0][1],
                "details": comparison["models"].get(best_model_name, {})
            }
        
        # Analysis
        comparison["analysis"] = self._generate_analysis(all_metrics, scores)
        
        return comparison
    
    def _generate_analysis(self, all_metrics: Dict, scores: Dict) -> Dict:
        """Build a detailed analysis"""
        analysis = {
            "summary": "",
            "strengths": {},
            "weaknesses": {},
            "recommendations": []
        }
        
        # Strengths and weaknesses of each model
        for model_name in self.model_names:
            metrics = all_metrics.get(model_name, {})
            strengths = []
            weaknesses = []
            
            # mAP50 analysis
            map50 = metrics.get("mAP50", 0)
            if map50 >= 80:
                strengths.append("High accuracy (mAP50 >= 80%)")
            elif map50 < 50:
                weaknesses.append("Low accuracy (mAP50 < 50%)")
            
            # FPS analysis
            fps = metrics.get("fps", 0)
            if fps >= 30:
                strengths.append("Real-time performance (FPS >= 30)")
            elif fps < 10 and fps > 0:
                weaknesses.append("Slow inference (FPS < 10)")
            
            # Precision-recall balance
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            
            if precision > 0 and recall > 0:
                if abs(precision - recall) < 10:
                    strengths.append("Balanced precision-recall")
                elif precision > recall + 20:
                    weaknesses.append("Low recall - may miss some weapons")
                elif recall > precision + 20:
                    weaknesses.append("Low precision - false positives are possible")
            
            analysis["strengths"][model_name] = strengths
            analysis["weaknesses"][model_name] = weaknesses
        
        # Recommendations
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_scores:
            best = sorted_scores[0][0]
            best_metrics = all_metrics.get(best, {})
            
            if best_metrics.get("mAP50", 0) >= 70 and best_metrics.get("fps", 0) >= 30:
                analysis["recommendations"].append(
                    f"The {best.upper()} model is suitable for production in both accuracy and speed."
                )
            elif best_metrics.get("mAP50", 0) >= 70:
                analysis["recommendations"].append(
                    f"The {best.upper()} model provides high accuracy, but speed optimization may be required."
                )
            elif best_metrics.get("fps", 0) >= 30:
                analysis["recommendations"].append(
                    f"The {best.upper()} model is fast, but more training data may be needed for accuracy."
                )
        
        analysis["recommendations"].append(
            "Run a thorough evaluation on the validation set before testing in real-world scenarios."
        )
        
        # Summary
        if sorted_scores:
            best = sorted_scores[0][0]
            best_score = sorted_scores[0][1]
            analysis["summary"] = (
                f"The comparison shows that the {best.upper()} model achieved the best performance "
                f"with a score of {best_score:.2f}."
            )
        
        return analysis
    
    def print_comparison_table(self, comparison: Dict):
        """Print the comparison table"""
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        
        # Header row
        header = f"{'Metric':<20}"
        for model in self.model_names:
            header += f"{model.upper():>18}"
        header += f"{'Best':>12}"
        print(header)
        print("-" * 80)
        
        # Metrics
        metrics_order = ["mAP50", "mAP50-95", "precision", "recall", "f1_score", "fps", "inference_time"]
        metric_labels = {
            "mAP50": "mAP@50 (%)",
            "mAP50-95": "mAP@50-95 (%)",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1_score": "F1 Score (%)",
            "fps": "FPS",
            "inference_time": "Inference (ms)"
        }
        
        metrics_comp = comparison.get("metrics_comparison", {})
        
        for metric in metrics_order:
            data = metrics_comp.get(metric, {})
            values = data.get("values", {})
            best = data.get("best_model", "")
            
            row = f"{metric_labels.get(metric, metric):<20}"
            for model in self.model_names:
                val = values.get(model, 0)
                if metric == "fps":
                    row += f"{val:>18.2f}"
                elif metric == "inference_time":
                    row += f"{val:>18.2f}"
                else:
                    row += f"{val:>18.2f}"
            row += f"{best.upper():>12}"
            print(row)
        
        print("-" * 80)
        
        # Scores
        print(f"\n{'OVERALL SCORE (0-100)':<20}", end="")
        for model in self.model_names:
            model_data = comparison.get("models", {}).get(model, {})
            score = model_data.get("score", 0)
            print(f"{score:>18.2f}", end="")
        print()
        
        # Ranking
        print(f"\n{'='*80}")
        print("RANKING")
        print(f"{'='*80}")
        
        for item in comparison.get("ranking", []):
            rank = item["rank"]
            model = item["model"]
            score = item["score"]
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            print(f"  {rank}. {medal} {model.upper()}: {score:.2f} points")
        
        # Best model
        best = comparison.get("best_model", {})
        if best:
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best.get('name', 'N/A').upper()}")
            print(f"SCORE: {best.get('score', 0):.2f}")
            print(f"{'='*80}")
        
        # Analysis
        analysis = comparison.get("analysis", {})
        
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")
        
        if analysis.get("summary"):
            print(f"\n{analysis['summary']}")
        
        print("\nSTRENGTHS:")
        for model, strengths in analysis.get("strengths", {}).items():
            if strengths:
                print(f"  {model.upper()}:")
                for s in strengths:
                    print(f"    ✓ {s}")
        
        print("\nWEAKNESSES:")
        for model, weaknesses in analysis.get("weaknesses", {}).items():
            if weaknesses:
                print(f"  {model.upper()}:")
                for w in weaknesses:
                    print(f"    ✗ {w}")
        
        print("\nRECOMMENDATIONS:")
        for rec in analysis.get("recommendations", []):
            print(f"  → {rec}")
    
    def save_comparison(self, comparison: Dict) -> str:
        """Save the comparison results"""
        output_path = BENCHMARK_DIR / "final_benchmark_comparison.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=4, ensure_ascii=False)
        
        print(f"\n[INFO] Comparison saved: {output_path}")
        
        return str(output_path)
    
    def generate_html_report(self, comparison: Dict) -> str:
        """Build a detailed report in HTML"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weapon Detection Model Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #4CAF50;
            border-left: 4px solid #4CAF50;
            padding-left: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f5e9;
        }}
        .best {{
            background-color: #c8e6c9 !important;
            font-weight: bold;
        }}
        .winner {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }}
        .winner h3 {{
            margin: 0;
            font-size: 24px;
        }}
        .winner p {{
            margin: 10px 0 0 0;
            font-size: 18px;
        }}
        .card {{
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .strengths {{
            border-left-color: #4CAF50;
        }}
        .weaknesses {{
            border-left-color: #f44336;
        }}
        .recommendations {{
            border-left-color: #2196F3;
        }}
        ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
        }}
        .score-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Weapon Detection Model Comparison</h1>
        
        <div class="winner">
            <h3>🏆 Best Model: {comparison.get('best_model', {}).get('name', 'N/A').upper()}</h3>
            <p>Overall Score: {comparison.get('best_model', {}).get('score', 0):.2f}/100</p>
        </div>
        
        <h2>📊 Metric Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                {"".join(f'<th>{m.upper()}</th>' for m in self.model_names)}
                <th>Best</th>
            </tr>
"""
        
        metrics_order = ["mAP50", "mAP50-95", "precision", "recall", "f1_score", "fps", "inference_time"]
        metric_labels = {
            "mAP50": "mAP@50 (%)",
            "mAP50-95": "mAP@50-95 (%)",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1_score": "F1 Score (%)",
            "fps": "FPS",
            "inference_time": "Inference (ms)"
        }
        
        metrics_comp = comparison.get("metrics_comparison", {})
        
        for metric in metrics_order:
            data = metrics_comp.get(metric, {})
            values = data.get("values", {})
            best = data.get("best_model", "")
            
            html_content += f"<tr><td><strong>{metric_labels.get(metric, metric)}</strong></td>"
            for model in self.model_names:
                val = values.get(model, 0)
                css_class = "best" if model == best else ""
                html_content += f'<td class="{css_class}">{val:.2f}</td>'
            html_content += f"<td><strong>{best.upper()}</strong></td></tr>"
        
        html_content += """
        </table>
        
        <h2>📈 Overall Scores</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Score</th>
                <th>Chart</th>
            </tr>
"""
        
        for item in comparison.get("ranking", []):
            rank = item["rank"]
            model = item["model"]
            score = item["score"]
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            html_content += f"""
            <tr>
                <td>{rank} {medal}</td>
                <td><strong>{model.upper()}</strong></td>
                <td>{score:.2f}</td>
                <td>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score}%">{score:.1f}</div>
                    </div>
                </td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>💪 Strengths</h2>
"""
        
        analysis = comparison.get("analysis", {})
        for model, strengths in analysis.get("strengths", {}).items():
            if strengths:
                html_content += f"""
        <div class="card strengths">
            <strong>{model.upper()}</strong>
            <ul>
                {"".join(f'<li>{s}</li>' for s in strengths)}
            </ul>
        </div>
"""
        
        html_content += """
        <h2>⚠️ Weaknesses</h2>
"""
        
        for model, weaknesses in analysis.get("weaknesses", {}).items():
            if weaknesses:
                html_content += f"""
        <div class="card weaknesses">
            <strong>{model.upper()}</strong>
            <ul>
                {"".join(f'<li>{w}</li>' for w in weaknesses)}
            </ul>
        </div>
"""
        
        html_content += """
        <h2>💡 Recommendations</h2>
        <div class="card recommendations">
            <ul>
"""
        
        for rec in analysis.get("recommendations", []):
            html_content += f"<li>{rec}</li>"
        
        html_content += f"""
            </ul>
        </div>
        
        <div class="timestamp">
            Report generated: {comparison.get('timestamp', 'N/A')}
        </div>
    </div>
</body>
</html>
"""
        
        output_path = BENCHMARK_DIR / "benchmark_report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[INFO] HTML report created: {output_path}")
        
        return str(output_path)


def run_all_benchmarks():
    """Train every model and build its benchmark"""
    print(f"\n{'='*60}")
    print("BUILDING BENCHMARKS FOR EVERY MODEL")
    print(f"{'='*60}\n")
    
    # YOLOv8
    print("\n[1/2] YOLOv8 Benchmark...")
    try:
        from train_yolov8 import YOLOv8WeaponDetector
        detector = YOLOv8WeaponDetector(model_size="n")
        detector.load_model()
        detector.generate_benchmark()
    except Exception as e:
        print(f"[ERROR] YOLOv8 error: {e}")
    
    # Faster R-CNN
    print("\n[2/2] Faster R-CNN Benchmark...")
    try:
        from train_faster_rcnn import FasterRCNNWeaponDetector
        detector = FasterRCNNWeaponDetector(backbone="R50-FPN")
        detector.generate_benchmark()
    except Exception as e:
        print(f"[ERROR] Faster R-CNN error: {e}")
    


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Weapon Detection Model Comparison")
    parser.add_argument("--run-all", action="store_true", 
                       help="Run the benchmarks for every model")
    parser.add_argument("--html", action="store_true", 
                       help="Create an HTML report")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("WEAPON DETECTION MODEL COMPARISON")
    print("YOLOv8 vs Faster R-CNN")
    print(f"{'='*60}")
    
    # Run every benchmark
    if args.run_all:
        run_all_benchmarks()
    
    # Create the comparator
    comparator = BenchmarkComparator()
    
    # Load the benchmarks
    if not comparator.load_benchmarks():
        print("\n[ERROR] No benchmark file was found!")
        print("Train the models first, or pass --run-all.")
        return
    
    # Run the comparison
    comparison = comparator.compare_models()
    
    # Print the table
    comparator.print_comparison_table(comparison)
    
    # Save as JSON
    comparator.save_comparison(comparison)
    
    # Create the HTML report
    if args.html:
        comparator.generate_html_report(comparison)
    else:
        # Create the HTML report by default
        comparator.generate_html_report(comparison)
    
    print(f"\n{'='*60}")
    print("COMPARISON FINISHED")
    print(f"{'='*60}")
    print(f"\nResult files:")
    print(f"  - {BENCHMARK_DIR / 'final_benchmark_comparison.json'}")
    print(f"  - {BENCHMARK_DIR / 'benchmark_report.html'}")


if __name__ == "__main__":
    main()
