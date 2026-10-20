#!/usr/bin/env python3
"""
Helmet Vision - Custom YOLO Model Fine-Tuning Pipeline
Fine-tunes YOLOv8 / YOLO11 on custom construction site datasets.

Usage:
    python train.py --data dataset.yaml --epochs 50 --batch 16 --device 0
    python train.py --generate-sample-config  # Creates sample dataset.yaml template
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on construction safety datasets")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Path to dataset configuration YAML (YOLO format)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained base model weights to fine-tune from (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device: '0' for CUDA GPU, or 'cpu'",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training checkpoints and validation curves",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="helmet_vision_custom",
        help="Run name",
    )
    parser.add_argument(
        "--export-to-app",
        action="store_true",
        help="Automatically copy best.pt weights to vison_/models/best.pt when complete",
    )
    parser.add_argument(
        "--generate-sample-config",
        action="store_true",
        help="Create a template dataset.yaml file matching classes.yaml taxonomy",
    )
    return parser.parse_args()


def generate_sample_dataset_yaml(output_path: str = "dataset.yaml") -> None:
    """Generate a boilerplate dataset.yaml template."""
    classes_path = os.path.join(os.path.dirname(__file__), "classes.yaml")
    names_dict: Dict[int, str] = {}
    if os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            names_dict = cfg.get("names", {})

    content = {
        "path": "./dataset",  # dataset root dir
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_dict or {
            0: "worker",
            1: "hard_hat",
            2: "no_hard_hat",
            3: "safety_vest",
            4: "no_safety_vest",
            5: "crane",
            6: "suspended_load",
            7: "excavator",
            8: "heavy_truck",
            9: "forklift",
            10: "traffic_cone",
            11: "barrier",
            12: "hazard_tape",
            13: "warning_sign",
            14: "no_entry_sign",
            15: "danger_sign",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False)
    print(f"[INFO] Template dataset configuration created at '{output_path}'.")


def run_training(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run within venv:")
        print("    ./venv/bin/pip install ultralytics")
        sys.exit(1)

    if args.generate_sample_config:
        generate_sample_dataset_yaml()
        return

    if not os.path.isfile(args.data):
        print(f"[ERROR] Dataset configuration file '{args.data}' not found!")
        print("Tip: Run `python train.py --generate-sample-config` to create a template.")
        sys.exit(1)

    print("=" * 60)
    print("  HELMET VISION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f" Base model:     {args.base_model}")
    print(f" Dataset config: {args.data}")
    print(f" Epochs:         {args.epochs}")
    print(f" Batch size:     {args.batch}")
    print(f" Image size:     {args.imgsz}")
    print(f" Device:         {args.device}")
    print("=" * 60)

    # Load base model
    model = YOLO(args.base_model)

    # Run fine-tuning
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        plots=True,
    )

    print("\n[INFO] Training complete!")
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"

    if best_weights.is_file():
        print(f"[SUCCESS] Best fine-tuned model saved at: {best_weights}")
        if args.export_to_app:
            dest_dir = Path(__file__).resolve().parent.parent / "models"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / "best.pt"
            shutil.copy(best_weights, dest_path)
            print(f"[SUCCESS] Copied model weights to application directory: {dest_path}")
            print("Tip: Update `vison_/config.yaml` with: `model_path: models/best.pt`")


if __name__ == "__main__":
    cli_args = parse_args()
    run_training(cli_args)
