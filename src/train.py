import os
from pathlib import Path
from ultralytics import YOLO
import argparse

def train_model(
    data_yaml: str,
    model_size: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str,
    project: str,
    name: str,
):
    root_dir = Path.cwd().absolute()
    model = YOLO(str(root_dir / "weights" / f"yolo11{model_size}.pt"))
    
    results = model.train(
        data=str(root_dir / data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(root_dir / project),
        name=name,
        exist_ok=True,
    )
    
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--device', type=str, default='0',
                      help='Device to use for training ("0" for GPU, "cpu" for CPU)')
    
    args = parser.parse_args()
    data_yaml = "datasets/merged/data.yaml"

    train_model(
        data_yaml=data_yaml,
        model_size="n",
        epochs=100,
        batch_size=16,
        img_size=640,
        device=args.device,
        project="runs/train",
        name="exp",
    )