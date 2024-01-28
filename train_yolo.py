import os
import torch
from time import time
from ultralytics import YOLO
import argparse


def train_yolo(dataset_yaml_path: str = None, imgsz: int = 640, batch: int = 8,
               epochs: int = 100, lr0: float = 0.0001, optimizer: str = 'Adam',
               project: str = './training_runs', name: str = f'training_yolo_{time()}'):
    os.makedirs(project, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current device is {device}')
    model = YOLO('yolov8s.pt').to(device)
    model.train(task='detect',
                mode='train',
                data=dataset_yaml_path,
                imgsz=imgsz,
                batch=batch,
                epochs=epochs,
                lr0=lr0,
                optimizer=optimizer,
                project=project,
                name=name)


if __name__ == "__main__":
    current_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_yaml_path", type=str, required=True)
    parser.add_argument("--imgsz", type=int, required=False, default=640)
    parser.add_argument("--batch", type=int, required=True, default=8)
    parser.add_argument("--epochs", type=int, required=True, default=100)
    parser.add_argument("--lr0", type=float, required=True, default=0.0001)
    parser.add_argument("--optimizer", type=str, required=False, default='Adam')
    parser.add_argument("--project", type=str, required=False, default='./training_runs')
    parser.add_argument("--name", type=str, required=False, default=f'training_yolo_{current_time}')
    args = parser.parse_args()
    train_yolo(**vars(args))
