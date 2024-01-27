import os
from ultralytics import YOLO
import torch
from utils import download_pretrained_yolo
import argparse
from time import time
import typing as tp


def upload_model(path_to_model: str = None, device: str = 'cuda'):
    model = YOLO(path_to_model).to(device)
    return model


def inference(path_to_model: str = None, path_to_image: str = '/defaults/img_1.jpg', imgsz: int = 640,
              project: str = './inference_runs', name: str = f'inference_yolo_{time()}', confidence: float = 0.5,
              show_conf: bool = True, show_labels: bool = True,
              save_result: bool = False, show_result: bool = True) -> tp.NoReturn:

    os.makedirs(project, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if path_to_model is not None:
        model = upload_model(path_to_model, device=device)
    else:
        models_folder = './models'
        os.makedirs(models_folder, exist_ok=True)
        if not os.path.exists(f"{models_folder}/yolov8s-human-v3.pt"):
            path_to_yolo = download_pretrained_yolo()
            model = upload_model(path_to_yolo, device=device)
        else:
            model = upload_model(f"{models_folder}/yolov8s-human-v3.pt", device=device)


    results = model.predict(source=path_to_image,
                            project=project,
                            name=name,
                            imgsz=imgsz,
                            conf=confidence,
                            save=save_result,
                            boxes=True,
                            show_conf=show_conf,
                            show_labels=show_labels,
                            show=show_result)


if __name__ == "__main__":
    current_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", type=str, required=False, default=None)
    parser.add_argument("--path_to_image", type=str, required=True, default='/defaults/img_1.jpg')
    parser.add_argument("--imgsz", type=int, required=False, default=640)
    parser.add_argument("--project", type=str, required=False, default='./inference_runs')
    parser.add_argument("--name", type=str, required=False, default=f'inference_yolo_{current_time}')
    parser.add_argument("--confidence", type=float, required=False, default=0.5)
    parser.add_argument("--show_conf", type=bool, required=False, default=True)
    parser.add_argument("--show_labels", type=bool, required=False, default=True)
    parser.add_argument("--save_result", type=bool, required=False, default=False)
    parser.add_argument("--show_result", type=bool, required=False, default=True)
    args = parser.parse_args()
    inference(**vars(args))








