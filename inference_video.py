import cv2
from ultralytics import YOLO
import argparse
import torch
import typing as tp
from time import time
import os
from utils import download_pretrained_yolo


def upload_model(path_to_model: str = None, device: str = 'cuda'):
    model = YOLO(path_to_model).to(device)
    return model


def inference(path_to_model: str = None, path_to_video: str = './defaults/vid_1.mp4',
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
        if not os.path.exists(f"{models_folder}/yolov8s-human"):
            path_to_yolo = download_pretrained_yolo()
            model = upload_model(path_to_yolo, device=device)
    cap = cv2.VideoCapture(path_to_video)

    while True:
        success, frame = cap.read()

        if success:
            results = model.predict(source=path_to_video,
                                    project=project,
                                    name=name,
                                    conf=confidence,
                                    save=save_result,
                                    boxes=True,
                                    show_conf=show_conf,
                                    show_labels=show_labels,
                                    show=show_result)
            try:
                annotated_frame = results[0].plot()
                cv2.imshow("Human detection result", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except AttributeError:
                continue
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    current_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", type=str, required=False, default=None)
    parser.add_argument("--path_to_video", type=int, required=True, default='./defaults/vid_1.mp4')
    parser.add_argument("--project", type=str, required=False, default='./inference_runs')
    parser.add_argument("--name", type=str, required=False, default=f'inference_yolo_{current_time}')
    parser.add_argument("--confidence", type=float, required=False, default=0.5)
    parser.add_argument("--show_conf", type=bool, required=False, default=True)
    parser.add_argument("--show_labels", type=bool, required=False, default=True)
    parser.add_argument("--save_result", type=bool, required=False, default=False)
    parser.add_argument("--show_result", type=bool, required=False, default=True)
    args = parser.parse_args()
    inference(**vars(args))
