import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import torch
from math import dist
import os
from utils import download_pretrained_yolo
import argparse
import typing as tp

font = cv2.FONT_HERSHEY_SIMPLEX


def upload_model(path_to_model: str = None, device: str = 'cuda'):
    model = YOLO(path_to_model).to(device)
    return model


def estimate_speed(point_1, point_2, ppm_rate: int = 8, fps: int = 15) -> int:
    d_pixel = dist(point_1, point_2)
    d_meters = d_pixel/ppm_rate
    time_constant = fps*3.6
    speed = d_meters * time_constant
    return int(speed)


def tracking(path_to_model: str = None, path_to_video: str = './defaults/video_1.mp4',
             confidence: float = 0.5) -> tp.NoReturn:

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
    cap = cv2.VideoCapture(path_to_video)

    track_history = defaultdict(lambda: [])
    while True:
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, conf=confidence, tracker='bytetrack.yaml')
            try:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((int(x), int(y)))
                    if len(track) >= 2:
                        speed = estimate_speed(track[-1], track[-2], ppm_rate=16)
                        cv2.putText(annotated_frame, f'{speed} km/h', track[-1], font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if len(track) > 50:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                cv2.imshow("infrared image tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except AttributeError:
                continue
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", type=str, required=False, default=None)
    parser.add_argument("--path_to_video", type=int, required=True, default='./defaults/vid_1.mp4')
    parser.add_argument("--confidence", type=float, required=False, default=0.5)
    args = parser.parse_args()
    tracking(**vars(args))
