import cv2
import os
import uuid
import requests


def download_pretrained_yolo(link: str = '', model_name: str = 'yolov8s-human') -> str:

    """
        Method to download pretrained YOLOv8
    """

    models_folder = './models'
    os.makedirs(models_folder, exist_ok=True)
    response = requests.get(link)
    open(f"{models_folder}/{model_name}", "wb").write(response.content)
    path_to_yolo = f"{models_folder}/{model_name}"
    return path_to_yolo


def video_to_frames(path_to_video: str = None, output_path: str = None):

    """
    Method to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        path_to_video: Input video file.
        output_path: Output directory to save the frames.
    Returns:
        None
    """

    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError:
        pass

    cap = cv2.VideoCapture(path_to_video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{output_path}/{uuid.uuid4()}.jpg', frame)

#
# if __name__ == "__main__":
#     base_path = 'stanforddata/videos/{name}/{video_num}/video.mp4'
#     base_output_path = './output_frames'
#     names = os.listdir('./stanforddata/videos')
#     for name in names:
#         for num in range(3):
#             path_to_video = base_path.replace('{name}', name).replace('{video_num}', f'video{num}')
#             output_path = f'{base_output_path}/{name}/video{num}'
#             video_to_frames(path_to_video, output_path)
