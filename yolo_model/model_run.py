import os
import yaml
import torch
import shutil
import numpy as np
from datetime import datetime
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INFO_PATH = os.path.join(CURRENT_DIR, "data.yaml")
YOLO_TRAIN_LOGS_DIR = os.path.join(
    os.path.dirname(CURRENT_DIR), "runs", "detect"
)
SAVED_MODELS_DIR = os.path.join(CURRENT_DIR, "saved_models")
IMAGE_CROPS_DIR = os.path.join(
    os.path.dirname(CURRENT_DIR), "image_crops", "yolo_crops"
)
IMAGE_NON_CROPS_DIR = os.path.join(
    os.path.dirname(CURRENT_DIR), "image_crops", "non_crops"
)


def train_model(save_model_dir: str = SAVED_MODELS_DIR, epochs: int = 100):
    """Train YOLO model.

    Args:
        save_model_dir (str, optional): folder to save the trained model to. Defaults to CURRENT_DIR.
        epochs (int, optional): Number of epochs to run. Defaults to 100.
    """
    model = YOLO(os.path.join(CURRENT_DIR, "yolov8n.pt"))
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.backends.cudnn.version())  # type: ignore
    model.train(
        batch=1,
        device=0,
        data=DATA_INFO_PATH,
        epochs=epochs,
        imgsz=640,
    )
    time = str(datetime.now()).replace(":", "_").replace(" ", "_")
    recent_training_dir = sorted(
        [
            name
            for name in os.listdir(YOLO_TRAIN_LOGS_DIR)
            if os.path.isdir(os.path.join(YOLO_TRAIN_LOGS_DIR, name))
            and "train" in name
        ],
        reverse=True,
    )[0]
    # print(recent_training_dir)
    best_model_path = os.path.join(
        YOLO_TRAIN_LOGS_DIR, recent_training_dir, "weights", "best.pt"
    )
    path_to_save = os.path.join(save_model_dir, f"yolo_{time}.pt")
    shutil.copyfile(best_model_path, path_to_save)


def create_local_yolo_settings():
    """Creates a local yolo settings with directory structure fitted for the used OS."""
    root_dir = os.path.dirname(CURRENT_DIR)
    yolo_data_path = os.path.join(
        root_dir, "data_fetch", "prepared_data", "yolo_data"
    )
    data_dirs_path = {
        "train": os.path.join(yolo_data_path, "train", "images"),
        "val": os.path.join(yolo_data_path, "val", "images"),
        "test": os.path.join(yolo_data_path, "test", "images"),
        "names": ["slide"],
    }
    with open(DATA_INFO_PATH, "w") as file:
        yaml.dump(data_dirs_path, file)


def detect(model_path):
    with open(DATA_INFO_PATH) as f:
        data_info = yaml.safe_load(f)
    data_dir = data_info["test"]
    model = YOLO(model_path)
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source=data_dir, save=True, save_crop=True)
    for result in results:
        print(result.boxes.xyxy)
    most_recent_predict = sorted(
        [
            name
            for name in os.listdir(YOLO_TRAIN_LOGS_DIR)
            if os.path.isdir(os.path.join(YOLO_TRAIN_LOGS_DIR, name))
            and "predict" in name
        ],
        reverse=True,
    )[0]
    cropped_images_path = os.path.join(
        YOLO_TRAIN_LOGS_DIR, most_recent_predict, "crops", "slide"
    )
    if not os.path.exists(IMAGE_CROPS_DIR):
        os.makedirs(IMAGE_CROPS_DIR)
    shutil.copytree(cropped_images_path, IMAGE_CROPS_DIR, dirs_exist_ok=True)
    shutil.copytree(data_dir, IMAGE_NON_CROPS_DIR, dirs_exist_ok=True)


def test(model_path):
    model = YOLO(model_path)
    metrics = model.val()
    precision, recall, mAP50, mAP50_95 = metrics.mean_results()
    print("box recall: ", metrics.box.r)
    print("box precision: ", metrics.box.p)
    print("map: ", metrics.box.map)
    print("map50: ", metrics.box.map50)  # map50
    print("map75: ", metrics.box.map75)  # map75


if __name__ == "__main__":
    # train_model()
    # path = os.path.abspath(os.getcwd())
    # print(path)
    # detect(
    #     "/home/kbaran/git/git-uni/SlideLink/yolo_model/saved_models/yolo_2023-06-15_23_02_47.597993.pt"
    # )
    test(
        "/home/kbaran/git/git-uni/SlideLink/yolo_model/saved_models/yolo_2023-06-16_22_20_56.831368.pt"
    )
# Print out system filepath
# print("System filepath: ", os.path.abspath(os.getcwd()))
# duplicates = process_images("matching_datasets/slides")
# print("Duplicates: ", duplicates)
# matched_images = match_slides()
# analyze_matches(matched_images, duplicates)
