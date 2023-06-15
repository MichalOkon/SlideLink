import os
import sys
import yaml
import cv2
import torch
import shutil
import numpy as np

# from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
# sys.path.insert(0, path)

# from LoFTR.src.loftr import LoFTR
# from LoFTR.src.loftr.utils.cvpr_ds_config import default_cfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INFO_PATH = os.path.join(CURRENT_DIR, "data.yaml")
YOLO_TRAIN_LOGS_DIR = os.path.join(
    os.path.dirname(CURRENT_DIR), "runs", "detect"
)
SAVED_MODELS_DIR = os.path.join(CURRENT_DIR, "saved_models")


def train_model(save_model_dir: str = SAVED_MODELS_DIR):
    """Train YOLO model.

    Args:
        save_model_dir (str, optional): folder to save the trained model to. Defaults to CURRENT_DIR.
    """
    model = YOLO(os.path.join(CURRENT_DIR, "yolov8n.pt"))
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.backends.cudnn.version())
    model.train(
        batch=1,
        device=0,
        data=DATA_INFO_PATH,
        epochs=100,
        imgsz=640,
    )
    time = str(datetime.now()).replace(":", "_").replace(" ", "_")
    recent_training_dir = sorted(
        [
            name
            for name in os.listdir(YOLO_TRAIN_LOGS_DIR)
            if os.path.isdir(os.path.join(YOLO_TRAIN_LOGS_DIR, name))
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


def detect(model_path, data_dir):
    model = YOLO(model_path)
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source=data_dir, save=True, save_crop=True)
    for i, result in enumerate(results):
        print(result.boxes.xyxy)


def compute_matching_score(mkpts0):
    """Compute matching score based on the number of matching points"""
    return len(mkpts0)


def match_slides():
    matcher = LoFTR(config=default_cfg)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(
        script_dir, "weights_loftr", "loftr_indoor.ckpt"
    )
    matcher.load_state_dict(torch.load(weights_path)["state_dict"])
    matcher = matcher.eval().cuda()

    slides_dir = os.path.join(script_dir, "matching_datasets", "trimmed_images")
    slides_filepaths = []
    # Add filepaths of the slides
    for filename in os.listdir(slides_dir):
        slides_filepaths.append(os.path.join(slides_dir, filename))

    crops_dir = os.path.join(script_dir, "matching_datasets", "crops")
    crops_filepaths = []
    # Add filepaths of the crops
    for filename in os.listdir(crops_dir):
        crops_filepaths.append(os.path.join(crops_dir, filename))

    matched_images = {}
    # Loop over all target images
    for crop_filepath in crops_filepaths:
        # Break if the filename is longer than 8 characters (incorrectly cropped slides)
        print(crop_filepath.split(sep="\\")[-1])
        if len(crop_filepath.split(sep="\\")[-1]) > 8:
            continue

        img0_raw = cv2.imread(crop_filepath, cv2.IMREAD_GRAYSCALE)
        img0_raw = cv2.resize(img0_raw, (640, 320))
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.0
        # Display the image
        # cv2.imshow("Image", img0_raw)
        matching_scores = []
        # Pause the program until a key is pressed
        # cv2.waitKey(0)
        for slide_filepath in slides_filepaths:
            img1_raw = cv2.imread(slide_filepath, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.resize(img1_raw, (640, 320))

            img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.0
            batch = {"image0": img0, "image1": img1}

            # Inference with LoFTR and get prediction
            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch["mkpts0_f"].cpu().numpy()

            # Compute a matching score
            matching_score = compute_matching_score(mkpts0)
            # Add the matching score to the list
            matching_scores.append(matching_score)
            print("Matched score: ", matching_score)

        if np.max(matching_scores) < 10:
            print("No match found")
            matched_images[crop_filepath] = "No match found"
            continue
        else:
            best_matching_image_path = slides_filepaths[
                np.argmax(matching_scores)
            ]
            print("Best matching image: ", best_matching_image_path)
            matched_images[crop_filepath] = best_matching_image_path

    return matched_images


def analyze_matches(matched_images, duplicates_dict):
    def print_match(crop_filepath, slide_filepath):
        print("Crop: ", crop_filepath)
        print("Slide: ", slide_filepath)
        print("----------------------------------------------------")

    print(path)
    matched_correctly = 0
    matched = 0
    for crop_filepath, slide_filepath in matched_images.items():
        if slide_filepath == "No match found":
            print_match(crop_filepath, slide_filepath)
            continue
        matched += 1
        crop_filename = crop_filepath.split(sep="\\")[-1][:4]
        slide_filename = slide_filepath.split(sep="\\")[-1][:4]
        if duplicates_dict[crop_filename] == duplicates_dict[slide_filename]:
            print("Matched!")
            matched_correctly += 1
        print_match(crop_filepath, slide_filepath)

    print("Matched correctly: ", matched_correctly)
    print("Total matched: ", matched)
    accuracy = matched_correctly / matched
    print("Accuracy: ", accuracy)
    print("Unidentified: ", len(matched_images) - matched)

    return accuracy


def process_images(images_filepath):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trimmed_dir = "matching_datasets\\trimmed_images"
    trimmed_path = os.path.join(script_dir, trimmed_dir)
    print(trimmed_path)
    if not os.path.exists(trimmed_path):
        print("Create")
        os.mkdir(trimmed_path)

    for filename in os.listdir(images_filepath):
        image_path = os.path.join(images_filepath, filename)
        print(image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Crop out the bottom of the image
        img_raw = img_raw[0 : img_raw.shape[0] - 50, 0 : img_raw.shape[1]]
        cv2.imwrite(os.path.join(trimmed_path, filename), img_raw)

    def find_duplicates(images_filepath):
        duplicates_dict = {}
        deleted = set()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_filepath = os.path.join(script_dir, images_filepath)
        print(images_filepath)
        # Find identical images
        for filename in os.listdir(images_filepath):
            if filename in deleted:
                continue
            print(os.path.join(images_filepath, filename))
            img1 = cv2.imread(os.path.join(images_filepath, filename))
            print(img1.shape)
            for filename2 in os.listdir(images_filepath):
                if filename == filename2 or filename2 in deleted:
                    continue

                img2 = cv2.imread(os.path.join(images_filepath, filename2))
                if (
                    img1.shape == img2.shape
                    and np.bitwise_xor(img1, img2).sum() < 5
                ):
                    print("Identical images: ", filename, filename2)
                    duplicates_dict[filename2[:4]] = filename[:4]
                    deleted.add(filename2)
                    os.remove(os.path.join(images_filepath, filename2))

            if filename not in duplicates_dict.keys():
                duplicates_dict[filename[:4]] = filename[:4]

        return duplicates_dict

    duplicates_dict = find_duplicates(trimmed_path)
    return duplicates_dict


if __name__ == "__main__":
    # train_model()
    # path = os.path.abspath(os.getcwd())
    # print(path)
    # detect("../runs/detect/train4/weights/best.pt", "test_dataset")
    # Print out system filepath
    # print("System filepath: ", os.path.abspath(os.getcwd()))
    duplicates = process_images("matching_datasets/slides")
    print("Duplicates: ", duplicates)
    matched_images = match_slides()
    analyze_matches(matched_images, duplicates)
