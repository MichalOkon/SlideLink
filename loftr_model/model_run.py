import os
import sys
import cv2
import torch
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
from LoFTR.src.loftr import LoFTR
from LoFTR.src.loftr.utils.cvpr_ds_config import default_cfg

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


def compute_matching_score(mkpts0):
    """Compute matching score based on the number of matching points"""
    return len(mkpts0)


def match_slides(slides_dir, crops_dir, is_weight_outdoor=True):
    matcher = LoFTR(config=default_cfg)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outin_dir = "outdoor" if is_weight_outdoor else "indoor"
    weights_path = os.path.join(
        script_dir, "weights_loftr", f"loftr_{outin_dir}.ckpt"
    )
    matcher.load_state_dict(torch.load(weights_path)["state_dict"])
    matcher = matcher.eval().cuda()

    slides_filepaths = []
    # Add filepaths of the slides
    for filename in os.listdir(slides_dir):
        if "png" in filename:
            slides_filepaths.append(os.path.join(slides_dir, filename))

    crops_filepaths = []
    # Add filepaths of the crops
    for filename in os.listdir(crops_dir):
        if "png" in filename:
            crops_filepaths.append(os.path.join(crops_dir, filename))

    matched_images = {}
    # Loop over all target images
    for crop_filepath in tqdm(crops_filepaths):
        # Break if the filename is longer than 8 characters (incorrectly cropped slides)

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

        if np.max(matching_scores) < 50:
            matched_images[crop_filepath] = "No match found"
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

    matched_correctly = 0
    matched = 0
    for crop_filepath, slide_filepath in matched_images.items():
        if slide_filepath == "No match found":
            print_match(crop_filepath, slide_filepath)
            continue
        matched += 1
        crop_filename = os.path.basename(crop_filepath)
        slide_filename = os.path.basename(crop_filepath)
        if duplicates_dict.get(crop_filename) == duplicates_dict.get(
            slide_filename
        ):
            print("Matched!")
            matched_correctly += 1
        print_match(crop_filepath, slide_filepath)

    print("Matched correctly: ", matched_correctly)
    print("Total matched: ", matched)
    accuracy = matched_correctly / matched
    print("Accuracy: ", accuracy)
    print("Unidentified: ", len(matched_images) - matched)

    # Write results to a file
    time = str(datetime.now()).replace(":", "_").replace(" ", "_")
    with open(f"results_{time}.txt", "w") as f:
        f.write("Matched correctly: " + str(matched_correctly) + "\n")
        f.write("Total matched: " + str(matched) + "\n")
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("Unidentified: " + str(len(matched_images) - matched) + "\n")
    return accuracy


if __name__ == "__main__":
    # train_model()
    # path = os.path.abspath(os.getcwd())
    # print(path)
    # detect("../runs/detect/train4/weights/best.pt", "test_dataset")
    # Print out system filepath
    # print("System filepath: ", os.path.abspath(os.getcwd()))
    # duplicates = process_images("matching_datasets/slides")
    # print("Duplicates: ", duplicates)
    root_dir_project = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    slides_dir = os.path.join(
        root_dir_project, "data_fetch", "prepared_data", "slides"
    )
    slides_duplicates_file = os.path.join(
        slides_dir,
        "duplicates.json",
    )
    crops_dir = os.path.join(root_dir_project, "image_crops", "non_crops")

    with open(slides_duplicates_file) as f:
        duplicates = json.load(f)
    matched_images = match_slides(slides_dir, crops_dir, is_weight_outdoor=True)
    analyze_matches(matched_images, duplicates)
