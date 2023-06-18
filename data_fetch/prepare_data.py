import os
import cv2
import json
import math
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from rich.console import Console
from rich.progress import track
from typing import Dict, Any, List, Tuple

####### Static variables #################################
LABELS_FILE_NAME = "labels.json"
CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CONSOLE = Console()

####### Custom functions #################################


def move_id_within_dict(key: str, label_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create a dict with key as id. This enables to create a list item.

    Args:
        key (str): key of the outside dict
        label_dict (Dict[str, Any]): dict to add key to

    Returns:
        Dict[str, Any]: _description_
    """
    label_dict["id"] = key
    return label_dict


def train_test_split(
    data: List[Dict[str, Any]], split_percent=0.8, random_state=0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits an list into train and test data.

    Args:
        data (List[Dict[str, Any]]): list of data
        split_percent (float, optional): data split percentage. Defaults to 0.8.
        random_state (int, optional): deterministic random state. Defaults to 0.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: train and test data split
    """
    assert 0.0 < split_percent < 1.0, "data split should be between 0 and 1"
    random.Random(random_state).shuffle(data)
    index_split = math.floor(split_percent * len(data))
    train_data = data[:index_split]
    test_data = data[index_split:]
    return train_data, test_data


def train_test_split_by_name(
    data: List[Dict[str, Any]], name: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits an list into train and test data by name of the lecture.

    Args:
        data (List[Dict[str, Any]]): list of data
        name (str): name of the lecture

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: train and test data split
    """
    train_data = []
    test_data = []
    for i, point in enumerate(data):
        if name in point["filename"]:
            test_data.append(point)
        else:
            train_data.append(point)
    return train_data, test_data


def detection_conversion(shape_attributes, img_width, img_height):
    x_center = (
        shape_attributes["x"] + shape_attributes["width"] / 2
    ) / img_width
    y_center = (
        shape_attributes["y"] + shape_attributes["height"] / 2
    ) / img_height
    width = shape_attributes["width"] / img_width
    height = shape_attributes["height"] / img_height

    return x_center, y_center, width, height


def json_to_yolo_format(json_data, image_dir):
    yolo_data = {}

    for _, value in json_data.items():
        # Get the filename without extension
        file_id = value["filename"].split(".")[0]

        image_path = os.path.join(image_dir, value["filename"])

        # Open the image to get its dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        for region in value["regions"]:
            shape_attributes = region["shape_attributes"]
            if shape_attributes["name"] == "rect":
                x_center, y_center, width, height = detection_conversion(
                    shape_attributes, img_width, img_height
                )

                # Save the converted data
                if file_id not in yolo_data:
                    yolo_data[file_id] = []

                # Append to list of rectangles for this image
                # Assuming a class id of 0 for all rectangles
                yolo_data[file_id].append(
                    f"0 {x_center} {y_center} {width} {height}"
                )
    return yolo_data


def find_duplicates(images_filepath):
    duplicates_dict = {}
    deleted = set()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_filepath = os.path.join(script_dir, images_filepath)
    # Find identical images
    for filename in track(
        os.listdir(images_filepath), description="Finding duplicates..."
    ):
        if filename in deleted:
            continue
        img1 = cv2.imread(os.path.join(images_filepath, filename))
        for filename2 in os.listdir(images_filepath):
            if filename == filename2 or filename2 in deleted:
                continue

            img2 = cv2.imread(os.path.join(images_filepath, filename2))
            if (
                img1.shape == img2.shape
                and np.bitwise_xor(img1, img2).sum() < 5
            ):
                duplicates_dict[filename2] = filename
                deleted.add(filename2)
                os.remove(os.path.join(images_filepath, filename2))

        if filename not in duplicates_dict.keys():
            duplicates_dict[filename] = filename

    return duplicates_dict


def process_images(images_filepath):
    for filename in os.listdir(images_filepath):
        image_path = os.path.join(images_filepath, filename)
        img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Crop out the bottom of the image
        img_raw = img_raw[0 : img_raw.shape[0] - 50, 0 : img_raw.shape[1]]
        cv2.imwrite(os.path.join(images_filepath, filename), img_raw)
    duplicates_dict = find_duplicates(images_filepath)
    return duplicates_dict


def copy_tree_contents(src: str, dst: str):
    """Copies the contents of a directory to a directory (without the file itself).

    Args:
        src (str): source directory
        dst (str): destination directory
    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


################################################################ MAIN PROGRAM #################################################################
def prepare_image_data():
    """Prepared image data."""
    with CONSOLE.status("[bold green]Preparing data..."):
        ####### Preparation before changing the folder structure #################################
        # Get original data folder path
        src_dir = os.path.join(CURRENT_DIR_PATH, "screenshots")
        # Preprocessed data folder path
        final_dist_root_path = os.path.join(CURRENT_DIR_PATH, "prepared_data")
        # Copy all files the new folder
        shutil.copytree(src_dir, final_dist_root_path)
        CONSOLE.log(
            "[green]Copied data to `prepared_data` directory...[/green]"
        )

        ####### Moving label and image data to new directory #################################
        # Get all lecture names
        lecture_names = [
            name
            for name in os.listdir(final_dist_root_path)
            if os.path.isdir(os.path.join(final_dist_root_path, name))
        ]
        slides_dir = os.path.join(final_dist_root_path, "slides")
        os.mkdir(slides_dir)
        # Move all slides to the newly created "prepared_data/slides" directory
        for lecture_name in lecture_names:
            lecture_slides_path = os.path.join(
                final_dist_root_path, lecture_name
            )
            move_src_path = os.path.join(lecture_slides_path, "slideshow")
            files_to_copy = [
                name
                for name in os.listdir(move_src_path)
                if os.path.isfile(os.path.join(move_src_path, name))
            ]
            for file in files_to_copy:
                shutil.move(
                    os.path.join(move_src_path, file),
                    os.path.join(slides_dir, f"{lecture_name}_{file}"),
                )
            shutil.rmtree(move_src_path)
        CONSOLE.log(
            "[green]Moved slides to `prepared_data/slides` directory...[/green]"
        )
        # Move all files to the newly created "prepared_data" directory
        for lecture_name in lecture_names:
            CONSOLE.log(
                f"[green]Moving presenter images of lecture {lecture_name}...[/green]"
            )
            move_dist_path = os.path.join(final_dist_root_path, lecture_name)
            move_src_path = os.path.join(move_dist_path, "presenter")

            files_to_copy = [
                name
                for name in os.listdir(move_src_path)
                if os.path.isfile(os.path.join(move_src_path, name))
            ]
            for file in files_to_copy:
                shutil.move(
                    os.path.join(move_src_path, file),
                    os.path.join(
                        final_dist_root_path, f"{lecture_name}_{file}"
                    ),
                )
            shutil.rmtree(move_src_path)
            os.rmdir(move_dist_path)

        CONSOLE.log(
            "[green]Moved presenter images to lecture directory...[/green]"
        )
        ####### Create one main labels file #################################
        # Get all json files in the directory
        json_label_files = [
            name
            for name in os.listdir(final_dist_root_path)
            if os.path.isfile(os.path.join(final_dist_root_path, name))
            and "json" in name
        ]

        # Move all labels to one file
        res_dict = dict()
        for file_json in json_label_files:
            with open(
                os.path.join(final_dist_root_path, file_json), "r"
            ) as json_file:
                data = json.load(json_file)
            new_file_name = file_json.split("_")[0]
            for key in data:
                data[key][
                    "filename"
                ] = f"{new_file_name}_{data[key]['filename']}"
            res_dict = {**res_dict, **data}
            os.remove(os.path.join(final_dist_root_path, file_json))

        labels_file_path = os.path.join(final_dist_root_path, LABELS_FILE_NAME)
        with open(labels_file_path, "w") as fp:
            json.dump(res_dict, fp)
        if os.path.exists(os.path.join(final_dist_root_path, ".gitkeep")):
            os.remove(os.path.join(final_dist_root_path, ".gitkeep"))

        CONSOLE.log("[green]Created global json labels file...[/green]")
        ####### Move Everything to new train, validation and test directories for Mask-RCNN model ###########
        maskrcnn_path = os.path.join(final_dist_root_path, "maskrcnn_data")
        os.mkdir(maskrcnn_path)

        data_folders = ["train", "val", "test"]
        for folder in data_folders:
            os.mkdir(os.path.join(maskrcnn_path, folder))

        # Divide the labels into train, validation and test sets
        with open(labels_file_path, "r") as json_file:
            labels_dict = json.load(json_file)

        labels_list = [
            move_id_within_dict(key, labels_dict[key]) for key in labels_dict
        ]

        # Divide up the images
        test_data_lecture = random.Random(1).choice(lecture_names)
        train_data, test_data = train_test_split_by_name(
            labels_list, test_data_lecture
        )
        train_data, val_data = train_test_split(train_data, split_percent=0.8)
        data_lists = [train_data, val_data, test_data]
        CONSOLE.log("[green]Data divided into train, val and test...[/green]")

        for file in os.listdir(slides_dir):
            if ".png" in file and test_data_lecture not in file:
                os.remove(os.path.join(slides_dir, file))

        duplicates_dict = process_images(slides_dir)

        with open(os.path.join(slides_dir, "duplicates.json"), "w") as fp:
            json.dump(duplicates_dict, fp)
        CONSOLE.log("[green]Found and deleted duplicate slides...[/green]")

        # Move files to the correct location
        for i, data_dicts_list in enumerate(data_lists):
            folder_dist = os.path.join(maskrcnn_path, data_folders[i])
            for data_dict in data_dicts_list:
                file_name = data_dict["filename"]
                file_src = os.path.join(final_dist_root_path, file_name)
                file_dist = os.path.join(folder_dist, file_name)
                shutil.move(file_src, file_dist)
            save_dict = {
                curr_dict["id"]: curr_dict for curr_dict in data_dicts_list
            }
            with open(os.path.join(folder_dist, LABELS_FILE_NAME), "w") as fp:
                json.dump(save_dict, fp)

        os.remove(os.path.join(final_dist_root_path, LABELS_FILE_NAME))

        CONSOLE.log("[green]Prepared data for Mask-RCNN model...[/green]")
        ####### Create data ready for YOLO model ###########
        yolo_path = os.path.join(final_dist_root_path, "yolo_data")
        os.mkdir(yolo_path)
        copy_tree_contents(maskrcnn_path, yolo_path)

        for folder in data_folders:
            curr_yolo_data_path = os.path.join(yolo_path, folder)
            labels_folder_path = os.path.join(curr_yolo_data_path, "labels")
            os.mkdir(labels_folder_path)
            labels_file_path = os.path.join(curr_yolo_data_path, "labels.json")
            with open(labels_file_path, "r") as f:
                labels_data = json.load(f)
            yolo_labels_dict = json_to_yolo_format(
                labels_data, curr_yolo_data_path
            )
            for file_id, lines in yolo_labels_dict.items():
                new_yolo_label_file_path = os.path.join(
                    labels_folder_path, f"{file_id}.txt"
                )
                with open(new_yolo_label_file_path, "w") as f:
                    f.write("\n".join(lines))
            os.remove(labels_file_path)

            images_names = [
                name
                for name in os.listdir(curr_yolo_data_path)
                if os.path.isfile(os.path.join(curr_yolo_data_path, name))
                and "png" in name
            ]

            images_folder_path = os.path.join(curr_yolo_data_path, "images")
            os.mkdir(images_folder_path)
            for image_name in images_names:
                os.rename(
                    os.path.join(curr_yolo_data_path, image_name),
                    os.path.join(images_folder_path, image_name),
                )
        CONSOLE.log("[green]Prepared data for YOLO model...[/green]")
        CONSOLE.log("[bold][red]Done!")
