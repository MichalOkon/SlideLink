import os
import json
import math
import random
import shutil
from PIL import Image
from typing import Dict, Any, List, Tuple

####### Static variables #################################
LABELS_FILE_NAME = "labels.json"
CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

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


####### Preparation before changing the folder structure #################################
# Get original data folder path
src_dir = os.path.join(CURRENT_DIR_PATH, "screenshots")
# Preprocessed data folder path
final_dist_root_path = os.path.join(CURRENT_DIR_PATH, "prepared_data")
# Copy all files the new folder
shutil.copytree(src_dir, final_dist_root_path)


####### Moving and label and image data to new directory #################################
# Get all lecture names
lecture_data_paths = [
    name
    for name in os.listdir(final_dist_root_path)
    if os.path.isdir(os.path.join(final_dist_root_path, name))
]
# Remove slideshow data: UNNECESSARY FOR Mask-RCNN
for data_path in lecture_data_paths:
    shutil.rmtree(
        os.path.join(os.path.join(final_dist_root_path, data_path), "slideshow")
    )
# Move all files to the newly created "prepared_data" directory
for data_path in lecture_data_paths:
    move_dist_path = os.path.join(final_dist_root_path, data_path)
    move_src_path = os.path.join(move_dist_path, "presenter")

    files_to_copy = [
        name
        for name in os.listdir(move_src_path)
        if os.path.isfile(os.path.join(move_src_path, name))
    ]
    for file in files_to_copy:
        shutil.move(
            os.path.join(move_src_path, file),
            os.path.join(final_dist_root_path, f"{data_path}_{file}"),
        )
    shutil.rmtree(move_src_path)
    os.rmdir(move_dist_path)

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
    with open(os.path.join(final_dist_root_path, file_json)) as json_file:
        data = json.load(json_file)
    new_file_name = file_json.split("_")[0]
    for key in data:
        data[key]["filename"] = f"{new_file_name}_{data[key]['filename']}"
    res_dict = {**res_dict, **data}
    os.remove(os.path.join(final_dist_root_path, file_json))

labels_file_path = os.path.join(final_dist_root_path, LABELS_FILE_NAME)
with open(labels_file_path, "w") as fp:
    json.dump(res_dict, fp)
if os.path.exists(os.path.join(final_dist_root_path, ".gitkeep")):
    os.remove(os.path.join(final_dist_root_path, ".gitkeep"))

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
train_data, test_data = train_test_split(labels_list, split_percent=0.8)
train_data, val_data = train_test_split(train_data, split_percent=0.8)
data_lists = [train_data, val_data, test_data]

# Move files to the correct location
for i, data_dicts_list in enumerate(data_lists):
    folder_dist = os.path.join(maskrcnn_path, data_folders[i])
    for data_dict in data_dicts_list:
        file_name = data_dict["filename"]
        file_src = os.path.join(final_dist_root_path, file_name)
        file_dist = os.path.join(folder_dist, file_name)
        shutil.move(file_src, file_dist)
    save_dict = {curr_dict["id"]: curr_dict for curr_dict in data_dicts_list}
    with open(os.path.join(folder_dist, LABELS_FILE_NAME), "w") as fp:
        json.dump(save_dict, fp)

os.remove(os.path.join(final_dist_root_path, LABELS_FILE_NAME))


def copy_tree_contents(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


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
    yolo_labels_dict = json_to_yolo_format(labels_data, curr_yolo_data_path)
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
