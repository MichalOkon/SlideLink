import os
import json
import math
import random
import shutil
from typing import Dict, Any, List, Tuple

LABELS_FILE_NAME = "labels.json"

curr_path = os.path.dirname(os.path.abspath(__file__))

copy_files_path = os.path.join(curr_path, "mrcnn_data")

src_dir = os.path.join(curr_path, "screenshots")

shutil.copytree(src_dir, copy_files_path)

data_paths = [name for name in os.listdir(copy_files_path) if os.path.isdir(os.path.join(copy_files_path, name))]

for data_path in data_paths:
    shutil.rmtree(os.path.join(os.path.join(copy_files_path, data_path), "slideshow"))

for data_path in data_paths:
    move_dist_path = os.path.join(copy_files_path, data_path)
    move_src_path = os.path.join(move_dist_path, "presenter")

    files_to_copy = [name for name in os.listdir(move_src_path) if os.path.isfile(os.path.join(move_src_path, name))]
    for file in files_to_copy:
        shutil.move(os.path.join(move_src_path, file), os.path.join(copy_files_path, f"{data_path}_{file}"))
    shutil.rmtree(move_src_path)
    os.rmdir(move_dist_path)

json_label_files = [
    name
    for name in os.listdir(copy_files_path)
    if os.path.isfile(os.path.join(copy_files_path, name)) and "json" in name
]

res_dict = dict()

for file_json in json_label_files:
    with open(os.path.join(copy_files_path, file_json)) as json_file:
        data = json.load(json_file)
    new_file_name = file_json.split("_")[0]
    for key in data:
        data[key]["filename"] = f"{new_file_name}_{data[key]['filename']}"
    res_dict = {**res_dict, **data}
    os.remove(os.path.join(copy_files_path, file_json))

labels_file_path = os.path.join(copy_files_path, LABELS_FILE_NAME)
with open(labels_file_path, "w") as fp:
    json.dump(res_dict, fp)

os.remove(os.path.join(copy_files_path, ".gitkeep"))

data_folders = ["train", "val", "test"]
for folder in data_folders:
    os.mkdir(os.path.join(copy_files_path, folder))


def add_key(key: str, label_dict: Dict[str, Any]) -> Dict[str, Any]:
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
    data: List[Dict[str, Any]], split_percent=0.8
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits an list into train and test data.

    Args:
        data (List[Dict[str, Any]]): list of data
        split_percent (float, optional): data split percentage. Defaults to 0.8.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: train and test data split
    """
    assert 0.0 < split_percent < 1.0, "data split should be between 0 and 1"
    random.Random(0).shuffle(data)
    index_split = math.floor(split_percent * len(data))
    train_data = data[:index_split]
    test_data = data[index_split:]
    return train_data, test_data


# Divide the labels into train, validation and test sets
with open(labels_file_path, "r") as json_file:
    labels_dict = json.load(json_file)

labels_list = [add_key(key, labels_dict[key]) for key in labels_dict]

train_data, test_data = train_test_split(labels_list, split_percent=0.8)
train_data, val_data = train_test_split(train_data, split_percent=0.8)
data_lists = [train_data, val_data, test_data]

# Move files to the correct location
for i, data_dicts_list in enumerate(data_lists):
    folder_dist = os.path.join(copy_files_path, data_folders[i])
    for data_dict in data_dicts_list:
        file_name = data_dict["filename"]
        file_src = os.path.join(copy_files_path, file_name)
        file_dist = os.path.join(folder_dist, file_name)
        shutil.move(file_src, file_dist)
    save_dict = {curr_dict["id"]: curr_dict for curr_dict in data_dicts_list}
    with open(os.path.join(folder_dist, LABELS_FILE_NAME), "w") as fp:
        json.dump(save_dict, fp)

os.remove(os.path.join(copy_files_path, LABELS_FILE_NAME))
