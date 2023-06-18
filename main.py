"""Main CLI Application file."""
import os
import json
import shutil
from enum import Enum
from typer import Typer, Argument, Option, BadParameter
from typing_extensions import Annotated
from yolo_model.model_run import (
    train_model,
    create_local_yolo_settings,
    detect as detect_yolo,
    test as test_yolo,
)
from data_fetch.prepare_data import prepare_image_data
from mrcnn_model.train_mrcnn_model import (
    train_mask_rcnn,
    detect_mask_rcnn,
    test as test_mask_rcnn,
)
from loftr_model.model_run import match_slides, analyze_matches
from rich import print as rprint
from rich.console import Console

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_NON_CROPS_DIR = os.path.join(CURRENT_PATH, "image_crops", "non_crops")
TEST_DATA_YOLO_PATH = os.path.join(
    CURRENT_PATH, "data_fetch", "prepared_data", "yolo_data", "test", "images"
)
MASKRCNN_DATA_PATH = os.path.join(
    CURRENT_PATH, "data_fetch", "prepared_data", "maskrcnn_data"
)
SLIDES_IMAGES_DIR = os.path.join(
    CURRENT_PATH, "data_fetch", "prepared_data", "slides"
)
DUPLICATES_DATA_PATH = os.path.join(
    SLIDES_IMAGES_DIR,
    "duplicates.json",
)


class TrainNetworkType(str, Enum):
    YOLO = "yolo"
    MASK_RCNN = "maskrcnn"
    ALL = "all"


class InferenceNetworkType(str, Enum):
    YOLO = "yolo"
    MASK_RCNN = "maskrcnn"
    NONE = "none"


class MaskRCNNWeights(str, Enum):
    COCO = "coco"
    IMAGENET = "imagenet"
    NONE = "none"


class LoFTRMatchingInput(str, Enum):
    MASK_RCNN = "maskrcnn"
    YOLO = "yolo"
    RAW = "non"


class LoFTRWeights(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


def train_yolo(epochs: int):
    """Train the YOLO model

    Args:
        epochs (int): number of epochs to train on.
    """
    create_local_yolo_settings()
    train_model(epochs=epochs)


CLI = Typer()
CONSOLE = Console()


@CLI.command()
def train(
    model_name: Annotated[
        TrainNetworkType,
        Argument(
            help="The name of the model to train.",
        ),
    ],
    epochs: Annotated[
        int, Option(help="Number of epochs to train model on.")
    ] = 100,
    weights: Annotated[
        MaskRCNNWeights, Option(help="Pretrained weights.")
    ] = MaskRCNNWeights.NONE,
):
    """Trains a specified model."""
    weights_str = (
        f", weights={weights.value}"
        if weights is not MaskRCNNWeights.NONE
        else ""
    )
    rprint(
        f"Traing [bold green]{model_name.value.upper()}[/bold green] model (epochs={epochs}{weights_str}) :boom:"
    )
    if model_name is TrainNetworkType.YOLO:
        if weights is not MaskRCNNWeights.NONE:
            raise BadParameter("Weights only for MASK RCNN.")
        train_yolo(epochs)
    elif model_name is TrainNetworkType.MASK_RCNN:
        if weights is MaskRCNNWeights.NONE:
            raise BadParameter(
                "MASK RCNN weights {coco|imagenet} MUST be applied to MASK RCNN."
            )
        train_mask_rcnn(epochs, weights)
    elif model_name is TrainNetworkType.ALL:
        for pre_weight in ["imagenet", "coco"]:
            train_mask_rcnn(epochs, pre_weight)
        train_yolo(epochs)
    else:
        raise BadParameter("You can only train YOLO or MASK RCNN or both.")


@CLI.command()
def evaluate(
    model_name: Annotated[
        InferenceNetworkType,
        Argument(
            help="The name of the model to make inference on.",
        ),
    ],
    model_path: Annotated[
        str, Option(help="Path to the trained model file.")
    ] = "",
):
    """Evaluates a model on the test set with mAP and F1 scores."""
    if model_name is InferenceNetworkType.YOLO:
        test_yolo(model_path)
    elif model_name is InferenceNetworkType.MASK_RCNN:
        test_mask_rcnn(model_path)
    else:
        raise BadParameter("Only YOLO and MASK RCNN models are supported.")


@CLI.command()
def prepare_data():
    """Prepares the data in the `data_fetch/prepared_data` folder.
    This will take the `data_fetch/screenshots` folder and prepare
    it for the `data_fetch/prepared_data` folder.
    This will also prepare the `data_fetch/prepared_data/maskrcnn_data`
    (data prepared for MASK R-CNN training and inference),
    `data_fetch/prepared_data/yolo_data` (data prepared for YOLOv8
    training and inference) and `data_fetch/prepared_data/slides/ (slides
    prepared for matching slides with LoFTR)

    Data folders for YOLOv8 and MASK-R-CNN each contain train, val
    and test folders. Slides contain test slides which will be used
    for matching slides on already trained LoFTR model."""
    prepare_image_data()


@CLI.command()
def detect_crop(
    model_name: Annotated[
        InferenceNetworkType,
        Argument(
            help="The name of the model to make inference on.",
        ),
    ],
    model_path: Annotated[
        str, Option(help="Path to the trained model file.")
    ] = "",
):
    """Detect areas and crop."""
    if model_name is InferenceNetworkType.YOLO:
        detect_yolo(model_path)
    elif model_name is InferenceNetworkType.MASK_RCNN:
        detect_mask_rcnn(model_path)
    elif model_name is InferenceNetworkType.NONE:
        shutil.copytree(
            TEST_DATA_YOLO_PATH, IMAGE_NON_CROPS_DIR, dirs_exist_ok=True
        )
    else:
        raise BadParameter("Only YOLO and MASK RCNN models are supported.")


@CLI.command()
def match_crops(
    crops: Annotated[
        LoFTRMatchingInput,
        Argument(
            help="Image crops to match projected slides with.",
        ),
    ],
    weights: Annotated[
        LoFTRWeights,
        Argument(
            help="LoFTR weights to use for matching.",
        ),
    ],
):
    """Match images of projections"""
    crops_dir = os.path.join(
        CURRENT_PATH, "image_crops", f"{crops.value}_crops"
    )
    with open(DUPLICATES_DATA_PATH, "r") as f:
        duplicates = json.load(f)
    is_weight_outdoor = True if weights is LoFTRWeights.OUTDOOR else False
    matched_images = match_slides(
        SLIDES_IMAGES_DIR, crops_dir, is_weight_outdoor=is_weight_outdoor
    )
    analyze_matches(matched_images, duplicates)


if __name__ == "__main__":
    CLI()
