"""Main CLI Application file."""
import os
from enum import Enum
from typer import Typer, Argument, Option, BadParameter
from typing_extensions import Annotated
from yolo_model.model_run import (
    train_model,
    create_local_yolo_settings,
    detect,
    test,
)
from data_fetch.prepare_data import prepare_image_data
from mrcnn_model.train_mrcnn_model import (
    DEFAULT_LOGS_DIR,
    SAVED_MODELS,
    DATASET,
    SAVED_MODELS,
    SlideConfig,
    get_weights_path,
    train as train_mrcnn,
    InferenceConfig,
    crop_predictions,
)
from mrcnn_model.mrcnn.model import MaskRCNN
from rich import print as rprint
from rich.console import Console


class TrainNetworkType(str, Enum):
    YOLO = "yolo"
    MASK_RCNN = "maskrcnn"
    ALL = "all"


class InferenceNetworkType(str, Enum):
    YOLO = "yolo"
    MASK_RCNN = "maskrcnn"


class MaskRCNNWeights(str, Enum):
    COCO = "coco"
    IMAGENET = "imagenet"
    NONE = "none"


def train_yolo(epochs: int):
    """Train the YOLO model

    Args:
        epochs (int): number of epochs to train on.
    """
    create_local_yolo_settings()
    train_model(epochs=epochs)


def train_mask_rcnn(epochs: int, weights_name: str):
    """Trains the MASK RNN model.

    Args:
        epochs (int): number of epochs to train on.
        weights_name (str): weights to train on.
    """
    config = SlideConfig()
    model_log_path = os.path.join(DEFAULT_LOGS_DIR, weights_name)
    if not os.path.exists(DEFAULT_LOGS_DIR):
        os.mkdir(DEFAULT_LOGS_DIR)
    if not os.path.exists(model_log_path):
        os.mkdir(model_log_path)
    model = MaskRCNN(mode="training", config=config, model_dir=model_log_path)
    weights_path = get_weights_path(weights_name, model)

    model.load_weights(
        weights_path,
        by_name=True,
        exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask",
        ],
    )
    train_mrcnn(model, epochs, dataset_path=DATASET, conf=config)


def detect_mask_rcnn(model_path: str):
    config = InferenceConfig()
    model = MaskRCNN(
        mode="inference", config=config, model_dir=os.path.dirname(SAVED_MODELS)
    )
    model.load_weights(
        model_path,
        by_name=True,
        exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask",
        ],
    )
    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(
        dir_path, "data_fetch", "prepared_data", "maskrcnn_data", "test"
    )
    save_path = os.path.join(dir_path, "image_crops", "maskrcnn_crops")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    crop_predictions(model, test_data_path, save_path)


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
                "MASK RCNN weights can only be applied to MASK RCNN."
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
        test(model_path)
    elif model_name is InferenceNetworkType.MASK_RCNN:
        raise BadParameter("Not implemented yet.")
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
        detect(model_path)
    elif model_name is InferenceNetworkType.MASK_RCNN:
        detect_mask_rcnn(model_path)
    else:
        raise BadParameter("Only YOLO and MASK RCNN models are supported.")


if __name__ == "__main__":
    CLI()
