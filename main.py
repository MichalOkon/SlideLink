"""Main CLI APPlication file."""
from enum import Enum
from typer import Typer, Argument, Option
from typing_extensions import Annotated
from yolo_model.model_run import train_model, create_local_yolo_settings


class NetworkType(str, Enum):
    YOLO = "yolo"
    MASK_RCNN = "maskrcnn"
    LOFTR = "loftr"


APP = Typer()


@APP.command()
def train(
    model_name: Annotated[
        NetworkType,
        Argument(
            metavar="model-type",
            help="The name of the model to train.",
        ),
    ],
    epochs: Annotated[
        int, Option(help="Number of epochs to train model on.")
    ] = 100,
):
    match model_name:
        case NetworkType.YOLO:
            create_local_yolo_settings()
            train_model(epochs=epochs)
        case NetworkType.MASK_RCNN:
            print(f"Picked {model_name.value}")
        case NetworkType.LOFTR:
            print(f"Picked {model_name.value}")


@APP.command()
def evaluate(model_name: str, model_path: str = "", verbose: bool = False):
    """Evaluates a model on the test set with mAP and F1 scores."""
    from_text = f" (from {model_path})" if model_path else ""
    print(f"Picked {model_name}{from_text}. Have a good day.")
    if verbose:
        print("Verbose: ON")


if __name__ == "__main__":
    APP()
