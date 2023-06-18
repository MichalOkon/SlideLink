"""
Mask R-CNN
Train on the Slides dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 train_mrcnn_model.py train --dataset=/path/to/model/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 train_mrcnn_model.py train --dataset=/path/to/model/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python train_mrcnn_model.py train --dataset=/path/to/model/dataset --weights=imagenet

    # Apply color splash to an image
    python3 train_mrcnn_model.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 train_mrcnn_model.py splash --weights=last --video=<URL or path to file>
"""

import os
import cv2
import sys
import json
import wandb
import argparse
import datetime
import numpy as np
import skimage.draw
from typing import Union
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# The template for the code comes from
# https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVED_MODELS = os.path.join(ROOT_DIR, "saved_models")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import evaluate_model

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


DATASET = os.path.join(
    os.path.dirname(ROOT_DIR), "data_fetch", "prepared_data", "maskrcnn_data"
)
############################################################
#  Configurations
############################################################


class SlideConfig(Config):
    """Configuration for training on the slide dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "slides"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + slides

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Number of epochs to train on
    EPOCHS = 100


class InferenceConfig(SlideConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


############################################################
#  Dataset
############################################################


class SlideDataset(utils.Dataset):
    def load_slides(self, dataset_dir, subset):
        """Load a subset of the Slide dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("slide", 1, "Slide")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        with open(os.path.join(dataset_dir, "labels.json")) as f:
            annotations = json.load(f)
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a["regions"]) is dict:
                polygons = [
                    r["shape_attributes"] for r in a["regions"].values()
                ]
            else:
                polygons = [r["shape_attributes"] for r in a["regions"]]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a["filename"])
            image = imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "slide",
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
             one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a slide dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "slide":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])],
            dtype=np.uint8,
        )
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["y"], p["x"])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "slide":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train(
    model,
    epochs,
    conf: Config,
    dataset_path="",
    custom_callbacks=None,
    weights_name="",
):
    """Train the model."""
    # Training dataset.
    dataset_train = SlideDataset()
    dataset_train.load_slides(dataset_path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SlideDataset()
    dataset_val.load_slides(dataset_path, "val")
    dataset_val.prepare()

    # ***  Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=conf.LEARNING_RATE,
        epochs=epochs,
        layers="heads",
        custom_callbacks=custom_callbacks,
    )

    time_stamp = (
        str(datetime.datetime.now()).replace(":", "_").replace(" ", "_")
    )
    model_save_path = os.path.join(
        SAVED_MODELS, f"slides_{weights_name}_mask_rcnn_{time_stamp}.h5"
    )
    model.keras_model.save_weights(model_save_path)
    with open(
        os.path.join(
            SAVED_MODELS, f"slides_{weights_name}_mask_rcnn_{time_stamp}.json"
        ),
        "w",
    ) as outfile:
        json.dump(conf.display_dict(), outfile, cls=NumpyEncoder)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    if image.shape[-1] == 4:
        image = image[..., :3]
    gray = gray2rgb(rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = np.sum(mask, -1, keepdims=True) >= 1
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    file_name = ""
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r["masks"])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now()
        )
        imsave(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(
            datetime.datetime.now()
        )
        vwriter = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r["masks"])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def visualize_detection(model, image_path=""):
    img_cvt = imread(image_path)
    results = model.detect([img_cvt], verbose=1)

    # Visualize results
    r = results[0]

    print(r)
    visualize.display_instances(
        img_cvt,
        r["rois"],
        r["masks"],
        r["class_ids"],
        ["Slide"],
        r["scores"],
        show_caption=False,
        show_mask_polygon=True,
        show_mask=False,
    )


def crop_predictions(model, input_dir, output_dir):
    image_files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name)) and "png" in name
    ]
    for image_file in tqdm(image_files):
        img_cvt = imread(os.path.join(input_dir, image_file))
        if img_cvt.shape[-1] == 4:
            img_cvt = img_cvt[..., :3]
        results = model.detect([img_cvt])
        if len(results) == 0 or len(results[0]["rois"]) == 0:
            continue
        # Visualize results
        r = results[0]
        highest_score_index = np.argmax(r["scores"])
        big_box = r["rois"][highest_score_index]
        x, y, width, height = big_box
        crop_img = img_cvt[x:width, y:height]

        image_save_path = os.path.join(output_dir, image_file)
        cv2.imwrite(image_save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))


def get_weights_path(weights_name: str, model):
    """Select weights file to load.

    Args:
        weights_name (str): _description_

    Returns:
        str: _description_
    """
    # Select weights file to load
    if weights_name.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights_name.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights_name.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights_name
    return weights_path


def train_mask_rcnn(epochs: int, weights_name: str):
    """Trains the MASK RNN model.

    Args:
        epochs (int): number of epochs to train on.
        weights_name (str): weights to train on.
    """
    config = SlideConfig()
    config.display()
    model_log_path = os.path.join(DEFAULT_LOGS_DIR, weights_name)
    if not os.path.exists(DEFAULT_LOGS_DIR):
        os.mkdir(DEFAULT_LOGS_DIR)
    if not os.path.exists(model_log_path):
        os.mkdir(model_log_path)
    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=model_log_path
    )
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
    train(model, epochs, config, dataset_path=DATASET)


############################################################
#  Training
############################################################

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect slides."
    )
    parser.add_argument(
        "command", metavar="<command>", help="'train' or 'splash' or 'dectect'"
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/slide/dataset/",
        help="Directory of the Slide dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--image",
        required=False,
        metavar="path or URL to image",
        help="Image to apply the color splash effect on",
    )
    parser.add_argument(
        "--video",
        required=False,
        metavar="path or URL to video",
        help="Video to apply the color splash effect on",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        metavar="use wandb.ai tool",
        help="Use wandb.ai tool",
    )
    parser.add_argument(
        "--test-images",
        required=False,
        metavar="path to test images",
        help="Images to test predictions against",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "predict":
        assert (
            args.test_images
        ), "Argument --test-image is required for predictions"
    elif args.command == "splash":
        assert (
            args.image or args.video
        ), "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SlideConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(
            mode="training", config=config, model_dir=args.logs
        )
    else:
        model = modellib.MaskRCNN(
            mode="inference", config=config, model_dir=args.logs
        )

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)

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

    custom_callbacks = None
    if args.wandb:
        # Start a run, tracking hyperparameters
        wandb.init(
            # set the wandb project where this run will be logged
            project="slidelink",
            # track hyperparameters and run metadata with wandb.config
            config=config.display_dict(),
        )
        custom_callbacks = [
            WandbMetricsLogger(log_freq=1),
            WandbModelCheckpoint("models"),
        ]

    # Train or evaluate
    if args.command == "train":
        train(
            model,
            100,
            config,
            dataset_path=args.dataset,
            custom_callbacks=custom_callbacks,
            weights_name=args.weights.lower(),
        )
    elif args.command == "splash":
        detect_and_color_splash(
            model, image_path=args.image, video_path=args.video
        )
    elif args.command == "detect":
        visualize_detection(model, image_path=args.image)
    elif args.command == "predict":
        dir_path = os.path.dirname(ROOT_DIR)
        test_data_path = os.path.join(
            dir_path, "data_fetch", "prepared_data", "maskrcnn_data", "test"
        )
        save_path = os.path.join(dir_path, "image_crops", "maskrcnn_crops")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        crop_predictions(model, test_data_path, save_path)
    elif args.command == "evaluate":
        test_set = SlideDataset()
        print(args.dataset)
        test_set.load_slides(args.dataset, "test")
        test_set.prepare()
        test_mAP, test_mAR, f1_test = evaluate_model(test_set, model, config)
        print(f"mAP: {test_mAP}")
        print(f"mAR: {test_mAR}")
        print(f"F1 : {f1_test}")
    else:
        print(
            "'{}' is not recognized. "
            "Use 'train' or 'splash'".format(args.command)
        )
