import os
import sys

import torch
from ultralytics import YOLO



if __name__ == '__main__':
        model = YOLO("yolov8n.pt")
        torch.cuda.empty_cache()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(torch.cuda.is_available())
        print(torch.__version__)
        print(torch.backends.cudnn.version())
        results = model.train(
                batch=2,
                device=0,
                data="data.yaml",
                epochs=7,
                imgsz=640,
        )