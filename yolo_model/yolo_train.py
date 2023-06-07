import os
import sys

import torch
from ultralytics import YOLO
from PIL import Image
import cv2

from LoFTR.src.loftr import LoFTR
from LoFTR.src.loftr.utils.cvpr_ds_config import default_cfg


def train_model():
    model = YOLO("yoyololov8n.pt")
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.backends.cudnn.version())
    results = model.train(
        batch=1,
        device=0,
        data="data.yaml",
        epochs=100,
        imgsz=640,
    )


def detect(model_path, data_dir):
    model = YOLO(model_path)
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source=data_dir, save=True, save_crop=True)
    for i, result in enumerate(results):
        print(result.boxes.xyxy)

# def pair_slides():
#     matcher = LoFTR(config=default_cfg)
#     matcher.load_state_dict(torch.load("weights/loftr_indoor_ds_new.ckpt")['state_dict'])
#     matcher = matcher.eval().cuda()
#
#
#     slides_dir = "../"
#     # Rerun this cell (and below) if a new image pair is uploaded.
#     img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
#     img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
#     img0_raw = cv2.resize(img0_raw, (640, 480))
#     img1_raw = cv2.resize(img1_raw, (640, 480))
#
#     img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
#     img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
#     batch = {'image0': img0, 'image1': img1}
#
#     # Inference with LoFTR and get prediction
#     with torch.no_grad():
#         matcher(batch)
#         mkpts0 = batch['mkpts0_f'].cpu().numpy()
#         mkpts1 = batch['mkpts1_f'].cpu().numpy()
#         mconf = batch['mconf'].cpu().numpy()

if __name__ == '__main__':
    # train_model()
    path = os.path.abspath(os.getcwd())
    print(path)
    detect("../runs/detect/train4/weights/best.pt", "test_dataset")
