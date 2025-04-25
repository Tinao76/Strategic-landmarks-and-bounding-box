
from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=60, imgsz=640)
