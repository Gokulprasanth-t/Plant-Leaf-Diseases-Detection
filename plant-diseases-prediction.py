# -*- coding: utf-8 -*-
"""Yolov8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CiCydFqiT08714cS2O3s3flxijeot6os
"""

!pip install ultralytics

import ultralytics
ultralytics.checks()

!curl -L "https://app.roboflow.com/ds/fH27xyiG8o?key=9TO0Gwshn6" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

from ultralytics import YOLO
model = YOLO("yolov8x.pt")

model.train(data="/content/data.yaml",epochs=50,imgsz=640)