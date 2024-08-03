from ultralytics import YOLO

YOLO("yolov8n", verbose=False).export(format="edgetpu")
