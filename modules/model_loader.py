# File: modules/model_loader.py

import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def load_yolo_model(model_path='yolov8n.pt'):
    """Load YOLOv8 model."""
    model = YOLO(model_path)
    return model


def init_tracker():
    """Initialize DeepSORT tracker."""
    tracker = DeepSort(max_age=30, n_init=1)
    return tracker
