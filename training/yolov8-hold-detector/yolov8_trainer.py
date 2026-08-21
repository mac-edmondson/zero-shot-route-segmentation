from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data/indoor-climbing-gym-hold-segmentation/yolov8_data/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    device=0,
    project="training",
    name="yolov8-hold-detector",
)