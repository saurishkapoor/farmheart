from ultralytics import YOLO

# Load YOLOv8 classification model (pretrained on ImageNet)
model = YOLO("yolov8n-cls.pt")  # 'n' is for nano (smallest), you can use yolov8s-cls.pt for a bigger model

# Train model
results = model.train(
    data="split_data",    # Path to dataset folder
    epochs=100,         # Number of training epochs
    imgsz=224,         # Image size (224x224 for classification)
    batch=32,          # Batch size
)