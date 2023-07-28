from ultralytics import YOLO


model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt", task="detect")

# Use the model
model.train(data="./bcc_yolo_config.yaml", epochs=300)  # train the model
