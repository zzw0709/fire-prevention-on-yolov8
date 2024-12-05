from ultralytics import YOLO
model = YOLO('best_cbam.pt')

#model.predict(source=0, imgsz=640, conf=0.6, show=True)
model.predict(source=0, imgsz=640, conf=0.6, show=True)
