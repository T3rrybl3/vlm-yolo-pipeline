from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):  # change model if needed
        self.model = YOLO(model_path)

    def detect(self, image_path: str):
        results = self.model(image_path)

        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    # convert from tensor to python values
                    "bbox": list(map(float, box.xyxy[0]))
                })
        print(f"Passed YOLO")
        return detections  # returns a list of 2 dicts
