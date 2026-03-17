from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):  # change model if needed
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold  # boxes below this score get ignored

    def detect(self, image_path: str):
        results = self.model(image_path)

        detections = []

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])

                if confidence < self.conf_threshold:  # skip weak detections before they reach VLM
                    continue
                detections.append({
                    "class": self.model.names[int(box.cls[0])],
                    "confidence": confidence,
                    # convert from tensor to python values
                    "bbox": list(map(float, box.xyxy[0])),
                    "id": None  # placeholder, will be set in pipeline
                })
        print(
            f"YOLO detected {len(detections)} objects above conf={self.conf_threshold}")
        return detections  # returns a list of 2 dicts
