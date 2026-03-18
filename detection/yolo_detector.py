from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, tracker_config="bytetrack.yaml"):  # change model if needed
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold  # boxes below this score get ignored
        self.tracker_config = tracker_config  # ByteTrack config used for persistent IDs

    def detect(self, image_path: str):
        results = self.model(image_path, verbose=False)

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

    def track_people(self, frame):
        try:
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                classes=[0],  # only track people so ByteTrack IDs map to human crops
                conf=self.conf_threshold,
                persist=True,
                verbose=False
            )
        except ModuleNotFoundError as e:
            if e.name == "lap":
                raise RuntimeError(
                    "ByteTrack requires the 'lap' package. Install it with 'pip install lap'."
                ) from e
            raise

        result = results[0]
        detections = []

        if result.boxes is None:
            print(
                f"YOLO+ByteTrack tracked 0 people above conf={self.conf_threshold}")
            return detections

        boxes = result.boxes
        track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)

        for i, box in enumerate(boxes):
            detections.append({
                "class": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                # keep bbox/id in plain python types so the pipeline can serialize them
                "bbox": list(map(float, box.xyxy[0])),
                "id": None if track_ids[i] is None else int(track_ids[i])
            })

        print(
            f"YOLO+ByteTrack tracked {len(detections)} people above conf={self.conf_threshold}")
        return detections

    def reset_tracker(self):
        predictor = self.model.predictor

        if predictor is None or not hasattr(predictor, "trackers"):
            return

        for tracker in predictor.trackers:
            tracker.reset()  # clear old state before a new video starts

        predictor.vid_path = [None] * len(predictor.trackers)
