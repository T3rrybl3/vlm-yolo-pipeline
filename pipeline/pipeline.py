from vlm.vlm import VLMClient
from detection.yolo_detector import YOLODetector


class PerceptionPipeline:
    def __init__(self):
        self.vlm = VLMClient()
        self.detector = YOLODetector()

    def run(self, image_path: str):
        detections = self.detector.detect(image_path)

        # Extract useful info
        object_classes = [d["class"] for d in detections]
        people_count = object_classes.count("person")  # number of people

        # Smarter prompt
        prompt = f"""
Detected objects: {set(object_classes)}
Number of people: {people_count}

Describe what each person is doing and any unusual behaviour.
"""

        description = self.vlm.describe(image_path, prompt)

        return {
            "detections": detections,
            "summary": description
        }
