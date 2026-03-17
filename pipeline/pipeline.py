from vlm.vlm import VLMClient, PersonDescription
from detection.yolo_detector import YOLODetector
from io import BytesIO
from PIL import Image


class PerceptionPipeline:
    def __init__(self, yolo_conf: float = 0.5):
        self.vlm = VLMClient()
        self.detector = YOLODetector(conf_threshold=yolo_conf)

    def _crop_person(self, image_path: str, bbox: list[float]) -> bytes:
        x1, y1, x2, y2 = map(int, bbox)  # convert bbox floats to pixel ints

        with Image.open(image_path) as img:
            pad = 10  # add padding so the person isn't cut off at the box edge
            width, height = img.size
            # clamp to image bounds so we don't crop outside
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)

            crop = img.crop((x1, y1, x2, y2))

        buf = BytesIO()
        crop.save(buf, format="JPEG")  # save to memory buffer instead of disk
        return buf.getvalue()

    def run(self, image_path: str):
        detections = self.detector.detect(image_path)

        # assign id to detected people
        person_id = 1
        for det in detections:
            if det["class"] == "person":
                det["id"] = person_id
                person_id += 1

        # filter to only people for VLM stage
        people_detections = [d for d in detections if d["class"] == "person"]

        if not people_detections:  # skip VLM entirely if no people found, saves time
            print("No people detected, skipping VLM.")
            return {"detections": detections, "people": []}

        # --- Stage 2: per-person crop + VLM inference ---
        people_results: list[PersonDescription] = []

        for det in people_detections:
            pid = det["id"]
            print(f"Running VLM on Person {pid}...")

            # crop just this person from the full image
            crop_bytes = self._crop_person(image_path, det["bbox"])
            description = self.vlm.describe_person_crop(
                crop_bytes, pid)  # VLM only sees one person at a time

            if description is not None:
                people_results.append(description)
            else:
                # bad response, move on
                print(
                    f"[Pipeline] Skipping Person {pid} due to VLM parse failure.")

        return {
            "detections": detections,
            # convert pydantic objects to plain dicts
            "people": [p.model_dump() for p in people_results]
        }
