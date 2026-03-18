from io import BytesIO
import os

import cv2
from PIL import Image

from detection.yolo_detector import YOLODetector
from vlm.vlm import PersonDescription, VLMClient


class PerceptionPipeline:
    def __init__(self, yolo_conf: float = 0.5):
        self.vlm = VLMClient()
        self.detector = YOLODetector(conf_threshold=yolo_conf)
        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
        self.webcam_sources = {"webcam", "camera", "cam"}
        self.max_crop_side = int(os.getenv("VLM_MAX_CROP_SIDE", "896"))

    def _crop_person_from_image(self, img: Image.Image, bbox: list[float]) -> bytes | None:
        x1, y1, x2, y2 = map(int, bbox)  # convert bbox floats to pixel ints

        pad = 10  # add padding so the person isn't cut off at the box edge
        width, height = img.size
        # clamp to image bounds so we don't crop outside
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return None  # skip invalid crops instead of sending junk to the VLM

        crop = img.crop((x1, y1, x2, y2))
        crop = self._resize_crop(crop)

        buf = BytesIO()
        crop.save(buf, format="JPEG")  # save to memory buffer instead of disk
        return buf.getvalue()

    def _resize_crop(self, crop: Image.Image) -> Image.Image:
        width, height = crop.size
        longest_side = max(width, height)

        if longest_side <= self.max_crop_side:
            return crop  # keep small crops untouched so we do not lose detail for no reason

        scale = self.max_crop_side / longest_side
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return crop.resize(new_size, Image.Resampling.LANCZOS)

    def _crop_person(self, image_path: str, bbox: list[float]) -> bytes | None:
        with Image.open(image_path) as img:
            return self._crop_person_from_image(img, bbox)

    def _crop_person_from_frame(self, frame, bbox: list[float]) -> bytes | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self._crop_person_from_image(image, bbox)

    def _is_video_source(self, source_path: str) -> bool:
        lowered = source_path.lower()
        return any(lowered.endswith(ext) for ext in self.video_extensions)

    def _is_webcam_source(self, source_path: str) -> bool:
        return source_path.lower() in self.webcam_sources or source_path.isdigit()

    def _annotate_frame(self, frame, people_in_frame: list[dict]):
        annotated = frame.copy()

        for person in people_in_frame:
            x1, y1, x2, y2 = map(int, person["bbox"])
            pid = person["id"]
            action = person["description"]["action"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # keep overlay short so it stays readable in the webcam window
            label = f"ID {pid}: {action}"
            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return annotated

    def _run_image(self, image_path: str):
        detections = self.detector.detect(image_path)

        # assign temporary ids so the image path still matches the old output format
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
            if crop_bytes is None:
                print(f"[Pipeline] Skipping Person {pid} due to invalid crop.")
                continue

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

    def _run_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else None

        frame_results = []
        people_by_id: dict[int, PersonDescription] = {}
        frame_index = 0

        self.detector.reset_tracker()  # start each video with a fresh ByteTrack state

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                detections = self.detector.track_people(frame)
                people_in_frame = []

                for det in detections:
                    pid = det["id"]

                    if pid is None:
                        continue  # tracker sometimes needs warmup before an ID is assigned

                    description = people_by_id.get(pid)

                    if description is None:
                        print(f"Running VLM on Person {pid}...")
                        crop_bytes = self._crop_person_from_frame(frame, det["bbox"])

                        if crop_bytes is None:
                            print(f"[Pipeline] Skipping Person {pid} due to invalid crop.")
                            continue

                        description = self.vlm.describe_person_crop(crop_bytes, pid)

                        if description is None:
                            print(
                                f"[Pipeline] Skipping Person {pid} due to VLM parse failure.")
                            continue

                        people_by_id[pid] = description  # cache once per tracked person

                    people_in_frame.append({
                        "id": pid,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det["class"],
                        "description": description.model_dump()
                    })

                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_sec": None if fps is None else frame_index / fps,
                    "people": people_in_frame
                })
                frame_index += 1
        finally:
            cap.release()
            self.detector.reset_tracker()  # avoid leaking IDs into the next run

        return {
            "video": video_path,
            "fps": fps,
            "total_frames": frame_index,
            "frames": frame_results,
            "people": [people_by_id[pid].model_dump() for pid in sorted(people_by_id)]
        }

    def _run_webcam(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise ValueError(f"Could not open webcam index: {camera_index}")

        people_by_id: dict[int, PersonDescription] = {}
        self.detector.reset_tracker()  # start each webcam session with a fresh ByteTrack state

        print("Webcam started. Press 'q' to quit.")

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    print("[Pipeline] Failed to read webcam frame, stopping.")
                    break

                detections = self.detector.track_people(frame)
                people_in_frame = []

                for det in detections:
                    pid = det["id"]

                    if pid is None:
                        continue  # tracker sometimes needs warmup before an ID is assigned

                    description = people_by_id.get(pid)

                    if description is None:
                        print(f"Running VLM on Person {pid}...")
                        crop_bytes = self._crop_person_from_frame(frame, det["bbox"])

                        if crop_bytes is None:
                            print(f"[Pipeline] Skipping Person {pid} due to invalid crop.")
                            continue

                        description = self.vlm.describe_person_crop(crop_bytes, pid)

                        if description is None:
                            print(
                                f"[Pipeline] Skipping Person {pid} due to VLM parse failure.")
                            continue

                        people_by_id[pid] = description  # cache once per tracked person
                        print(
                            f"[Pipeline] Person {pid}: action={description.action}, attributes={description.attributes}")

                    people_in_frame.append({
                        "id": pid,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det["class"],
                        "description": description.model_dump()
                    })

                annotated = self._annotate_frame(frame, people_in_frame)
                cv2.imshow("AStar Webcam Tracking", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.reset_tracker()  # avoid leaking IDs into the next run

        return {
            "source": f"webcam:{camera_index}",
            "people": [people_by_id[pid].model_dump() for pid in sorted(people_by_id)]
        }

    def run(self, source_path: str):
        if self._is_webcam_source(source_path):
            camera_index = int(source_path) if source_path.isdigit() else 0
            return self._run_webcam(camera_index)

        if self._is_video_source(source_path):
            return self._run_video(source_path)

        return self._run_image(source_path)
