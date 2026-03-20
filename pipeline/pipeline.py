from io import BytesIO
import os
from queue import Full, Queue
from threading import Lock, Thread
import time

import cv2
import numpy as np
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
        self.vlm_queue_size = int(os.getenv("VLM_QUEUE_SIZE", "8"))
        # 5 seconds is long enough to survive short occlusion/re-entry, but short enough to avoid stale rematches
        self.rematch_timeout_sec = float(os.getenv("PERSON_REMATCH_TIMEOUT_SEC", "20"))
        self.rematch_similarity_threshold = float(
            os.getenv("PERSON_REMATCH_THRESHOLD", "0.5"))

    def _extract_crop_image(self, img: Image.Image, bbox: list[float]) -> Image.Image | None:
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

        return img.crop((x1, y1, x2, y2))

    def _encode_crop_image(self, crop: Image.Image) -> bytes:
        resized_crop = self._resize_crop(crop)
        buf = BytesIO()
        resized_crop.save(buf, format="JPEG")  # save to memory buffer instead of disk
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
            crop = self._extract_crop_image(img, bbox)

        if crop is None:
            return None

        return self._encode_crop_image(crop)

    def _extract_crop_from_frame(self, frame, bbox: list[float]) -> Image.Image | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self._extract_crop_image(image, bbox)

    def _crop_person_from_frame(self, frame, bbox: list[float]) -> bytes | None:
        crop = self._extract_crop_from_frame(frame, bbox)

        if crop is None:
            return None

        return self._encode_crop_image(crop)

    def _normalize_feature_vector(self, feature: np.ndarray) -> np.ndarray:
        feature = feature.astype(np.float32).flatten()
        norm = np.linalg.norm(feature)

        if norm == 0:
            return feature

        return feature / norm

    def _compute_appearance_embedding(self, crop: Image.Image) -> np.ndarray:
        # use a mix of color and texture so rematching is less brittle than hsv alone
        resized = crop.convert("RGB").resize((96, 192), Image.Resampling.BILINEAR)
        rgb = np.array(resized)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # keep some color information, but split the crop so upper and lower clothing matter separately
        upper_hsv = hsv[: hsv.shape[0] // 2, :, :]
        lower_hsv = hsv[hsv.shape[0] // 2:, :, :]
        upper_hist = cv2.calcHist([upper_hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        lower_hist = cv2.calcHist([lower_hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        upper_hist = self._normalize_feature_vector(upper_hist)
        lower_hist = self._normalize_feature_vector(lower_hist)

        # add coarse rgb statistics so overall brightness and channel balance are represented
        rgb_mean = rgb.mean(axis=(0, 1)).astype(np.float32) / 255.0
        rgb_std = rgb.std(axis=(0, 1)).astype(np.float32) / 255.0

        # add gradient orientation features so body outline and clothing texture help rematching
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        orientation_hist, _ = np.histogram(
            angle,
            bins=16,
            range=(0.0, 360.0),
            weights=magnitude
        )
        orientation_hist = self._normalize_feature_vector(orientation_hist)

        # include simple shape cues so a very different crop is less likely to steal an id
        aspect_ratio = np.array(
            [rgb.shape[1] / max(rgb.shape[0], 1)],
            dtype=np.float32
        )

        feature = np.concatenate([
            upper_hist,
            lower_hist,
            rgb_mean,
            rgb_std,
            orientation_hist,
            aspect_ratio,
        ]).astype(np.float32)
        return self._normalize_feature_vector(feature)

    def _appearance_similarity(self, left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None:
            return -1.0

        return float(np.dot(left, right))

    def _blend_embeddings(self, current: np.ndarray | None, new: np.ndarray) -> np.ndarray:
        if current is None:
            return new

        blended = (0.7 * current) + (0.3 * new)
        norm = np.linalg.norm(blended)

        if norm == 0:
            return blended

        return blended / norm

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
            attributes = person["description"]["attributes"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # keep overlay short so it stays readable in the webcam window
            label1 = f"ID {pid}: {action}"
            label2 = f"Attributes: {attributes}"

            # First line
            cv2.putText(
                annotated,
                label1,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Second line (shifted down)
            cv2.putText(
                annotated,
                label2,
                (x1, max(25, y1 - 10) + 25),  # adjust spacing here
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return annotated

    def _new_identity(self, person_id: int, embedding: np.ndarray, current_time_sec: float):
        return {
            "id": person_id,
            "embedding": embedding,
            "description": None,
            "last_seen": current_time_sec
        }

    def _prune_stale_identities(
        self,
        identities: dict[int, dict],
        track_to_person: dict[int, int],
        pending_ids: set[int],
        current_time_sec: float
    ):
        expired_ids = []

        for person_id, identity in identities.items():
            if person_id in pending_ids:
                continue  # keep identities alive until a queued VLM request finishes

            if current_time_sec - identity["last_seen"] > self.rematch_timeout_sec:
                expired_ids.append(person_id)

        for person_id in expired_ids:
            del identities[person_id]
            print(
                f"[Pipeline] Deleted Person {person_id} after {self.rematch_timeout_sec:.1f}s without a match.")

        for track_id, person_id in list(track_to_person.items()):
            if person_id not in identities:
                del track_to_person[track_id]

    def _resolve_person_id(
        self,
        track_id: int,
        crop: Image.Image,
        current_time_sec: float,
        identities: dict[int, dict],
        track_to_person: dict[int, int],
        next_person_id: int,
        reserved_person_ids: set[int]
    ) -> tuple[int, int]:
        embedding = self._compute_appearance_embedding(crop)

        if track_id in track_to_person and track_to_person[track_id] in identities:
            person_id = track_to_person[track_id]
            identity = identities[person_id]
            identity["embedding"] = self._blend_embeddings(
                identity["embedding"], embedding)
            identity["last_seen"] = current_time_sec
            reserved_person_ids.add(person_id)
            return person_id, next_person_id

        best_person_id = None
        best_score = -1.0

        for person_id, identity in identities.items():
            if person_id in reserved_person_ids:
                continue  # do not let two tracks in one frame claim the same logical person

            age_sec = current_time_sec - identity["last_seen"]
            if age_sec > self.rematch_timeout_sec:
                continue

            score = self._appearance_similarity(identity["embedding"], embedding)
            if score > best_score:
                best_score = score
                best_person_id = person_id

        if best_person_id is not None and best_score >= self.rematch_similarity_threshold:
            person_id = best_person_id
            identity = identities[person_id]
            identity["embedding"] = self._blend_embeddings(
                identity["embedding"], embedding)
            identity["last_seen"] = current_time_sec
            track_to_person[track_id] = person_id
            reserved_person_ids.add(person_id)
            print(
                f"[Pipeline] Re-matched ByteTrack ID {track_id} to Person {person_id} (similarity={best_score:.2f}).")
            return person_id, next_person_id

        # for testing remove later: show why an eligible old identity was not reused
        if best_person_id is not None:
            print(
                f"[Pipeline] ByteTrack ID {track_id} did not rematch. "
                f"Best candidate was Person {best_person_id} with similarity={best_score:.2f} "
                f"below threshold={self.rematch_similarity_threshold:.2f}.")

        person_id = next_person_id
        identities[person_id] = self._new_identity(
            person_id, embedding, current_time_sec)
        track_to_person[track_id] = person_id
        reserved_person_ids.add(person_id)
        print(f"[Pipeline] Created Person {person_id} for ByteTrack ID {track_id}.")
        return person_id, next_person_id + 1

    def _build_person_payload(self, person_id: int, description: PersonDescription | None, is_pending: bool):
        if description is not None:
            return description.model_dump()

        if is_pending:
            return {
                "id": person_id,
                "action": "Analyzing...",
                "attributes": "pending..."
            }

        return {
            "id": person_id,
            "action": "Unknown",
            "attributes": "no VLM result yet"
        }

    def _start_vlm_worker(self, identities: dict[int, dict], pending_ids: set[int], state_lock: Lock):
        task_queue: Queue = Queue(maxsize=self.vlm_queue_size)
        stop_token = object()

        def worker():
            while True:
                task = task_queue.get()

                if task is stop_token:
                    task_queue.task_done()
                    break

                person_id = task["person_id"]
                crop_bytes = task["crop_bytes"]

                try:
                    description = self.vlm.describe_person_crop(crop_bytes, person_id)

                    with state_lock:
                        if person_id in identities and description is not None:
                            identities[person_id]["description"] = description
                            print(
                                f"[Pipeline] Person {person_id}: action={description.action}, attributes={description.attributes}")
                        elif description is None:
                            print(
                                f"[Pipeline] Skipping Person {person_id} due to VLM parse failure.")

                        pending_ids.discard(person_id)  # mark job finished whether it succeeded or not
                finally:
                    task_queue.task_done()

        worker_thread = Thread(target=worker, daemon=True)
        worker_thread.start()

        return task_queue, stop_token, worker_thread

    def _stop_vlm_worker(self, task_queue: Queue, stop_token, worker_thread: Thread):
        try:
            task_queue.put(stop_token, timeout=1)  # ask the background worker to exit cleanly
        except Full:
            return

        worker_thread.join(timeout=2)

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
        identities: dict[int, dict] = {}
        track_to_person: dict[int, int] = {}
        next_person_id = 1
        frame_index = 0

        self.detector.reset_tracker()  # start each video with a fresh ByteTrack state

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                current_time_sec = frame_index if fps is None else frame_index / fps
                self._prune_stale_identities(
                    identities, track_to_person, set(), current_time_sec)

                detections = self.detector.track_people(frame)
                people_in_frame = []
                reserved_person_ids = set()

                for det in detections:
                    track_id = det["id"]

                    if track_id is None:
                        continue  # tracker sometimes needs warmup before an ID is assigned

                    crop = self._extract_crop_from_frame(frame, det["bbox"])
                    if crop is None:
                        print(
                            f"[Pipeline] Skipping ByteTrack ID {track_id} due to invalid crop.")
                        continue

                    person_id, next_person_id = self._resolve_person_id(
                        track_id,
                        crop,
                        current_time_sec,
                        identities,
                        track_to_person,
                        next_person_id,
                        reserved_person_ids
                    )

                    identity = identities[person_id]
                    description = identity["description"]

                    if description is None:
                        print(f"Running VLM on Person {person_id}...")
                        crop_bytes = self._encode_crop_image(crop)
                        description = self.vlm.describe_person_crop(
                            crop_bytes, person_id)

                        if description is None:
                            print(
                                f"[Pipeline] Skipping Person {person_id} due to VLM parse failure.")
                        else:
                            identity["description"] = description

                    people_in_frame.append({
                        "id": person_id,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det["class"],
                        "description": self._build_person_payload(
                            person_id, identity["description"], False)
                    })

                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_sec": current_time_sec,
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
            "people": [identities[pid]["description"].model_dump() for pid in sorted(identities)
                       if identities[pid]["description"] is not None]
        }

    def _run_webcam(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise ValueError(f"Could not open webcam index: {camera_index}")

        identities: dict[int, dict] = {}
        track_to_person: dict[int, int] = {}
        pending_ids: set[int] = set()
        state_lock = Lock()
        next_person_id = 1
        task_queue, stop_token, worker_thread = self._start_vlm_worker(
            identities, pending_ids, state_lock)
        self.detector.reset_tracker()  # start each webcam session with a fresh ByteTrack state

        print("Webcam started. Press 'q' to quit.")

        try:
            while True:
                ok, frame = cap.read()

                if not ok:
                    print("[Pipeline] Failed to read webcam frame, stopping.")
                    break

                current_time_sec = time.monotonic()
                with state_lock:
                    self._prune_stale_identities(
                        identities, track_to_person, pending_ids, current_time_sec)

                detections = self.detector.track_people(frame)
                people_in_frame = []
                reserved_person_ids = set()

                for det in detections:
                    track_id = det["id"]

                    if track_id is None:
                        continue  # tracker sometimes needs warmup before an ID is assigned

                    crop = self._extract_crop_from_frame(frame, det["bbox"])
                    if crop is None:
                        print(
                            f"[Pipeline] Skipping ByteTrack ID {track_id} due to invalid crop.")
                        continue

                    with state_lock:
                        person_id, next_person_id = self._resolve_person_id(
                            track_id,
                            crop,
                            current_time_sec,
                            identities,
                            track_to_person,
                            next_person_id,
                            reserved_person_ids
                        )
                        identity = identities[person_id]
                        description = identity["description"]
                        is_pending = person_id in pending_ids

                    if description is None and not is_pending:
                        print(f"Queueing VLM for Person {person_id}...")
                        crop_bytes = self._encode_crop_image(crop)

                        queued = False
                        with state_lock:
                            if identities[person_id]["description"] is None and person_id not in pending_ids:
                                try:
                                    task_queue.put_nowait({
                                        "person_id": person_id,
                                        "crop_bytes": crop_bytes
                                    })
                                    pending_ids.add(
                                        person_id)  # prevent duplicate jobs while the worker is busy
                                    queued = True
                                except Full:
                                    print(
                                        f"[Pipeline] VLM queue is full, skipping Person {person_id} for now.")

                        if queued:
                            is_pending = True

                    with state_lock:
                        description = identities[person_id]["description"]

                    people_in_frame.append({
                        "id": person_id,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class": det["class"],
                        "description": self._build_person_payload(
                            person_id, description, is_pending)
                    })

                annotated = self._annotate_frame(frame, people_in_frame)
                cv2.imshow("AStar Webcam Tracking", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            self._stop_vlm_worker(task_queue, stop_token, worker_thread)
            cap.release()
            cv2.destroyAllWindows()
            self.detector.reset_tracker()  # avoid leaking IDs into the next run

        return {
            "source": f"webcam:{camera_index}",
            "people": [identities[pid]["description"].model_dump() for pid in sorted(identities)
                       if identities[pid]["description"] is not None]
        }

    def run(self, source_path: str):
        if self._is_webcam_source(source_path):
            camera_index = int(source_path) if source_path.isdigit() else 0
            return self._run_webcam(camera_index)

        if self._is_video_source(source_path):
            return self._run_video(source_path)

        return self._run_image(source_path)
