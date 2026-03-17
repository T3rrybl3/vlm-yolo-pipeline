from vlm.vlm import VLMClient
from detection.yolo_detector import YOLODetector


class PerceptionPipeline:
    def __init__(self):
        self.vlm = VLMClient()
        self.detector = YOLODetector()

    def run(self, image_path: str):
        detections = self.detector.detect(image_path)

        # assign id to detected people
        person_id = 1
        for det in detections:
            if det["class"] == "person":
                det["id"] = person_id
                person_id += 1

        # Extract useful info
        object_classes = [d["class"] for d in detections]
        people_count = object_classes.count("person")  # number of people

        # list all people with ids
        person_lines = []
        for det in detections:
            if det["class"] == "person":
                person_lines.append(f"Person {det['id']}")

        person_list_str = "\n".join(person_lines)

        prompt = f"""
Detected objects: {set(object_classes)}
Number of people: {people_count}

Describe each person listed below individually.
{person_list_str} 

Return ONLY JSON in this format:
{{
"people": [
    {{"id": int, "action": str, "attributes": str}}
]
}}
No extra text, no markdown.
"""

        vlm_output_obj = self.vlm.describe_structured(image_path, prompt)

        vlm_output = vlm_output_obj.people
        return {
            "detections": detections,
            "people": vlm_output
        }
