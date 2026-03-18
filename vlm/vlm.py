import base64
import httpx
import json
import os
from pydantic import BaseModel


class PersonDescription(BaseModel):  # used as a check for datatype
    id: int
    action: str
    attributes: str


class PeopleDescription(BaseModel):
    people: list[PersonDescription]


class VLMClient:
    def __init__(self, model=None):
        # 3B fits much better on a 6GB GPU, 7B was falling back to CPU on this machine
        self.model = model or os.getenv("VLM_MODEL", "qwen2.5vl:3b")
        self.url = os.getenv("VLM_URL", "http://localhost:11434/v1/chat/completions")
        self.timeout_sec = int(os.getenv("VLM_TIMEOUT_SEC", "120"))

    def _encode_image(self, path: str):  # convert image to format suitable for vlm
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"

        return f"data:{mime};base64,{b64}"

    # same as above but takes raw bytes instead of a path, used for crops
    def _encode_image_bytes(self, image_bytes: bytes, mime: str = "image/jpeg") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        return f"data:{mime};base64,{b64}"

    # single place that actually talks to the VLM, both file and crop paths go through here
    def _call(self, image_data_url: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "stream": False
        }

        # blocks until timeout or response
        resp = httpx.post(self.url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()  # used to check if smt wrong, if yes, throw an error

        return resp.json()["choices"][0]["message"]["content"]

    def describe(self, path: str, prompt: str) -> str:  # send a full image file to the VLM
        return self._call(self._encode_image(path), prompt)

    # send a cropped image (as bytes) to the VLM
    def describe_crop(self, image_bytes: bytes, prompt: str) -> str:
        return self._call(self._encode_image_bytes(image_bytes), prompt)

    # old method, kept in case full-image inference is needed elsewhere
    def describe_structured(self, path: str, prompt: str) -> PeopleDescription:

        raw = self.describe(path, prompt)
        return self._parse_people(raw)

    def describe_person_crop(self, image_bytes: bytes, person_id: int) -> PersonDescription | None:
        # focused prompt for a single person crop, reduces hallucination vs describing the full scene
        prompt = f"""You are looking at a cropped image of a single person (Person {person_id}).
 
Describe this person only. Return ONLY JSON in this format:
{{"id": {person_id}, "action": str, "attributes": str}}
 
- "action": what the person is doing (e.g. "walking", "sitting", "carrying a bag")
- "attributes": visible physical traits or clothing (e.g. "wearing red jacket, carrying backpack")
 
No extra text, no markdown, no explanation."""

        try:
            raw = self.describe_crop(image_bytes, prompt)
            clean = raw.strip().replace("```json", "").replace(
                "```", "")  # strip markdown fences if VLM adds them
            data = json.loads(clean)
            return PersonDescription(**data)  # validate types before returning
        except Exception as e:
            print(
                f"[VLM] Failed to parse response for Person {person_id}: {e}")
            return None  # return None so pipeline can skip this person cleanly

    # shared parser for full-image responses
    def _parse_people(self, raw: str) -> PeopleDescription:
        try:
            clean = raw.strip().replace("```json", "").replace("```", "")
            return PeopleDescription(**json.loads(clean))
        except Exception as e:
            print(f"[VLM] Failed to parse structured output: {e}")
            # return empty list instead of crashing
            return PeopleDescription(people=[])
