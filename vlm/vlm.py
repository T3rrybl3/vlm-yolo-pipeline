import base64
import httpx
import json
from pydantic import BaseModel


class SceneDescription(BaseModel):  # used as a check for datatype
    people_count: int
    objects: list[str]
    scene_summary: str
    confidence: str


class VLMClient:
    def __init__(self, model="qwen2.5vl:7b"):
        self.model = model
        self.url = "http://localhost:11434/v1/chat/completions"

    def _encode_image(self, path: str):  # convert image to format suitable for vlm
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"

        return f"data:{mime};base64,{b64}"

    def describe(self, path: str, prompt: str):
        image_data = self._encode_image(path)

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "stream": False
        }

        # blocks until timeout or response
        resp = httpx.post(self.url, json=payload, timeout=240)
        resp.raise_for_status()  # used to check if smt wrong, if yes, throw an error

        return resp.json()["choices"][0]["message"]["content"]

    def describe_structured(self, path: str) -> SceneDescription:  # work in progress
        prompt = """Return ONLY JSON:
{
  "people_count": int,
  "objects": [strings],
  "scene_summary": "one sentence",
  "confidence": "high|medium|low"
}
"""

        raw = self.describe(path, prompt)

        clean = raw.strip().replace("```json", "").replace("```", "")
        return SceneDescription(**json.loads(clean))
