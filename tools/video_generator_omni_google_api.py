"""Gemini Omni video through the Gemini API Interactions endpoint (Google AI Studio key).

https://ai.google.dev/gemini-api/docs/interactions
The call is synchronous: the response carries the finished video as base64 inside steps[].
Frames: none (text to video), one (first frame) or two (first and last frame, 3-10 s).
"""

import asyncio
import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import List

import requests

from interfaces.video_output import VideoOutput

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GoogleVideoAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Gemini Interactions video request failed with HTTP {status}: {body[:600]}")
        self.status = status


class VideoGeneratorOmniGoogleAPI:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-omni-1.2-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        attempts: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[str] = [],
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> VideoOutput:
        duration = int(kwargs.get("duration") or os.environ.get("VIMAX_VIDEO_DURATION", "8"))
        resolution = kwargs.get("resolution") or os.environ.get("VIMAX_VIDEO_RESOLUTION", "720p")
        data, meta = await self.generate_video_with_response(prompt, reference_image_paths, aspect_ratio, duration, resolution)
        return VideoOutput(fmt="bytes", ext="mp4", data=data, metadata=meta)

    def request_body(self, prompt: str, frames: List[str], aspect_ratio: str, duration: int, resolution: str) -> dict:
        if len(frames) > 2:
            raise ValueError("Gemini Omni takes at most two frames (first and last)")
        return {
            "model": self.model,
            "input": [_image_part(path) for path in frames] + [{"type": "text", "text": prompt}],
            "response_format": {"type": "video", "duration": f"{duration}s", "aspect_ratio": aspect_ratio, "resolution": resolution},
        }

    async def generate_video_with_response(self, prompt: str, frames: List[str], aspect_ratio: str, duration: int, resolution: str = "720p") -> tuple[bytes, dict]:
        """Returns the mp4 bytes and the response without the video payload, for bookkeeping."""
        body = self.request_body(prompt, frames, aspect_ratio, duration, resolution)
        return await asyncio.to_thread(self._post, body)

    def _post(self, body: dict) -> tuple[bytes, dict]:
        url = f"{self.base_url}/interactions"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        timeout = float(os.environ.get("VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS", "900"))
        for attempt in range(1, self.attempts + 1):
            logging.info(f"Calling {self.model} through the Gemini Interactions API, attempt {attempt}")
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                encoded = _find_video(payload)
                if encoded is None:
                    raise RuntimeError(f"Gemini Interactions response has no video, status={payload.get('status')}: {str(_strip_media(payload))[:600]}")
                return base64.b64decode(encoded), _strip_media(payload)
            if response.status_code not in RETRYABLE_STATUS or attempt == self.attempts:
                raise GoogleVideoAPIError(response.status_code, response.text)
            time.sleep(10 * attempt)
        raise AssertionError("unreachable")


def _image_part(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return {"type": "image", "data": base64.b64encode(Path(path).read_bytes()).decode(), "mime_type": mime}


def _find_video(node):
    if isinstance(node, dict):
        if node.get("type") == "video" and node.get("data"):
            return node["data"]
        for value in node.values():
            if (found := _find_video(value)) is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            if (found := _find_video(value)) is not None:
                return found
    return None


def _strip_media(node):
    if isinstance(node, dict):
        return {k: (f"<omitted {len(v)} chars>" if k == "data" and isinstance(v, str) else _strip_media(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_media(v) for v in node]
    return node
