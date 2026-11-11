import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_runtime.vimax_adapters import _build_image_generator, _build_video_generator
from tools.image_generator_nanobanana_google_api import ImageGeneratorNanobananaGoogleAPI
from tools.video_generator_omni_google_api import GoogleVideoAPIError, VideoGeneratorOmniGoogleAPI
from tools.video_generator_veo_google_api import VideoGeneratorVeoGoogleAPI


def _response(status, payload=None, text=""):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload or {}
    response.text = text
    return response


class GoogleGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def test_factories_select_google_by_provider(self):
        with patch("agent_runtime.vimax_adapters.image_provider", return_value="google"), \
             patch("agent_runtime.vimax_adapters.image_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="gemini-nano-banana-2.1"):
            image = _build_image_generator()
        self.assertIsInstance(image, ImageGeneratorNanobananaGoogleAPI)
        self.assertEqual(image.model, "gemini-nano-banana-2.1")
        common = [patch("agent_runtime.vimax_adapters.video_provider", return_value="google"),
                  patch("agent_runtime.vimax_adapters.video_api_key", return_value="secret"),
                  patch("agent_runtime.vimax_adapters.video_base_url", return_value="https://generativelanguage.googleapis.com/v1beta")]
        for item in common:
            item.start()
        self.addCleanup(patch.stopall)
        with patch("agent_runtime.vimax_adapters.video_model", return_value="gemini-omni-1.2-flash"):
            self.assertIsInstance(_build_video_generator(), VideoGeneratorOmniGoogleAPI)
        with patch("agent_runtime.vimax_adapters.video_model", return_value="veo-3.1-generate-preview"):
            self.assertIsInstance(_build_video_generator(), VideoGeneratorVeoGoogleAPI)

    async def test_omni_sends_frames_and_decodes_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, last = Path(tmp) / "a.png", Path(tmp) / "b.png"
            first.write_bytes(b"first")
            last.write_bytes(b"last")
            video = base64.b64encode(b"mp4-bytes").decode()
            payload = {"id": "int-1", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "video", "data": video, "mime_type": "video/mp4"}]}]}
            post = MagicMock(return_value=_response(200, payload))
            generator = VideoGeneratorOmniGoogleAPI(api_key="secret")
            with patch("tools.video_generator_omni_google_api.requests.post", post):
                data, meta = await generator.generate_video_with_response("walks", [str(first), str(last)], "9:16", 6)
        self.assertEqual(data, b"mp4-bytes")
        self.assertEqual(meta["id"], "int-1")
        self.assertNotIn(video, str(meta))
        body = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "secret")
        self.assertEqual([part["type"] for part in body["input"]], ["image", "image", "text"])
        self.assertEqual(body["response_format"], {"type": "video", "duration": "6s", "aspect_ratio": "9:16", "resolution": "720p"})

    async def test_omni_does_not_retry_client_errors(self):
        post = MagicMock(return_value=_response(400, text="bad"))
        with patch("tools.video_generator_omni_google_api.requests.post", post):
            with self.assertRaises(GoogleVideoAPIError):
                await VideoGeneratorOmniGoogleAPI(api_key="secret").generate_video_with_response("x", [], "9:16", 5)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
