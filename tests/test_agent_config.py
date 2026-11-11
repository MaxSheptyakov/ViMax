import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agent_runtime.config import (
    api_provider_from_base_url,
    embedding_api_key,
    embedding_base_url,
    embedding_model,
    embedding_model_provider,
    image_api_key,
    image_base_url,
    image_model,
    image_num_candidates,
    image_provider,
    llm_provider,
    llm_api_key,
    llm_base_url,
    llm_model,
    llm_model_provider,
    load_agent_config,
    reranker_api_key,
    reranker_base_url,
    reranker_model,
    video_api_key,
    video_base_url,
    video_model,
    video_provider,
    music_api_key,
    music_base_url,
    music_model,
    music_provider,
    voice_api_key,
    voice_base_url,
    voice_model,
    voice_provider,
)


class AgentConfigTests(unittest.TestCase):
    def setUp(self):
        load_agent_config.cache_clear()

    def tearDown(self):
        load_agent_config.cache_clear()

    def test_reads_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "llm": {"model_provider": "openai", "model": "config-llm", "base_url": "https://config.test/v1", "api_key": "config-key"},
                "image": {"model": "config-image", "base_url": "https://image.test", "api_key": "image-key"},
                "video": {"model": "config-video", "base_url": "https://openrouter.ai/api/v1", "api_key": "video-key"},
                "embedding": {"model_provider": "openai", "model": "config-embedding", "base_url": "https://embedding.test/v1", "api_key": "embedding-key"},
                "reranker": {"model": "config-reranker", "base_url": "https://reranker.test", "api_key": "reranker-key"},
            }), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(llm_model(tmp), "config-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_base_url(tmp), "https://config.test/v1")
                self.assertEqual(llm_api_key(tmp), "config-key")
                self.assertEqual(image_model(tmp), "config-image")
                self.assertEqual(image_base_url(tmp), "https://image.test")
                self.assertEqual(image_api_key(tmp), "image-key")
                self.assertEqual(video_model(tmp), "config-video")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(video_api_key(tmp), "video-key")
                self.assertEqual(embedding_model_provider(tmp), "openai")
                self.assertEqual(embedding_model(tmp), "config-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "embedding-key")
                self.assertEqual(reranker_model(tmp), "config-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://reranker.test")
                self.assertEqual(reranker_api_key(tmp), "reranker-key")

    def test_environment_overrides_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"llm": {"model": "config-llm", "api_key": "config-key"}}), encoding="utf-8")
            with patch.dict(os.environ, {"VIMAX_LLM_MODEL": "env-llm", "VIMAX_LLM_MODEL_PROVIDER": "openai", "VIMAX_LLM_API_KEY": "env-key", "VIMAX_VIDEO_BASE_URL": "https://openrouter.ai/api/v1", "VIMAX_EMBEDDING_MODEL": "env-embedding", "VIMAX_EMBEDDING_BASE_URL": "https://env-embedding.test/v1", "VIMAX_EMBEDDING_API_KEY": "env-embedding-key", "VIMAX_RERANKER_MODEL": "env-reranker", "VIMAX_RERANKER_BASE_URL": "https://env-reranker.test", "VIMAX_RERANKER_API_KEY": "env-reranker-key"}, clear=True):
                self.assertEqual(llm_model(tmp), "env-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_api_key(tmp), "env-key")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(embedding_model(tmp), "env-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://env-embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "env-embedding-key")
                self.assertEqual(reranker_model(tmp), "env-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://env-reranker.test")
                self.assertEqual(reranker_api_key(tmp), "env-reranker-key")

    def _write(self, tmp, payload):
        config_dir = Path(tmp) / "configs"
        config_dir.mkdir()
        (config_dir / "agent.local.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    def test_blocks_borrow_llm_key_only_for_the_same_provider(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self._write(tmp, {
                "llm": {"provider": "google", "api_key": "google-key"},
                "image": {"provider": "google"},
                "video": {"provider": "openrouter"},
            })
            self.assertEqual(image_api_key(tmp), "google-key")
            self.assertEqual(video_api_key(tmp), "")

    def test_explicit_provider_picks_default_base_urls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self._write(tmp, {
                "llm": {"provider": "google", "model": "gemini-3.5-flash-lite", "api_key": "g"},
                "image": {"provider": "google", "model": "gemini-nano-banana-2.1", "api_key": "g"},
                "video": {"provider": "google", "model": "gemini-omni-1.2-flash", "api_key": "g"},
                "voice": {"provider": "google", "model": "gemini-3.8-flash-tts", "api_key": "g"},
            })
            self.assertEqual(llm_provider(tmp), "google")
            self.assertEqual(llm_base_url(tmp), "https://generativelanguage.googleapis.com/v1beta/openai/")
            self.assertEqual(llm_model_provider(tmp), "openai")
            self.assertEqual(image_provider(tmp), "google")
            self.assertEqual(image_base_url(tmp), "https://generativelanguage.googleapis.com/v1beta")
            self.assertEqual(video_provider(tmp), "google")
            self.assertEqual(voice_provider(tmp), "google")
            self.assertEqual(voice_model(tmp), "gemini-3.8-flash-tts")
            self.assertEqual(voice_base_url(tmp), "https://generativelanguage.googleapis.com/v1beta")
            self.assertEqual(voice_api_key(tmp), "g")

    def test_music_block_has_its_own_provider_model_and_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self._write(tmp, {"music": {"provider": "openrouter", "model": "google/lyria-3-pro-preview", "api_key": "or"}})
            self.assertEqual(music_provider(tmp), "openrouter")
            self.assertEqual(music_model(tmp), "google/lyria-3-pro-preview")
            self.assertEqual(music_base_url(tmp), "https://openrouter.ai/api/v1")
            self.assertEqual(music_api_key(tmp), "or")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(music_provider(tmp), "google")
            self.assertEqual(music_model(tmp), "lyria-3.5")
            self.assertEqual(music_base_url(tmp), "https://generativelanguage.googleapis.com/v1beta")

    def test_explicit_base_url_and_environment_override_provider_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, {"video": {"provider": "openrouter", "base_url": "https://proxy.test/v1", "api_key": "k"}})
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(video_base_url(tmp), "https://proxy.test/v1")
            with patch.dict(os.environ, {"VIMAX_VIDEO_PROVIDER": "google", "VIMAX_VIDEO_API_KEY": "env"}, clear=True):
                self.assertEqual(video_provider(tmp), "google")
                self.assertEqual(video_api_key(tmp), "env")

    def test_unknown_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self._write(tmp, {"image": {"provider": "openai"}})
            with self.assertRaisesRegex(RuntimeError, "image.provider must be one of"):
                image_provider(tmp)

    def test_blocks_without_provider_keep_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_base_url(tmp), "https://yunwu.ai/v1")
            self.assertEqual(image_base_url(tmp), "https://yunwu.ai")
            self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
            self.assertEqual(voice_provider(tmp), "openrouter")

    def test_video_provider_is_inferred_from_base_url(self):
        self.assertEqual(api_provider_from_base_url("https://openrouter.ai/api/v1"), "openrouter")
        self.assertEqual(api_provider_from_base_url("https://yunwu.ai/v1"), "yunwu")
        self.assertEqual(api_provider_from_base_url("https://generativelanguage.googleapis.com/v1beta"), "google")
        self.assertEqual(api_provider_from_base_url("https://example.com/v1"), "")

    def test_image_candidates_default_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(image_num_candidates(tmp), 2)
            with patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": "1"}):
                self.assertEqual(image_num_candidates(tmp), 1)

    def test_image_candidates_use_workspace_config(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text("image_selection:\n  num_candidates: 3\n", encoding="utf-8")
            self.assertEqual(image_num_candidates(tmp), 3)
            with patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": "1"}):
                self.assertEqual(image_num_candidates(tmp), 1)

    def test_invalid_image_candidate_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("0", "-1", "1.5", "abc", ""):
                with self.subTest(value=value), patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": value}, clear=True):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        image_num_candidates(tmp)


if __name__ == "__main__":
    unittest.main()
