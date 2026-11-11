from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from utils.image_selection import image_candidate_count_from_config

# Each generation block (llm, image, video, voice, music) in configs/agent.local.yaml names its
# provider explicitly and carries its own model and api_key; base_url is optional:
#
#   video:
#     provider: google        # openrouter | google | yunwu
#     model: gemini-omni-1.2-flash
#     api_key: ...
#
# Environment overrides: VIMAX_<BLOCK>_PROVIDER / _MODEL / _BASE_URL / _API_KEY.
PROVIDERS = ("openrouter", "google", "yunwu")
PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "yunwu": "https://yunwu.ai/v1",
}
# Where a block talks to a different endpoint of the same provider.
SECTION_BASE_URLS = {
    ("llm", "google"): "https://generativelanguage.googleapis.com/v1beta/openai/",
    ("image", "yunwu"): "https://yunwu.ai",
}
# Used only when a block names neither provider nor base_url (configs written before `provider` existed).
DEFAULT_PROVIDERS = {"llm": "yunwu", "image": "yunwu", "video": "openrouter", "voice": "openrouter", "music": "google"}

DEFAULT_LLM_MODEL = "gpt-5.5"
# LangChain chat protocol, not the vendor: every supported llm provider speaks the OpenAI API.
DEFAULT_LLM_MODEL_PROVIDER = "openai"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo3.1-fast"
DEFAULT_VOICE_MODEL = "google/gemini-3.8-flash-tts"
DEFAULT_MUSIC_MODEL = "lyria-3.5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL_PROVIDER = "openai"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=4)
def load_agent_config(workspace_root: str | Path = ".") -> dict[str, Any]:
    path = Path(workspace_root).resolve() / "configs" / "agent.local.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid configs/agent.local.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("configs/agent.local.yaml must be a YAML mapping")
    return payload


def config_value(section: str, key: str, env_names: list[str], default: str = "", workspace_root: str | Path = ".") -> str:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    section_payload = load_agent_config(workspace_root).get(section, {})
    if isinstance(section_payload, dict):
        value = section_payload.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _env(section: str, field: str) -> str:
    return f"VIMAX_{section.upper()}_{field.upper()}"


def section_provider(section: str, workspace_root: str | Path = ".") -> str:
    """The provider a generation block uses: explicit `provider`, else inferred from `base_url`."""
    provider = config_value(section, "provider", [_env(section, "provider")], "", workspace_root).strip().lower()
    if provider:
        if provider not in PROVIDERS:
            raise RuntimeError(f"configs/agent.local.yaml {section}.provider must be one of {', '.join(PROVIDERS)}, got {provider!r}")
        return provider
    explicit_url = config_value(section, "base_url", [_env(section, "base_url")], "", workspace_root)
    return api_provider_from_base_url(explicit_url) or DEFAULT_PROVIDERS[section]


def section_base_url(section: str, workspace_root: str | Path = ".") -> str:
    explicit = config_value(section, "base_url", [_env(section, "base_url")], "", workspace_root)
    if explicit:
        return explicit
    provider = section_provider(section, workspace_root)
    return SECTION_BASE_URLS.get((section, provider), PROVIDER_BASE_URLS[provider])


def section_api_key(section: str, workspace_root: str | Path = ".") -> str:
    """The block's own api_key; a block without one borrows the llm key only when both use the same provider."""
    key = config_value(section, "api_key", [_env(section, "api_key")], "", workspace_root)
    if key or section == "llm":
        return key or os.environ.get("VIMAX_API_KEY", "")
    if section_provider(section, workspace_root) == section_provider("llm", workspace_root):
        key = llm_api_key(workspace_root)
    return key or os.environ.get("VIMAX_API_KEY", "")


def llm_provider(workspace_root: str | Path = ".") -> str:
    return section_provider("llm", workspace_root)


def llm_model(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model", ["VIMAX_LLM_MODEL"], DEFAULT_LLM_MODEL, workspace_root)


def llm_model_provider(workspace_root: str | Path = ".") -> str:
    """LangChain chat protocol for init_chat_model; `openai` for every supported provider."""
    return config_value("llm", "model_provider", ["VIMAX_LLM_MODEL_PROVIDER"], DEFAULT_LLM_MODEL_PROVIDER, workspace_root)


def llm_base_url(workspace_root: str | Path = ".") -> str:
    return section_base_url("llm", workspace_root)


def llm_api_key(workspace_root: str | Path = ".") -> str:
    return section_api_key("llm", workspace_root)


def image_provider(workspace_root: str | Path = ".") -> str:
    return section_provider("image", workspace_root)


def image_model(workspace_root: str | Path = ".") -> str:
    return config_value("image", "model", ["VIMAX_IMAGE_MODEL"], DEFAULT_IMAGE_MODEL, workspace_root)


def image_base_url(workspace_root: str | Path = ".") -> str:
    return section_base_url("image", workspace_root)


def image_api_key(workspace_root: str | Path = ".") -> str:
    return section_api_key("image", workspace_root)


def voice_provider(workspace_root: str | Path = ".") -> str:
    return section_provider("voice", workspace_root)


def voice_model(workspace_root: str | Path = ".") -> str:
    return config_value("voice", "model", ["VIMAX_VOICE_MODEL"], DEFAULT_VOICE_MODEL, workspace_root)


def voice_base_url(workspace_root: str | Path = ".") -> str:
    return section_base_url("voice", workspace_root)


def voice_api_key(workspace_root: str | Path = ".") -> str:
    return section_api_key("voice", workspace_root)


# Background music for the series-video skill (studio.py music stage): Lyria on google or openrouter.
def music_provider(workspace_root: str | Path = ".") -> str:
    return section_provider("music", workspace_root)


def music_model(workspace_root: str | Path = ".") -> str:
    return config_value("music", "model", ["VIMAX_MUSIC_MODEL"], DEFAULT_MUSIC_MODEL, workspace_root)


def music_base_url(workspace_root: str | Path = ".") -> str:
    return section_base_url("music", workspace_root)


def music_api_key(workspace_root: str | Path = ".") -> str:
    return section_api_key("music", workspace_root)


def image_num_candidates(workspace_root: str | Path = ".") -> int:
    return image_candidate_count_from_config(load_agent_config(workspace_root))



def embedding_model(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model", ["VIMAX_EMBEDDING_MODEL"], DEFAULT_EMBEDDING_MODEL, workspace_root)


def embedding_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model_provider", ["VIMAX_EMBEDDING_MODEL_PROVIDER"], DEFAULT_EMBEDDING_MODEL_PROVIDER, workspace_root)


def embedding_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "base_url", ["VIMAX_EMBEDDING_BASE_URL"], "", workspace_root)


def embedding_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "api_key", ["VIMAX_EMBEDDING_API_KEY"], "", workspace_root)


def reranker_model(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "model", ["VIMAX_RERANKER_MODEL"], DEFAULT_RERANKER_MODEL, workspace_root)


def reranker_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "base_url", ["VIMAX_RERANKER_BASE_URL"], "", workspace_root)


def reranker_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "api_key", ["VIMAX_RERANKER_API_KEY"], "", workspace_root)


def video_model(workspace_root: str | Path = ".") -> str:
    return config_value("video", "model", ["VIMAX_VIDEO_MODEL"], DEFAULT_VIDEO_MODEL, workspace_root)


def video_base_url(workspace_root: str | Path = ".") -> str:
    return section_base_url("video", workspace_root)


def video_api_key(workspace_root: str | Path = ".") -> str:
    return section_api_key("video", workspace_root)


def api_provider_from_base_url(base_url: str) -> str:
    normalized = base_url.strip().lower()
    if "openrouter.ai" in normalized:
        return "openrouter"
    if "yunwu.ai" in normalized:
        return "yunwu"
    if "generativelanguage.googleapis.com" in normalized:
        return "google"
    return ""


def video_provider(workspace_root: str | Path = ".") -> str:
    return section_provider("video", workspace_root)
