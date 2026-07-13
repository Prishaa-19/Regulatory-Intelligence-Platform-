"""
Shared LLM helper for the Nexus pipeline (extract_document.py,
structure_document.py, interpret_document.py).

Why a provider *chain* and not a single endpoint:
the Azure AI Foundry grok deployment was observed returning HTTP 200 with a
completely empty body in sustained multi-minute streaks — no error code, no
content — regardless of payload size (a 1-word prompt failed as readily as a
36k-char document). NVIDIA's OpenAI-compatible endpoint, tested under the
same rapid-fire pattern, returned real content 5/5. So call_chat tries
providers in priority order and falls through to the next one whenever a
provider errors, returns an empty body, or returns an empty content field —
only raising if every configured provider fails.

Priority order:
  1. NVIDIA   (NVIDIA_API_KEY / NVIDIA_BASE_URL / NVIDIA_MODEL) — reliable
  2. Azure    (AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY)    — flaky fallback

Both speak the OpenAI chat-completions shape; they differ only in URL form
and auth header, captured per-provider below.
"""

import os
import time

import httpx

DEFAULT_RETRIES = 2
BASE_DELAY_SECONDS = 2
MAX_DELAY_SECONDS = 10

AZURE_DEFAULT_MODEL = "grok-4-20-non-reasoning"


class LLMError(RuntimeError):
    """Raised only when every configured provider has failed."""


def _providers() -> list[dict]:
    """Build the ordered provider list from the current environment.

    Read at call time (not import time) so scripts that call load_dotenv()
    before invoking the pipeline get their credentials picked up.
    """
    providers: list[dict] = []

    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    nvidia_base = os.environ.get("NVIDIA_BASE_URL")
    nvidia_model = os.environ.get("NVIDIA_MODEL")
    if nvidia_key and nvidia_base and nvidia_model:
        providers.append({
            "name": "nvidia",
            "url": nvidia_base.strip().strip('"').rstrip("/") + "/chat/completions",
            "headers": {"Authorization": f"Bearer {nvidia_key.strip()}", "Content-Type": "application/json"},
            "model": nvidia_model.strip().strip('"'),
        })

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_endpoint and azure_key:
        providers.append({
            "name": "azure",
            "url": azure_endpoint.strip().strip('"'),
            "headers": {"api-key": azure_key.strip(), "Content-Type": "application/json"},
            "model": (os.environ.get("AZURE_OPENAI_MODEL") or AZURE_DEFAULT_MODEL).strip().strip('"'),
        })

    return providers


def is_configured() -> bool:
    """True if at least one provider has complete credentials."""
    return bool(_providers())


def configured_summary() -> str:
    """Human-readable list of providers that will be tried, in order."""
    names = [p["name"] for p in _providers()]
    return " -> ".join(names) if names else "(none configured)"


def call_chat(messages: list[dict], retries: int = DEFAULT_RETRIES) -> str:
    """Return the assistant's text, trying each provider in priority order.

    A provider is considered failed (and the next one tried) if it errors,
    returns an empty HTTP body, or returns an empty content field. Within a
    provider, transient failures are retried up to `retries` times with
    exponential backoff before falling through.
    """
    providers = _providers()
    if not providers:
        raise LLMError(
            "No LLM provider configured — set NVIDIA_API_KEY/NVIDIA_BASE_URL/NVIDIA_MODEL "
            "or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY in .env"
        )

    errors: list[str] = []
    with httpx.Client(timeout=120) as client:
        for provider in providers:
            body = {"model": provider["model"], "messages": messages}
            for attempt in range(retries):
                try:
                    resp = client.post(provider["url"], headers=provider["headers"], json=body)
                    resp.raise_for_status()
                    if not resp.text.strip():
                        errors.append(f"{provider['name']}: empty response body (attempt {attempt + 1})")
                    else:
                        content = resp.json()["choices"][0]["message"]["content"] or ""
                        if content.strip():
                            return content
                        errors.append(f"{provider['name']}: 200 with empty content field (attempt {attempt + 1})")
                except Exception as e:
                    errors.append(f"{provider['name']}: {type(e).__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(min(BASE_DELAY_SECONDS * (2 ** attempt), MAX_DELAY_SECONDS))

    raise LLMError("all providers failed [" + " | ".join(errors) + "]")
