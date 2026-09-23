"""
core/llm_gateway.py
===================
Optional **remote LLM transport** for CAT-Pal, with *fail-closed* secret handling.

Golden rules enforced here
--------------------------
1. **No credentials in code.** Keys are read at runtime from ``st.secrets`` (preferred,
   injected by the caller as a plain mapping) or the process environment / ``.env``.
   :func:`load_llm_settings` is the *only* place a key is read, and nothing logs it:
   :meth:`LLMSettings.redacted_api_key` exists for the audit panel.
2. **No network by default.** ``mode="offline"`` (the shipped default) never opens a socket;
   the deterministic knowledge base answers instead. This keeps the edge node air-gap
   friendly and the test-suite hermetic.
3. **Wrapped prompts only.** Callers must pass a message list built by
   :func:`core.copilot.build_messages` - the system prompt is always message[0] and the
   operator text is always inside ``<operator_query>`` tags.
4. **Bounded responses.** HTTP timeouts, output length caps and output sanitisation
   (an LLM response is untrusted text too) are applied before anything reaches the UI.

Transport is injectable (``transport=...``) so tests can exercise the full request/response
path without a network and without mocking ``requests`` internals.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from config import (
    COPILOT_MODE,
    LLM_HTTP_TIMEOUT_S,
    LLM_MAX_OUTPUT_CHARS,
)
from core.security import redact_secret, sanitize_text

#: Providers we are willing to talk to (any OpenAI-compatible endpoint qualifies).
SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "openai-compatible", "azure-openai", "local-openai", "ollama", "vllm"}
)

#: Environment / secrets keys (names only - values are never literals in this file).
SECRET_KEYS: dict[str, str] = {
    "mode": "CAT_COPILOT_MODE",
    "provider": "CAT_LLM_PROVIDER",
    "model": "CAT_LLM_MODEL",
    "api_key": "CAT_LLM_API_KEY",
    "base_url": "CAT_LLM_BASE_URL",
}

_ENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load ``.env`` if python-dotenv is installed (never fatal when it is absent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:  # pragma: no cover - depends on the optional dependency being present
        from dotenv import load_dotenv  # type: ignore import-not-found

        load_dotenv(override=False)
    except Exception:  # noqa: BLE001 - optional dependency, any failure is non-fatal
        return


@dataclass(frozen=True)
class LLMSettings:
    """Resolved, validated LLM configuration (never serialised to the UI in full)."""

    mode: str = "offline"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1/chat/completions"
    timeout_s: float = LLM_HTTP_TIMEOUT_S
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def configured(self) -> bool:
        """True when remote mode is requested *and* every required field is valid."""
        return self.mode == "remote" and self.api_key != "" and not self.problems

    @property
    def effective_mode(self) -> str:
        """``remote`` only when fully configured; otherwise the safe ``offline`` fallback."""
        return "remote" if self.configured else "offline"

    def redacted_api_key(self) -> str:
        """Masked key for the audit panel (``sk-…9f2a`` / ``<unset>``)."""
        return redact_secret(self.api_key)

    def describe(self) -> str:
        """One-line, secret-free description used by the UI."""
        if self.effective_mode == "offline":
            return "offline deterministic knowledge base (no API key required)"
        return f"remote · {self.provider} · {self.model} · key {self.redacted_api_key()}"


def load_llm_settings(
    secrets: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMSettings:
    """
    Resolve LLM settings from ``secrets`` (``st.secrets``) then environment variables.

    Parameters
    ----------
    secrets:
        A mapping such as ``st.secrets`` or ``{"CAT_LLM_API_KEY": "..."}``. Read-only.
    environ:
        Override for the process environment (used by the unit tests).

    Precedence: explicit ``secrets`` > environment > ``.env`` > defaults.
    Validation failures are reported in ``problems`` instead of raising, so the UI can
    degrade to offline mode and *explain* what is missing.
    """
    _load_dotenv_once()
    env: Mapping[str, str] = environ if environ is not None else os.environ

    def get(setting: str, default: str = "") -> str:
        """
        Resolve one setting by its *short* name (``mode``, ``api_key``, ...).

        The short name is translated through :data:`SECRET_KEYS` into the real
        ``CAT_*`` key before lookup, so a caller cannot accidentally read the wrong key.
        """
        key = SECRET_KEYS.get(setting, setting)
        # NOTE: ``st.secrets`` is a lazy mapping - merely *touching* it raises
        # StreamlitSecretNotFoundError when no secrets.toml exists, which is the normal state
        # on a fresh edge node. Any secrets-backend failure therefore means "not configured"
        # and we fall through to the environment. This broad catch is intentional and is the
        # only one in the module.
        if secrets is not None:
            try:
                if key in secrets:  # type: ignore[operator]
                    value = secrets[key]  # type: ignore[index]
                    if value is not None:
                        return str(value).strip()
            except Exception:  # noqa: BLE001 - missing/short-typed secrets == unset
                # nosec B110 - the silent fallback is the intended behaviour: a missing or
                # malformed secrets store must never be fatal, and there is nothing to log
                # that would not itself risk leaking a key fragment.
                pass
        return str(env.get(key, default)).strip()

    problems: list[str] = []

    mode = (get("mode") or COPILOT_MODE or "offline").lower()
    if mode not in {"offline", "remote"}:
        problems.append(f"unknown CAT_COPILOT_MODE '{mode}' - falling back to offline")
        mode = "offline"

    provider = (get("provider") or "openai").lower()
    if provider not in SUPPORTED_PROVIDERS:
        problems.append(f"unsupported CAT_LLM_PROVIDER '{provider}'")

    model = get("model") or "gpt-4o-mini"
    base_url = get("base_url") or "https://api.openai.com/v1/chat/completions"
    if not base_url.lower().startswith(("http://", "https://")):
        problems.append("CAT_LLM_BASE_URL must be an http(s) URL")

    api_key = get("api_key")
    if mode == "remote":
        if not api_key:
            problems.append("remote mode requested but no API key is configured")
        else:
            if len(api_key) < 16:
                problems.append("API key looks too short (<16 chars)")
            if any(ch.isspace() for ch in api_key):
                problems.append("API key contains whitespace - check for a copy/paste error")
            if api_key.lower().startswith(("your-", "changeme", "placeholder", "sk-xxxx")):
                problems.append("API key is still a placeholder value")

    return LLMSettings(
        mode=mode,
        provider=provider,
        model=model,
        api_key=api_key if mode == "remote" else "",
        base_url=base_url,
        timeout_s=float(LLM_HTTP_TIMEOUT_S),
        problems=tuple(problems),
    )


# --------------------------------------------------------------------------------------
# Request / response handling
# --------------------------------------------------------------------------------------
@dataclass
class LLMResult:
    """Outcome of a remote completion call."""

    ok: bool
    text: str = ""
    error: str = ""
    model: str = ""
    latency_ms: float = 0.0
    sanitization_note: str = ""


def build_request_payload(
    messages: Sequence[Mapping[str, str]],
    model: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> dict[str, object]:
    """
    Build an OpenAI-compatible chat-completions payload.

    Low ``temperature`` is deliberate: this assistant must be conservative and repeatable,
    not creative. ``max_tokens`` bounds cost and response size.
    """
    return {
        "model": model,
        "messages": [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                     for m in messages],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }


def parse_response(payload: object) -> str:
    """
    Extract the assistant text from an OpenAI-compatible response body.

    Accepts the dict form or a JSON string. Returns ``""`` for anything unrecognised -
    a malformed response must never crash the chat, and must never be echoed verbatim.
    """
    data = payload
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            data = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
    if not isinstance(data, dict):
        return ""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    # Ollama-style native response
    if isinstance(data.get("response"), str):
        return data["response"]
    return ""


def sanitize_model_output(text: str) -> tuple[str, str]:
    """
    Sanitise model output before it reaches the chat renderer.

    The model is *not* a trusted party: strip HTML/script payloads and control characters,
    enforce a newline-preserving length cap. Returns ``(clean_text, note)``.
    """
    cleaned = sanitize_text(text, max_chars=LLM_MAX_OUTPUT_CHARS, allow_newlines=True)
    return cleaned.text, cleaned.report.describe()


#: Only these URL schemes may ever be opened (blocks ``file://``, ``ftp://``, ``data:``
#: and other scheme-smuggling attempts from a hostile configuration).
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _validate_endpoint(url: str) -> str:
    """
    Validate the LLM endpoint before use and return the normalised URL.

    Defence in depth for SSRF-style abuse and for ``urllib`` scheme smuggling: the endpoint
    must be an absolute ``http(s)`` URL with a host. Raises :class:`ValueError` otherwise -
    callers convert that into a fail-closed "not configured" state.
    """
    parts = urllib.parse.urlsplit(str(url).strip())
    if parts.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {parts.scheme or '<none>'}")
    if not parts.netloc:
        raise ValueError("LLM endpoint must include a host")
    return urllib.parse.urlunsplit(parts)


def _http_post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout: float,
) -> object:
    """Stdlib JSON POST (no ``requests`` dependency on the edge node)."""
    endpoint = _validate_endpoint(url)
    request = urllib.request.Request(  # noqa: S310 - scheme is restricted by _validate_endpoint
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    # nosec B310 - ``_validate_endpoint`` has already restricted the scheme to http/https
    # with a non-empty host, so file:/custom schemes cannot reach this call.
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        body = response.read(2 * 1024 * 1024)
    return body


def chat_completion(
    messages: Sequence[Mapping[str, str]],
    settings: LLMSettings,
    *,
    transport: Callable[[str, Mapping[str, str], Mapping[str, object], float], object] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> LLMResult:
    """
    Perform one remote completion (or fail safely).

    Never raises: every failure mode (no key, DNS failure, HTTP 4xx/5xx, timeout, malformed
    body, oversized output) is returned as ``LLMResult(ok=False, error=...)`` so the caller
    can fall back to the knowledge base and tell the operator plainly.
    """
    started = time.perf_counter()
    if not settings.configured:
        return LLMResult(
            ok=False,
            error="remote LLM not configured - offline knowledge base in use",
            model=settings.model,
        )

    payload = build_request_payload(messages, settings.model, temperature=temperature,
                                    max_tokens=max_tokens)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.api_key}",
        "User-Agent": "CAT-Smart-Operator-Assistant/1.0",
    }
    sender = transport or _http_post_json
    try:
        raw = sender(settings.base_url, headers, payload, settings.timeout_s)
    except urllib.error.HTTPError as exc:
        detail = "auth/rate/quota error" if exc.code in {401, 403, 429} else "HTTP error"
        return LLMResult(ok=False, error=f"{detail} ({exc.code})", model=settings.model,
                         latency_ms=round((time.perf_counter() - started) * 1000, 1))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return LLMResult(
            ok=False,
            error=f"network unavailable ({type(exc).__name__})",
            model=settings.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    except Exception as exc:  # noqa: BLE001 - transport is injectable/untrusted
        return LLMResult(
            ok=False,
            error=f"transport failure ({type(exc).__name__})",
            model=settings.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    text = parse_response(raw)
    latency = round((time.perf_counter() - started) * 1000, 1)
    if not text.strip():
        return LLMResult(ok=False, error="empty or unparsable model response",
                         model=settings.model, latency_ms=latency)

    clean, note = sanitize_model_output(text)
    return LLMResult(ok=True, text=clean, model=settings.model, latency_ms=latency,
                    sanitization_note=note)


def check_connection(
    settings: LLMSettings,
    *,
    transport: Callable[[str, Mapping[str, str], Mapping[str, object], float], object] | None = None,
) -> LLMResult:
    """
    Lightweight connectivity probe for the sidebar ("test connection" button).

    Sends the smallest legal request and reports latency - it never prints the key.
    """
    probe = [
        {"role": "system", "content": "Reply with the single word: OK"},
        {"role": "user", "content": "ping"},
    ]
    return chat_completion(probe, settings, transport=transport, temperature=0.0, max_tokens=8)


__all__ = [
    "SUPPORTED_PROVIDERS",
    "ALLOWED_URL_SCHEMES",
    "LLMSettings",
    "LLMResult",
    "load_llm_settings",
    "build_request_payload",
    "parse_response",
    "sanitize_model_output",
    "chat_completion",
    "check_connection",
]
