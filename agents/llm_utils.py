import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE = "http://127.0.0.1:11434"
PREFERRED_MODELS = [
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "phi4-mini",
    "qwen2.5:3b",
]


def normalize_base_url(value: str | None) -> str:
    """Normalize common Ollama base-URL mistakes to the native server root."""
    base = (value or DEFAULT_BASE).strip().rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/tags", "/api/version", "/api", "/v1"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
    if base in {"http://localhost", "https://localhost"}:
        base += ":11434"
    return base


def _request(url: str, payload: dict[str, Any] | None = None, timeout: float = 4.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if payload is None else "POST"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "status": int(getattr(resp, "status", 200)),
                "data": json.loads(raw) if raw else {},
                "raw": raw,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return {
            "ok": False,
            "status": int(exc.code),
            "error": f"HTTP {exc.code}",
            "raw": raw[:500],
            "reason": exc.reason,
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "status": None,
            "error": "connection_error",
            "raw": str(getattr(exc, "reason", exc)),
        }
    except TimeoutError:
        return {"ok": False, "status": None, "error": "timeout", "raw": "request timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "error": type(exc).__name__, "raw": str(exc)[:500]}


def server_status(base_url: str | None = None) -> dict[str, Any]:
    """Check both Ollama version and model-list endpoints with precise diagnostics."""
    base = normalize_base_url(base_url)
    version = _request(base + "/api/version", timeout=3.0)
    if not version["ok"]:
        # Windows users sometimes resolve localhost differently. Try loopback once.
        alt = "http://127.0.0.1:11434"
        if base != alt:
            version_alt = _request(alt + "/api/version", timeout=3.0)
            if version_alt["ok"]:
                base = alt
                version = version_alt
    tags = _request(base + "/api/tags", timeout=4.0) if version["ok"] else {"ok": False, "data": {}}
    models = []
    if tags.get("ok"):
        for item in tags.get("data", {}).get("models", []) or []:
            name = item.get("name")
            if name:
                models.append(name)
    return {
        "base_url": base,
        "server_ok": bool(version.get("ok")),
        "version": version.get("data", {}).get("version"),
        "version_error": None if version.get("ok") else _friendly_error(version),
        "tags_ok": bool(tags.get("ok")),
        "tags_error": None if tags.get("ok") else _friendly_error(tags),
        "models": models,
        "raw_tags_status": tags.get("status"),
    }


def _friendly_error(result: dict[str, Any]) -> str:
    status = result.get("status")
    error = result.get("error")
    raw = (result.get("raw") or "").strip()
    if status is not None:
        detail = f"{error or 'HTTP error'}"
        if raw:
            detail += f": {raw[:300]}"
        return detail
    return f"{error or 'request_failed'}: {raw[:300]}" if raw else str(error or "request_failed")


def choose_model(status: dict[str, Any], requested: str | None = None) -> tuple[str | None, str]:
    models = status.get("models", [])
    if requested:
        req = requested.strip()
        for name in models:
            if name == req or name.split(":", 1)[0] == req.split(":", 1)[0]:
                return name, "configured model"
        if status.get("server_ok"):
            return None, f"Configured Ollama model '{req}' is not installed. Installed models: {', '.join(models) or 'none'}."
    for preferred in PREFERRED_MODELS:
        for name in models:
            if name == preferred or name.split(":", 1)[0] == preferred.split(":", 1)[0]:
                return name, "preferred installed model"
    return (models[0], "first installed model") if models else (None, "no local model installed")


def ollama_json(prompt: str, model: str, base_url: str | None = None, timeout: float = 25.0) -> tuple[dict[str, Any] | None, str | None]:
    base = normalize_base_url(base_url)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 180},
    }
    result = _request(base + "/api/generate", payload, timeout=timeout)
    if not result.get("ok"):
        return None, _friendly_error(result)
    response = result.get("data", {}).get("response", "")
    try:
        return json.loads(response), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON returned by Ollama: {exc}. Response: {response[:300]}"


def ollama_text(
    prompt: str,
    model: str,
    base_url: str | None = None,
    timeout: float = 20.0,
    num_predict: int = 280,
) -> tuple[str | None, str | None]:
    """Generate short local text with a bounded latency/output budget.

    The insight stage should never become the bottleneck for an analytical
    workflow, so callers can keep the response intentionally compact.
    """
    base = normalize_base_url(base_url)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": int(num_predict)},
    }
    result = _request(base + "/api/generate", payload, timeout=timeout)
    if not result.get("ok"):
        return None, _friendly_error(result)
    text = (result.get("data", {}).get("response") or "").strip()
    return (text or None), (None if text else "Ollama returned an empty response.")
