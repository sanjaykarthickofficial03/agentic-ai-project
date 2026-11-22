"""Quick Windows-friendly Ollama connectivity and model diagnostic."""
import os
from dotenv import load_dotenv
from agents.llm_utils import choose_model, normalize_base_url, server_status

load_dotenv()

base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
requested = os.getenv("OLLAMA_MODEL", "")
print("Ollama base URL:", normalize_base_url(base))
status = server_status(base)
print("Server reachable:", status["server_ok"])
print("Version:", status.get("version") or "unknown")
if not status["server_ok"]:
    print("Server error:", status.get("version_error"))
    raise SystemExit(1)
print("Tags endpoint:", status["tags_ok"])
print("Installed models:", ", ".join(status.get("models", [])) or "none")
model, reason = choose_model(status, requested)
print("Requested model:", requested or "(auto)")
print("Chosen model:", model or "none")
print("Selection reason:", reason)
if not model:
    raise SystemExit(2)
print("Diagnosis: Ollama server and a usable local model are available.")
