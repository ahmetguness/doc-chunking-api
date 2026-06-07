#!/usr/bin/env python3
"""Interactive setup wizard for the Document Chunking & Embedding API.

Walks you through choosing an embedding backend and produces a ready-to-run
``.env`` file. No model registry is hardcoded: model names are validated live
against OpenRouter (cloud mode) or HuggingFace (local mode).

Flow
----
1. Choose backend:  local  |  cloud
2. Enter a model name.
3. Validate it:
     * cloud  -> a real embeddings call to OpenRouter (also checks the key)
     * local  -> HuggingFace repo existence check
4. Collect secrets (OpenRouter key / HF token) and, for local, cpu vs gpu.
5. Resolve + download the tokenizer used for chunk sizing.
6. Write .env and print the docker compose command to run.

Run:  python setup.py
"""

import sys
import urllib.error
import urllib.request
import json
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"
DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
HF_API_MODEL = "https://huggingface.co/api/models/"


# ---------------------------------------------------------------------------
# Small prompt helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    opts = "/".join(c if c != default else c.upper() for c in choices)
    while True:
        val = input(f"{prompt} ({opts}): ").strip().lower() or default
        if val in choices:
            return val
        print(f"  Geçersiz seçim. Şunlardan biri olmalı: {', '.join(choices)}")


def fail(msg: str) -> None:
    print(f"\n[HATA] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Validation against live services
# ---------------------------------------------------------------------------

def validate_cloud_model(base_url: str, api_key: str, model: str) -> None:
    """Make a tiny embeddings call; verifies model + key in one shot."""
    url = f"{base_url.rstrip('/')}/embeddings"
    body = json.dumps({"model": model, "input": "ping"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dim = len(data["data"][0]["embedding"])
        print(f"  OK: '{model}' OpenRouter'da çalışıyor (dimension={dim}).")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")
        if e.code in (401, 403):
            fail(f"OpenRouter API key reddedildi ({e.code}). Key'i kontrol et.")
        if e.code == 404:
            fail(f"'{model}' OpenRouter'da bulunamadı (404). Model adını kontrol et.")
        fail(f"OpenRouter doğrulama hatası ({e.code}): {detail}")
    except urllib.error.URLError as e:
        fail(f"OpenRouter'a ulaşılamadı: {e.reason}")


def validate_hf_model(model: str, hf_token: str = "") -> bool:
    """Return True if the model repo exists on HuggingFace."""
    req = urllib.request.Request(HF_API_MODEL + model)
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        if e.code in (401, 403):
            fail(f"HuggingFace erişimi reddedildi ({e.code}). Gated model olabilir, "
                 f"HF_TOKEN gerekiyor.")
        fail(f"HuggingFace doğrulama hatası ({e.code}).")
    except urllib.error.URLError as e:
        fail(f"HuggingFace'e ulaşılamadı: {e.reason}")
    return False


def download_tokenizer(tokenizer_id: str, hf_token: str = "") -> bool:
    """Download only the tokenizer for chunk sizing. Returns success."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ! transformers yüklü değil; tokenizer indirme atlanıyor "
              "(ilk istekte container içinde inecek).")
        return False
    kwargs = {"token": hf_token} if hf_token else {}
    try:
        AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
        print(f"  OK: tokenizer '{tokenizer_id}' indirildi/önbelleğe alındı.")
        return True
    except Exception as e:  # noqa: BLE001 - surface any HF error to the user
        print(f"  ! tokenizer '{tokenizer_id}' indirilemedi: {e}")
        return False


def resolve_cloud_tokenizer(model: str, hf_token: str) -> str:
    """Resolve the HuggingFace tokenizer repo for an OpenRouter model slug.

    Tries the slug as-is and common case variants. If none exist on HF, asks
    the user to provide the tokenizer repo explicitly.
    """
    org, _, name = model.partition("/")
    candidates = [model]
    if name:
        candidates += [f"{org.upper()}/{name}", f"{org.capitalize()}/{name}", name]
    for cand in dict.fromkeys(candidates):  # dedupe, keep order
        print(f"  HuggingFace'de tokenizer aranıyor: '{cand}'...")
        if validate_hf_model(cand, hf_token):
            print(f"  OK: tokenizer reposu bulundu -> '{cand}'")
            return cand
    print("\n  OpenRouter modeli için HuggingFace tokenizer reposu otomatik "
          "bulunamadı.")
    while True:
        manual = ask("  Tokenizer için HuggingFace repo adı gir (örn. BAAI/bge-m3)")
        if manual and validate_hf_model(manual, hf_token):
            return manual
        print("  Bulunamadı, tekrar dene.")


# ---------------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------------

def write_env(values: dict[str, str]) -> None:
    lines = ["# Generated by setup.py - do not commit secrets.\n"]
    for key, val in values.items():
        lines.append(f"{key}={val}\n")
    ENV_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\n.env yazıldı -> {ENV_PATH}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" Document Chunking & Embedding API - Kurulum Sihirbazı")
    print("=" * 60)

    if ENV_PATH.exists():
        if ask_choice("\n.env zaten var. Üzerine yazılsın mı?", ["y", "n"], "n") == "n":
            print("İptal edildi.")
            return

    env: dict[str, str] = {}

    # Shared settings
    api_port = ask("\nAPI portu", "8001")
    auth_token = ask("Servis auth token (boş = auth kapalı)", "change-this-token")

    # 1) backend
    backend = ask_choice("\nEmbedding kaynağı?  local=serviste model, "
                         "cloud=OpenRouter API", ["local", "cloud"], "local")

    if backend == "cloud":
        base_url = ask("OpenRouter base URL", DEFAULT_OPENROUTER_BASE)
        model = ask("Model adı (OpenRouter slug, örn. baai/bge-m3)")
        if not model:
            fail("Model adı boş olamaz.")
        api_key = ask("OpenRouter API key")
        if not api_key:
            fail("Cloud modda OpenRouter API key zorunlu.")

        print("\n[1/2] OpenRouter'da model doğrulanıyor...")
        validate_cloud_model(base_url, api_key, model)

        hf_token = ask("\nHuggingFace token (tokenizer indirmek için; "
                       "public ise boş geç)", "")
        print("\n[2/2] Tokenizer çözümleniyor...")
        tokenizer_id = resolve_cloud_tokenizer(model, hf_token)
        download_tokenizer(tokenizer_id, hf_token)

        env.update({
            "EMBEDDING_MODE": "cloud",
            "EMBEDDING_API_BASE_URL": base_url,
            "EMBEDDING_API_MODEL": model,
            "EMBEDDING_API_KEY": api_key,
            "TOKENIZER_ID": tokenizer_id,
            "HF_TOKEN": hf_token,
        })
        compose_cmd = "docker compose up -d --build"

    else:  # local
        model = ask("Model adı (HuggingFace repo, örn. BAAI/bge-m3)")
        if not model:
            fail("Model adı boş olamaz.")
        hf_token = ask("HuggingFace token (gated/private model için; "
                       "public ise boş geç)", "")

        print("\n[1/3] HuggingFace'de model doğrulanıyor...")
        if not validate_hf_model(model, hf_token):
            fail(f"'{model}' HuggingFace'de bulunamadı. Repo adını kontrol et.")
        print(f"  OK: '{model}' HuggingFace'de mevcut.")

        device = ask_choice("\n[2/3] Donanım?", ["cpu", "gpu"], "gpu")

        print("\n[3/3] Tokenizer indiriliyor...")
        download_tokenizer(model, hf_token)

        env.update({
            "EMBEDDING_MODE": "local",
            "EMBEDDING_DEVICE": device,
            "MODEL_NAME": model,
            "TOKENIZER_ID": model,
            "HF_TOKEN": hf_token,
        })
        compose_cmd = (
            "docker compose up -d --build" if device == "gpu"
            else "docker compose -f docker-compose.cpu.yml up -d --build"
        )

    env.update({
        "API_PORT": api_port,
        "CHUNKING_AUTH_TOKEN": auth_token,
    })

    write_env(env)

    print("\nKurulum tamam. Servisi başlatmak için:\n")
    print(f"    {compose_cmd}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nİptal edildi.")
        sys.exit(1)
