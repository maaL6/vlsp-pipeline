"""HTTP service around the pipeline (what the Docker image runs).

  python -m docmt.server --model MODEL_DIR [--port 8000]

  GET  /health      -> 200 {"status": "ok"} once the model is loaded
  POST /translate   body: UTF-8 Markdown document; the direction is detected from the text
                    (override with ?direction=en-vi|vi-en)
                    -> 200 UTF-8 Markdown translation; statistics in the X-* response headers

Requests are served one at a time (one GPU). Nothing leaves the machine: the model is loaded from MODEL_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .core.config import RunConfig
from .core.tok import HyTok
from .engine.api import translate_document

ROOT = Path(__file__).resolve().parents[1]
VIETNAMESE_LETTERS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")


def detect_direction(text: str) -> str:
    """"vi-en" for Vietnamese source text, otherwise "en-vi" (Vietnamese text has ~28% diacritic letters, English < 1%)."""
    letters = [c for c in text.lower() if c.isalpha()]
    share = sum(c in VIETNAMESE_LETTERS for c in letters) / max(len(letters), 1)
    return "vi-en" if share > 0.05 else "en-vi"


class Translator:
    def __init__(self, cfg: RunConfig, tok, backend):
        self.cfg, self.tok, self.backend = cfg, tok, backend
        self.lock = threading.Lock()
        self.counter = 0

    def translate(self, text: str, direction: str | None = None) -> tuple[str, dict]:
        direction = direction or detect_direction(text)
        src, tgt = direction.split("-")
        started = time.monotonic()
        with self.lock:
            self.counter += 1
            with tempfile.TemporaryDirectory() as tmp:
                source, target = Path(tmp) / "in" / "doc.md", Path(tmp) / "out.md"
                source.parent.mkdir()
                source.write_bytes(text.encode("utf-8"))
                result = translate_document(source, target, src, tgt, self.cfg, self.backend, tok=self.tok,
                                            doc_id=f"request/{self.counter}")
                if result.status == "failed" or not target.exists():
                    raise RuntimeError(f"translation failed: {result.diagnostics.get('error', result.status)}")
                output = target.read_bytes().decode("utf-8")
            peak = self.backend.peak_vram() if hasattr(self.backend, "peak_vram") else None
        seconds = time.monotonic() - started
        words = len(text.split())
        return output, {"direction": direction, "status": result.status, "input_words": words,
                        "seconds": round(seconds, 2), "words_per_second": round(words / max(seconds, 1e-9), 2),
                        "peak_vram_bytes": peak}


def make_handler(translator: Translator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, content_type: str, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str):
            self._send(code, json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            if urlparse(self.path).path in ("/", "/health"):
                self._send(200, b'{"status": "ok"}', "application/json")
            else:
                self._error(404, "use POST /translate or GET /health")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/translate":
                return self._error(404, "use POST /translate or GET /health")
            direction = (parse_qs(url.query).get("direction") or [None])[0]
            if direction not in (None, "en-vi", "vi-en"):
                return self._error(400, "direction must be en-vi or vi-en")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                text = self.rfile.read(length).decode("utf-8-sig")
            except UnicodeDecodeError:
                return self._error(400, "body must be UTF-8")
            if not text.strip():
                return self._error(400, "empty document")
            try:
                output, info = translator.translate(text, direction)
            except Exception as exc:  # never kill the service on one bad document
                print(f"[server] error: {exc!r}", flush=True)
                return self._error(500, str(exc))
            print("[server] " + json.dumps(info), flush=True)
            self._send(200, output.encode("utf-8"), "text/markdown; charset=utf-8", {
                "X-Direction": info["direction"], "X-Status": info["status"], "X-Input-Words": info["input_words"],
                "X-Seconds": info["seconds"], "X-Words-Per-Second": info["words_per_second"],
                "X-Peak-Vram-Bytes": info["peak_vram_bytes"] if info["peak_vram_bytes"] is not None else ""})

        def log_message(self, *args):
            pass

    return Handler


def build_translator(model: str, tokenizer: str, config: str) -> Translator:
    from .backends import make_backend
    cfg = RunConfig().apply(json.loads(Path(config).read_text(encoding="utf-8")))
    overrides = {"run_id": "vlsp26-service", "backend": "hf", "model": model, "tokenizer_path": tokenizer}
    for env, key in (("HF_DEVICE", "hf_device"), ("HF_DTYPE", "hf_dtype")):  # e.g. HF_DEVICE=cpu HF_DTYPE=float32 for a smoke test without GPU
        if os.environ.get(env):
            overrides[key] = os.environ[env]
    cfg.apply(overrides)
    if cfg.reference_available_at_inference or cfg.normalize_reference:
        raise SystemExit("inference must not use or normalize references")
    if cfg.source_context or cfg.soft_glossary or cfg.target_history:
        raise SystemExit("source_context, soft_glossary and target_history are not supported")
    tok = HyTok(tokenizer)
    backend = make_backend(cfg.backend, cfg, tok)
    backend.prepare()
    return Translator(cfg, tok, backend)


def serve(translator: Translator, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(translator))
    print(f"[server] listening on {host}:{port}", flush=True)
    server.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL", "models/hy_mt_merged"))
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER"))
    ap.add_argument("--config", default=os.environ.get("CONFIG", str(ROOT / "configs" / "hy_mt_vlsp26.json")))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop: as PID 1 the process would otherwise ignore SIGTERM
    print(f"[server] loading model from {args.model} ...", flush=True)
    translator = build_translator(args.model, args.tokenizer or str(Path(args.model) / "tokenizer.json"), args.config)
    serve(translator, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
