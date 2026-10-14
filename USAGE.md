# Hy-MT2 VLSP 2026 — document-level legal MT (EN↔VI)

Self-contained Docker image: model and dependencies are inside, no network is needed to run it, no external API is called.
Model: Hy-MT2-1.8B + LoRA adapter (fewer than 3B parameters), fp16, greedy decoding.

## Contents of the archive

| File | Purpose |
|---|---|
| `image.tar` | the Docker image (`docker load -i image.tar`) |
| `translate.sh` | client script: `./translate.sh HOST:PORT INPUT OUTPUT` |
| `README.md` | this guide |

## Requirements

Linux, an NVIDIA GPU with at least about 5 GB free memory (fp16), a recent NVIDIA driver, Docker and the NVIDIA Container Toolkit
(`docker run --gpus all ...`). `translate.sh` needs `bash` and `curl`.

## Run

```bash
docker load -i image.tar
docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26        # loads the model, then logs "listening on 0.0.0.0:8000"
```

The service is ready when `curl localhost:8000/health` returns `{"status": "ok"}` (the container also reports `healthy`).
Another port: `-e PORT=9000 -p 9000:9000`.

## Translate

```bash
./translate.sh localhost:8000 input.md output.md      # one document -> UTF-8 Markdown translation
./translate.sh localhost:8000 private_test.csv results.csv   # private test CSV -> submission CSV
```

- The direction is detected from the text (Vietnamese source → English, otherwise English → Vietnamese).
- `.md` in/out: any UTF-8 Markdown document; the output is the complete translated document.
- `.csv` in: columns `sample_id, category, source_language, source_markdown` (`en` or `vi`); `.csv` out: columns
  `sample_id, direction, translation`, same order, one row per input row.
- The script waits for the model to load (up to `WAIT_SECONDS`, default 1800 s) and prints statistics:
  direction, input words, seconds, words per second and peak GPU memory.

Without HTTP, directly on a CSV:

```bash
docker run --rm --gpus all -v /path/to/in:/in:ro -v /path/to/out:/out hy-mt-vlsp26 \
    --csv /in/private_test.csv --out /out/results.csv
```

## HTTP API

| Request | Result |
|---|---|
| `GET /health` | `200 {"status": "ok"}` once the model is loaded |
| `POST /translate` (body = Markdown, optional `?direction=en-vi\|vi-en`) | `200` Markdown translation; statistics in `X-*` headers |
| `POST /translate_csv` (body = private test CSV) | `200` submission CSV |

Requests are served one at a time. Errors are JSON `{"error": ...}` with status 400 (bad input) or 500 (translation failed).
In CSV mode a document that fails twice is copied unchanged so the file always has every row.

## Notes

- Translating a full document-level test set takes several hours on one GPU; long documents (hundreds of KB) take minutes each.
  Do not set a short client timeout.
- Quick check: `docker run --rm --gpus all --entrypoint python hy-mt-vlsp26 -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## Method in brief

The document is parsed into structural nodes (headings, paragraphs, tables, signatures), adjacent legal prose is grouped into blocks,
blocks are packed into translation units and translated with Hy-MT2-1.8B + LoRA adapter (fp16, greedy, repeated-token detection).
Each unit is validated against the source (numbers, headings, loops, untranslated text); failed units are re-translated node by node,
and the translated nodes are reassembled on the original Markdown layout. Public test (100 documents), BLEU / chrF:
EN→VI 70.87 / 84.48, VI→EN 47.44 / 79.50.
