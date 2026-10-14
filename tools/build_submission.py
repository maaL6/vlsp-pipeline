"""Package translated Markdown files as an AIHub submission (standard library only)."""
from __future__ import annotations

import argparse
import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


COLUMNS = ["sample_id", "direction", "translation"]
# A private test input (one source language per ID) also works as template: en -> en-vi, vi -> vi-en.
INPUT_COLUMNS = ["sample_id", "category", "source_language", "source_markdown"]
SOURCE_DIRECTION = {"en": "en-vi", "vi": "vi-en"}


def build_submission(template: Path, roots: dict[str, Path], out: Path) -> int:
    csv.field_size_limit(100_000_000)
    rows = []
    seen = set()
    # The template supplies required keys and ordering, never translation content.
    with template.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        from_input = reader.fieldnames == INPUT_COLUMNS
        if reader.fieldnames != COLUMNS and not from_input:
            raise ValueError(f"Template columns must be: {','.join(COLUMNS)} (or the private input columns)")
        for line, row in enumerate(reader, start=2):
            if None in row or any(row.get(column) is None for column in (INPUT_COLUMNS if from_input else COLUMNS)):
                raise ValueError(f"Malformed CSV record {line}")
            sid = row["sample_id"]
            direction = SOURCE_DIRECTION.get(row["source_language"].strip().lower(), "") if from_input else row["direction"]
            key = (sid, direction)
            if direction not in ("en-vi", "vi-en"):
                raise ValueError(f"Invalid direction for {sid!r}: {direction!r}")
            if key in seen:
                raise ValueError(f"Duplicate template key: {key}")
            seen.add(key)
            relative = PurePosixPath(sid)
            if (not sid or relative.is_absolute() or ".." in relative.parts
                    or "\\" in sid or ":" in sid):
                raise ValueError(f"Invalid sample_id: {sid!r}")
            root = roots.get(direction)
            if root is None:
                raise ValueError(f"Missing output directory for {direction}")
            prediction = root / (sid + ".md")
            if not prediction.is_file():
                raise ValueError(f"Missing prediction: {prediction}")
            with prediction.open(encoding="utf-8", newline="") as source_prediction:
                translation = source_prediction.read()
            if not translation.strip():
                raise ValueError(f"Empty prediction: {prediction}")
            rows.append(dict(sample_id=sid, direction=direction, translation=translation))
    if not rows:
        raise ValueError("Template has no prediction rows")
    if out.suffix.lower() != ".zip":
        raise ValueError("Output must have a .zip extension")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Finish the archive before replacing an existing submission.
    fd, temporary = tempfile.mkstemp(suffix=".zip", dir=out.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("results.csv", buffer.getvalue().encode("utf-8"))
        os.replace(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("dataset/private_test/private_test_input.csv"))
    parser.add_argument("--en-vi", type=Path, help="Directory containing Vietnamese output .md files")
    parser.add_argument("--vi-en", type=Path, help="Directory containing English output .md files")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    args = parser.parse_args(argv)
    roots = {direction: root for direction, root in
             (("en-vi", args.en_vi), ("vi-en", args.vi_en)) if root is not None}
    try:
        count = build_submission(args.template, roots, args.out)
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        parser.exit(1, f"Submission error: {exc}\n")
    print(f"Created {args.out}: {count} prediction rows, results.csv at ZIP root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
