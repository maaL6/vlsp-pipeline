"""Merge the LoRA adapter into Hy-MT2-1.8B and save a model dir usable as MODEL by run.sh / the Docker image.

  python tools/merge_adapter.py --adapter ADAPTER_DIR --out models/hy_mt_merged

ADAPTER_DIR holds adapter_model.safetensors + adapter_config.json. Needs torch, transformers, peft and
internet access (or a local HF cache) for the base model.
"""
import argparse
import hashlib
from pathlib import Path

BASE = "tencent/Hy-MT2-1.8B"
REVISION = "9a341cd1b679d3efd23b46e847b01745a71ed792"
ADAPTER_SHA256 = "30b5fc68a31c3de3d722a6dee5484aaf98c4f6b9d7b41fb55ac70f6d23a3ba89"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0", help="device_map for the merge (cpu works, slower)")
    ap.add_argument("--skip-sha-check", action="store_true")
    args = ap.parse_args()

    weights = args.adapter / "adapter_model.safetensors"
    if not args.skip_sha_check and sha256(weights) != ADAPTER_SHA256:
        raise SystemExit(f"adapter sha256 differs from {ADAPTER_SHA256}; pass --skip-sha-check to merge anyway")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION, dtype=torch.float16,
                                                 device_map=args.device, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(BASE, revision=REVISION, trust_remote_code=True).save_pretrained(args.out)
    print(f"merged model: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
