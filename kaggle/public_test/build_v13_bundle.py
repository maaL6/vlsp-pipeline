"""Stage the current working tree and the previously downloaded official evaluator."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "kaggle/public_test/build"


def main():
    bundle = BUILD / "v13_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    pipe = bundle / "pipeline-v1"
    pipe.mkdir(exist_ok=True)
    for name in ("docmt", "configs"):
        shutil.copytree(ROOT / name, pipe / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("requirements.txt", "run.sh"):
        shutil.copy2(ROOT / name, pipe / name)
    shutil.copytree(BUILD / "original/benchmark", bundle / "benchmark", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(bundle.rglob("*")) if p.is_file()
             and p.name not in ("pipeline_bundle_manifest.json", "dataset-metadata.json")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"git_head": head, "working_tree": "uncommitted v1.3", "files": files}
    (bundle / "pipeline_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    owner, slug = "wann179", "vlsp26-pipeline-v13-pt50"
    (bundle / "dataset-metadata.json").write_text(json.dumps({"title": slug, "id": f"{owner}/{slug}",
                                                            "licenses": [{"name": "other"}]}), encoding="utf-8")
    kernel = BUILD / "v13_kernel"
    kernel.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "kaggle/public_test/kernel_v13.py", kernel / "kernel_v13.py")
    kernel_slug = "vlsp26-v13-batching-glossary-pt50"
    metadata = {"id": f"{owner}/{kernel_slug}", "title": kernel_slug, "code_file": "kernel_v13.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{owner}/vlsp26-hy-a1-merged",
                                                                          f"{owner}/{slug}"],
                "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(bundle), "files": len(files), "kernel": str(kernel)}, indent=2))


if __name__ == "__main__":
    main()
