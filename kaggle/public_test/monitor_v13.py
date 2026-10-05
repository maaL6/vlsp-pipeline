"""Keep auth in memory; follow live logs and fetch completed output automatically."""
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = "wann179/vlsp26-v13-batching-glossary-pt50"
OUT = ROOT / "results/v13_kaggle_20261004"


def main():
    token = getpass.getpass("Kaggle token (memory only): ")
    env = dict(os.environ, KAGGLE_API_TOKEN=token, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    OUT.mkdir(parents=True, exist_ok=True)
    def cli(args):
        r = subprocess.run([sys.executable, "-m", "kaggle", *args], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        return r.returncode, (r.stdout + r.stderr).replace(token, "[REDACTED]")
    with (OUT / "live.log").open("a", encoding="utf-8") as logfile:
        def start_stream():
            return subprocess.Popen([sys.executable, "-m", "kaggle", "kernels", "logs", KERNEL, "--follow"],
                                    env=env, stdout=logfile, stderr=subprocess.STDOUT)
        stream = start_stream()
        try:
            while True:
                try:
                    rc, output = cli(["kernels", "status", KERNEL])
                except (subprocess.TimeoutExpired, OSError) as exc:
                    rc, output = 1, type(exc).__name__
                state = {"kernel": KERNEL, "checked_unix": time.time(), "returncode": rc,
                         "status": output.strip(), "monitor_pid": os.getpid()}
                (OUT / "monitor_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
                print(output.strip(), flush=True)
                terminal = any(word in output.upper() for word in (".COMPLETE", ".ERROR", ".CANCEL", "\"COMPLETE\"", "\"ERROR\""))
                if rc == 0 and terminal:
                    # Outputs can take a little longer to publish than the status transition.
                    for attempt in range(6):
                        rc, downloaded = cli(["kernels", "output", KERNEL, "-p", str(OUT), "-q", "-o"])
                        (OUT / "download.log").write_text(downloaded, encoding="utf-8")
                        if rc == 0 and (OUT / "kernel_status.json").exists():
                            break
                        time.sleep(60)
                    rc_log, logs = cli(["kernels", "logs", KERNEL])
                    (OUT / "kernel.log").write_text(logs, encoding="utf-8")
                    state.update(download_returncode=rc, finished_unix=time.time())
                    if (OUT / "kernel_status.json").exists():
                        r = subprocess.run([sys.executable, str(ROOT / "kaggle/public_test/summarize_v13.py"),
                                            str(OUT)], cwd=ROOT, capture_output=True, text=True,
                                           encoding="utf-8", errors="replace")
                        state["summary_returncode"] = r.returncode
                        (OUT / "summary.log").write_text(r.stdout + r.stderr, encoding="utf-8")
                    (OUT / "monitor_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
                    return
                if stream.poll() is not None:
                    # Kaggle closes idle SSE streams; this does not stop the GPU session.
                    stream = start_stream()
                time.sleep(300)
        finally:
            stream.terminate()
            try:
                stream.wait(10)
            except subprocess.TimeoutExpired:
                stream.kill()


if __name__ == "__main__":
    main()
