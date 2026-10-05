"""Run Kaggle CLI commands with credentials kept only in this process's memory.

Send one JSON object per stdin line. First: {"tokens": {"account": "token"}}.
Then: {"account": "account", "args": ["kernels", "status", "owner/slug"]}.
Replies are JSON; credentials are redacted from captured CLI output.
"""
import json
import getpass
import os
import subprocess
import sys


def main():
    tokens = {}
    if "--interactive" in sys.argv:
        tokens = {"a": getpass.getpass("Kaggle token A: "),
                  "b": getpass.getpass("Kaggle token B: ")}
        print(json.dumps({"accounts": sorted(tokens)}), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if "tokens" in command:
                tokens.update(command["tokens"])
                reply = {"accounts": sorted(tokens)}
            elif command.get("exit"):
                break
            else:
                args = command["args"]
                if not args or args[0] not in ("datasets", "kernels"):
                    raise ValueError("only datasets/kernels commands are supported")
                env = dict(os.environ, KAGGLE_API_TOKEN=tokens[command["account"]],
                           PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                result = subprocess.run([sys.executable, "-m", "kaggle", *args],
                                        env=env, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace")
                output = result.stdout + result.stderr
                for token in tokens.values():
                    output = output.replace(token, "[REDACTED]")
                reply = {"args": args, "returncode": result.returncode, "output": output}
        except Exception as exc:
            reply = {"error": type(exc).__name__}
        print(json.dumps(reply, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
