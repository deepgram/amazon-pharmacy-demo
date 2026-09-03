"""Fail if the Dockerfile would ship a container that cannot import.

server/ loads catalog.json, knowledge-base.json and account.json at import time.
Adding a data file without adding it to the Dockerfile builds fine and then
crashes on boot, where Fly reports only "the app appears to be crashing". This
reproduces the image's file set in a temp dir and imports the app.

  uv run python scripts/check_container_files.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def copied_paths() -> list[str]:
    """Everything the Dockerfile COPYs from the build context."""
    out: list[str] = []
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        line = line.strip()
        if not line.startswith("COPY") or "--from=" in line:
            continue
        parts = re.split(r"\s+", line)[1:-1]   # drop COPY and the destination
        out.extend(parts)
    return out


def main() -> None:
    paths = copied_paths()
    print(f"Dockerfile copies: {', '.join(paths)}\n")
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "app"
        sandbox.mkdir()
        for rel in paths:
            src = ROOT / rel.rstrip("/")
            if not src.exists():
                sys.exit(f"FAIL  Dockerfile copies {rel!r} but it does not exist")
            dst = sandbox / src.name
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        # the venv is built inside the image; reuse this one just to import
        (sandbox / ".venv").symlink_to(ROOT / ".venv")

        proc = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-c",
             "import sys; sys.path.insert(0, '.'); import server.app; print('OK')"],
            cwd=sandbox, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        print("FAIL  the container would crash on boot:\n")
        for line in tail:
            print(f"    {line}")
        print("\n  Add the missing file to the Dockerfile's COPY line.")
        sys.exit(1)
    print("OK  the app imports with only the files the image ships")


if __name__ == "__main__":
    main()
