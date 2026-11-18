"""Copy this repository into a cloned Hugging Face Space directory.

Usage:
    python deploy/hf/sync_to_space.py D:/path/to/cloned-space

The target must be a clone of the Space repository (it already contains the
README.md with the Space metadata).  This script copies the runtime files and
replaces the Space README with deploy/hf/README.md, then prints the git commands
to push.  It never touches .git and never copies private data.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_DIRS = ["app", "diff_srt"]
INCLUDE_FILES = ["Dockerfile", "requirements.txt", "launcher.py", "LICENSE"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    if not (target / ".git").exists():
        raise SystemExit(f"{target} 不是 git 仓库（请先 git clone Space 仓库）")

    for name in INCLUDE_DIRS:
        src = ROOT / name
        dst = target / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in INCLUDE_FILES:
        shutil.copy2(ROOT / name, target / name)
    shutil.copy2(ROOT / "deploy" / "hf" / "README.md", target / "README.md")

    print(f"copied runtime files into {target}")
    print("next:")
    print(f"  git -C {target} add -A")
    print(f'  git -C {target} commit -m "deploy: sync from GitHub"')
    print(f"  git -C {target} push")


if __name__ == "__main__":
    main()
