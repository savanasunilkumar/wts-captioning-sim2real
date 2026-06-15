"""Smoke-test the environment without needing a GPU.

Run on the login node to confirm all imports work.
"""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def check_import(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except ImportError as e:
        return False, str(e)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Executable: {sys.executable}")

    pkgs = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "PIL",
        "cv2",
        "decord",
        "av",
        "pycocoevalcap",
        "nltk",
        "wandb",
        "tqdm",
        "yaml",
        "jsonlines",
    ]
    print("\n--- Packages ---")
    failures: list[str] = []
    for p in pkgs:
        ok, info = check_import(p)
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {p:18s} {info}")
        if not ok:
            failures.append(p)

    import torch  # already verified above
    print("\n--- Torch / CUDA ---")
    print(f"  torch.__version__       = {torch.__version__}")
    print(f"  torch.version.cuda      = {torch.version.cuda}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device_count            = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  device[{i}]: {torch.cuda.get_device_name(i)}")
    else:
        print("  (No CUDA on login node — expected. Test on a GPU node.)")

    print("\n--- Paths ---")
    for path in [
        "/work/anujs/savana/aicity-track2",
        "/ptmp/anujs/savana/aicity-data",
        "/ptmp/anujs/savana/aicity-outputs",
        "/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct",
    ]:
        p = Path(path)
        size = "?"
        if p.exists():
            try:
                size = shutil.disk_usage(p).free // (1024**3)
                size = f"free={size} GB"
            except OSError:
                pass
        print(f"  [{'EXISTS' if p.exists() else 'MISS  '}] {p}  {size}")

    if failures:
        print(f"\n❌ {len(failures)} import failure(s): {failures}")
        return 1
    print("\n✅ Environment looks healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
