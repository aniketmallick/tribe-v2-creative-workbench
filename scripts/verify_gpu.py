from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python platform: {platform.platform()}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name: no gpu")


if __name__ == "__main__":
    main()
