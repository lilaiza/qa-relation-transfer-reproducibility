"""Check that the active PyTorch interpreter can execute on ROCm."""

from __future__ import annotations

import sys

import torch


def main() -> None:
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm/HIP: {torch.version.hip}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: this process cannot access a ROCm GPU. "
            "Check the interpreter, /dev/kfd and /dev/dri devices, "
            "and the current user's permissions."
        )

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    left = torch.randn((1024, 1024), device=device)
    result = left @ left
    torch.cuda.synchronize(device)
    assert result.is_cuda
    print(f"Matrix multiplication completed on: {result.device}")


if __name__ == "__main__":
    main()
