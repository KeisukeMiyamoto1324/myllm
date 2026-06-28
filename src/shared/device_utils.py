from pathlib import Path
import os
import time

import torch


DevicesValue = int | str


def resolve_device() -> torch.device:
    # ---------------------------------------------------------
    # Select the fastest available runtime in priority order so
    # training and inference can share the same device policy.
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_accelerator() -> str:
    # ---------------------------------------------------------
    # Convert the resolved torch device into the accelerator name
    # expected by Lightning's Trainer configuration.
    # ---------------------------------------------------------
    return resolve_device().type


def resolve_precision(accelerator: str) -> str:
    # ---------------------------------------------------------
    # Use mixed precision on CUDA to reduce activation memory while
    # keeping other backends on Lightning's default 32-bit behavior.
    # ---------------------------------------------------------
    precision_by_accelerator = {"cuda": "bf16-mixed"}
    return precision_by_accelerator.get(accelerator, "32-true")


def resolve_devices(devices: str) -> DevicesValue:
    # ---------------------------------------------------------
    # Convert the CLI device selector into the value expected by
    # Lightning while rejecting invalid numeric values early.
    # ---------------------------------------------------------
    if devices == "auto":
        return devices

    device_count = int(devices)

    if device_count < 1:
        raise ValueError("--devices must be auto or a positive integer")

    return device_count


def resolve_device_count(accelerator: str, devices: DevicesValue) -> int:
    # ---------------------------------------------------------
    # Resolve the effective device count used for metadata and
    # validation budgets before Lightning builds the trainer.
    # ---------------------------------------------------------
    if isinstance(devices, int):
        return devices

    if accelerator == "cuda":
        return max(1, torch.cuda.device_count())

    return 1


def resolve_strategy(accelerator: str, device_count: int) -> str | None:
    # ---------------------------------------------------------
    # Use explicit DDP only for CUDA multi-GPU runs. Other runtime
    # backends keep the existing single-process behavior.
    # ---------------------------------------------------------
    if accelerator == "cuda" and device_count > 1:
        return "ddp"

    return None


def is_global_zero_process() -> bool:
    # ---------------------------------------------------------
    # Check the distributed rank from the environment before the
    # Lightning trainer is available.
    # ---------------------------------------------------------
    return int(os.environ.get("RANK", "0")) == 0


def wait_for_file(path: Path, timeout_seconds: int = 3600) -> None:
    # ---------------------------------------------------------
    # Let non-zero ranks wait until rank zero finishes creating
    # shared files such as validation caches.
    # ---------------------------------------------------------
    start_time = time.monotonic()

    while not path.exists():
        if time.monotonic() - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {path}")

        time.sleep(1)
