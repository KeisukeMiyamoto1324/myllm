import torch


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
