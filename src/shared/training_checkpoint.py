from pathlib import Path

import torch


def resolve_resume_shuffle_seed(base_seed: int, checkpoint_path: str) -> int:
    # ---------------------------------------------------------
    # Move resumed streaming jobs away from the initial shuffle
    # order by deriving a deterministic offset from global_step.
    # ---------------------------------------------------------
    if checkpoint_path == "":
        return base_seed

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    return base_seed + int(checkpoint["global_step"])
