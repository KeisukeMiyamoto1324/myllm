from pathlib import Path
import os
import tempfile

import torch


def _fsync_parent_directory(path: Path) -> None:
    # ---------------------------------------------------------
    # Flush the parent directory entry so the atomic rename is
    # durable after the process returns.
    # ---------------------------------------------------------
    directory_fd = os.open(path.parent, os.O_RDONLY)

    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_torch_save(payload: object, path: Path) -> None:
    # ---------------------------------------------------------
    # Write the tensor payload to a same-directory temporary file
    # before exposing it at the final path atomically.
    # ---------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)

    try:
        with temporary_file:
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
        _fsync_parent_directory(path=path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise
