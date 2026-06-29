from pathlib import Path
import tempfile
import unittest

import torch

from src.shared.training_checkpoint import resolve_resume_shuffle_seed


class TrainingCheckpointTest(unittest.TestCase):
    def test_resolve_resume_shuffle_seed_keeps_base_seed_without_checkpoint(self) -> None:
        # ---------------------------------------------------------
        # Start fresh training runs from the configured base seed
        # when no Lightning checkpoint is being resumed.
        # ---------------------------------------------------------
        shuffle_seed = resolve_resume_shuffle_seed(
            base_seed=17,
            checkpoint_path="",
        )

        self.assertEqual(shuffle_seed, 17)

    def test_resolve_resume_shuffle_seed_offsets_by_global_step(self) -> None:
        # ---------------------------------------------------------
        # Use Lightning global_step as a deterministic offset so
        # resumed streams do not reuse the initial shuffle order.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "step.ckpt"
            torch.save({"global_step": 123}, checkpoint_path)

            shuffle_seed = resolve_resume_shuffle_seed(
                base_seed=17,
                checkpoint_path=str(checkpoint_path),
            )

        self.assertEqual(shuffle_seed, 140)


if __name__ == "__main__":
    unittest.main()
