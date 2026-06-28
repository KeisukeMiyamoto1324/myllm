import unittest
from unittest.mock import patch

from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_strategy


class DeviceUtilsTest(unittest.TestCase):
    def test_resolve_devices_accepts_auto_and_positive_integer(self) -> None:
        # ---------------------------------------------------------
        # Keep the CLI device selector compatible with Lightning's
        # auto mode while accepting explicit positive counts.
        # ---------------------------------------------------------
        self.assertEqual(resolve_devices(devices="auto"), "auto")
        self.assertEqual(resolve_devices(devices="4"), 4)

    def test_resolve_devices_rejects_non_positive_integer(self) -> None:
        # ---------------------------------------------------------
        # Reject invalid device counts before the trainer starts.
        # ---------------------------------------------------------
        with self.assertRaises(ValueError):
            resolve_devices(devices="0")

    def test_resolve_device_count_uses_cuda_count_for_auto(self) -> None:
        # ---------------------------------------------------------
        # Resolve auto CUDA devices from torch so metadata matches
        # the actual number of visible GPUs.
        # ---------------------------------------------------------
        with patch("torch.cuda.device_count", return_value=4):
            self.assertEqual(resolve_device_count(accelerator="cuda", devices="auto"), 4)

    def test_resolve_strategy_uses_ddp_for_cuda_multi_gpu(self) -> None:
        # ---------------------------------------------------------
        # Use explicit DDP only when CUDA has more than one active
        # training device.
        # ---------------------------------------------------------
        self.assertEqual(resolve_strategy(accelerator="cuda", device_count=2), "ddp")
        self.assertIsNone(resolve_strategy(accelerator="cuda", device_count=1))
        self.assertIsNone(resolve_strategy(accelerator="mps", device_count=2))


if __name__ == "__main__":
    unittest.main()
