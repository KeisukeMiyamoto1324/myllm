import unittest

from src.shared.console import ProgressManager


class ProgressManagerTest(unittest.TestCase):
    def test_progress_manager_supports_nested_tasks(self) -> None:
        # ---------------------------------------------------------
        # Keep one long-running task active while a temporary task
        # starts, updates, and finishes in the same live display.
        # ---------------------------------------------------------
        manager = ProgressManager()
        training_task_id = manager.add_task("Training", total=10)
        validation_task_id = manager.add_task("Validation", total=2)

        manager.update(training_task_id, completed=3, metrics="train_loss=1.000")
        manager.update(validation_task_id, advance=2)
        manager.finish_task(validation_task_id)

        self.assertIn(training_task_id, manager.task_ids)
        self.assertNotIn(validation_task_id, manager.task_ids)

        manager.finish_task(training_task_id)
        self.assertEqual(manager.task_ids, set())


if __name__ == "__main__":
    unittest.main()
