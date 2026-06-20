from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn


console = Console(stderr=True)


class ProgressManager:
    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=console,
            refresh_per_second=4,
        )
        self.task_ids: set[int] = set()

    def add_task(
        self,
        description: str,
        total: int,
        completed: int = 0,
        metrics: str = "",
    ) -> int:
        # ---------------------------------------------------------
        # Start the shared live display when its first task arrives
        # and add one independently updateable progress row.
        # ---------------------------------------------------------
        if len(self.task_ids) == 0:
            self.progress.start()

        task_id = self.progress.add_task(
            description=description,
            total=total,
            completed=completed,
            metrics=metrics,
        )
        self.task_ids.add(task_id)
        return task_id

    def update(
        self,
        task_id: int,
        completed: int | None = None,
        advance: int = 0,
        metrics: str | None = None,
    ) -> None:
        # ---------------------------------------------------------
        # Update task progress and optional metrics without creating
        # a second live display that could corrupt terminal output.
        # ---------------------------------------------------------
        fields = {} if metrics is None else {"metrics": metrics}
        self.progress.update(
            task_id=task_id,
            completed=completed,
            advance=advance,
            refresh=True,
            **fields,
        )

    def finish_task(self, task_id: int) -> None:
        # ---------------------------------------------------------
        # Remove completed nested tasks and stop the live display
        # after the final active task has finished.
        # ---------------------------------------------------------
        if task_id not in self.task_ids:
            return

        self.progress.stop_task(task_id=task_id)
        self.progress.refresh()
        self.task_ids.remove(task_id)

        if len(self.task_ids) == 0:
            self.progress.stop()
            self.progress.remove_task(task_id)
            return

        self.progress.remove_task(task_id)


progress_manager = ProgressManager()
