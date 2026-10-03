"""Optional MLflow experiment tracking.

MLflow writes to ``./mlruns`` by default; set ``MLFLOW_TRACKING_URI`` to use a remote server.
"""

from __future__ import annotations

from typing import Any


class Tracker:
    """No-op tracker used when MLflow logging is disabled."""

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass


class MlflowTracker(Tracker):
    def __init__(self, experiment_name: str, run_name: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)
        self._run_name = run_name

    def __enter__(self) -> MlflowTracker:
        self._mlflow.start_run(run_name=self._run_name)
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)
