"""
Almacenamiento incremental de resultados (CSV) con soporte de reanudación.

Cada fila es una combinación única (tarea, N_train, semilla, T, modelo). Las filas se
escriben en cuanto termina cada job, por lo que una interrupción solo pierde el job en curso.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

RESULT_FIELDS = [
    "timestamp", "mode", "task", "task_label", "classes", "n_train", "n_per_class", "seed",
    "budget_T", "model_name", "model_family", "input_view", "trainable_params",
    "n_qubits", "circuit_depth", "n_gates",
    "epochs_run", "best_epoch", "val_accuracy",
    "test_accuracy", "test_balanced_acc", "test_f1", "test_precision", "test_recall", "test_loss",
    "train_time_sec", "infer_time_sec",
]

ResultKey = tuple[str, int, int, int, str]


def result_key(task: str, n_train: int, seed: int, budget: Optional[float], model: str) -> ResultKey:
    """Clave única de una corrida. Los modelos fijos (sin T) usan budget = 0."""
    b = 0 if budget is None or (isinstance(budget, float) and math.isnan(budget)) else int(budget)
    return (str(task), int(n_train), int(seed), b, str(model))


class ResultStore:
    def __init__(self, csv_path: Path, resume: bool = True):
        self.path = Path(csv_path)
        if self.path.exists() and not resume:
            backup = self.path.with_name(f"{self.path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
            self.path.rename(backup)

    def load(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=RESULT_FIELDS)
        return pd.read_csv(self.path)

    def completed_keys(self) -> set[ResultKey]:
        df = self.load()
        return {
            result_key(r.task, r.n_train, r.seed, r.budget_T, r.model_name)
            for r in df.itertuples(index=False)
        }

    def append(self, records: Iterable[dict]) -> None:
        records = list(records)
        if not records:
            return
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(records)
