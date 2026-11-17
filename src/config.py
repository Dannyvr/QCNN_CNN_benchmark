"""
Carga, validación y resolución del plan experimental a partir de config.yaml.

Toda la lógica de "qué se ejecuta" vive aquí: el resto del código recibe un
`ExperimentPlan` inmutable y ya validado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_MODES = ("pilot", "data_size", "param_budget", "grid")
SUPPORTED_BUDGETS = (15, 30, 45, 90)
VARIATIONAL_MODELS = ("QCNN", "CNN_Matched", "MLP_Compact")
FIXED_MODELS = ("Logistic_Regression", "SVM_Linear", "SVM_RBF", "CNN_Full")
ALL_MODELS = VARIATIONAL_MODELS + FIXED_MODELS


class ConfigError(ValueError):
    """Error de configuración detectado antes de iniciar cualquier entrenamiento."""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    classes: tuple[int, int]
    description: str = ""

    @property
    def label(self) -> str:
        return f"MNIST {self.classes[0]} vs {self.classes[1]}"


@dataclass(frozen=True)
class ExperimentPlan:
    mode: str
    description: str
    tasks: tuple[TaskSpec, ...]
    train_sizes: tuple[int, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    epochs_scale: float

    @property
    def variational_models(self) -> tuple[str, ...]:
        return tuple(m for m in self.models if m in VARIATIONAL_MODELS)

    @property
    def fixed_models(self) -> tuple[str, ...]:
        return tuple(m for m in self.models if m in FIXED_MODELS)

    @property
    def n_jobs_total(self) -> int:
        return len(self.tasks) * len(self.train_sizes) * len(self.seeds)

    @property
    def fits_per_job(self) -> int:
        return len(self.fixed_models) + len(self.variational_models) * len(self.budgets)

    def describe(self) -> str:
        lines = [
            f"Modo           : {self.mode} — {self.description}",
            f"Tareas         : {', '.join(t.label for t in self.tasks)}",
            f"N_train        : {list(self.train_sizes)}",
            f"Presupuestos T : {list(self.budgets)}",
            f"Semillas       : {len(self.seeds)} {list(self.seeds)}",
            f"Modelos        : {', '.join(self.models)}",
            f"Escala épocas  : {self.epochs_scale}",
            f"Total          : {self.n_jobs_total} jobs x {self.fits_per_job} ajustes = "
            f"{self.n_jobs_total * self.fits_per_job} entrenamientos",
        ]
        return "\n".join(lines)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuración inválida en {path}")
    return cfg


def _require(cfg: dict, *keys: str) -> Any:
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Falta la clave obligatoria '{'.'.join(keys)}' en la configuración")
        node = node[key]
    return node


def build_plan(cfg: dict[str, Any]) -> ExperimentPlan:
    """Resuelve el modo activo y valida exhaustivamente la configuración."""
    mode = _require(cfg, "run", "mode")
    if mode not in VALID_MODES:
        raise ConfigError(f"run.mode='{mode}' inválido. Opciones: {VALID_MODES}")
    mode_cfg = _require(cfg, "modes", mode)

    # --- Tareas ---------------------------------------------------------------
    task_defs = _require(cfg, "tasks")
    task_names = _require(cfg, "run", "tasks")
    if not task_names:
        raise ConfigError("run.tasks no puede estar vacío")
    tasks = []
    for name in task_names:
        if name not in task_defs:
            raise ConfigError(f"La tarea '{name}' no está definida en la sección 'tasks'")
        classes = tuple(int(c) for c in task_defs[name]["classes"])
        if len(classes) != 2 or classes[0] == classes[1] or not all(0 <= c <= 9 for c in classes):
            raise ConfigError(f"tasks.{name}.classes debe tener dos dígitos distintos entre 0 y 9")
        tasks.append(TaskSpec(name, classes, task_defs[name].get("description", "")))

    # --- Modelos --------------------------------------------------------------
    models = tuple(_require(cfg, "run", "models"))
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown or not models:
        raise ConfigError(f"Modelos desconocidos {unknown}. Disponibles: {ALL_MODELS}")
    for m in models:
        _require(cfg, "models", m)

    # --- Tamaños de entrenamiento ----------------------------------------------
    n_components = int(_require(cfg, "preprocessing", "pca_components"))
    train_sizes = tuple(sorted(int(n) for n in _require(mode_cfg, "train_sizes")))
    for n in train_sizes:
        if n <= 0 or n % 2:
            raise ConfigError(f"N_train={n} debe ser positivo y par (muestreo balanceado por clase)")
        if n < n_components:
            raise ConfigError(f"N_train={n} es menor que pca_components={n_components}")

    # --- Presupuestos ------------------------------------------------------------
    budgets = tuple(sorted(int(t) for t in _require(mode_cfg, "budgets")))
    if any(m in VARIATIONAL_MODELS for m in models):
        bad = [t for t in budgets if t not in SUPPORTED_BUDGETS]
        if bad or not budgets:
            raise ConfigError(f"Presupuestos T no soportados {bad}. Soportados: {SUPPORTED_BUDGETS}")

    # --- Semillas -----------------------------------------------------------------
    seeds = tuple(int(s) for s in _require(mode_cfg, "seeds"))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigError("La lista de semillas debe ser no vacía y sin duplicados")

    epochs_scale = float(mode_cfg.get("epochs_scale", 1.0))
    if epochs_scale <= 0:
        raise ConfigError("epochs_scale debe ser > 0")

    if cfg["models"].get("QCNN", {}).get("n_qubits", 8) != n_components and "QCNN" in models:
        raise ConfigError("models.QCNN.n_qubits debe coincidir con preprocessing.pca_components")

    return ExperimentPlan(
        mode=mode,
        description=mode_cfg.get("description", ""),
        tasks=tuple(tasks),
        train_sizes=train_sizes,
        budgets=budgets,
        seeds=seeds,
        models=models,
        epochs_scale=epochs_scale,
    )

