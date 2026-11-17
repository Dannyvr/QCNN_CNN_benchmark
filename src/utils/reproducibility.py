"""Control de semillas y registro del entorno de ejecución (reproducibilidad)."""

from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime
from importlib import metadata

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch.

    Se invoca antes de construir y entrenar CADA modelo, de modo que el resultado
    de una combinación (tarea, N, semilla, T, modelo) no depende del orden de
    ejecución ni de si la corrida se reanuda o se paraleliza.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_info() -> dict:
    packages = ["torch", "pennylane", "pennylane-lightning", "scikit-learn", "numpy",
                "scipy", "pandas", "matplotlib", "pyyaml"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "packages": versions,
    }

