"""
Baselines clásicos:
1. MLP compacto con exactamente T in {15, 30, 45, 90} parámetros (PyTorch).
2. Regresión logística, SVM lineal y SVM RBF (scikit-learn).

    MLP T=15: Linear(8,1,sin bias)=8 -> Linear(1,2)=4 -> Linear(2,1)=3
    MLP T=30: Linear(8,3)=27 -> Linear(3,1,sin bias)=3
    MLP T=45: Linear(8,4,sin bias)=32 -> Linear(4,2)=10 -> Linear(2,1)=3
    MLP T=90: Linear(8,9)=81 -> Linear(9,1,sin bias)=9
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _mlp_15() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 1, bias=False), nn.ReLU(), nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))


def _mlp_30() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 3), nn.ReLU(), nn.Linear(3, 1, bias=False))


def _mlp_45() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 4, bias=False), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))


def _mlp_90() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 9), nn.ReLU(), nn.Linear(9, 1, bias=False))


MLP_ARCHITECTURES: dict[int, Callable[[], nn.Sequential]] = {15: _mlp_15, 30: _mlp_30, 45: _mlp_45, 90: _mlp_90}


class MLPCompact(nn.Module):
    """MLP con exactamente T parámetros entrenables. `forward` devuelve logits."""

    def __init__(self, target_params: int = 45):
        super().__init__()
        if target_params not in MLP_ARCHITECTURES:
            raise ValueError(f"T={target_params} no soportado para MLPCompact. Opciones: {sorted(MLP_ARCHITECTURES)}")
        self.target_params = target_params
        self.net = MLP_ARCHITECTURES[target_params]()

        actual = self.count_parameters()
        if actual != target_params:
            raise AssertionError(f"[MLPCompact] se esperaban {target_params} parámetros, obtenidos {actual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_logistic_regression(C: float = 1.0, max_iter: int = 1000, seed: int = 42) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs")


def get_svm_linear(C: float = 1.0, seed: int = 42) -> SVC:
    # Sin probability=True: está deprecado y ejecuta una CV interna de 5 folds que no se usa.
    return SVC(kernel="linear", C=C, random_state=seed)


def get_svm_rbf(C: float = 1.0, gamma: str | float = "scale", seed: int = 42) -> SVC:
    return SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)


def count_sklearn_params(model: Any) -> int:
    """Parámetros ajustados por un modelo scikit-learn.

    - Lineales (LogReg, SVM lineal): coeficientes + intercepto.
    - SVM RBF (no paramétrico): coeficientes duales (uno por vector de soporte) + intercepto.
      Crece con N, por lo que NO es comparable con el presupuesto T de los modelos variacionales.
    """
    if isinstance(model, SVC) and model.kernel != "linear":
        return int(model.dual_coef_.size + model.intercept_.size)
    return int(model.coef_.size + model.intercept_.size)
