"""Registro central de modelos: construcción por nombre y tipo de entrada requerida."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from src.models.baselines import MLPCompact, get_logistic_regression, get_svm_linear, get_svm_rbf
from src.models.cnn_full import CNNFull
from src.models.cnn_matched import CNNMatched
from src.models.qcnn import QCNN

# Representación de entrada que consume cada modelo (ver FoldViews)
INPUT_VIEW = {
    "QCNN": "angles",
    "CNN_Matched": "pca",
    "MLP_Compact": "pca",
    "Logistic_Regression": "pca",
    "SVM_Linear": "pca",
    "SVM_RBF": "pca",
    "CNN_Full": "image",
}

TORCH_MODELS = ("QCNN", "CNN_Matched", "MLP_Compact", "CNN_Full")


def is_torch_model(name: str) -> bool:
    return name in TORCH_MODELS


def build_torch_model(name: str, budget: Optional[int], model_cfg: dict[str, Any]) -> nn.Module:
    """Construye un modelo PyTorch. Llamar después de fijar la semilla global."""
    if name == "QCNN":
        return QCNN(
            target_params=int(budget),
            device_name=model_cfg.get("device", "default.qubit"),
            n_qubits=int(model_cfg.get("n_qubits", 8)),
            init_scale=float(model_cfg.get("init_scale", 0.785398)),
        )
    if name == "CNN_Matched":
        return CNNMatched(target_params=int(budget))
    if name == "MLP_Compact":
        return MLPCompact(target_params=int(budget))
    if name == "CNN_Full":
        return CNNFull(
            conv1_channels=int(model_cfg.get("conv1_channels", 8)),
            conv2_channels=int(model_cfg.get("conv2_channels", 16)),
            fc_dim=int(model_cfg.get("fc_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"'{name}' no es un modelo PyTorch registrado")


def build_sklearn_model(name: str, model_cfg: dict[str, Any], seed: int):
    if name == "Logistic_Regression":
        return get_logistic_regression(C=float(model_cfg.get("C", 1.0)),
                                       max_iter=int(model_cfg.get("max_iter", 1000)), seed=seed)
    if name == "SVM_Linear":
        return get_svm_linear(C=float(model_cfg.get("C", 1.0)), seed=seed)
    if name == "SVM_RBF":
        return get_svm_rbf(C=float(model_cfg.get("C", 1.0)), gamma=model_cfg.get("gamma", "scale"), seed=seed)
    raise ValueError(f"'{name}' no es un modelo scikit-learn registrado")

