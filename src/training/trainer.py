"""
Bucle de entrenamiento único para todos los modelos PyTorch (QCNN, CNN igualada,
MLP compacto y CNN completa). Todos devuelven logits y se entrenan con
BCEWithLogitsLoss + Adam bajo exactamente el mismo protocolo.

Selección de modelo: si `select_best_on_val` está activo, se evalúa la exactitud en
el conjunto de validación al final de cada época y se restauran los pesos de la mejor
época. El conjunto de prueba nunca interviene en el entrenamiento ni en la selección.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    epochs_run: int
    best_epoch: int
    val_accuracy: float
    train_time_sec: float  # Solo pasos de optimización (excluye la evaluación de validación)


def _make_train_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    bs = max(1, min(int(batch_size), len(dataset)))
    # Evita un último batch de tamaño 1 (rompe BatchNorm y añade ruido innecesario)
    drop_last = len(dataset) > bs and len(dataset) % bs == 1
    generator = torch.Generator().manual_seed(int(seed))  # shuffle reproducible e independiente del orden
    return DataLoader(dataset, batch_size=bs, shuffle=True, generator=generator, drop_last=drop_last)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    outputs = [model(X_t[i:i + batch_size]).reshape(-1) for i in range(0, len(X_t), batch_size)]
    return torch.cat(outputs).cpu().numpy()


def logits_to_predictions(logits: np.ndarray) -> np.ndarray:
    return (logits >= 0.0).astype(np.int64)


def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    return float(nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    ))


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hparams: dict,
    *,
    seed: int,
    epochs_scale: float = 1.0,
    select_best_on_val: bool = True,
    eval_batch_size: int = 256,
) -> TrainResult:
    epochs = max(1, int(round(float(hparams["epochs"]) * epochs_scale)))
    loader = _make_train_loader(X_train, y_train, int(hparams["batch_size"]), seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams.get("weight_decay", 0.0))
    )
    criterion = nn.BCEWithLogitsLoss()

    best_acc, best_epoch, best_state = -1.0, epochs, None
    train_time = 0.0

    for epoch in range(1, epochs + 1):
        t0 = perf_counter()
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).reshape(-1), yb)
            loss.backward()
            optimizer.step()
        train_time += perf_counter() - t0

        if select_best_on_val:
            val_acc = float(np.mean(logits_to_predictions(predict_logits(model, X_val, eval_batch_size)) == y_val))
            if val_acc > best_acc:  # '>' conserva la época más temprana ante empates
                best_acc, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(model.state_dict())

    if select_best_on_val and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_acc = float(np.mean(logits_to_predictions(predict_logits(model, X_val, eval_batch_size)) == y_val))

    return TrainResult(epochs_run=epochs, best_epoch=best_epoch, val_accuracy=best_acc, train_time_sec=train_time)

