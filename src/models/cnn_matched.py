"""
CNN clásica de capacidad igualada (Matched-Capacity CNN, PyTorch).

Opera sobre las 8 features PCA como una señal 1D e iguala EXACTAMENTE el número de
parámetros entrenables T de la QCNN para T in {15, 30, 45, 90}:

    T=15: Conv1d(1,1,k3)=4  | MaxPool(8->4) | Conv1d(1,1,k5)=6  | Linear(4,1)=5
    T=30: Conv1d(1,2,k3)=8  | MaxPool(8->4) | Conv1d(2,2,k3)=14 | Linear(8,1,sin bias)=8
    T=45: Conv1d(1,2,k3)=8  | MaxPool(8->4) | Conv1d(2,4,k3)=28 | MaxPool(4->2) | Linear(8,1)=9
    T=90: Conv1d(1,3,k3)=12 | MaxPool(8->4) | Conv1d(3,7,k3)=70 | AvgPool global | Linear(7,1)=8
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _arch_15() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(1, 1, kernel_size=5, padding=2), nn.ReLU(),
        nn.Flatten(), nn.Linear(4, 1),
    )


def _arch_30() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(2, 2, kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(8, 1, bias=False),
    )


def _arch_45() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(2, 4, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(), nn.Linear(8, 1),
    )


def _arch_90() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(3, 7, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Linear(7, 1),
    )


ARCHITECTURES: dict[int, Callable[[], nn.Sequential]] = {15: _arch_15, 30: _arch_30, 45: _arch_45, 90: _arch_90}


class CNNMatched(nn.Module):
    """CNN 1D con exactamente T parámetros entrenables. `forward` devuelve logits."""

    def __init__(self, target_params: int = 45):
        super().__init__()
        if target_params not in ARCHITECTURES:
            raise ValueError(f"T={target_params} no soportado para CNNMatched. Opciones: {sorted(ARCHITECTURES)}")
        self.target_params = target_params
        self.net = ARCHITECTURES[target_params]()

        actual = self.count_parameters()
        if actual != target_params:
            raise AssertionError(f"[CNNMatched] se esperaban {target_params} parámetros, obtenidos {actual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 8) -> (B, 1, 8)
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
