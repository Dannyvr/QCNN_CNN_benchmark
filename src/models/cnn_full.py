"""
CNN clásica de capacidad completa (techo de referencia) sobre imágenes 28x28.

Con la configuración por defecto (8, 16, 32) tiene 26,449 parámetros entrenables:
    Conv2d(1,8,3)=80 + BN(8)=16 + Conv2d(8,16,3)=1,168 + BN(16)=32
    + Linear(784,32)=25,120 + Linear(32,1)=33
No participa en la comparación de presupuesto igualado: su entrada y capacidad son distintas.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNFull(nn.Module):
    """CNN 2D estándar para clasificación binaria. `forward` devuelve logits."""

    def __init__(self, conv1_channels: int = 8, conv2_channels: int = 16, fc_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
