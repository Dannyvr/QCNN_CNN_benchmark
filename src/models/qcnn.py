"""
Red Neuronal Convolucional Cuántica (QCNN) en PennyLane.

Ansatz inspirado en Cong et al. (Nature Physics, 2019): tres etapas jerárquicas de
convolución cuántica (unitarias de 2 qubits con pesos compartidos por etapa) y
pooling (rotaciones controladas que "concentran" la información en la mitad de
los qubits), 8 -> 4 -> 2 -> 1 qubits, con lectura <Z> en el qubit 7.

Presupuestos soportados con conteo exacto de parámetros variacionales:
    T = 3 etapas x (p_conv + p_pool)
    T=15: (3 + 2) | T=30: (7 + 3) | T=45: (10 + 5) | T=90: 2 repeticiones de (10 + 5)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Bloques de 2 qubits por presupuesto
# ---------------------------------------------------------------------------


def _conv_15(w, a, b):
    qml.RY(w[0], wires=a)
    qml.RY(w[1], wires=b)
    qml.CRZ(w[2], wires=[a, b])


def _pool_15(w, c, t):
    qml.CRX(w[0], wires=[c, t])
    qml.RY(w[1], wires=t)


def _conv_30(w, a, b):
    qml.RZ(w[0], wires=a)
    qml.RY(w[1], wires=a)
    qml.RZ(w[2], wires=b)
    qml.RY(w[3], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(w[4], wires=a)
    qml.RY(w[5], wires=b)
    qml.CRZ(w[6], wires=[a, b])


def _pool_30(w, c, t):
    qml.CRZ(w[0], wires=[c, t])
    qml.CRY(w[1], wires=[c, t])
    qml.RY(w[2], wires=t)


def _conv_45(w, a, b):
    qml.RZ(w[0], wires=a)
    qml.RY(w[1], wires=a)
    qml.RZ(w[2], wires=a)
    qml.RZ(w[3], wires=b)
    qml.RY(w[4], wires=b)
    qml.RZ(w[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(w[6], wires=a)
    qml.RZ(w[7], wires=a)
    qml.RY(w[8], wires=b)
    qml.RZ(w[9], wires=b)


def _pool_45(w, c, t):
    qml.RZ(w[0], wires=c)
    qml.RX(w[1], wires=c)
    qml.CRZ(w[2], wires=[c, t])
    qml.CRY(w[3], wires=[c, t])
    qml.RY(w[4], wires=t)


def _conv_90(w, a, b):
    _conv_45(w[:10], a, b)
    _conv_45(w[10:], a, b)


def _pool_90(w, c, t):
    _pool_45(w[:5], c, t)
    _pool_45(w[5:], c, t)


# T -> (params conv por etapa, params pool por etapa, bloque conv, bloque pool)
ANSATZ: dict[int, tuple[int, int, Callable, Callable]] = {
    15: (3, 2, _conv_15, _pool_15),
    30: (7, 3, _conv_30, _pool_30),
    45: (10, 5, _conv_45, _pool_45),
    90: (20, 10, _conv_90, _pool_90),
}

# Topología fija de 8 qubits: (pares de convolución, pares (control, objetivo) de pooling)
STAGES = (
    ([(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6), (7, 0)], [(0, 1), (2, 3), (4, 5), (6, 7)]),
    ([(1, 3), (5, 7), (3, 5), (7, 1)], [(1, 3), (5, 7)]),
    ([(3, 7)], [(3, 7)]),
)
N_QUBITS = 8
READOUT_WIRE = 7


def create_qcnn_qnode(target_params: int = 45, device_name: str = "default.qubit", n_qubits: int = N_QUBITS):
    """Construye el QNode de la QCNN para el presupuesto T indicado."""
    if n_qubits != N_QUBITS:
        raise ValueError(f"La topología implementada requiere {N_QUBITS} qubits (recibido {n_qubits}).")
    if target_params not in ANSATZ:
        raise ValueError(f"Presupuesto T={target_params} no soportado. Opciones: {sorted(ANSATZ)}")

    p_conv, p_pool, conv_block, pool_block = ANSATZ[target_params]
    stride = p_conv + p_pool
    dev = qml.device(device_name, wires=n_qubits)
    diff_method = "backprop" if device_name == "default.qubit" else "best"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # Angle encoding: una feature PCA por qubit
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        # Convolución + pooling jerárquicos (pesos compartidos dentro de cada etapa)
        for stage, (conv_pairs, pool_pairs) in enumerate(STAGES):
            w = weights[stage * stride:(stage + 1) * stride]
            for a, b in conv_pairs:
                conv_block(w[:p_conv], a, b)
            for c, t in pool_pairs:
                pool_block(w[p_conv:], c, t)
        return qml.expval(qml.PauliZ(READOUT_WIRE))

    return circuit


class QCNN(nn.Module):
    """QCNN envuelta como módulo PyTorch. `forward` devuelve logits.

    p(y=1) = (<Z> + 1) / 2 ; logit = log(p / (1 - p)).
    Entrenar con BCEWithLogitsLoss sobre este logit es matemáticamente equivalente a
    BCE sobre p, y permite un único bucle de entrenamiento para todos los modelos.
    """

    _EPS = 1e-6

    def __init__(
        self,
        target_params: int = 45,
        device_name: str = "default.qubit",
        n_qubits: int = N_QUBITS,
        init_scale: float = np.pi / 4,
    ):
        super().__init__()
        self.target_params = target_params
        self.n_qubits = n_qubits
        self.qnode = create_qcnn_qnode(target_params, device_name, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(self.qnode, {"weights": (target_params,)})
        with torch.no_grad():
            self.q_layer.weights.uniform_(-init_scale, init_scale)

        actual = self.count_parameters()
        if actual != target_params:
            raise AssertionError(f"[QCNN] se esperaban {target_params} parámetros, obtenidos {actual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        expval = self.q_layer(x)
        probs = torch.clamp((expval + 1.0) / 2.0, self._EPS, 1.0 - self._EPS)
        return torch.log(probs) - torch.log1p(-probs)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def circuit_resources(self) -> dict:
        """Recursos cuánticos del circuito (qubits, profundidad, número de compuertas)."""
        info = {"n_qubits": self.n_qubits}
        try:
            x = torch.zeros(self.n_qubits)
            specs = qml.specs(self.qnode)(x, self.q_layer.weights.detach())
            res = specs["resources"]
            depth = getattr(res, "depth", None)
            n_gates = getattr(res, "num_gates", None)
            if n_gates is None and hasattr(res, "gate_types"):
                n_gates = sum(res.gate_types.values())
            info.update({"circuit_depth": depth, "n_gates": n_gates})
        except Exception:  # La API de qml.specs cambia entre versiones; no es crítico
            pass
        return info
