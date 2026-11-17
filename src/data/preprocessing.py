"""
Módulo de Preprocesamiento Científico con Prevención Rigurosa de Data Leakage.
Ajusta transformaciones (PCA, Scaler) EXCLUSIVAMENTE sobre el subconjunto de entrenamiento
de cada ejecución/semilla, proyectando posteriormente validación y prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ArrayTriple = tuple[np.ndarray, np.ndarray, np.ndarray]


def _to_unit_float(X: np.ndarray) -> np.ndarray:
    """Convierte imágenes a float32 en [0, 1].

    La decisión se basa en el dtype (uint8 => /255) y no en los valores de los datos,
    de modo que la transformación es idéntica para train, validación y test.
    """
    X_float = X.astype(np.float32)
    if np.issubdtype(X.dtype, np.integer):
        X_float /= 255.0
    return X_float


class PreprocessingPipeline:
    """
    Pipeline que encapsula la reducción de dimensionalidad y normalización.
    Garantiza aislamiento estricto entre entrenamiento, validación y prueba.
    """

    def __init__(
        self,
        n_components: int = 8,
        scaler_type: str = "standard",
        quantum_angle_range: str = "0_to_pi",
        seed: int = 42,
    ):
        self.n_components = n_components
        self.scaler_type = scaler_type.lower()
        self.quantum_angle_range = quantum_angle_range
        self.seed = seed

        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.is_fitted: bool = False

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        X_unit = _to_unit_float(X)
        return X_unit.reshape(len(X_unit), -1)

    def fit(self, X_train: np.ndarray) -> "PreprocessingPipeline":
        """Ajusta PCA y el escalador ÚNICAMENTE sobre los datos de entrenamiento."""
        X_train_flat = self._flatten(X_train)
        if X_train_flat.shape[0] < self.n_components:
            raise ValueError(
                f"El tamaño de entrenamiento ({X_train_flat.shape[0]}) es menor "
                f"que el número de componentes PCA solicitadas ({self.n_components})."
            )

        self.pca = PCA(n_components=self.n_components, random_state=self.seed)
        X_train_pca = self.pca.fit_transform(X_train_flat)

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Escalador desconocido: {self.scaler_type}")

        self.scaler.fit(X_train_pca)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma datos con el PCA y el escalador previamente ajustados en train."""
        if not self.is_fitted or self.pca is None or self.scaler is None:
            raise RuntimeError("El pipeline debe ser ajustado con fit() antes de invocar transform().")
        return self.scaler.transform(self.pca.transform(self._flatten(X))).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def to_quantum_angles(self, X_features: np.ndarray) -> np.ndarray:
        """
        Mapea las características reducidas a un rango de ángulos válido para
        rotaciones cuánticas. Es una función fija (sin parámetros ajustados),
        por lo que no introduce fuga de información.
        """
        if self.quantum_angle_range == "0_to_pi":
            out = np.pi / (1.0 + np.exp(-X_features))
        elif self.quantum_angle_range == "minus_pi_to_pi":
            out = np.pi * np.tanh(X_features)
        elif self.quantum_angle_range == "minmax_clip":
            out = (np.clip(X_features, -3.0, 3.0) + 3.0) / 6.0 * np.pi
        else:
            raise ValueError(f"Rango cuántico desconocido: {self.quantum_angle_range}")
        return out.astype(np.float32)

    @staticmethod
    def prepare_for_cnn_full(X: np.ndarray) -> np.ndarray:
        """Imágenes brutas -> tensor (N, 1, 28, 28) en [0, 1] para la CNN completa."""
        X_unit = _to_unit_float(X)
        if X_unit.ndim == 3:
            return X_unit[:, None, :, :]
        if X_unit.ndim == 2:
            return X_unit.reshape(-1, 1, 28, 28)
        return X_unit


@dataclass
class FoldViews:
    """Representaciones (train, val, test) de un fold para cada tipo de entrada.

    - "pca"    : features PCA estandarizadas (baselines compactos)
    - "angles" : features PCA mapeadas a ángulos (QCNN)
    - "image"  : imágenes completas 28x28 (CNN de resolución completa)
    """

    views: dict[str, ArrayTriple]

    def __getitem__(self, key: str) -> ArrayTriple:
        return self.views[key]


def build_fold_views(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    prep_cfg: dict,
    seed: int,
    need_images: bool,
) -> FoldViews:
    """Ajusta el pipeline SOLO con train y genera todas las vistas del fold."""
    pipe = PreprocessingPipeline(
        n_components=int(prep_cfg.get("pca_components", 8)),
        scaler_type=prep_cfg.get("scaler", "standard"),
        quantum_angle_range=prep_cfg.get("quantum_angle_range", "0_to_pi"),
        seed=seed,
    ).fit(X_train_raw)

    raw = (X_train_raw, X_val_raw, X_test_raw)
    pca = tuple(pipe.transform(X) for X in raw)
    views: dict[str, ArrayTriple] = {
        "pca": pca,
        "angles": tuple(pipe.to_quantum_angles(X) for X in pca),
    }
    if need_images:
        views["image"] = tuple(PreprocessingPipeline.prepare_for_cnn_full(X) for X in raw)
    return FoldViews(views)
