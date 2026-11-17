"""
Módulo de Carga y Particionamiento Estático para tareas binarias de MNIST (p. ej. 0 vs 1, 3 vs 8).
Garantiza aislamiento estricto del conjunto de prueba y reproducibilidad.
"""

from __future__ import annotations
import os
import json
import numpy as np
import torchvision
from typing import Tuple, Dict, Any


def split_file_path(splits_dir: str, classes: Tuple[int, int]) -> str:
    """Ruta del archivo .npz de partición estática para un par de clases."""
    return os.path.join(splits_dir, f"mnist_{classes[0]}{classes[1]}_static_test.npz")


def load_raw_mnist(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset MNIST original en memoria como arrays NumPy de tipo uint8.
    """
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)

    X_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    X_test = test_set.data.numpy()
    y_test = test_set.targets.numpy()

    return X_train, y_train, X_test, y_test


def filter_binary_classes(
    X: np.ndarray,
    y: np.ndarray,
    classes: Tuple[int, int] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra las muestras correspondientes exclusivamente a las dos clases indicadas
    y remapea las etiquetas a 0 y 1.
    """
    c0, c1 = classes
    mask = (y == c0) | (y == c1)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # Remapear a {0, 1}
    y_binary = np.where(y_filtered == c0, 0, 1).astype(np.int64)
    return X_filtered, y_binary


def initialize_static_splits(
    data_dir: str = "./data",
    splits_dir: str = "./data/splits",
    classes: Tuple[int, int] = (0, 1),
    test_size: int = 1000,
    val_size: int = 200,
    seed: int = 42,
    force_recreate: bool = False
) -> Dict[str, Any]:
    """
    Genera y almacena en disco el archivo de partición estática para MNIST binario.
    El conjunto de prueba y validación se fijan de forma inmutable antes de cualquier
    fase de entrenamiento o ajuste de preprocesamiento.

    Guarda:
    - mnist_01_static_test.npz: Arrays de X_test, y_test, X_val, y_val, X_train_pool, y_train_pool.
    - mnist_01_static_meta.json: Metadatos del particionamiento, tamaños y semillas.
    """
    os.makedirs(splits_dir, exist_ok=True)
    npz_path = split_file_path(splits_dir, classes)
    meta_path = npz_path.replace("_static_test.npz", "_static_meta.json")

    if os.path.exists(npz_path) and os.path.exists(meta_path) and not force_recreate:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mismatch = {
            k: (meta.get(k), v)
            for k, v in {"seed": seed, "test_size": test_size, "val_size": val_size}.items()
            if meta.get(k) != v
        }
        if mismatch:
            raise ValueError(
                f"La partición estática existente {npz_path} fue creada con parámetros distintos "
                f"(guardado, solicitado): {mismatch}. Elimine el archivo o use force_recreate=True."
            )
        print(f"[DataLoader] Cargando partición estática existente desde: {npz_path}")
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}

    print(f"[DataLoader] Creando nueva partición estática estricta (Seed: {seed})...")
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = load_raw_mnist(data_dir=data_dir)

    # Filtrar las dos clases de la tarea y remapear a {0, 1}
    X_tr, y_tr = filter_binary_classes(X_tr_raw, y_tr_raw, classes=classes)
    X_te, y_te = filter_binary_classes(X_te_raw, y_te_raw, classes=classes)

    # Unir todo el pool disponible para particionar de forma balanceada y sin fugas
    X_total = np.concatenate([X_tr, X_te], axis=0)
    y_total = np.concatenate([y_tr, y_te], axis=0)

    rng = np.random.RandomState(seed)

    idx_c0 = np.where(y_total == 0)[0]
    idx_c1 = np.where(y_total == 1)[0]

    rng.shuffle(idx_c0)
    rng.shuffle(idx_c1)

    n_test_half = test_size // 2
    n_val_half = val_size // 2

    # Partición de Test (50% clase 0, 50% clase 1)
    test_idx = np.concatenate([idx_c0[:n_test_half], idx_c1[:n_test_half]])
    rng.shuffle(test_idx)

    # Partición de Validación (50% clase 0, 50% clase 1)
    val_idx = np.concatenate([
        idx_c0[n_test_half:n_test_half + n_val_half],
        idx_c1[n_test_half:n_test_half + n_val_half]
    ])
    rng.shuffle(val_idx)

    # Pool restante para Entrenamiento
    pool_idx = np.concatenate([
        idx_c0[n_test_half + n_val_half:],
        idx_c1[n_test_half + n_val_half:]
    ])
    rng.shuffle(pool_idx)

    X_test = X_total[test_idx]
    y_test = y_total[test_idx]

    X_val = X_total[val_idx]
    y_val = y_total[val_idx]

    X_train_pool = X_total[pool_idx]
    y_train_pool = y_total[pool_idx]

    # Guardar en archivo binario inmutable .npz
    np.savez_compressed(
        npz_path,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        X_train_pool=X_train_pool,
        y_train_pool=y_train_pool
    )

    metadata = {
        "dataset": "mnist_binary",
        "classes": list(classes),
        "seed": seed,
        "test_size": int(len(y_test)),
        "test_distribution": {"c0": int((y_test == 0).sum()), "c1": int((y_test == 1).sum())},
        "val_size": int(len(y_val)),
        "val_distribution": {"c0": int((y_val == 0).sum()), "c1": int((y_val == 1).sum())},
        "train_pool_size": int(len(y_train_pool)),
        "train_pool_distribution": {"c0": int((y_train_pool == 0).sum()), "c1": int((y_train_pool == 1).sum())},
        "file_path": os.path.abspath(npz_path)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[DataLoader] Partición estática guardada exitosamente en {npz_path}")
    print(f"             Test: {len(y_test)} | Val: {len(y_val)} | Pool Train: {len(y_train_pool)}")

    return {
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_pool": X_train_pool,
        "y_train_pool": y_train_pool
    }


def sample_train_subset(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_samples: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Submuestrea un subconjunto de entrenamiento perfectamente balanceado de tamaño n_samples.
    Garantiza n_samples/2 para clase 0 y n_samples/2 para clase 1 con semilla determinista.
    """
    assert n_samples % 2 == 0, f"n_samples debe ser par para balance estricto, recibido: {n_samples}"
    half = n_samples // 2

    idx_0 = np.where(y_pool == 0)[0]
    idx_1 = np.where(y_pool == 1)[0]

    assert len(idx_0) >= half, f"No hay suficientes muestras de clase 0 en el pool ({len(idx_0)} < {half})"
    assert len(idx_1) >= half, f"No hay suficientes muestras de clase 1 en el pool ({len(idx_1)} < {half})"

    rng = np.random.RandomState(seed)
    chosen_0 = rng.choice(idx_0, size=half, replace=False)
    chosen_1 = rng.choice(idx_1, size=half, replace=False)

    chosen_indices = np.concatenate([chosen_0, chosen_1])
    rng.shuffle(chosen_indices)

    return X_pool[chosen_indices].copy(), y_pool[chosen_indices].copy()

