"""
Ejecutor de Jobs Experimentales.

Cada job corresponde a una tupla (TaskSpec, n_train, seed):
1. Submuestrea el conjunto de entrenamiento balanceado (n_train/2 por clase).
2. Ajusta el pipeline de preprocesamiento (PCA + Scaler) EXCLUSIVAMENTE con train.
3. Ejecuta los modelos fijos (una vez por job) y los modelos variacionales para
   cada presupuesto T en `plan.budgets`, saltando combinaciones ya registradas
   cuando `resume=True`.
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any, Optional

import numpy as np

from src.config import ExperimentPlan, TaskSpec
from src.data.loader import sample_train_subset
from src.data.preprocessing import FoldViews, build_fold_views
from src.models.baselines import count_sklearn_params
from src.models.registry import INPUT_VIEW, build_sklearn_model, build_torch_model, is_torch_model
from src.training.trainer import bce_from_logits, logits_to_predictions, predict_logits, train_torch_model
from src.utils.logger import ResultKey, result_key
from src.utils.metrics import compute_classification_metrics
from src.utils.reproducibility import set_global_seed


def _evaluate_and_pack(
    *,
    plan: ExperimentPlan,
    task: TaskSpec,
    n_train: int,
    seed: int,
    budget_t: Optional[int],
    model_name: str,
    model_family: str,
    trainable_params: int,
    q_resources: dict[str, Any],
    epochs_run: Optional[int],
    best_epoch: Optional[int],
    val_accuracy: Optional[float],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_loss: Optional[float],
    train_time_sec: float,
    infer_time_sec: float,
) -> dict[str, Any]:
    metrics = compute_classification_metrics(y_test, y_pred, loss=test_loss)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": plan.mode,
        "task": task.name,
        "task_label": task.label,
        "classes": f"{task.classes[0]}_{task.classes[1]}",
        "n_train": int(n_train),
        "n_per_class": int(n_train // 2),
        "seed": int(seed),
        "budget_T": budget_t,
        "model_name": model_name,
        "model_family": model_family,
        "input_view": INPUT_VIEW[model_name],
        "trainable_params": int(trainable_params),
        "n_qubits": q_resources.get("n_qubits"),
        "circuit_depth": q_resources.get("circuit_depth"),
        "n_gates": q_resources.get("n_gates"),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "val_accuracy": None if val_accuracy is None else round(float(val_accuracy), 4),
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_balanced_acc": round(metrics["balanced_accuracy"], 4),
        "test_f1": round(metrics["f1_score"], 4),
        "test_precision": round(metrics["precision"], 4),
        "test_recall": round(metrics["recall"], 4),
        "test_loss": None if metrics["loss"] is None else round(metrics["loss"], 4),
        "train_time_sec": round(float(train_time_sec), 4),
        "infer_time_sec": round(float(infer_time_sec), 5),
    }


def _run_single_model(
    *,
    model_name: str,
    budget_t: Optional[int],
    model_family: str,
    fold_views: FoldViews,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    plan: ExperimentPlan,
    task: TaskSpec,
    n_train: int,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    set_global_seed(seed)
    model_cfg = cfg["models"][model_name]
    train_cfg = cfg.get("training", {})
    select_best = bool(train_cfg.get("select_best_on_val", True))
    eval_bs = int(train_cfg.get("eval_batch_size", 256))

    X_tr, X_va, X_te = fold_views[INPUT_VIEW[model_name]]

    if is_torch_model(model_name):
        model = build_torch_model(model_name, budget_t, model_cfg)
        train_res = train_torch_model(
            model,
            X_tr,
            y_train,
            X_va,
            y_val,
            model_cfg,
            seed=seed,
            epochs_scale=plan.epochs_scale,
            select_best_on_val=select_best,
            eval_batch_size=eval_bs,
        )
        t_inf0 = perf_counter()
        logits_te = predict_logits(model, X_te, batch_size=eval_bs)
        infer_time = perf_counter() - t_inf0

        preds_te = logits_to_predictions(logits_te)
        loss_te = bce_from_logits(logits_te, y_test)
        q_res = model.circuit_resources() if hasattr(model, "circuit_resources") else {}

        return _evaluate_and_pack(
            plan=plan,
            task=task,
            n_train=n_train,
            seed=seed,
            budget_t=budget_t,
            model_name=model_name,
            model_family=model_family,
            trainable_params=model.count_parameters(),
            q_resources=q_res,
            epochs_run=train_res.epochs_run,
            best_epoch=train_res.best_epoch,
            val_accuracy=train_res.val_accuracy,
            y_test=y_test,
            y_pred=preds_te,
            test_loss=loss_te,
            train_time_sec=train_res.train_time_sec,
            infer_time_sec=infer_time,
        )

    # Modelo scikit-learn
    clf = build_sklearn_model(model_name, model_cfg, seed=seed)
    t0 = perf_counter()
    clf.fit(X_tr, y_train)
    train_time = perf_counter() - t0

    val_acc = float(np.mean(clf.predict(X_va) == y_val))

    t_inf0 = perf_counter()
    preds_te = clf.predict(X_te)
    infer_time = perf_counter() - t_inf0

    return _evaluate_and_pack(
        plan=plan,
        task=task,
        n_train=n_train,
        seed=seed,
        budget_t=None,
        model_name=model_name,
        model_family=model_family,
        trainable_params=count_sklearn_params(clf),
        q_resources={},
        epochs_run=None,
        best_epoch=None,
        val_accuracy=val_acc,
        y_test=y_test,
        y_pred=preds_te,
        test_loss=None,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
    )


def execute_fold_job(
    *,
    task: TaskSpec,
    n_train: int,
    seed: int,
    splits: dict[str, np.ndarray],
    plan: ExperimentPlan,
    cfg: dict[str, Any],
    completed: set[ResultKey],
) -> list[dict[str, Any]]:
    """Ejecuta todos los modelos pendientes para un fold (task, n_train, seed)."""
    pending_fixed = [
        m for m in plan.fixed_models if result_key(task.name, n_train, seed, None, m) not in completed
    ]
    pending_var = [
        (t, m)
        for t in plan.budgets
        for m in plan.variational_models
        if result_key(task.name, n_train, seed, t, m) not in completed
    ]
    if not pending_fixed and not pending_var:
        return []

    X_train_raw, y_train = sample_train_subset(
        splits["X_train_pool"], splits["y_train_pool"], n_samples=n_train, seed=seed
    )
    need_images = "CNN_Full" in pending_fixed
    fold_views = build_fold_views(
        X_train_raw=X_train_raw,
        X_val_raw=splits["X_val"],
        X_test_raw=splits["X_test"],
        prep_cfg=cfg["preprocessing"],
        seed=seed,
        need_images=need_images,
    )

    records: list[dict[str, Any]] = []

    # 1. Modelos de capacidad fija (independientes de T)
    for m_name in pending_fixed:
        family = "ceiling" if m_name == "CNN_Full" else "classical_fixed"
        records.append(
            _run_single_model(
                model_name=m_name,
                budget_t=None,
                model_family=family,
                fold_views=fold_views,
                y_train=y_train,
                y_val=splits["y_val"],
                y_test=splits["y_test"],
                plan=plan,
                task=task,
                n_train=n_train,
                seed=seed,
                cfg=cfg,
            )
        )

    # 2. Modelos de presupuesto igualado para cada T en plan.budgets
    for budget_t, m_name in pending_var:
        family = "quantum" if m_name == "QCNN" else "classical_matched"
        records.append(
            _run_single_model(
                model_name=m_name,
                budget_t=budget_t,
                model_family=family,
                fold_views=fold_views,
                y_train=y_train,
                y_val=splits["y_val"],
                y_test=splits["y_test"],
                plan=plan,
                task=task,
                n_train=n_train,
                seed=seed,
                cfg=cfg,
            )
        )

    return records

