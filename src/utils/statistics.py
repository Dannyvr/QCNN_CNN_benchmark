"""
Análisis Estadístico Riguroso para el Benchmark QCNN vs Baselines Clásicos.

Incluye:
1. Intervalos de confianza (t-Student al nivel configurable, p. ej. 95%).
2. Tablas agregadas por (tarea, N_train, presupuesto T, modelo).
3. Pruebas pareadas por semilla (Wilcoxon signed-rank) entre el modelo de
   referencia (QCNN) y cada baseline, con tamaño de efecto (correlación
   rango-biserial y d de Cohen pareado) y corrección por comparaciones
   múltiples (Holm-Bonferroni o Bonferroni).
4. Identificación de puntos de cruce (ventaja QCNN - CNN_Matched en la malla N_train x T).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_ci(values: np.ndarray | list[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Retorna (media, std muestral, ci_lower, ci_upper)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    if sem == 0.0:
        return mean, 0.0, mean, mean
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return mean, std, mean - t_crit * sem, mean + t_crit * sem


def build_summary_table(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Agrupa las corridas por (task, n_train, budget_T, model_name) y calcula IC."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["budget_T_key"] = work["budget_T"].fillna(0).astype(int)

    rows = []
    group_cols = ["task", "task_label", "n_train", "budget_T_key", "model_name", "model_family"]
    for keys, grp in work.groupby(group_cols, sort=True):
        task, task_label, n_train, budget_key, model_name, model_family = keys
        acc_m, acc_s, ci_l, ci_u = compute_ci(grp["test_accuracy"].values, confidence=confidence)
        bacc_m, _, _, _ = compute_ci(grp["test_balanced_acc"].values, confidence=confidence)
        f1_m, _, _, _ = compute_ci(grp["test_f1"].values, confidence=confidence)
        params = int(round(grp["trainable_params"].mean()))
        acc_per_param = (acc_m * 100.0) / max(1, params)

        rows.append({
            "task": task,
            "task_label": task_label,
            "n_train": int(n_train),
            "budget_T": None if budget_key == 0 else int(budget_key),
            "model_name": model_name,
            "model_family": model_family,
            "trainable_params": params,
            "n_seeds": int(len(grp)),
            "accuracy_mean": round(acc_m, 4),
            "accuracy_std": round(acc_s, 4),
            "ci_lower": round(ci_l, 4),
            "ci_upper": round(ci_u, 4),
            "balanced_acc_mean": round(bacc_m, 4),
            "f1_mean": round(f1_m, 4),
            "acc_per_param_pct": round(acc_per_param, 4),
            "train_time_mean_sec": round(float(grp["train_time_sec"].mean()), 4),
            "infer_time_mean_sec": round(float(grp["infer_time_sec"].mean()), 5),
        })

    return pd.DataFrame(rows)


def _adjust_pvalues(pvals: list[float], method: str = "holm") -> list[float]:
    """Corrección por comparaciones múltiples (Holm-Bonferroni o Bonferroni)."""
    m = len(pvals)
    if m == 0:
        return []
    method = (method or "none").lower()
    if method == "none":
        return list(pvals)
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvals]
    if method == "holm":
        order = np.argsort(pvals)
        adjusted = np.empty(m, dtype=np.float64)
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = pvals[idx] * (m - rank)
            running_max = max(running_max, adj)
            adjusted[idx] = min(1.0, running_max)
        return adjusted.tolist()
    raise ValueError(f"Método de corrección desconocido: {method}")


def run_paired_tests(df: pd.DataFrame, stats_cfg: dict) -> pd.DataFrame:
    """
    Ejecuta pruebas pareadas por semilla entre `reference_model` (p. ej. QCNN)
    y los modelos listados en `comparisons`.

    Para modelos variacionales (CNN_Matched, MLP_Compact) se empareja a igual
    (task, n_train, budget_T, seed). Para modelos fijos (Logistic_Regression,
    SVM_*, CNN_Full) se empareja por (task, n_train, seed) contra cada T de QCNN.
    """
    if df.empty:
        return pd.DataFrame()

    ref_name = stats_cfg.get("reference_model", "QCNN")
    comparisons = stats_cfg.get("comparisons", ["CNN_Matched", "MLP_Compact", "Logistic_Regression", "SVM_RBF", "CNN_Full"])
    correction = stats_cfg.get("correction", "holm")
    alpha = float(stats_cfg.get("alpha", 0.05))

    ref_df = df[df["model_name"] == ref_name].copy()
    if ref_df.empty:
        return pd.DataFrame()

    records = []
    for (task, n_train, budget_t), ref_grp in ref_df.groupby(["task", "n_train", "budget_T"], dropna=False):
        ref_by_seed = ref_grp.set_index("seed")["test_accuracy"]

        for comp_model in comparisons:
            comp_sub = df[(df["task"] == task) & (df["n_train"] == n_train) & (df["model_name"] == comp_model)]
            if comp_sub.empty:
                continue
            # Si el modelo comparado también depende de T, exigir el mismo budget_T
            if comp_sub["budget_T"].notna().any():
                comp_sub = comp_sub[comp_sub["budget_T"] == budget_t]
            if comp_sub.empty:
                continue

            comp_by_seed = comp_sub.set_index("seed")["test_accuracy"]
            common_seeds = sorted(set(ref_by_seed.index).intersection(comp_by_seed.index))
            if len(common_seeds) < 2:
                continue

            a = ref_by_seed.loc[common_seeds].to_numpy(dtype=np.float64)
            b = comp_by_seed.loc[common_seeds].to_numpy(dtype=np.float64)
            diff = a - b
            mean_diff = float(np.mean(diff))
            median_diff = float(np.median(diff))
            std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            cohen_dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0

            nonzero = diff[np.abs(diff) > 1e-12]
            if len(nonzero) >= 2:
                try:
                    stat_w, p_val = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", mode="auto")
                    stat_w = float(stat_w)
                    p_val = float(p_val)
                except ValueError:
                    stat_w, p_val = 0.0, 1.0
            else:
                stat_w, p_val = 0.0, 1.0

            records.append({
                "task": task,
                "n_train": int(n_train),
                "budget_T": None if pd.isna(budget_t) else int(budget_t),
                "reference_model": ref_name,
                "compared_model": comp_model,
                "n_paired_seeds": len(common_seeds),
                "ref_acc_mean": round(float(np.mean(a)), 4),
                "comp_acc_mean": round(float(np.mean(b)), 4),
                "diff_mean_pct": round(mean_diff * 100.0, 3),
                "diff_median_pct": round(median_diff * 100.0, 3),
                "cohen_dz": round(cohen_dz, 3),
                "wilcoxon_stat": round(stat_w, 3),
                "p_value_raw": round(p_val, 5),
            })

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["p_value_adj"] = np.round(_adjust_pvalues(out["p_value_raw"].tolist(), method=correction), 5)
    out["significant"] = out["p_value_adj"] < alpha
    out["winner"] = np.where(
        out["diff_mean_pct"] > 0,
        out["reference_model"],
        np.where(out["diff_mean_pct"] < 0, out["compared_model"], "Tie"),
    )
    return out


def compute_crossover_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de diferencia de exactitud (QCNN - CNN_Matched y QCNN - MLP_Compact)
    para cada celda (task, n_train, budget_T), permitiendo identificar puntos de cruce.
    """
    var_df = df[df["model_name"].isin(["QCNN", "CNN_Matched", "MLP_Compact"])].copy()
    if var_df.empty:
        return pd.DataFrame()

    piv = (
        var_df.groupby(["task", "task_label", "n_train", "budget_T", "model_name"])["test_accuracy"]
        .mean()
        .unstack("model_name")
        .reset_index()
    )
    if "QCNN" in piv.columns and "CNN_Matched" in piv.columns:
        piv["delta_qcnn_vs_cnn_pct"] = ((piv["QCNN"] - piv["CNN_Matched"]) * 100.0).round(2)
    if "QCNN" in piv.columns and "MLP_Compact" in piv.columns:
        piv["delta_qcnn_vs_mlp_pct"] = ((piv["QCNN"] - piv["MLP_Compact"]) * 100.0).round(2)
    return piv

