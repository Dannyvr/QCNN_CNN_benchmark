"""
Módulo de Visualización Científica en Alta Resolución (600 DPI).

Diseñado para comparar simultáneamente:
- Las 2 tareas binarias (p. ej. MNIST 0 vs 1 y MNIST 3 vs 8) lado a lado.
- El presupuesto de parámetros variacionales T in {15, 30, 45, 90}.
- La cantidad de imágenes de entrenamiento N_train.
- Todas las corridas individuales (semillas) y líneas continuas de comparación.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Paleta científica accesible (Okabe-Ito extendida)
MODEL_COLORS: dict[str, str] = {
    "QCNN": "#6A1B9A",                 # Púrpura cuántico
    "CNN_Matched": "#D55E00",          # Naranja bermellón
    "MLP_Compact": "#E69F00",          # Ámbar
    "Logistic_Regression": "#0072B2",  # Azul
    "SVM_Linear": "#56B4E9",           # Celeste
    "SVM_RBF": "#009E73",              # Verde azulado
    "CNN_Full": "#CC79A7",             # Rosa (techo 28x28)
    "CNN_Full_Resolution": "#CC79A7",  # Alias retrocompatible
}

MODEL_MARKERS: dict[str, str] = {
    "QCNN": "o",
    "CNN_Matched": "s",
    "MLP_Compact": "^",
    "Logistic_Regression": "D",
    "SVM_Linear": "v",
    "SVM_RBF": "P",
    "CNN_Full": "*",
    "CNN_Full_Resolution": "*",
}

MODEL_LABELS: dict[str, str] = {
    "QCNN": "QCNN (Cuántica)",
    "CNN_Matched": "CNN Matched (1D)",
    "MLP_Compact": "MLP Compact",
    "Logistic_Regression": "Reg. Logística (T=9)",
    "SVM_Linear": "SVM Lineal (T=9)",
    "SVM_RBF": "SVM RBF",
    "CNN_Full": "CNN Full 28x28 (Techo)",
    "CNN_Full_Resolution": "CNN Full 28x28 (Techo)",
}

BUDGET_LINESTYLES = ["-", "--", "-.", ":"]


def _apply_style(dpi: int = 600) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "figure.titlesize": 14,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": 0.32,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza columnas estándar incluso si se lee un CSV legado."""
    out = df.copy()
    if "model_name" in out.columns:
        out["model_name"] = out["model_name"].replace({"CNN_Full_Resolution": "CNN_Full"})
    if "task" not in out.columns:
        out["task"] = "mnist_binary"
    if "task_label" not in out.columns:
        out["task_label"] = out["task"].map({
            "mnist_01": "MNIST 0 vs 1",
            "mnist_38": "MNIST 3 vs 8",
        }).fillna(out["task"])
    if "budget_T" not in out.columns:
        var_mask = out["model_name"].isin(["QCNN", "CNN_Matched", "MLP_Compact"])
        out["budget_T"] = np.where(var_mask, out["trainable_params"], np.nan)
    return out


def _closest_value(values: list[int], target: int) -> int:
    return min(values, key=lambda v: abs(int(v) - int(target)))


def _task_list(df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = df[["task", "task_label"]].drop_duplicates()
    return [(str(r.task), str(r.task_label)) for r in pairs.itertuples(index=False)]


def plot_learning_curves_by_task(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
    ref_budget: int = 45,
) -> Path:
    """
    Panel lado a lado por tarea: Exactitud vs N_train (cantidad de imágenes).
    Muestra todas las semillas individuales, bandas de error y líneas por modelo y T.
    """
    _apply_style(dpi)
    tasks = _task_list(df)
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(7.2 * n_tasks, 5.8), squeeze=False, sharey=True)
    axes_row = axes[0]

    budgets = sorted(int(b) for b in df["budget_T"].dropna().unique())
    chosen_b = _closest_value(budgets, ref_budget) if budgets else ref_budget

    for ax, (task_name, task_label) in zip(axes_row, tasks):
        sub_t = df[df["task"] == task_name]
        n_vals = sorted(int(n) for n in sub_t["n_train"].unique())
        ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="Azar (50%)")

        # 1. Modelos variacionales: graficar cada presupuesto T con estilo de línea distinto
        for m in ["QCNN", "CNN_Matched", "MLP_Compact"]:
            m_df = sub_t[sub_t["model_name"] == m]
            if m_df.empty:
                continue
            color = MODEL_COLORS[m]
            marker = MODEL_MARKERS[m]
            for b_idx, b_val in enumerate(sorted(int(b) for b in m_df["budget_T"].dropna().unique())):
                b_df = m_df[m_df["budget_T"] == b_val]
                grp = b_df.groupby("n_train")["test_accuracy"].agg(["mean", "std"]).reset_index()
                xs = grp["n_train"].to_numpy(dtype=float)
                means = grp["mean"].to_numpy(dtype=float) * 100.0
                stds = grp["std"].fillna(0.0).to_numpy(dtype=float) * 100.0

                is_primary = (b_val == chosen_b) or (len(budgets) == 1)
                ls = "-" if is_primary else BUDGET_LINESTYLES[(b_idx + 1) % len(BUDGET_LINESTYLES)]
                lw = 2.3 if is_primary else 1.5
                alpha_line = 0.95 if is_primary else 0.65

                # Puntos individuales de cada semilla
                rng = np.random.RandomState(42 + b_val)
                for row in b_df.itertuples(index=False):
                    jit = rng.uniform(0.96, 1.04)
                    ax.scatter(
                        float(row.n_train) * jit,
                        float(row.test_accuracy) * 100.0,
                        color=color,
                        alpha=0.28 if is_primary else 0.16,
                        s=22 if is_primary else 15,
                        edgecolors="none",
                        zorder=2,
                    )

                ax.plot(xs, means, color=color, linestyle=ls, linewidth=lw, alpha=alpha_line, zorder=4)
                if is_primary and len(xs) > 1:
                    ax.fill_between(
                        xs,
                        np.clip(means - stds, 0, 100),
                        np.clip(means + stds, 0, 100),
                        color=color,
                        alpha=0.12,
                        zorder=1,
                    )
                lbl = f"{MODEL_LABELS[m]} (T={b_val})"
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds if is_primary else None,
                    fmt=marker,
                    color=color,
                    markersize=7.5 if is_primary else 5.5,
                    capsize=3,
                    elinewidth=1.3,
                    markeredgecolor="black",
                    markeredgewidth=0.9,
                    alpha=alpha_line,
                    label=lbl,
                    zorder=5,
                )

        # 2. Modelos fijos (independientes de T)
        for m in ["Logistic_Regression", "SVM_Linear", "SVM_RBF", "CNN_Full"]:
            m_df = sub_t[sub_t["model_name"] == m]
            if m_df.empty:
                continue
            color = MODEL_COLORS[m]
            marker = MODEL_MARKERS[m]
            grp = m_df.groupby("n_train")["test_accuracy"].agg(["mean", "std"]).reset_index()
            xs = grp["n_train"].to_numpy(dtype=float)
            means = grp["mean"].to_numpy(dtype=float) * 100.0
            stds = grp["std"].fillna(0.0).to_numpy(dtype=float) * 100.0

            for row in m_df.itertuples(index=False):
                ax.scatter(
                    float(row.n_train),
                    float(row.test_accuracy) * 100.0,
                    color=color,
                    alpha=0.22,
                    s=18,
                    edgecolors="none",
                    zorder=2,
                )
            ax.plot(xs, means, color=color, linestyle="--", linewidth=1.7, alpha=0.85, zorder=3)
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                fmt=marker,
                color=color,
                markersize=6.5,
                capsize=2.5,
                elinewidth=1.1,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=MODEL_LABELS[m],
                zorder=4,
            )

        if len(n_vals) > 1:
            ax.set_xscale("log")
            ax.set_xticks(n_vals)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        else:
            ax.set_xticks(n_vals)
        ax.set_xlabel("Cantidad de Imágenes de Entrenamiento (N_train)")
        ax.set_title(f"{task_label}")
        ax.set_ylim(42, 103)
        ax.legend(loc="lower right", framealpha=0.92, ncol=1)

    axes_row[0].set_ylabel("Exactitud en Prueba / Test Accuracy (%)")
    fig.suptitle("Curvas de Aprendizaje por Tarea: Exactitud vs Cantidad de Imágenes (N_train)", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "01_learning_curves_by_task_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_parameter_efficiency_by_task(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
    ref_n_train: int = 50,
) -> Path:
    """
    Panel lado a lado por tarea: Exactitud vs Presupuesto de Parámetros T.
    Muestra todas las corridas individuales (semillas), dodge horizontal para evitar
    solapamiento, líneas conectando los presupuestos T y líneas horizontales para
    los modelos de capacidad fija (CNN Full, LogReg, SVM).
    """
    _apply_style(dpi)
    tasks = _task_list(df)
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(7.2 * n_tasks, 5.8), squeeze=False, sharey=True)
    axes_row = axes[0]

    dodge_map = {"QCNN": -1.4, "CNN_Matched": 0.0, "MLP_Compact": 1.4}

    for ax, (task_name, task_label) in zip(axes_row, tasks):
        sub_t = df[df["task"] == task_name]
        n_vals = sorted(int(n) for n in sub_t["n_train"].unique())
        chosen_n = _closest_value(n_vals, ref_n_train) if n_vals else ref_n_train

        ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="Azar (50%)")

        # Líneas horizontales de referencia para modelos fijos en N = chosen_n
        sub_ref_n = sub_t[sub_t["n_train"] == chosen_n]
        for fixed_m, ls in [("CNN_Full", "--"), ("SVM_RBF", "-."), ("Logistic_Regression", ":")]:
            f_df = sub_ref_n[sub_ref_n["model_name"] == fixed_m]
            if not f_df.empty:
                mean_f = f_df["test_accuracy"].mean() * 100.0
                ax.axhline(
                    mean_f,
                    color=MODEL_COLORS[fixed_m],
                    linestyle=ls,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{MODEL_LABELS[fixed_m]} @N={chosen_n}: {mean_f:.1f}%",
                )

        # Modelos variacionales a través de T (y por cada N_train si hay varios)
        for m in ["QCNN", "CNN_Matched", "MLP_Compact"]:
            m_df = sub_t[sub_t["model_name"] == m]
            if m_df.empty:
                continue
            color = MODEL_COLORS[m]
            marker = MODEL_MARKERS[m]
            dodge = dodge_map.get(m, 0.0)

            for n_idx, n_val in enumerate(n_vals):
                mn_df = m_df[m_df["n_train"] == n_val]
                if mn_df.empty:
                    continue
                grp = mn_df.groupby("budget_T")["test_accuracy"].agg(["mean", "std"]).reset_index()
                ts = grp["budget_T"].to_numpy(dtype=float) + dodge
                means = grp["mean"].to_numpy(dtype=float) * 100.0
                stds = grp["std"].fillna(0.0).to_numpy(dtype=float) * 100.0

                is_primary = (n_val == chosen_n) or (len(n_vals) == 1)
                ls = "-" if is_primary else BUDGET_LINESTYLES[(n_idx + 1) % len(BUDGET_LINESTYLES)]
                lw = 2.3 if is_primary else 1.4
                alpha_line = 0.95 if is_primary else 0.55

                # Puntos individuales de cada semilla
                rng = np.random.RandomState(100 + n_val)
                for row in mn_df.itertuples(index=False):
                    jit = rng.uniform(-0.45, 0.45)
                    ax.scatter(
                        float(row.budget_T) + dodge + jit,
                        float(row.test_accuracy) * 100.0,
                        color=color,
                        alpha=0.32 if is_primary else 0.16,
                        s=24 if is_primary else 14,
                        edgecolors="none",
                        zorder=3,
                    )

                if len(ts) > 1:
                    ax.plot(ts, means, color=color, linestyle=ls, linewidth=lw, alpha=alpha_line, zorder=4)

                lbl = f"{MODEL_LABELS[m]} (N={n_val})"
                ax.errorbar(
                    ts,
                    means,
                    yerr=stds if is_primary else None,
                    fmt=marker,
                    color=color,
                    markersize=8.0 if is_primary else 5.5,
                    capsize=3.5,
                    elinewidth=1.5,
                    markeredgecolor="black",
                    markeredgewidth=0.9,
                    alpha=alpha_line,
                    label=lbl,
                    zorder=5,
                )

        budgets_t = sorted(int(b) for b in sub_t["budget_T"].dropna().unique())
        if budgets_t:
            ax.set_xticks(budgets_t)
            for b in budgets_t:
                ax.axvline(b, color="gray", linestyle=":", alpha=0.3, zorder=1)
        ax.set_xlabel("Presupuesto de Parámetros Variacionales (T)")
        ax.set_title(f"{task_label}")
        ax.set_ylim(42, 103)
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8.3)

    axes_row[0].set_ylabel("Exactitud en Prueba / Test Accuracy (%)")
    fig.suptitle("Eficiencia Paramétrica por Tarea: Exactitud vs Presupuesto T (Todas las Corridas)", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "02_parameter_efficiency_by_task_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_per_param_by_task(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
    ref_n_train: int = 50,
) -> Path:
    """Panel lado a lado: Exactitud por parámetro (% Accuracy / T) frente a T."""
    _apply_style(dpi)
    tasks = _task_list(df)
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(6.8 * n_tasks, 5.4), squeeze=False, sharey=True)
    axes_row = axes[0]

    for ax, (task_name, task_label) in zip(axes_row, tasks):
        sub_t = df[(df["task"] == task_name) & (df["model_name"].isin(["QCNN", "CNN_Matched", "MLP_Compact"]))].copy()
        if sub_t.empty:
            continue
        n_vals = sorted(int(n) for n in sub_t["n_train"].unique())
        chosen_n = _closest_value(n_vals, ref_n_train)
        sub_t = sub_t[sub_t["n_train"] == chosen_n].copy()
        sub_t["acc_per_param"] = (sub_t["test_accuracy"] * 100.0) / sub_t["trainable_params"]

        for m in ["QCNN", "CNN_Matched", "MLP_Compact"]:
            m_df = sub_t[sub_t["model_name"] == m]
            if m_df.empty:
                continue
            color = MODEL_COLORS[m]
            marker = MODEL_MARKERS[m]
            grp = m_df.groupby("budget_T")["acc_per_param"].agg(["mean", "std"]).reset_index()
            ts = grp["budget_T"].to_numpy(dtype=float)
            means = grp["mean"].to_numpy(dtype=float)
            stds = grp["std"].fillna(0.0).to_numpy(dtype=float)

            for row in m_df.itertuples(index=False):
                ax.scatter(float(row.budget_T), float(row.acc_per_param), color=color, alpha=0.3, s=22)

            ax.plot(ts, means, color=color, linewidth=2.2, label=f"{MODEL_LABELS[m]} (N={chosen_n})")
            ax.errorbar(ts, means, yerr=stds, fmt=marker, color=color, markersize=7.5, capsize=3.5,
                        markeredgecolor="black", markeredgewidth=0.9)

        budgets_t = sorted(int(b) for b in sub_t["budget_T"].dropna().unique())
        if budgets_t:
            ax.set_xticks(budgets_t)
        ax.set_xlabel("Presupuesto de Parámetros (T)")
        ax.set_title(f"{task_label} (N_train = {chosen_n})")
        ax.legend(loc="upper right", framealpha=0.92)

    axes_row[0].set_ylabel("Exactitud por Parámetro (% Accuracy / T)")
    fig.suptitle("Eficiencia de Representación: Exactitud Normalizada por Parámetro", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "03_accuracy_per_parameter_by_task_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_joint_grid_heatmap(df: pd.DataFrame, output_dir: Path, dpi: int = 600) -> Path:
    """
    Mapa de calor 2D (N_train x T) lado a lado para cada tarea mostrando:
    Delta Exactitud (%) = Acc(QCNN) - Acc(CNN_Matched).
    Permite visualizar de un vistazo las 2 tareas, T y N_train al mismo tiempo.
    """
    _apply_style(dpi)
    tasks = _task_list(df)
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(6.5 * n_tasks, 5.2), squeeze=False)
    axes_row = axes[0]

    for ax, (task_name, task_label) in zip(axes_row, tasks):
        sub_t = df[(df["task"] == task_name) & (df["model_name"].isin(["QCNN", "CNN_Matched"]))].copy()
        if sub_t.empty:
            continue
        piv = (
            sub_t.groupby(["n_train", "budget_T", "model_name"])["test_accuracy"]
            .mean()
            .unstack("model_name")
        )
        if "QCNN" not in piv.columns or "CNN_Matched" not in piv.columns:
            continue
        piv["delta"] = (piv["QCNN"] - piv["CNN_Matched"]) * 100.0
        grid = piv["delta"].unstack("budget_T")

        n_vals = [int(n) for n in grid.index]
        t_vals = [int(t) for t in grid.columns]
        mat = grid.to_numpy(dtype=float)

        vmax = max(5.0, float(np.nanmax(np.abs(mat))) if mat.size else 5.0)
        im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")

        ax.set_xticks(np.arange(len(t_vals)))
        ax.set_xticklabels([f"T={t}" for t in t_vals])
        ax.set_yticks(np.arange(len(n_vals)))
        ax.set_yticklabels([f"N={n}" for n in n_vals])
        ax.set_xlabel("Presupuesto de Parámetros (T)")
        ax.set_ylabel("Imágenes de Entrenamiento (N_train)")
        ax.set_title(f"{task_label}\nΔ Accuracy = QCNN − CNN Matched (%)")
        ax.grid(False)

        for i in range(len(n_vals)):
            for j in range(len(t_vals)):
                val = mat[i, j]
                if not np.isnan(val):
                    q_acc = piv.loc[(n_vals[i], t_vals[j]), "QCNN"] * 100.0
                    c_acc = piv.loc[(n_vals[i], t_vals[j]), "CNN_Matched"] * 100.0
                    txt_color = "white" if abs(val) > 0.55 * vmax else "black"
                    sign = "+" if val >= 0 else ""
                    ax.text(
                        j, i, f"{sign}{val:.1f}%\n({q_acc:.0f} vs {c_acc:.0f})",
                        ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold",
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Exactitud (%) [Azul: gana QCNN]")

    fig.suptitle("Comparación Conjunta (Tarea × N_train × Presupuesto T): Ventaja QCNN vs CNN Igualada", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "04_joint_grid_qcnn_vs_cnn_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_paired_seed_trajectories(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
    ref_n_train: int = 50,
    ref_budget: int = 45,
) -> Path:
    """
    Gráfico de coordenadas paralelas por tarea: conecta cada semilla individual a
    través de todos los modelos evaluados sobre la misma partición de entrenamiento.
    """
    _apply_style(dpi)
    tasks = _task_list(df)
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(7.2 * n_tasks, 5.6), squeeze=False, sharey=True)
    axes_row = axes[0]

    ordered_models = ["QCNN", "CNN_Matched", "MLP_Compact", "Logistic_Regression", "SVM_Linear", "SVM_RBF", "CNN_Full"]

    for ax, (task_name, task_label) in zip(axes_row, tasks):
        sub_t = df[df["task"] == task_name]
        n_vals = sorted(int(n) for n in sub_t["n_train"].unique())
        budgets = sorted(int(b) for b in sub_t["budget_T"].dropna().unique())
        chosen_n = _closest_value(n_vals, ref_n_train)
        chosen_b = _closest_value(budgets, ref_budget) if budgets else ref_budget

        sub_sel = sub_t[
            (sub_t["n_train"] == chosen_n)
            & ((sub_t["budget_T"] == chosen_b) | (sub_t["budget_T"].isna()))
        ]
        models = [m for m in ordered_models if m in sub_sel["model_name"].unique()]
        x_idx = np.arange(len(models))
        m_to_x = {m: i for i, m in enumerate(models)}

        seeds = sorted(int(s) for s in sub_sel["seed"].unique())
        cmap = matplotlib.colormaps["tab20"].resampled(max(1, len(seeds)))

        for s_i, seed in enumerate(seeds):
            s_df = sub_sel[sub_sel["seed"] == seed]
            xs, ys = [], []
            for m in models:
                vals = s_df[s_df["model_name"] == m]["test_accuracy"].values
                if len(vals) > 0:
                    xs.append(m_to_x[m])
                    ys.append(float(vals[0]) * 100.0)
            ax.plot(xs, ys, color=cmap(s_i), alpha=0.55, linewidth=1.3, marker="o", markersize=4.5,
                    label=f"Seed {seed}" if len(seeds) <= 6 else None)

        means = [sub_sel[sub_sel["model_name"] == m]["test_accuracy"].mean() * 100.0 for m in models]
        stds = [sub_sel[sub_sel["model_name"] == m]["test_accuracy"].std() * 100.0 for m in models]
        ax.plot(x_idx, means, color="black", linewidth=2.8, marker="s", markersize=8,
                label="Media Global (±Std)", zorder=10)
        ax.errorbar(x_idx, means, yerr=stds, fmt="none", ecolor="black", elinewidth=2.0, capsize=4, zorder=11)

        if all(m in m_to_x for m in ["QCNN", "CNN_Matched", "MLP_Compact"]):
            ax.axvspan(-0.3, 2.3, color="gray", alpha=0.11, label=f"Capacidad Igualada (T={chosen_b})")

        ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.2)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=25, ha="right")
        ax.set_ylim(42, 103)
        ax.set_title(f"{task_label} (N_train={chosen_n}, T={chosen_b}, {len(seeds)} semillas)")
        ax.legend(loc="lower right", framealpha=0.92)

    axes_row[0].set_ylabel("Exactitud en Prueba / Test Accuracy (%)")
    fig.suptitle("Trayectorias Pareadas por Semilla: Comparación Modelo a Modelo en cada Corrida", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "05_paired_seed_trajectories_by_task_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_master_dashboard(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
    ref_n_train: int = 50,
    ref_budget: int = 45,
) -> Path:
    """
    Dashboard ejecutivo 2x2 a 600 DPI que sintetiza las dos tareas, N_train, T y tiempos:
    - Panel A: Comparación simultánea de tareas (0 vs 1 vs 3 vs 8) en función de N_train.
    - Panel B: Comparación simultánea de tareas en función del presupuesto T.
    - Panel C: Diferencia Δ Accuracy (QCNN - CNN_Matched) para cada tarea, N_train y T.
    - Panel D: Costo computacional (tiempo de entrenamiento medio en escala log).
    """
    _apply_style(dpi)
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.2))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    tasks = _task_list(df)
    task_styles = {tasks[0][0]: "-" if tasks else "-"}
    if len(tasks) > 1:
        task_styles[tasks[1][0]] = "--"

    budgets = sorted(int(b) for b in df["budget_T"].dropna().unique())
    n_vals = sorted(int(n) for n in df["n_train"].unique())
    chosen_b = _closest_value(budgets, ref_budget) if budgets else ref_budget
    chosen_n = _closest_value(n_vals, ref_n_train) if n_vals else ref_n_train

    # --- PANEL A: Ambas tareas vs N_train (en T = chosen_b) ---
    for task_name, task_label in tasks:
        ls = task_styles.get(task_name, "-")
        sub = df[(df["task"] == task_name) & ((df["budget_T"] == chosen_b) | (df["budget_T"].isna()))]
        for m in ["QCNN", "CNN_Matched", "MLP_Compact", "CNN_Full"]:
            m_df = sub[sub["model_name"] == m]
            if m_df.empty:
                continue
            grp = m_df.groupby("n_train")["test_accuracy"].agg(["mean", "std"]).reset_index()
            ax_a.errorbar(
                grp["n_train"], grp["mean"] * 100.0, yerr=grp["std"].fillna(0.0) * 100.0,
                fmt=MODEL_MARKERS[m], linestyle=ls, color=MODEL_COLORS[m],
                linewidth=2.0, markersize=6.5, capsize=3,
                label=f"{MODEL_LABELS[m]} [{task_label}]",
            )
    if len(n_vals) > 1:
        ax_a.set_xscale("log")
        ax_a.set_xticks(n_vals)
        ax_a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_a.set_xlabel("Imágenes de Entrenamiento (N_train)")
    ax_a.set_ylabel("Test Accuracy (%)")
    ax_a.set_ylim(42, 103)
    ax_a.set_title(f"A) Ambas Tareas vs Cantidad de Imágenes (T={chosen_b})")
    ax_a.legend(loc="lower right", fontsize=7.8, ncol=1)

    # --- PANEL B: Ambas tareas vs Presupuesto T (en N_train = chosen_n) ---
    for task_name, task_label in tasks:
        ls = task_styles.get(task_name, "-")
        sub = df[(df["task"] == task_name) & (df["n_train"] == chosen_n)]
        for m in ["QCNN", "CNN_Matched", "MLP_Compact"]:
            m_df = sub[sub["model_name"] == m]
            if m_df.empty:
                continue
            grp = m_df.groupby("budget_T")["test_accuracy"].agg(["mean", "std"]).reset_index()
            ax_b.errorbar(
                grp["budget_T"], grp["mean"] * 100.0, yerr=grp["std"].fillna(0.0) * 100.0,
                fmt=MODEL_MARKERS[m], linestyle=ls, color=MODEL_COLORS[m],
                linewidth=2.0, markersize=6.5, capsize=3,
                label=f"{MODEL_LABELS[m]} [{task_label}]",
            )
    if budgets:
        ax_b.set_xticks(budgets)
    ax_b.set_xlabel("Presupuesto de Parámetros Variacionales (T)")
    ax_b.set_ylabel("Test Accuracy (%)")
    ax_b.set_ylim(42, 103)
    ax_b.set_title(f"B) Ambas Tareas vs Presupuesto de Parámetros T (N={chosen_n})")
    ax_b.legend(loc="lower right", fontsize=8.0)

    # --- PANEL C: Ventaja QCNN - CNN_Matched por (Tarea, N_train, T) ---
    var_sub = df[df["model_name"].isin(["QCNN", "CNN_Matched"])]
    piv = (
        var_sub.groupby(["task_label", "n_train", "budget_T", "model_name"])["test_accuracy"]
        .mean()
        .unstack("model_name")
        .reset_index()
    )
    if not piv.empty and "QCNN" in piv.columns and "CNN_Matched" in piv.columns:
        piv["delta"] = (piv["QCNN"] - piv["CNN_Matched"]) * 100.0
        piv["cond"] = [f"N={int(r.n_train)}, T={int(r.budget_T)}" for r in piv.itertuples(index=False)]
        conds = list(dict.fromkeys(piv["cond"].tolist()))
        x_c = np.arange(len(conds))
        t_labels = [tl for _, tl in tasks]
        width = 0.8 / max(1, len(t_labels))
        task_palette = ["#5E35B1", "#00897B", "#E53935"]

        for i, tl in enumerate(t_labels):
            t_piv = piv[piv["task_label"] == tl].set_index("cond").reindex(conds)
            vals = t_piv["delta"].fillna(0.0).to_numpy(dtype=float)
            offset = (i - (len(t_labels) - 1) / 2.0) * width
            bars = ax_c.bar(x_c + offset, vals, width=width * 0.9, label=tl,
                            color=task_palette[i % len(task_palette)], alpha=0.85, edgecolor="black")
            for b, v in zip(bars, vals):
                ax_c.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + (0.6 if v >= 0 else -1.8),
                    f"{v:+.1f}%",
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                )
        ax_c.axhline(0.0, color="black", linewidth=1.2)
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(conds, rotation=25, ha="right")
        ax_c.set_ylabel("Δ Accuracy = QCNN − CNN Matched (%)")
        ax_c.set_title("C) Ventaja de la QCNN frente a CNN de Igual Capacidad")
        ax_c.legend(loc="best")

    # --- PANEL D: Tiempo de entrenamiento (escala log) ---
    ordered = ["QCNN", "CNN_Matched", "MLP_Compact", "Logistic_Regression", "SVM_Linear", "SVM_RBF", "CNN_Full"]
    present = [m for m in ordered if m in df["model_name"].unique()]
    x_d = np.arange(len(present))
    t_means = [df[df["model_name"] == m]["train_time_sec"].mean() for m in present]
    bars_d = ax_d.bar(
        x_d, t_means,
        color=[MODEL_COLORS[m] for m in present],
        alpha=0.85, edgecolor="black", width=0.6,
    )
    ax_d.set_yscale("log")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([MODEL_LABELS[m] for m in present], rotation=25, ha="right")
    ax_d.set_ylabel("Tiempo Medio de Entrenamiento (s, Log)")
    ax_d.set_title("D) Costo Computacional de Entrenamiento")
    for b, t in zip(bars_d, t_means):
        ax_d.text(
            b.get_x() + b.get_width() / 2.0,
            max(t * 1.18, 1e-4),
            f"{t:.3f}s" if t < 0.1 else f"{t:.2f}s",
            ha="center", va="bottom", fontsize=8.2, fontweight="bold",
        )

    fig.suptitle(
        "DASHBOARD COMPARATIVO INTEGRAL: QCNN vs BASELINES CLÁSICOS (Tareas × N_train × Presupuesto T)",
        fontsize=14.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / "06_master_dashboard_600dpi.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_benchmark_plots(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 600,
    ref_budget: int = 45,
    ref_n_train: int = 50,
) -> list[Path]:
    """Genera todos los gráficos PNG a 600 DPI en la carpeta de resultados."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo de resultados: {csv_path}")

    out_dir = Path(output_dir) if output_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _normalize_df(pd.read_csv(csv_path))
    if df.empty:
        return []

    generated = [
        plot_learning_curves_by_task(df, out_dir, dpi=dpi, ref_budget=ref_budget),
        plot_parameter_efficiency_by_task(df, out_dir, dpi=dpi, ref_n_train=ref_n_train),
        plot_accuracy_per_param_by_task(df, out_dir, dpi=dpi, ref_n_train=ref_n_train),
        plot_joint_grid_heatmap(df, out_dir, dpi=dpi),
        plot_paired_seed_trajectories(df, out_dir, dpi=dpi, ref_n_train=ref_n_train, ref_budget=ref_budget),
        plot_master_dashboard(df, out_dir, dpi=dpi, ref_n_train=ref_n_train, ref_budget=ref_budget),
    ]
    for p in generated:
        print(f"[Visualizer] Figura 600 DPI guardada: {p}")
    return generated
