"""
Punto de Entrada Único del Benchmark QCNN vs Baselines Clásicos.

Todo el comportamiento experimental se define en `config.yaml` (`run.mode`):
    - pilot        : Validación rápida extremo a extremo (ambas tareas, varios N y T)
    - data_size    : Barrido de cantidad de imágenes N_train (curvas de aprendizaje)
    - param_budget : Barrido de presupuesto de parámetros T (eficiencia paramétrica)
    - grid         : Barrido cruzado completo (Tareas x N_train x T x Semillas)

Uso:
    .venv\\Scripts\\python.exe main.py
    .venv\\Scripts\\python.exe main.py --config config.yaml
    .venv\\Scripts\\python.exe main.py --mode param_budget
    .venv\\Scripts\\python.exe main.py --report-only results/pilot
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import pandas as pd
import yaml
from tabulate import tabulate

from src.config import ExperimentPlan, TaskSpec, build_plan, load_config
from src.data.loader import initialize_static_splits
from src.training.runner import execute_fold_job
from src.utils.logger import ResultStore
from src.utils.reproducibility import environment_info
from src.utils.statistics import build_summary_table, compute_crossover_grid, run_paired_tests
from src.utils.visualizer import generate_benchmark_plots


def _resolve_run_dir(cfg: dict[str, Any], plan: ExperimentPlan) -> Path:
    base_dir = Path(cfg["run"].get("results_dir", "./results"))
    run_name = cfg["run"].get("run_name") or plan.mode
    run_dir = base_dir / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _worker_entry(args: tuple) -> tuple[str, int, int, list[dict[str, Any]]]:
    task, n_train, seed, splits, plan, cfg, completed = args
    recs = execute_fold_job(
        task=task,
        n_train=n_train,
        seed=seed,
        splits=splits,
        plan=plan,
        cfg=cfg,
        completed=completed,
    )
    return task.label, n_train, seed, recs


def _print_reports(
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    crossover_df: pd.DataFrame,
) -> None:
    if not summary_df.empty:
        disp = pd.DataFrame({
            "Tarea": summary_df["task_label"],
            "N_train": summary_df["n_train"],
            "T": summary_df["budget_T"].apply(lambda v: "-" if pd.isna(v) else f"{int(v)}"),
            "Modelo": summary_df["model_name"],
            "Params": summary_df["trainable_params"],
            "Seeds": summary_df["n_seeds"],
            "Test Acc (Media ± Std)": [
                f"{m * 100:.2f}% ± {s * 100:.2f}%"
                for m, s in zip(summary_df["accuracy_mean"], summary_df["accuracy_std"])
            ],
            "IC 95%": [
                f"[{l * 100:.1f}%, {u * 100:.1f}%]"
                for l, u in zip(summary_df["ci_lower"], summary_df["ci_upper"])
            ],
            "Acc/Param": [f"{v:.2f}%/p" for v in summary_df["acc_per_param_pct"]],
            "Train (s)": [f"{t:.2f}s" for t in summary_df["train_time_mean_sec"]],
        })
        print("\n" + "=" * 108)
        print(" TABLA RESUMEN DE DESEMPEÑO (MEDIA, DESVIACIÓN ESTÁNDAR E IC 95%)")
        print("=" * 108)
        print(tabulate(disp, headers="keys", tablefmt="fancy_grid", showindex=False))

    if not crossover_df.empty:
        c_disp = pd.DataFrame({
            "Tarea": crossover_df["task_label"],
            "N_train": crossover_df["n_train"],
            "Presupuesto T": crossover_df["budget_T"].astype(int),
            "QCNN (%)": (crossover_df.get("QCNN", pd.Series(dtype=float)) * 100).round(2),
            "CNN_Matched (%)": (crossover_df.get("CNN_Matched", pd.Series(dtype=float)) * 100).round(2),
            "MLP_Compact (%)": (crossover_df.get("MLP_Compact", pd.Series(dtype=float)) * 100).round(2),
            "Δ (QCNN - CNN)": [
                f"{v:+.2f}%" if pd.notna(v) else "-"
                for v in crossover_df.get("delta_qcnn_vs_cnn_pct", pd.Series(dtype=float))
            ],
            "Δ (QCNN - MLP)": [
                f"{v:+.2f}%" if pd.notna(v) else "-"
                for v in crossover_df.get("delta_qcnn_vs_mlp_pct", pd.Series(dtype=float))
            ],
        })
        print("\n" + "=" * 108)
        print(" MATRIZ DE PUNTOS DE CRUCE: VENTAJA DE CAPACIDAD IGUALADA (QCNN vs CNN_Matched / MLP_Compact)")
        print("=" * 108)
        print(tabulate(c_disp, headers="keys", tablefmt="fancy_grid", showindex=False))

    if not tests_df.empty:
        t_disp = pd.DataFrame({
            "Tarea": tests_df["task"],
            "N": tests_df["n_train"],
            "T": tests_df["budget_T"].apply(lambda v: "-" if pd.isna(v) else f"{int(v)}"),
            "Comparación": tests_df["reference_model"] + " vs " + tests_df["compared_model"],
            "Δ Media": [f"{v:+.2f}%" for v in tests_df["diff_mean_pct"]],
            "Cohen dz": tests_df["cohen_dz"],
            "p-raw": tests_df["p_value_raw"],
            "p-Holm": tests_df["p_value_adj"],
            "Ganador": tests_df["winner"],
        })
        print("\n" + "=" * 108)
        print(" PRUEBAS ESTADÍSTICAS PAREADAS POR SEMILLA (WILCOXON + CORRECCIÓN HOLM-BONFERRONI)")
        print("=" * 108)
        print(tabulate(t_disp, headers="keys", tablefmt="fancy_grid", showindex=False))


def generate_reports_and_plots(raw_csv: Path, out_dir: Path, cfg: dict[str, Any]) -> None:
    df = pd.read_csv(raw_csv)
    if df.empty:
        print("[Main] No hay registros para generar reportes.")
        return

    stats_cfg = cfg.get("statistics", {})
    plots_cfg = cfg.get("plots", {})
    conf = float(stats_cfg.get("confidence_level", 0.95))

    summary_df = build_summary_table(df, confidence=conf)
    tests_df = run_paired_tests(df, stats_cfg)
    crossover_df = compute_crossover_grid(df)

    summary_path = out_dir / "summary_ci95.csv"
    tests_path = out_dir / "statistical_tests_wilcoxon.csv"
    crossover_path = out_dir / "crossover_grid.csv"

    summary_df.to_csv(summary_path, index=False)
    if not tests_df.empty:
        tests_df.to_csv(tests_path, index=False)
    if not crossover_df.empty:
        crossover_df.to_csv(crossover_path, index=False)

    _print_reports(summary_df, tests_df, crossover_df)

    dpi = int(plots_cfg.get("dpi", 600))
    ref_b = int(plots_cfg.get("reference_budget", 45))
    ref_n = int(plots_cfg.get("reference_train_size", 50))

    print(f"\n[Main] Generando gráficos comparativos en PNG a {dpi} DPI...")
    generate_benchmark_plots(
        csv_path=raw_csv,
        output_dir=out_dir,
        dpi=dpi,
        ref_budget=ref_b,
        ref_n_train=ref_n,
    )
    print(f"\n[Main] Todos los artefactos y gráficos se guardaron en: {out_dir.resolve()}")


def run_benchmark(config_path: str = "config.yaml", mode_override: str | None = None) -> Path:
    cfg = load_config(config_path)
    if mode_override:
        cfg["run"]["mode"] = mode_override

    plan = build_plan(cfg)
    run_dir = _resolve_run_dir(cfg, plan)

    print("\n" + "=" * 108)
    print(" BENCHMARK RIGUROSO QCNN vs BASELINES CLÁSICOS — PIPELINE UNIFICADO")
    print("=" * 108)
    print(plan.describe())
    print(f"Directorio de salida : {run_dir.resolve()}")
    print("=" * 108)

    # Guardar snapshot de configuración y entorno para reproducibilidad
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(environment_info(), f, indent=2, ensure_ascii=False)

    # 1. Inicializar particiones estáticas inmutables para todas las tareas configuradas
    data_cfg = cfg["data"]
    splits_by_task: dict[str, dict[str, Any]] = {}
    for task in plan.tasks:
        splits_by_task[task.name] = initialize_static_splits(
            data_dir=data_cfg["data_dir"],
            splits_dir=data_cfg["splits_dir"],
            classes=task.classes,
            test_size=int(data_cfg["test_size"]),
            val_size=int(data_cfg["val_size"]),
            seed=int(data_cfg.get("split_seed", 42)),
        )

    # 2. Preparar almacén de resultados con soporte de reanudación
    raw_csv = run_dir / "raw_results.csv"
    resume = bool(cfg["run"].get("resume", True))
    store = ResultStore(raw_csv, resume=resume)
    completed = store.completed_keys() if resume else set()
    if completed:
        print(f"[Main] Reanudación activa: {len(completed)} registros previos detectados en {raw_csv.name}")

    jobs = [
        (task, n_train, seed)
        for task in plan.tasks
        for n_train in plan.train_sizes
        for seed in plan.seeds
    ]

    n_jobs_cfg = int(cfg["run"].get("n_jobs", 1))
    n_workers = (os.cpu_count() or 1) if n_jobs_cfg == -1 else max(1, n_jobs_cfg)

    t_start = perf_counter()
    if n_workers == 1:
        for idx, (task, n_train, seed) in enumerate(jobs, start=1):
            t_job0 = perf_counter()
            recs = execute_fold_job(
                task=task,
                n_train=n_train,
                seed=seed,
                splits=splits_by_task[task.name],
                plan=plan,
                cfg=cfg,
                completed=completed,
            )
            store.append(recs)
            if recs:
                q_Summary = ", ".join(
                    f"{r['model_name']}{'(T=' + str(r['budget_T']) + ')' if r['budget_T'] else ''}: "
                    f"{r['test_accuracy']*100:.1f}%"
                    for r in recs
                )
                print(
                    f"[{idx:02d}/{len(jobs):02d}] {task.label} | N={n_train:<3d} | Seed={seed:<4d} "
                    f"({perf_counter() - t_job0:.1f}s) -> {q_Summary}"
                )
            else:
                print(f"[{idx:02d}/{len(jobs):02d}] {task.label} | N={n_train:<3d} | Seed={seed:<4d} -> (ya completado)")
    else:
        print(f"[Main] Ejecutando en paralelo con {n_workers} procesos...")
        worker_args = [
            (task, n_train, seed, splits_by_task[task.name], plan, cfg, completed)
            for (task, n_train, seed) in jobs
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_entry, arg): i for i, arg in enumerate(worker_args, start=1)}
            for fut in as_completed(futures):
                idx = futures[fut]
                task_label, n_train, seed, recs = fut.result()
                store.append(recs)
                print(f"[{idx:02d}/{len(jobs):02d}] {task_label} | N={n_train} | Seed={seed} -> +{len(recs)} filas")

    print(f"\n[Main] Fase de entrenamiento completada en {perf_counter() - t_start:.1f}s.")
    generate_reports_and_plots(raw_csv=raw_csv, out_dir=run_dir, cfg=cfg)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Punto de entrada único del Benchmark QCNN vs Baselines Clásicos."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Ruta al archivo YAML de configuración (por defecto: config.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["pilot", "data_size", "param_budget", "grid"],
        help="Sobrescribe opcionalmente `run.mode` del config.yaml.",
    )
    parser.add_argument(
        "--report-only",
        type=str,
        default=None,
        help="Ruta a una carpeta de resultados o CSV existente para regenerar tablas y gráficos sin reentrenar.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.report_only:
        cfg_loaded = load_config(args.config)
        target = Path(args.report_only)
        csv_file = target / "raw_results.csv" if target.is_dir() else target
        out_folder = target if target.is_dir() else target.parent
        generate_reports_and_plots(raw_csv=csv_file, out_dir=out_folder, cfg=cfg_loaded)
    else:
        run_benchmark(config_path=args.config, mode_override=args.mode)

