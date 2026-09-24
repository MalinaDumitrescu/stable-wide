from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def display_notebook_outputs(out_dir: str | Path, *, max_rows: int = 20) -> None:
    """Render saved experiment artifacts inline in a Jupyter notebook."""
    if not _in_notebook():
        return
    try:
        from IPython.display import Image, display
    except Exception:
        return

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return

    files = sorted(
        [p for p in out_dir.iterdir() if p.is_file()],
        key=lambda p: (0 if p.name.startswith("aggregate_") else 1, p.name.lower()),
    )
    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp"}:
            print(f"\n[{path.name}]")
            display(Image(filename=str(path), retina=True))
        elif suffix == ".csv":
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    print(f"\n[{path.name}]")
                    display(df.head(max_rows))
            except Exception:
                print(f"\n[{path.name}] could not be displayed as a DataFrame.")
        elif suffix in {".txt", ".json", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    print(f"\n[{path.name}]")
                    print(text)
            except Exception:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        print(f"\n[{path.name}]")
                        print(text)
                except Exception:
                    pass


def find_project_root(start: str | Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "run_experiment.py").exists() and (candidate / "src" / "stablewide").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing run_experiment.py")


def ensure_src_on_path(root: Path) -> None:
    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def run_experiment(root: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    src = str((root / "src").resolve())
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(root / "run_experiment.py"), *map(str, args)]
    print(" ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if _in_notebook():
        out_arg = None
        for i, arg in enumerate(args):
            if arg == "--out" and i + 1 < len(args):
                out_arg = Path(args[i + 1])
                break
        if out_arg is not None:
            display_notebook_outputs(out_arg)
    return completed


def read_outputs(out_dir: Path) -> dict[str, object]:
    out_dir = Path(out_dir)
    result: dict[str, object] = {}
    csvs = [
        "aggregate_results.csv",
        "aggregate_fixed_k_pairwise.csv",
        "aggregate_tuned_performance.csv",
        "results_outer.csv",
        "support_results_outer.csv",
        "fixed_k_pairwise_outer.csv",
        "pairwise_matched_budget_outer.csv",
    ]
    for name in csvs:
        path = out_dir / name
        if path.exists() and path.stat().st_size > 0:
            try:
                result[name] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                result[name] = pd.DataFrame()
    decision = out_dir / "preregistered_decision.txt"
    if decision.exists():
        result["decision"] = decision.read_text(encoding="utf-8")
    meta = out_dir / "meta.json"
    if meta.exists():
        result["meta"] = json.loads(meta.read_text(encoding="utf-8"))
    if _in_notebook():
        display_notebook_outputs(out_dir)
    return result


def core_subsample_table(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sub = df[(df["experiment"] == "subsample") & (df["k"] == k)].copy()
    wanted = [
        "method", "n_train",
        "auroc_mean_mean", "auroc_mean_ci_low", "auroc_mean_ci_high",
        "jaccard_gene_mean", "jaccard_gene_ci_low", "jaccard_gene_ci_high",
        "jaccard_module_mean", "jaccard_module_ci_low", "jaccard_module_ci_high",
    ]
    cols = [c for c in wanted if c in sub.columns]
    return sub[cols].sort_values(["n_train", "method"], ascending=[False, True])
