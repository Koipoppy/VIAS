# -*- coding: utf-8 -*-
"""Aggregate the local concat baseline and server cross-attention predictions.

All predictions are aligned to the same joint manifest, temperature-scaled on
the calib split, and compared on the official PAN18 test split only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pan18_joint_fusion import (  # noqa: E402
    _logit, binary_metrics, bootstrap_delta, fit_temperature, labels_of,
    load_data, mcnemar,
)

BASELINE_METHODS = ["text", "text_tfidf", "image", "image_vit",
                    "late_logpool", "late_stack", "concat"]


def read_probs(path: Path, manifest: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    key = pd.MultiIndex.from_arrays([manifest["author_id"], manifest["lang"]])
    series = df.set_index(["author_id", "lang"])["p_female"]
    aligned = series.reindex(key)
    if aligned.isna().any():
        raise SystemExit(f"{path}: {int(aligned.isna().sum())} rows missing after alignment")
    return aligned.to_numpy(dtype=float)


def calibrate(y: np.ndarray, p: np.ndarray, calib: np.ndarray) -> tuple[float, np.ndarray]:
    temperature = fit_temperature(y[calib], p[calib])
    calibrated = 1.0 / (1.0 + np.exp(-_logit(p) / temperature))
    return temperature, calibrated


def summarize(y: np.ndarray, p: np.ndarray, test: np.ndarray,
              manifest: pd.DataFrame) -> dict:
    out = {"test": binary_metrics(y[test], p[test])}
    languages = manifest["lang"].to_numpy()
    out["per_lang"] = {
        lang: binary_metrics(y[test & (languages == lang)], p[test & (languages == lang)])
        for lang in ["en", "es", "ar"]
    }
    return out


def comparison(y: np.ndarray, test: np.ndarray, base: np.ndarray,
               new: np.ndarray, n_boot: int) -> dict:
    return {
        "delta": bootstrap_delta(y[test], base[test], new[test], n_boot),
        "mcnemar": mcnemar(y[test], base[test], new[test]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pan18_joint")
    ap.add_argument("--baseline", default="outputs/pan18_joint/baseline_2026-11-27")
    ap.add_argument("--crossattn", default="outputs/pan18_joint/crossattn_2026-11-27")
    ap.add_argument("--out", default="outputs/pan18_joint/crossattn_2026-11-27")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    manifest, _, _, _ = load_data(Path(args.data), 512, chunks=4)
    y = labels_of(manifest)
    calib = (manifest["split"] == "calib").to_numpy()
    test = (manifest["split"] == "test").to_numpy()

    cal: dict[str, np.ndarray] = {}
    temps: dict[str, float] = {}
    results: dict[str, dict] = {}

    baseline_dir = Path(args.baseline)
    for method in BASELINE_METHODS:
        path = baseline_dir / f"{method}_probs.csv"
        if not path.exists():
            print(f"[skip] {path}", flush=True)
            continue
        p = read_probs(path, manifest)
        temperature, p_cal = calibrate(y, p, calib)
        cal[method] = p_cal
        temps[method] = temperature
        results[method] = {"temperature": temperature,
                           **summarize(y, p_cal, test, manifest)}

    seed_files = sorted(Path(args.crossattn).glob("seed*/crossattn_probs.csv"))
    if not seed_files:
        raise SystemExit(f"no crossattn seed predictions under {args.crossattn}")
    seed_cal: dict[str, np.ndarray] = {}
    seed_results: dict[str, dict] = {}
    for path in seed_files:
        seed = path.parent.name
        p = read_probs(path, manifest)
        temperature, p_cal = calibrate(y, p, calib)
        seed_cal[seed] = p_cal
        seed_results[seed] = {"temperature": temperature,
                              **summarize(y, p_cal, test, manifest)}
        print(f"[seed] {seed} acc={seed_results[seed]['test']['accuracy']:.4f}", flush=True)

    p_ens_raw = np.mean(np.stack(list(seed_cal.values())), axis=0)
    temperature, p_ens = calibrate(y, p_ens_raw, calib)
    cal["crossattn_ens"] = p_ens
    temps["crossattn_ens"] = temperature
    results["crossattn_ens"] = {"temperature": temperature,
                                **summarize(y, p_ens, test, manifest)}

    accs = np.array([r["test"]["accuracy"] for r in seed_results.values()])
    lls = np.array([r["test"]["log_loss"] for r in seed_results.values()])
    seed_summary = {
        "n_seeds": int(len(accs)),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "log_loss_mean": float(lls.mean()),
        "log_loss_std": float(lls.std(ddof=1)) if len(lls) > 1 else 0.0,
    }

    comparisons: dict[str, dict] = {}
    for method in cal:
        if method == "concat":
            continue
        row = {}
        for base in ["text", "late_stack", "concat"]:
            if base in cal:
                row[f"vs_{base}"] = comparison(y, test, cal[base], cal[method],
                                               args.n_boot)
        comparisons[method] = row

    out = {
        "n": {"fit": int((manifest["split"] == "fit").sum()),
              "calib": int(calib.sum()), "test": int(test.sum())},
        "baseline_dir": str(baseline_dir),
        "crossattn_dir": str(args.crossattn),
        "seed_summary": seed_summary,
        "results": results,
        "seeds": seed_results,
        "comparisons": comparisons,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_crossattn.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for method, r in results.items():
        row = {"method": method, **r["test"]}
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")
    table.to_csv(out_dir / "method_summary.csv")
    print(table[["accuracy", "macro_f1", "auc", "log_loss", "ece"]]
          .to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(json.dumps(seed_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
