# -*- coding: utf-8 -*-
"""Compare fusion schemes on the CREMA-D audio+face+text test set.

Schemes
-------
single: text / face / audio age evidence (calibrated on calib)
late:   equal average, ridge stacking, text+face ridge
concat: PCA(audio,face,text) -> ridge
gated:  learned per-sample modality gate (3 seeds)
attn:   1-layer self-attention over 3 modality tokens (3 seeds)
oracle: per-case best single modality (upper bound, not achievable)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
CENTERS = np.array([21.0, 27.0, 32.0, 37.5])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bucket_of(age):
    return BUCKETS[int(np.argmin(np.abs(CENTERS - age)))]


def age_metrics(y_age, y_b, pred):
    pb = [bucket_of(v) for v in pred]
    return {
        "n": int(len(y_age)),
        "mae": float(np.mean(np.abs(pred - y_age))),
        "bias": float(np.mean(pred - y_age)),
        "bucket_accuracy": float(accuracy_score(y_b, pb)),
        "pm1": float(np.mean([abs(BUCKETS.index(a) - BUCKETS.index(b)) <= 1
                              for a, b in zip(y_b, pb)])),
    }


def load_matrix(cases: pd.DataFrame, path: Path, emb_key: str, case_col: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    mapping = {str(k): i for i, k in enumerate(z[emb_key])}
    return np.stack([z["emb"][mapping[str(k)]] for k in cases[case_col]]).astype(np.float32)


def build_features(cases: pd.DataFrame, n_components: int = 32):
    af = load_matrix(cases, OUT / "audio_embeddings.npz", "case_id", "case_id")
    ff = load_matrix(cases, OUT / "face_embeddings.npz", "case_id", "case_id")
    tf = load_matrix(cases, OUT / "text_embeddings.npz", "uid", "text_uid")
    split = cases.split.to_numpy()
    fit = split == "fit"
    feats = {}
    for name, x in [("audio", af), ("face", ff), ("text", tf)]:
        sc = StandardScaler().fit(x[fit])
        xs = sc.transform(x).astype(np.float32)
        pca = PCA(n_components=n_components, random_state=20261128).fit(xs[fit])
        feats[name] = pca.transform(xs).astype(np.float32)
    return feats


class GatedFusion(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 3), nn.Softmax(-1))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, 1))

    def forward(self, xs):
        hs = [p(x) for p, x in zip(self.proj, xs)]
        w = self.gate(torch.cat(hs, dim=1))
        fused = sum(w[:, i:i + 1] * hs[i] for i in range(3))
        return self.head(fused).squeeze(1), w


class AttnFusion(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.mod = nn.Parameter(torch.zeros(1, 3, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 2,
                                           dropout=0.1, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(dim))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, 1))

    def forward(self, xs):
        tokens = torch.stack([p(x) for p, x in zip(self.proj, xs)], dim=1) + self.mod
        h = self.encoder(tokens).mean(dim=1)
        return self.head(h).squeeze(1), None


def train_torch(model_cls, feats, y_age, split, seed, epochs=250, bs=16, lr=1e-3):
    fit = split == "fit"; cal = split == "calib"
    xs = [torch.tensor(feats[k], dtype=torch.float32) for k in ["audio", "face", "text"]]
    y = torch.tensor((y_age - 30.0) / 10.0, dtype=torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best = (1e9, None)
    for epoch in range(epochs):
        model.train()
        idx = np.flatnonzero(fit); np.random.shuffle(idx)
        for start in range(0, len(idx), bs):
            b = idx[start:start + bs]
            xb = [x[b].to(DEVICE) for x in xs]
            yb = y[b].to(DEVICE)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pc = model([x[cal].to(DEVICE) for x in xs])[0].cpu().numpy() * 10 + 30
        mae = float(np.mean(np.abs(pc - y_age[cal])))
        if mae < best[0] - 1e-4:
            best = (mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        elif epoch > 80:
            break
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pred = model([x.to(DEVICE) for x in xs])[0].cpu().numpy() * 10 + 30
    return pred, best[0]


def boot_ci(y_age, y_b, pred, actors, n_boot=2000, seed=20261128):
    rng = np.random.default_rng(seed)
    test_actors = np.array(sorted(set(actors)))
    vals_mae, vals_bacc = [], []
    for _ in range(n_boot):
        pick = rng.choice(test_actors, len(test_actors), replace=True)
        idx = np.concatenate([np.flatnonzero(actors == a) for a in pick])
        vals_mae.append(np.mean(np.abs(pred[idx] - y_age[idx])))
        vals_bacc.append(accuracy_score(y_b[idx], [bucket_of(v) for v in pred[idx]]))
    return {"mae_ci": [float(np.percentile(vals_mae, 2.5)), float(np.percentile(vals_mae, 97.5))],
            "bucket_ci": [float(np.percentile(vals_bacc, 2.5)), float(np.percentile(vals_bacc, 97.5))]}


def pair_boot_delta(y_age, pred_a, pred_b, actors, n_boot=2000, seed=20261129):
    """Actor-level paired bootstrap of ΔMAE (a - b)."""
    rng = np.random.default_rng(seed)
    test_actors = np.array(sorted(set(actors)))
    vals = []
    ea = np.abs(pred_a - y_age); eb = np.abs(pred_b - y_age)
    for _ in range(n_boot):
        pick = rng.choice(test_actors, len(test_actors), replace=True)
        idx = np.concatenate([np.flatnonzero(actors == a) for a in pick])
        vals.append(float(np.mean(ea[idx]) - np.mean(eb[idx])))
    return {"delta_mae": float(np.mean(ea) - np.mean(eb)),
            "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OUT / "audio_multimodal_cases.csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    d = pd.read_csv(args.cases)
    h = pd.read_csv(OUT / "audio_age_head_predictions.csv", encoding="utf-8-sig")
    h = h[["case_id", "audio_head_age", "face_head_age"]]
    d = d.merge(h, on="case_id", how="left")
    split = d.split.to_numpy(); fit = split == "fit"; cal = split == "calib"; test = split == "test"
    y_age = d.age.to_numpy(float); y_b = d.age_bucket.to_numpy()
    feats = build_features(d)
    actors = d.actor_id.to_numpy()

    # Per-modality age calibrations on calib.
    raw = {"text": d.text_age_expected.to_numpy(float),
           "face": d.face_head_age.to_numpy(float),
           "audio": d.audio_head_age.to_numpy(float)}
    cal_age = {}
    for k, v in raw.items():
        m = LinearRegression().fit(v[cal].reshape(-1, 1), y_age[cal])
        cal_age[k] = m.predict(v.reshape(-1, 1))
    methods = {f"single_{k}": v for k, v in cal_age.items()}
    methods["late_equal"] = (cal_age["text"] + cal_age["face"] + cal_age["audio"]) / 3
    X = np.column_stack([cal_age["text"], cal_age["face"], cal_age["audio"]])
    methods["late_ridge"] = Ridge(alpha=1.0).fit(X[cal], y_age[cal]).predict(X)
    X2 = np.column_stack([cal_age["text"], cal_age["face"]])
    methods["late_text_face"] = Ridge(alpha=1.0).fit(X2[cal], y_age[cal]).predict(X2)

    # Feature concat: PCA embeddings -> ridge; alpha on calib.
    C = np.column_stack([feats["audio"], feats["face"], feats["text"]])
    best = (1e9, None)
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        m = Ridge(alpha=alpha).fit(C[fit], y_age[fit])
        mae = float(np.mean(np.abs(m.predict(C[cal]) - y_age[cal])))
        if mae < best[0]:
            best = (mae, alpha)
    methods["concat_ridge"] = Ridge(alpha=best[1]).fit(C[fit], y_age[fit]).predict(C)

    # Feature concat without audio (missing-audio fallback).
    C2 = np.column_stack([feats["face"], feats["text"]])
    best2 = (1e9, None)
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        m = Ridge(alpha=alpha).fit(C2[fit], y_age[fit])
        mae = float(np.mean(np.abs(m.predict(C2[cal]) - y_age[cal])))
        if mae < best2[0]:
            best2 = (mae, alpha)
    methods["concat_text_face"] = Ridge(alpha=best2[1]).fit(C2[fit], y_age[fit]).predict(C2)

    # Gated / attention: 3 seeds, report mean prediction and per-seed test.
    seed_metrics = {}
    for name, cls in [("gated", GatedFusion), ("attention", AttnFusion)]:
        preds, sm = [], []
        for seed in [20261127, 20261128, 20261129]:
            p, _ = train_torch(cls, feats, y_age, split, seed)
            preds.append(p)
            sm.append(age_metrics(y_age[test], y_b[test], p[test]))
        methods[name] = np.mean(preds, axis=0)
        seed_metrics[name] = sm

    # Oracle over the three single modalities (per-case best).
    methods["oracle"] = np.min(np.column_stack([cal_age["text"], cal_age["face"], cal_age["audio"]]), axis=1)
    # Fix oracle: choose modality with smallest absolute error, not min prediction.
    single = np.column_stack([cal_age["text"], cal_age["face"], cal_age["audio"]])
    err = np.abs(single - y_age[:, None])
    methods["oracle"] = single[np.arange(len(d)), err.argmin(axis=1)]

    metrics = {}
    for k, p in methods.items():
        metrics[k] = {"test": age_metrics(y_age[test], y_b[test], p[test]),
                      "calib": age_metrics(y_age[cal], y_b[cal], p[cal])}
        if k not in ("oracle",):
            metrics[k]["test_ci"] = boot_ci(y_age[test], y_b[test], p[test], actors[test])

    # Gender: late fusion vs feature concat.
    yg = (d.gender == "female").astype(int).to_numpy()
    probs = {"text": d.text_p_female.to_numpy(float),
             "face": d.face_p_female.to_numpy(float),
             "audio": d.audio_p_female.to_numpy(float)}
    def platt(p):
        z_cal = np.log(np.clip(p[cal], 1e-6, 1 - 1e-6) / (1 - np.clip(p[cal], 1e-6, 1 - 1e-6)))
        m = LogisticRegression(max_iter=1000).fit(z_cal.reshape(-1, 1), yg[cal])
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return m.predict_proba(z.reshape(-1, 1))[:, 1]
    pc = {k: platt(v) for k, v in probs.items()}
    lg = {"text": pc["text"], "face": pc["face"], "audio": pc["audio"],
          "late_text_face": (pc["text"] + pc["face"]) / 2,
          "late_3": (pc["text"] + pc["face"] + pc["audio"]) / 3}
    for name, x in [("concat_logreg", C)]:
        m = LogisticRegression(max_iter=3000, C=0.1).fit(x[fit], yg[fit])
        lg[name] = m.predict_proba(x)[:, 1]
    gender = {k: {"test": {"accuracy": float(accuracy_score(yg[test], v[test] >= .5)),
                           "auc": float(roc_auc_score(yg[test], v[test]))},
                  "calib": {"accuracy": float(accuracy_score(yg[cal], v[cal] >= .5))}}
              for k, v in lg.items()}

    comparisons = {}
    for name in ["late_equal", "late_ridge", "late_text_face", "concat_ridge",
                 "concat_text_face", "gated", "attention"]:
        comparisons[name] = {
            "vs_text": pair_boot_delta(y_age[test], methods[name][test], methods["single_text"][test], actors[test]),
            "vs_late_text_face": pair_boot_delta(y_age[test], methods[name][test], methods["late_text_face"][test], actors[test]),
        }
    out = {"n": {"fit": int(fit.sum()), "calib": int(cal.sum()), "test": int(test.sum())},
           "age": metrics, "gender": gender, "comparisons": comparisons,
           "seed_metrics": seed_metrics, "concat_alpha": best[1], "concat_no_audio_alpha": best2[1],
           "notes": [
               "actor-disjoint splits; test 15 actors / 60 clips",
               "CI is actor-level bootstrap (2000) on test",
               "oracle selects the best single modality per case, not deployable",
               "text is a synthetic Chinese pairing; audio/face same CREMA-D actor",
           ]}
    (Path(args.out) / "audio_fusion_schemes_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred = d[["case_id", "actor_id", "split", "gender", "age", "age_bucket"]].copy()
    for k, v in methods.items():
        pred[f"age_{k}"] = v
    pred.to_csv(Path(args.out) / "audio_fusion_schemes_predictions.csv", index=False, encoding="utf-8-sig")
    print("AGE test")
    print(pd.DataFrame({k: v["test"] for k, v in metrics.items()}).T[
        ["mae", "bias", "bucket_accuracy", "pm1"]].to_string(float_format=lambda x: f"{x:.3f}"))
    print("GENDER test")
    print(pd.DataFrame({k: v["test"] for k, v in gender.items()}).T.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
