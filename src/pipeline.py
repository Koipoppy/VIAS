# -*- coding: utf-8 -*-
"""End-to-end pilot pipeline.

Stages:
  1. build unified evidence units from text/face/self-report predictions;
  2. score six-dimension credibility;
  3. fuse evidence with a weighted logarithmic opinion pool
     (simplified hierarchical Bayesian);
  4. detect conflicts, downweight the weaker source, iterate;
  5. leave-one-out contribution + evidence graph for interpretability;
  6. evaluate clean / injected scenarios and save metrics.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credibility import recompute_score, score_evidence
from fusion import (AGE_BUCKETS, GENDERS, conflict_handle, max_pairwise_conflict,
                    norm_probs, weighted_fuse)
from schema import EvidenceUnit

AGE_MID = {"18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-40": 37.5}
GENDER_PRIOR = {g: 0.5 for g in GENDERS}
AGE_PRIOR = {b: 0.25 for b in AGE_BUCKETS}
TAU = 0.50
INJECT_CASES = [f"C{i:03d}" for i in range(5, 51, 5)]  # 10 cases


# ---------------------------------------------------------------- evidence
def build_evidence(row, text_row, face_row, self_row, inject=False):
    cid = row["case_id"]
    n_chars = int(row["n_chars"])
    ev = []

    ev.append(EvidenceUnit(
        evidence_id=f"{cid}-T-G", case_id=cid, modality="text",
        source_type="text_model", source_name="TF-IDF char n-gram + LogisticRegression",
        content={"model": "text_gender", "n_chars": n_chars,
                 "snippet": str(row["text_clean"])[:160]},
        gender_probs={"male": float(text_row["text_gender_male"]),
                      "female": float(text_row["text_gender_female"])},
        credibility=score_evidence("text_model", n_chars=n_chars),
    ))
    ev.append(EvidenceUnit(
        evidence_id=f"{cid}-T-A", case_id=cid, modality="text",
        source_type="text_model", source_name="TF-IDF char n-gram + LogisticRegression",
        content={"model": "text_age", "n_chars": n_chars,
                 "snippet": str(row["text_clean"])[:160]},
        age_probs={b: float(text_row[f"text_age_{b}"]) for b in AGE_BUCKETS},
        credibility=score_evidence("text_model", n_chars=n_chars),
    ))
    ev.append(EvidenceUnit(
        evidence_id=f"{cid}-F-G", case_id=cid, modality="image",
        source_type="face_model", source_name="rizvandwiki/gender-classification (ViT)",
        content={"image": row["image_path"], "afad_age": int(row["afad_age"]),
                 "afad_gender": row["afad_gender"]},
        gender_probs={"male": float(face_row["face_gender_male"]),
                      "female": float(face_row["face_gender_female"])},
        credibility=score_evidence("face_model", face_quality=0.80),
    ))
    ev.append(EvidenceUnit(
        evidence_id=f"{cid}-F-A", case_id=cid, modality="image",
        source_type="face_model", source_name="nateraw/vit-age-classifier (ViT)",
        content={"image": row["image_path"], "afad_age": int(row["afad_age"]),
                 "raw_top_bucket": face_row["face_age_raw_top"]},
        age_probs={b: float(face_row[f"face_age_{b}"]) for b in AGE_BUCKETS},
        credibility=score_evidence("face_model", face_quality=0.70),
    ))

    snippet = "" if pd.isna(self_row.get("snippets")) else str(self_row.get("snippets"))
    if pd.notna(self_row.get("self_gender")):
        g = self_row["self_gender"]
        probs = {x: 0.15 for x in GENDERS}
        probs[g] = 0.85
        conf = float(self_row.get("self_gender_conf", 0.0))
        ev.append(EvidenceUnit(
            evidence_id=f"{cid}-S-G", case_id=cid, modality="text",
            source_type="self_report", source_name="regex self-report (gender)",
            content={"snippet": snippet, "claimed": g},
            gender_probs=norm_probs(probs),
            credibility=score_evidence("self_report", self_conf=conf),
        ))
    if pd.notna(self_row.get("self_age")):
        b = self_row["self_age_bucket"]
        probs = {x: 0.05 for x in AGE_BUCKETS}
        probs[b] = 0.85
        conf = float(self_row.get("self_age_conf", 0.0))
        ev.append(EvidenceUnit(
            evidence_id=f"{cid}-S-A", case_id=cid, modality="text",
            source_type="self_report", source_name="regex self-report (age)",
            content={"snippet": snippet,
                     "claimed_age": int(self_row["self_age"]), "claimed_bucket": b},
            age_probs=norm_probs(probs),
            credibility=score_evidence("self_report", self_conf=conf),
        ))

    if inject:
        true_g = row["gender"]
        wrong_g = "female" if true_g == "male" else "male"
        ti = AGE_BUCKETS.index(row["age_bucket"])
        wrong_b = AGE_BUCKETS[(ti + 2) % len(AGE_BUCKETS)]
        conf = 0.80
        gp = {x: 0.15 for x in GENDERS}
        gp[wrong_g] = 0.85
        ap = {x: 0.05 for x in AGE_BUCKETS}
        ap[wrong_b] = 0.85
        ev.append(EvidenceUnit(
            evidence_id=f"{cid}-X-G", case_id=cid, modality="text",
            source_type="synthetic_conflict", source_name="injected contradictory self-report",
            content={"injected": True, "claimed_gender": wrong_g},
            gender_probs=norm_probs(gp),
            credibility=score_evidence("synthetic_conflict", self_conf=conf),
            injected=True,
        ))
        ev.append(EvidenceUnit(
            evidence_id=f"{cid}-X-A", case_id=cid, modality="text",
            source_type="synthetic_conflict", source_name="injected contradictory self-report",
            content={"injected": True, "claimed_bucket": wrong_b},
            age_probs=norm_probs(ap),
            credibility=score_evidence("synthetic_conflict", self_conf=conf),
            injected=True,
        ))
    return ev


# ---------------------------------------------------------------- metrics
def score_predictions(df, pred_col, true_col, age=False):
    y, p = df[true_col].astype(str), df[pred_col].astype(str)
    acc = float((y == p).mean())
    labels = sorted(set(y) | set(p))
    f1s = []
    for lab in labels:
        tp = int(((y == lab) & (p == lab)).sum())
        fp = int(((y != lab) & (p == lab)).sum())
        fn = int(((y == lab) & (p != lab)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out = {"n": int(len(df)), "accuracy": round(acc, 4),
           "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4)}
    if age:
        out["mae_years"] = round(float(np.mean([
            abs(AGE_MID.get(pred, 0) - int(true))
            for pred, true in zip(df[pred_col], df["true_age"])
        ])), 2)
    return out


# ---------------------------------------------------------------- interpretability
def leave_one_out(evidence, attr, pred_label, true_label):
    prior = GENDER_PRIOR if attr == "gender" else AGE_PRIOR
    full, _ = weighted_fuse(prior, evidence, attr, use_credibility=True)
    rows = []
    for e in evidence:
        p = e.gender_probs if attr == "gender" else e.age_probs
        if not p:
            continue
        without = [x for x in evidence if x.evidence_id != e.evidence_id]
        post, _ = weighted_fuse(prior, without, attr, use_credibility=True)
        rows.append({
            "evidence_id": e.evidence_id,
            "source_type": e.source_type,
            "weight": round(e.effective_weight(), 4),
            "p_pred_all": round(full.get(pred_label, 0.0), 4),
            "p_pred_without": round(post.get(pred_label, 0.0), 4),
            "contribution_pred": round(full.get(pred_label, 0.0) - post.get(pred_label, 0.0), 4),
            "p_true_all": round(full.get(true_label, 0.0), 4),
            "p_true_without": round(post.get(true_label, 0.0), 4),
            "contribution_true": round(full.get(true_label, 0.0) - post.get(true_label, 0.0), 4),
            "conflict_flag": e.conflict_flag,
            "injected": e.injected,
        })
    rows.sort(key=lambda r: abs(r["contribution_pred"]), reverse=True)
    return full, rows


def graph_data(case_id, evidence, gender_post, age_post, gender_pred, age_pred, contributions):
    nodes = [{"id": e.evidence_id, "type": "evidence", "source_type": e.source_type,
              "weight": round(e.effective_weight(), 4), "conflict": e.conflict_flag,
              "injected": e.injected} for e in evidence]
    nodes += [{"id": f"H_G_{g}", "type": "hypothesis", "attr": "gender", "value": g}
              for g in GENDERS]
    nodes += [{"id": f"H_A_{b}", "type": "hypothesis", "attr": "age", "value": b}
              for b in AGE_BUCKETS]
    nodes += [{"id": "T", "type": "target", "gender": gender_pred, "age_bucket": age_pred}]
    edges = []
    cmap = {r["evidence_id"]: r for r in contributions}
    for e in evidence:
        c = cmap.get(e.evidence_id, {})
        if e.gender_probs:
            edges.append({"source": e.evidence_id, "target": f"H_G_{gender_pred}",
                          "weight": abs(c.get("contribution_pred", 0.0)),
                          "kind": "supports" if e.gender_probs.get(gender_pred, 0) >= 0.5 else "contradicts"})
        if e.age_probs:
            edges.append({"source": e.evidence_id, "target": f"H_A_{age_pred}",
                          "weight": abs(c.get("contribution_pred", 0.0)),
                          "kind": "supports" if e.age_probs.get(age_pred, 0) >= 0.25 else "contradicts"})
    for g in GENDERS:
        edges.append({"source": f"H_G_{g}", "target": "T", "weight": round(gender_post.get(g, 0.0), 4), "kind": "posterior"})
    for b in AGE_BUCKETS:
        edges.append({"source": f"H_A_{b}", "target": "T", "weight": round(age_post.get(b, 0.0), 4), "kind": "posterior"})
    return {"case_id": case_id, "nodes": nodes, "edges": edges}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/processed/pilot_cases.csv")
    ap.add_argument("--text", default="data/processed/text_predictions.csv")
    ap.add_argument("--face", default="data/processed/face_predictions.csv")
    ap.add_argument("--self", default="data/processed/self_reports.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--downweight", type=float, default=0.25)
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    text = pd.read_csv(args.text)
    face = pd.read_csv(args.face)
    selfrep = pd.read_csv(args.self)
    df = cases.merge(text, on="uid", how="left", suffixes=("", "_t"))
    df = df.merge(face.drop(columns=["true_gender", "true_age", "true_age_bucket"], errors="ignore"),
                  on=["case_id", "uid"], how="left")
    df = df.merge(selfrep, on="uid", how="left", suffixes=("", "_s"))
    assert df["text_gender_male"].notna().all()
    assert df["face_gender_male"].notna().all()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "evidence_graphs"), exist_ok=True)

    # Build evidence for clean and injected scenarios
    clean_ev = {}
    inj_ev = {}
    for _, row in df.iterrows():
        tr = text[text["uid"] == row["uid"]].iloc[0]
        fr = face[face["uid"] == row["uid"]].iloc[0]
        sr = selfrep[selfrep["uid"] == row["uid"]].iloc[0]
        clean_ev[row["case_id"]] = build_evidence(row, tr, fr, sr, inject=False)
        inj_ev[row["case_id"]] = build_evidence(row, tr, fr, sr, inject=row["case_id"] in INJECT_CASES)

    methods = ["text_only", "face_only", "fused_equal", "fused_cred", "fused_conflict"]
    results = {m: [] for m in methods}
    results_inj = {m: [] for m in methods}
    evidence_dump_clean = []
    evidence_dump_inj = []
    case_results_clean = []
    case_results_inj = []
    conflict_rows = []

    for scenario, ev_map, dump, case_out in [
        ("clean", clean_ev, evidence_dump_clean, case_results_clean),
        ("injected", inj_ev, evidence_dump_inj, case_results_inj),
    ]:
        for _, row in df.iterrows():
            cid = row["case_id"]
            ev = ev_map[cid]
            true_g, true_b = row["gender"], row["age_bucket"]
            for e in ev:
                dump.append({**e.model_dump(), "effective_weight": round(e.effective_weight(), 4)})

            # baselines
            text_g = text[text["uid"] == row["uid"]].iloc[0]["pred_gender"]
            text_b = text[text["uid"] == row["uid"]].iloc[0]["pred_age_bucket"]
            face_g = face[face["uid"] == row["uid"]].iloc[0]["face_pred_gender"]
            face_b = face[face["uid"] == row["uid"]].iloc[0]["face_age_pred_bucket"]

            g_equal, _ = weighted_fuse(GENDER_PRIOR, ev, "gender", use_credibility=False)
            g_cred, _ = weighted_fuse(GENDER_PRIOR, ev, "gender", use_credibility=True)
            a_equal, _ = weighted_fuse(AGE_PRIOR, ev, "age", use_credibility=False)
            a_cred, _ = weighted_fuse(AGE_PRIOR, ev, "age", use_credibility=True)
            g_conf, glog, gev_g, gmax, gpair, _ = conflict_handle(
                ev, "gender", tau=args.tau, downweight=args.downweight)
            a_conf, alog, gev_a, amax, apair, _ = conflict_handle(
                ev, "age", tau=args.tau, downweight=args.downweight)

            preds = {
                "text_only": (text_g, text_b, {g: float(text[text["uid"] == row["uid"]].iloc[0][f"text_gender_{g}"]) for g in GENDERS},
                              {b: float(text[text["uid"] == row["uid"]].iloc[0][f"text_age_{b}"]) for b in AGE_BUCKETS}),
                "face_only": (face_g, face_b, {g: float(face[face["uid"] == row["uid"]].iloc[0][f"face_gender_{g}"]) for g in GENDERS},
                              {b: float(face[face["uid"] == row["uid"]].iloc[0][f"face_age_{b}"]) for b in AGE_BUCKETS}),
                "fused_equal": (max(g_equal, key=g_equal.get), max(a_equal, key=a_equal.get), g_equal, a_equal),
                "fused_cred": (max(g_cred, key=g_cred.get), max(a_cred, key=a_cred.get), g_cred, a_cred),
                "fused_conflict": (max(g_conf, key=g_conf.get), max(a_conf, key=a_conf.get), g_conf, a_conf),
            }
            for m in methods:
                pg, pb, gp, ap = preds[m]
                rec = {"case_id": cid, "true_gender": true_g, "pred_gender": pg,
                       "true_age_bucket": true_b, "pred_age_bucket": pb,
                       "true_age": int(row["age"]), "injected": cid in INJECT_CASES}
                (results if scenario == "clean" else results_inj)[m].append(rec)

            # interpretability on the conflict-handled evidence (gender and age separately)
            g_full, g_contrib = leave_one_out(gev_g, "gender", preds["fused_conflict"][0], true_g)
            a_full, a_contrib = leave_one_out(gev_a, "age", preds["fused_conflict"][1], true_b)
            combined = {}
            for e in gev_g + gev_a:
                combined[e.evidence_id] = e
            graph = graph_data(cid, list(combined.values()),
                               g_full, a_full, preds["fused_conflict"][0], preds["fused_conflict"][1],
                               g_contrib + a_contrib)
            with open(os.path.join(args.out_dir, "evidence_graphs", f"{cid}.json"), "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            case_out.append({
                "case_id": cid,
                "uid": row["uid"],
                "true_gender": true_g,
                "true_age": int(row["age"]),
                "true_age_bucket": true_b,
                "predicted_gender": preds["fused_conflict"][0],
                "predicted_age_bucket": preds["fused_conflict"][1],
                "gender_posterior": g_full,
                "age_posterior": a_full,
                "conflict_detected": max(gmax, amax) > args.tau,
                "max_conflict": round(max(gmax, amax), 4),
                "max_conflict_gender": round(gmax, 4),
                "max_conflict_age": round(amax, 4),
                "conflict_rounds": len(glog) + len(alog),
                "conflict_log_gender": glog,
                "conflict_log_age": alog,
                "top_evidence_gender": g_contrib[:3],
                "top_evidence_age": a_contrib[:3],
                "injected": cid in INJECT_CASES,
            })
            if cid in INJECT_CASES:
                conflict_rows.append({
                    "case_id": cid, "scenario": scenario,
                    "max_conflict_gender": round(gmax, 4), "max_conflict_age": round(amax, 4),
                    "detected": max(gmax, amax) > args.tau,
                    "pred_gender": preds["fused_conflict"][0], "true_gender": true_g,
                    "pred_age": preds["fused_conflict"][1], "true_age": true_b,
                })

    # Save jsonl
    def dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(os.path.join(args.out_dir, "evidence_units_clean.jsonl"), evidence_dump_clean)
    dump_jsonl(os.path.join(args.out_dir, "evidence_units_injected.jsonl"), evidence_dump_inj)
    dump_jsonl(os.path.join(args.out_dir, "case_results_clean.jsonl"), case_results_clean)
    dump_jsonl(os.path.join(args.out_dir, "case_results_injected.jsonl"), case_results_inj)

    # Save method-level predictions for figures / report
    for scenario, res in [("clean", results), ("injected", results_inj)]:
        frames = []
        for m in methods:
            d = pd.DataFrame(res[m])
            d["method"] = m
            frames.append(d)
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(args.out_dir, f"method_predictions_{scenario}.csv"), index=False)

    # Metrics
    metrics = {"clean": {}, "injected": {}, "conflict": {}, "coverage": {}}
    for m in methods:
        d = pd.DataFrame(results[m])
        metrics["clean"][m] = {
            "gender": score_predictions(d, "pred_gender", "true_gender"),
            "age": score_predictions(d, "pred_age_bucket", "true_age_bucket", age=True),
        }
        di = pd.DataFrame(results_inj[m])
        metrics["injected"][m] = {
            "all": {
                "gender": score_predictions(di, "pred_gender", "true_gender"),
                "age": score_predictions(di, "pred_age_bucket", "true_age_bucket", age=True),
            },
            "injected_subset": {
                "gender": score_predictions(di[di["injected"]], "pred_gender", "true_gender"),
                "age": score_predictions(di[di["injected"]], "pred_age_bucket", "true_age_bucket", age=True),
            },
            "clean_subset": {
                "gender": score_predictions(di[~di["injected"]], "pred_gender", "true_gender"),
                "age": score_predictions(di[~di["injected"]], "pred_age_bucket", "true_age_bucket", age=True),
            },
        }

    # conflict stats (use injected scenario)
    cr = pd.DataFrame(conflict_rows)
    cr = cr[cr["scenario"] == "injected"]
    injected_detected = int(cr["detected"].sum()) if len(cr) else 0
    # false positives on clean cases from clean scenario
    clean_flags = [c for c in case_results_clean if c["conflict_detected"]]
    clean_cases = [c for c in case_results_clean]
    metrics["conflict"] = {
        "tau": args.tau,
        "injected_cases": len(INJECT_CASES),
        "injected_detected": injected_detected,
        "injected_detection_rate": round(injected_detected / max(1, len(cr)), 4),
        "clean_false_positive_rate": round(len(clean_flags) / max(1, len(clean_cases)), 4),
        "clean_flagged_cases": [c["case_id"] for c in clean_flags],
        "injected_recovery_rate": round(float(np.mean([
            (r["pred_gender"] == r["true_gender"]) and (r["pred_age"] == r["true_age"])
            for _, r in cr.iterrows()
        ])) if len(cr) else 0.0, 4),
    }
    metrics["coverage"] = {
        "n_cases": int(len(df)),
        "self_report_gender": float(selfrep["self_gender"].notna().mean()),
        "self_report_age": float(selfrep["self_age"].notna().mean()),
        "text_gender_evidence": 1.0,
        "text_age_evidence": 1.0,
        "face_gender_evidence": 1.0,
        "face_age_evidence": 1.0,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved outputs to", args.out_dir)


if __name__ == "__main__":
    main()
