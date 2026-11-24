# -*- coding: utf-8 -*-
"""Build 30 synthetic case packages from the 50 pilot cases.

Selection: for each age x gender cell, take the 1st, 3rd and 5th case.
The 5th case in each cell is one of the 10 injected-conflict cases, so the
30-case package contains 20 clean + 10 injected cases.

Each package is clearly marked synthetic: the text user and the face have
the same age/gender but are different people; this is not a real identity.
"""
import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

import pandas as pd

CASE_ROOT = "cases_synthetic"


def sanitize(text: str, max_chars: int = 5000) -> str:
    t = str(text).replace("\xa0", " ")
    t = re.sub(r"https?://\S+", "[URL]", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", t)
    t = re.sub(r"@[\w\u4e00-\u9fff-]+", "[MENTION]", t)
    t = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", t)
    t = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID]", t)
    t = re.sub(r"(?<!\d)\d{6,}(?!\d)", "[NUM]", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars] + "\n[TRUNCATED]"
    return t


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/processed/pilot_cases.csv")
    ap.add_argument("--evidence", default="outputs/evidence_units_injected.jsonl")
    ap.add_argument("--results", default="outputs/case_results_injected.jsonl")
    ap.add_argument("--out", default=CASE_ROOT)
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    evidence = load_jsonl(args.evidence)
    results = {r["case_id"]: r for r in load_jsonl(args.results)}

    # group and select 1st, 3rd, 5th in each age x gender cell
    selected = []
    for (age, gender), group in cases.groupby(["age", "gender"], sort=True):
        group = group.sort_values("case_id").reset_index(drop=True)
        for idx in [0, 2, 4]:
            if idx < len(group):
                selected.append(group.iloc[idx])
    selected = selected[:30]
    if len(selected) != 30:
        raise RuntimeError(f"expected 30 selected cases, got {len(selected)}")

    os.makedirs(args.out, exist_ok=True)
    # only clear old SYN-* dirs; keep README.md / DATA_NOTICE.md
    for d in glob.glob(os.path.join(args.out, "SYN-*")):
        if os.path.isdir(d):
            shutil.rmtree(d)

    manifest = []
    injected_count = 0
    for i, row in enumerate(selected, start=1):
        syn_id = f"SYN-{i:03d}"
        source_id = row["case_id"]
        gender = row["gender"]
        age = int(row["age"])
        bucket = row["age_bucket"]
        injected = source_id in {f"C{j:03d}" for j in range(5, 51, 5)}
        injected_count += int(injected)

        cdir = os.path.join(args.out, syn_id)
        os.makedirs(cdir, exist_ok=True)

        # text (sanitized, truncated)
        text = sanitize(row["text_clean"], max_chars=5000)
        with open(os.path.join(cdir, "text.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")

        # face image
        face_name = "face" + os.path.splitext(row["image_path"])[1].lower()
        shutil.copy2(row["image_path"], os.path.join(cdir, face_name))

        # evidence units: rewrite ids, mark synthetic / redacted
        ev_rows = []
        for e in evidence:
            if e.get("case_id") != source_id:
                continue
            e = json.loads(json.dumps(e))  # deep copy
            e["case_id"] = syn_id
            e["evidence_id"] = e["evidence_id"].replace(source_id + "-", syn_id + "-", 1)
            e["synthetic"] = True
            e["pii_redacted"] = True
            if e.get("modality") == "text":
                e["raw_ref"] = f"{args.out}/{syn_id}/text.txt"
            elif e.get("modality") == "image":
                e["raw_ref"] = f"{args.out}/{syn_id}/{face_name}"
            if isinstance(e.get("content"), dict) and "snippet" in e["content"]:
                e["content"]["snippet"] = sanitize(str(e["content"]["snippet"]), max_chars=200)
            ev_rows.append(e)
        with open(os.path.join(cdir, "evidence_units.jsonl"), "w", encoding="utf-8") as f:
            for e in ev_rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        # result
        r = json.loads(json.dumps(results[source_id]))
        r["case_id"] = syn_id
        r["uid"] = syn_id  # remove original SMP uid
        r["source_case_id"] = source_id
        r["synthetic"] = True
        r["not_real_identity"] = True
        r["pairing"] = "same age/gender, different person"
        r["ground_truth"] = {"gender": gender, "age": age, "age_bucket": bucket}
        for key in ["top_evidence_gender", "top_evidence_age"]:
            for item in r.get(key, []):
                item["evidence_id"] = item["evidence_id"].replace(source_id + "-", syn_id + "-", 1)
        r["method"] = "fused_conflict"
        for key in ["conflict_log_gender", "conflict_log_age"]:
            for log in r.get(key, []):
                for k in ["evidence_a", "evidence_b", "downweighted"]:
                    if k in log:
                        log[k] = log[k].replace(source_id + "-", syn_id + "-", 1)
        # apply conflict flags / downweights to the evidence units
        logs = r.get("conflict_log_gender", []) + r.get("conflict_log_age", [])
        by_id = {e["evidence_id"]: e for e in ev_rows}
        for log in logs:
            eid = log.get("downweighted")
            if eid in by_id:
                e = by_id[eid]
                e["conflict_flag"] = True
                e["conflict_degree"] = max(float(e.get("conflict_degree", 0)), float(log.get("conflict", 0)))
                e["weight_multiplier"] = float(log.get("weight_multiplier", 1.0))
                e["note"] = (str(e.get("note", "")) + f" downweighted: conflict={log.get('conflict')}").strip()
        with open(os.path.join(cdir, "evidence_units.jsonl"), "w", encoding="utf-8") as f:
            for e in ev_rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(os.path.join(cdir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

        # case.json
        case = {
            "case_id": syn_id,
            "source_case_id": source_id,
            "synthetic": True,
            "not_real_identity": True,
            "pairing": "same age/gender, different person",
            "ground_truth": {"gender": gender, "age": age, "age_bucket": bucket},
            "sources": {
                "text": "SMP CUP 2016 (public research dataset, sanitized)",
                "face": "AFAD-Lite (academic research dataset)",
            },
            "files": {
                "text": "text.txt",
                "face": face_name,
                "evidence_units": "evidence_units.jsonl",
                "result": "result.json",
                "card": "case_card.md",
            },
            "injected_conflict": injected,
            "pii_redacted": True,
            "note": "Synthetic pairing for pilot only; not a real person and not a real identity.",
        }
        with open(os.path.join(cdir, "case.json"), "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)

        # case card
        pred_g = r["predicted_gender"]
        pred_a = r["predicted_age_bucket"]
        ev_lines = []
        for e in ev_rows:
            ev_lines.append(
                f"| {e['evidence_id']} | {e['source_type']} | {e['modality']} | "
                f"{e['credibility']['score']:.3f} | {e['conflict_flag']} |"
            )
        top_g = r.get("top_evidence_gender", [])[:3]
        top_a = r.get("top_evidence_age", [])[:3]
        card = [
            f"# 合成样例案件 {syn_id}",
            "",
            f"- 来源案件：{source_id}（pilot 内部编号）",
            "- 类型：合成多模态配对（SMP 2016 中文文本 + AFAD-Lite 亚洲人脸）",
            "- **不是真实身份**：文本用户与人脸同年龄、同性别，但不是同一个人。",
            f"- Ground truth：**{gender} / {age} 岁 / {bucket}**",
            f"- 是否注入冲突：{'是' if injected else '否'}",
            "",
            "## 模型结论",
            f"- 预测性别：**{pred_g}**（后验：{ {k: round(v,3) for k,v in r['gender_posterior'].items()} }）",
            f"- 预测年龄段：**{pred_a}**（后验：{ {k: round(v,3) for k,v in r['age_posterior'].items()} }）",
            f"- 冲突检出：{r['conflict_detected']}，最大冲突：{r['max_conflict']}",
            "",
            "## 证据",
            "| evidence_id | source_type | modality | credibility | conflict |",
            "|---|---|---|---|---|",
            *ev_lines,
            "",
            "## Top 证据（性别）",
            *[f"- {x['evidence_id']}：贡献 {x['contribution_pred']:+.3f}" for x in top_g],
            "",
            "## Top 证据（年龄）",
            *[f"- {x['evidence_id']}：贡献 {x['contribution_pred']:+.3f}" for x in top_a],
            "",
            "## 使用限制",
            "- 仅用于内部 pilot 验证；**不得**当作真实案件或对外分发。",
            "- 文本已做基础脱敏（URL/@/邮箱/电话/证件号/长数字），但不保证完全匿名。",
            "- 人脸来自 AFAD-Lite（学术研究用途）；不要发布或再分发。",
        ]
        with open(os.path.join(cdir, "case_card.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(card) + "\n")

        manifest.append({
            "case_id": syn_id,
            "source_case_id": source_id,
            "age": age,
            "gender": gender,
            "age_bucket": bucket,
            "injected_conflict": injected,
            "predicted_gender": pred_g,
            "predicted_age_bucket": pred_a,
            "gender_correct": pred_g == gender,
            "age_correct": pred_a == bucket,
            "conflict_detected": r["conflict_detected"],
            "max_conflict": r["max_conflict"],
            "n_evidence": len(ev_rows),
            "text_chars": len(text),
            "png_card": "case_card.md",
        })

    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "manifest.csv"), index=False, encoding="utf-8-sig")
    print("cases:", len(manifest), "injected:", injected_count)
    print("gender correct:", sum(m["gender_correct"] for m in manifest), "/", len(manifest))
    print("age correct:", sum(m["age_correct"] for m in manifest), "/", len(manifest))
    print("output:", args.out)


if __name__ == "__main__":
    main()
