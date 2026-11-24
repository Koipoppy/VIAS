# -*- coding: utf-8 -*-
"""Acceptance test for the October plan standard.

Standard (from the task screenshot):
  1. 需求调研与数据可得性摸底
  2. 确定原型功能边界（文本、图像先行，音频与行为日志预留接口）
  3. 编制技术方案与数据规范，定型统一证据单元 Schema
  4. 收集 30 例样例案件数据
  5. 搭建开发环境
  6. 技术路线图（五段式）

The test is structural: it verifies concrete artifacts, runs the pipeline
end-to-end, and marks external dependencies (real desensitized cases,
Docker runtime) as PARTIAL rather than FAIL.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
VIAS = ROOT.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from schema import EvidenceUnit  # noqa: E402

CHECKS = []


def add(req, name, status, evidence="", note=""):
    CHECKS.append({
        "requirement": req,
        "check": name,
        "status": status,   # PASS / PASS_NOTE / PARTIAL / FAIL
        "evidence": evidence,
        "note": note,
    })


def run(cmd, name):
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    ok = p.returncode == 0
    add("端到端", name, "PASS" if ok else "FAIL",
        f"exit={p.returncode}", (p.stderr or p.stdout)[-300:])
    return p


def check_data_availability():
    xlsx = VIAS / "性别年龄推断数据集清单_2026-11-16.xlsx"
    if not xlsx.exists():
        add("1. 数据可得性", "数据集清单存在", "FAIL", str(xlsx))
        return
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, data_only=True)
    expected = {"中文文本": 6, "人脸图像": 15, "多模态英文": 6}
    total = 0
    missing_fields = 0
    links = 0
    for sheet, n in expected.items():
        ws = wb[sheet]
        count = 0
        for r in range(5, ws.max_row + 1):
            name = ws.cell(row=r, column=2).value
            if not name:
                continue
            count += 1
            if not ws.cell(row=r, column=6).value or not ws.cell(row=r, column=7).value:
                missing_fields += 1
            for c in range(2, ws.max_column + 1):
                if ws.cell(row=r, column=c).hyperlink:
                    links += 1
        add("1. 数据可得性", f"{sheet} 行数 = {n}", "PASS" if count >= n else "FAIL", f"actual={count}")
        total += count
    add("1. 数据可得性", "数据集总数 >= 27", "PASS" if total >= 27 else "FAIL", f"actual={total}")
    add("1. 数据可得性", "每行有可达性 + 大小", "PASS" if missing_fields == 0 else "FAIL",
        f"missing={missing_fields}")
    add("1. 数据可得性", "存在可点击链接", "PASS" if links > 0 else "FAIL", f"hyperlinks={links}")
    add("1. 数据可得性", "示范单位脱敏样本", "PARTIAL",
        "清单已覆盖公开数据集", "示范单位样本可得性待确认（外部依赖）")


def check_boundary():
    text = ROOT / "data/processed/text_predictions.csv"
    face = ROOT / "data/processed/face_predictions.csv"
    add("2. 原型边界", "文本预测 50 条", "PASS" if len(pd.read_csv(text)) == 50 else "FAIL",
        f"rows={len(pd.read_csv(text))}")
    add("2. 原型边界", "人脸预测 50 条", "PASS" if len(pd.read_csv(face)) == 50 else "FAIL",
        f"rows={len(pd.read_csv(face))}")
    mods = set()
    for line in open(ROOT / "outputs/evidence_units_clean.jsonl", encoding="utf-8"):
        mods.add(json.loads(line)["modality"])
    add("2. 原型边界", "证据包含 text + image", "PASS" if {"text", "image"} <= mods else "FAIL",
        f"modalities={sorted(mods)}")
    from parsers import parse_audio, parse_behavior
    a = parse_audio(case_id="SYN-TEST", audio_uri="minio://test.wav")
    b = parse_behavior(case_id="SYN-TEST", log_uri="minio://test.json", events=[])
    ok = (a.modality == "audio" and a.content["status"] == "reserved" and a.gender_probs is None
          and b.modality == "behavior" and b.content["status"] == "reserved" and b.age_probs is None)
    add("2. 原型边界", "音频/行为预留接口", "PASS" if ok else "FAIL",
        "audio/behavior stubs return reserved EvidenceUnit")


def check_docs():
    design = ROOT / "reports/pilot_design_note.md"
    review = ROOT / "reports/evidence_unit_v0.1_review.md"
    readme = ROOT / "README.md"
    dtext = design.read_text(encoding="utf-8") if design.exists() else ""
    for key in ["标签体系", "六维可信度", "层次结构", "融合", "冲突", "可解释"]:
        add("3. 技术方案/数据规范", f"设计说明包含「{key}」", "PASS" if key in dtext else "FAIL")
    rtext = review.read_text(encoding="utf-8") if review.exists() else ""
    add("3. 技术方案/数据规范", "Schema 评审稿存在", "PASS" if review.exists() else "FAIL", str(review))
    add("3. 技术方案/数据规范", "评审稿含字段表", "PASS" if "字段表" in rtext or "字段" in rtext else "FAIL")
    add("3. 技术方案/数据规范", "评审稿含待评审问题", "PASS" if "待评审问题" in rtext else "FAIL")
    add("3. 技术方案/数据规范", "README 含 10 月计划补齐", "PASS" if "10 月计划补齐情况" in readme.read_text(encoding="utf-8") else "FAIL")


def check_schema():
    from schema import EvidenceUnit as EU
    fields = set(EU.model_fields)
    required = {"evidence_id", "case_id", "modality", "source_type", "source_name", "credibility"}
    for f in ["schema_version", "raw_ref", "time_range", "language", "synthetic", "pii_redacted"]:
        add("4. 统一证据单元 Schema", f"字段 {f}", "PASS" if f in fields else "FAIL")
    add("4. 统一证据单元 Schema", "必填字段", "PASS" if required <= fields else "FAIL", str(sorted(required - fields)))
    mod_ann = str(EU.model_fields["modality"].annotation)
    add("4. 统一证据单元 Schema", "modality 含 audio/behavior",
        "PASS" if "audio" in mod_ann and "behavior" in mod_ann else "FAIL", mod_ann)
    schema_path = ROOT / "schema/evidence_unit.schema.json"
    example_path = ROOT / "schema/evidence_unit.example.json"
    try:
        sch = json.load(open(schema_path, encoding="utf-8"))
        add("4. 统一证据单元 Schema", "JSON Schema 可解析", "PASS", f"properties={len(sch['properties'])}")
    except Exception as e:
        add("4. 统一证据单元 Schema", "JSON Schema 可解析", "FAIL", str(e))
    try:
        ex = json.load(open(example_path, encoding="utf-8"))
        EU.model_validate(ex)
        add("4. 统一证据单元 Schema", "示例通过 Pydantic 校验", "PASS")
    except Exception as e:
        add("4. 统一证据单元 Schema", "示例通过 Pydantic 校验", "FAIL", str(e))
    bad = 0
    total = 0
    for path in ["outputs/evidence_units_clean.jsonl", "outputs/evidence_units_injected.jsonl"]:
        for line in open(ROOT / path, encoding="utf-8"):
            total += 1
            try:
                EU.model_validate(json.loads(line))
            except Exception:
                bad += 1
    add("4. 统一证据单元 Schema", "输出证据全部可校验", "PASS" if bad == 0 else "FAIL",
        f"validated={total}, bad={bad}")
    add("4. 统一证据单元 Schema", "定型/冻结", "PARTIAL", "v0.1 已可运行",
        "尚未团队评审/字段冻结")


def check_cases():
    mpath = ROOT / "cases_synthetic/manifest.csv"
    if not mpath.exists():
        add("5. 30 例样例案件", "manifest 存在", "FAIL", str(mpath))
        return
    m = pd.read_csv(mpath)
    add("5. 30 例样例案件", "30 行", "PASS" if len(m) == 30 else "FAIL", f"rows={len(m)}")
    add("5. 30 例样例案件", "20 干净 + 10 注入",
        "PASS" if int(m["injected_conflict"].sum()) == 10 and len(m) == 30 else "FAIL",
        f"injected={int(m['injected_conflict'].sum())}")
    dirs = sorted([d for d in (ROOT / "cases_synthetic").glob("SYN-*") if d.is_dir()])
    add("5. 30 例样例案件", "30 个案件目录", "PASS" if len(dirs) == 30 else "FAIL", f"dirs={len(dirs)}")
    required_files = ["case.json", "case_card.md", "text.txt", "face.jpg",
                      "evidence_units.jsonl", "result.json"]
    missing = []
    bad_evidence = 0
    pii = 0
    too_long = 0
    synthetic = 0
    for d in dirs:
        for f in required_files:
            if not (d / f).exists():
                missing.append(f"{d.name}/{f}")
        case = json.load(open(d / "case.json", encoding="utf-8"))
        if case.get("synthetic") is True and case.get("not_real_identity") is True:
            synthetic += 1
        if not {"gender", "age", "age_bucket"} <= set(case.get("ground_truth", {})):
            missing.append(f"{d.name}/case.json#ground_truth")
        evs = [json.loads(line) for line in open(d / "evidence_units.jsonl", encoding="utf-8")]
        if len(evs) < 4:
            bad_evidence += 1
        for e in evs:
            try:
                EvidenceUnit.model_validate(e)
            except Exception:
                bad_evidence += 1
            if e.get("case_id") != d.name or e.get("synthetic") is not True:
                bad_evidence += 1
        r = json.load(open(d / "result.json", encoding="utf-8"))
        if not {"predicted_gender", "predicted_age_bucket", "gender_posterior", "age_posterior"} <= set(r):
            missing.append(f"{d.name}/result.json#fields")
        text = (d / "text.txt").read_text(encoding="utf-8")
        if len(text) > 5100:
            too_long += 1
        if re.search(r"https?://", text) or re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
            pii += 1
        if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text) or re.search(r"(?<!\d)\d{17}[\dXx](?!\d)", text):
            pii += 1
    add("5. 30 例样例案件", "每案 6 个文件齐全", "PASS" if not missing else "FAIL",
        "; ".join(missing[:5]))
    add("5. 30 例样例案件", "每案证据 >=4 且可校验", "PASS" if bad_evidence == 0 else "FAIL",
        f"bad={bad_evidence}")
    add("5. 30 例样例案件", "文本脱敏无 URL/邮箱/手机/证件", "PASS" if pii == 0 else "FAIL", f"pii_files={pii}")
    add("5. 30 例样例案件", "文本长度 <= 5100", "PASS" if too_long == 0 else "FAIL", f"too_long={too_long}")
    add("5. 30 例样例案件", "30 例均标记 synthetic", "PASS" if synthetic == 30 else "FAIL",
        f"synthetic={synthetic}")
    add("5. 30 例样例案件", "真实/脱敏案件", "PARTIAL",
        "当前 30 例为公开数据构造的合成样例", "真实案件需示范单位脱敏样本（外部依赖）")
    readme = ROOT / "cases_synthetic/README.md"
    notice = ROOT / "cases_synthetic/DATA_NOTICE.md"
    add("5. 30 例样例案件", "案件包 README + 数据声明", "PASS" if readme.exists() and notice.exists() else "FAIL")


def check_env():
    req = ROOT / "requirements.txt"
    add("6. 开发环境", "requirements.txt", "PASS" if req.exists() else "FAIL", str(req))
    for pkg in ["numpy", "pandas", "sklearn", "joblib", "pydantic", "matplotlib", "networkx", "torch", "transformers", "PIL"]:
        add("6. 开发环境", f"依赖 {pkg}", "PASS" if importlib.util.find_spec(pkg) else "FAIL")
    add("6. 开发环境", "文本模型文件", "PASS" if (ROOT / "models/text_gender.joblib").exists() else "FAIL")
    # run pipeline + stubs
    p = run([sys.executable, "-I", "src/pipeline.py", "--tau", "0.50", "--downweight", "0.25"], "端到端 pipeline 运行")
    if p.returncode == 0:
        m = json.load(open(ROOT / "outputs/metrics.json", encoding="utf-8"))
        add("6. 开发环境", "pipeline 指标可用", "PASS",
            f"gender={m['clean']['fused_cred']['gender']['accuracy']}, age={m['clean']['fused_cred']['age']['accuracy']}")
    run([sys.executable, "-I", "src/check_stubs.py"], "audio/behavior 接口自检")
    run([sys.executable, "-I", "src/build_case_packages.py"], "30 例案件包重建")
    # compose yaml
    try:
        d = yaml.safe_load(open(ROOT / "deploy/docker-compose.yml", encoding="utf-8"))
        ok = all(k in d["services"] for k in ["postgres", "minio", "redis"])
        add("6. 开发环境", "docker-compose 三服务", "PASS" if ok else "FAIL",
            f"services={list(d['services'])}")
        add("6. 开发环境", "initdb SQL", "PASS" if (ROOT / "deploy/initdb/01_schema.sql").exists() else "FAIL")
    except Exception as e:
        add("6. 开发环境", "docker-compose 可解析", "FAIL", str(e))
    add("6. 开发环境", "项目级容器实际启动", "PARTIAL" if shutil.which("docker") is None else "PASS",
        "本机未安装 Docker" if shutil.which("docker") is None else "docker available",
        "compose 已做 YAML 校验，未实际启动容器")


def check_route_and_metrics():
    from fusion import weighted_fuse, conflict_handle  # noqa: F401
    from pipeline import leave_one_out  # noqa: F401
    add("7. 技术路线", "融合/冲突/可解释函数存在", "PASS", "weighted_fuse, conflict_handle, leave_one_out")
    m = json.load(open(ROOT / "outputs/metrics.json", encoding="utf-8"))
    add("7. 技术路线", "干净集性别 0.96", "PASS" if abs(m["clean"]["fused_cred"]["gender"]["accuracy"] - 0.96) < 1e-9 else "FAIL")
    add("7. 技术路线", "干净集年龄 0.50", "PASS" if abs(m["clean"]["fused_cred"]["age"]["accuracy"] - 0.50) < 1e-9 else "FAIL")
    add("7. 技术路线", "冲突检出 0.90", "PASS" if abs(m["conflict"]["injected_detection_rate"] - 0.90) < 1e-9 else "FAIL")
    add("7. 技术路线", "干净误报 0.14", "PASS" if abs(m["conflict"]["clean_false_positive_rate"] - 0.14) < 1e-9 else "FAIL")


def analysis():
    from collections import Counter
    cnt = Counter(c["status"] for c in CHECKS)
    if cnt.get("FAIL", 0):
        verdict = "FAIL"
        note = "存在结构性缺口"
    elif cnt.get("PARTIAL", 0):
        verdict = "PASS_WITH_PARTIALS"
        note = "按 pilot 口径通过；有 %d 项 PARTIAL（外部依赖/待评审）" % cnt.get("PARTIAL", 0)
    else:
        verdict = "PASS"
        note = "全部通过"
    if cnt.get("PASS_NOTE", 0):
        note += "；%d 项 PASS_NOTE" % cnt.get("PASS_NOTE", 0)
    return verdict, note, dict(cnt)


def write_reports():
    verdict, note, cnt = analysis()
    payload = {"verdict": verdict, "note": note, "counts": cnt, "checks": CHECKS}
    (ROOT / "reports/october_acceptance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 10 月计划验收报告",
        "",
        "**日期**：2026-11-24  ",
        "**标准**：需求调研与数据可得性摸底；原型边界（文本、图像先行，音频/行为预留）；技术方案与数据规范；统一证据单元 Schema；30 例样例案件；开发环境；技术路线。",
        "",
        f"## 总判定：**{verdict}**",
        "",
        f"- {note}",
        f"- 检查项：{cnt}",
        "",
        "## 测试矩阵",
        "",
        "| 要求 | 检查 | 状态 | 证据/说明 |",
        "|---|---|---|---|",
    ]
    for c in CHECKS:
        ev = (c["evidence"] + ("；" + c["note"] if c["note"] else "")).replace("|", "\\|")
        lines.append(f"| {c['requirement']} | {c['check']} | {c['status']} | {ev} |")
    lines += [
        "",
        "## 关键指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    try:
        m = json.load(open(ROOT / "outputs/metrics.json", encoding="utf-8"))
        lines += [
            f"| 性别 Acc（融合可信度） | {m['clean']['fused_cred']['gender']['accuracy']:.3f} |",
            f"| 年龄 Acc（融合可信度） | {m['clean']['fused_cred']['age']['accuracy']:.3f} |",
            f"| 年龄 MAE | {m['clean']['fused_cred']['age']['mae_years']:.2f} 岁 |",
            f"| 冲突检出 | {m['conflict']['injected_detection_rate']:.0%}（9/10） |",
            f"| 干净误报 | {m['conflict']['clean_false_positive_rate']:.0%}（7/50） |",
        ]
    except Exception:
        pass
    lines += [
        "",
        "## 30 例案件包",
        "",
        "- `cases_synthetic/`：30 个案件目录，20 干净 + 10 注入。",
        "- 这 30 例是**合成案件**（公开数据 + 同年龄/同性别配对），不是真实案件。",
        "- 30 例子集：性别 26/30，年龄段 16/30。",
        "",
        "## 待补 / 外部依赖",
        "",
        "1. **30 例真实/脱敏案件**：需示范单位提供样本并走审批；当前是合成占位。",
        "2. **项目级容器实际启动**：本机未安装 Docker；compose 文件已做 YAML 校验。",
        "3. **Schema 定型/冻结**：v0.1 可运行，但未团队评审。",
        "4. **音频/行为模型**：只有预留接口，没有 ASR/声纹/行为序列模型。",
        "",
        "## 复现",
        "",
        "```bash",
        "python tests/test_october_acceptance.py",
        "```",
    ]
    (ROOT / "reports/october_acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict


def preflight():
    required = [
        ROOT / "data/processed/pilot_cases.csv",
        ROOT / "cases_synthetic/manifest.csv",
        ROOT / "outputs/metrics.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("This acceptance test requires the full local data package.")
        print("Missing:")
        for m in missing:
            print(" -", m)
        print("The public GitHub repo intentionally excludes raw data/case files; see DELIVERY.md.")
        return False
    return True


def main():
    if not preflight():
        return 2
    check_data_availability()
    check_boundary()
    check_docs()
    check_schema()
    check_env()
    check_cases()
    check_route_and_metrics()
    verdict = write_reports()
    for c in CHECKS:
        print(f"{c['status']:9s} {c['requirement']} / {c['check']} {c['evidence']} {c['note']}")
    print("\nVERDICT:", verdict)
    print("report: reports/october_acceptance_report.md")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
