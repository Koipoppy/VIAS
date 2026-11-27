# -*- coding: utf-8 -*-
"""PAN18 joint text+image fusion: P1a embedding concat, with controls.

Stages (run in order):
  image     fit image-only logistic regression on frozen image embeddings
  text      fine-tune XLM-R text-only classifier
  concat    fine-tune XLM-R + frozen image embedding -> projection -> concat -> MLP
  late      calibrate/stack the text and image posteriors on the calib split
  evaluate  metrics, McNemar, bootstrap CI, per-language tables

The official PAN18 test split is only touched in `evaluate`.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

TEXT_MODEL = "FacebookAI/xlm-roberta-base"
LABELS = {"male": 0, "female": 1}


# --------------------------------------------------------------------------- data
def load_data(data_dir: Path, max_len: int, chunks: int = 1
              ) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray]:
    manifest = pd.read_csv(data_dir / "manifest_joint.csv")
    images = np.load(data_dir / "image_features.npz")
    token_path = data_dir / f"tokens_{max_len}_c{chunks}.npz"
    input_ids = attention_mask = None
    if token_path.exists():
        tokens = np.load(token_path)
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
        if len(manifest) != len(input_ids):
            raise SystemExit("manifest/token rows are not aligned")
    if len(manifest) != len(images["emb"]):
        raise SystemExit("manifest/image rows are not aligned")
    return manifest, input_ids, attention_mask, images


def split_index(manifest: pd.DataFrame, split: str) -> np.ndarray:
    return np.flatnonzero((manifest["split"] == split).to_numpy())


def labels_of(manifest: pd.DataFrame) -> np.ndarray:
    return manifest["gender"].map(LABELS).to_numpy(dtype=np.int64)


# ------------------------------------------------------------------------ metrics
def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        z = logits / t
        q = 1.0 / (1.0 + np.exp(-z))
        return float(-np.mean(y * np.log(np.clip(q, 1e-9, 1)) +
                              (1 - y) * np.log(np.clip(1 - q, 1e-9, 1))))

    res = minimize_scalar(nll, bounds=(-2.0, 3.0), method="bounded")
    return float(math.exp(res.x))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece_score(y, p),
    }


def mcnemar(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    pred_a, pred_b = (p_a >= 0.5).astype(int), (p_b >= 0.5).astype(int)
    b = int(np.sum((pred_a == y) & (pred_b != y)))
    c = int(np.sum((pred_a != y) & (pred_b == y)))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    res = stats.binomtest(min(b, c), b + c, 0.5)
    return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(res.pvalue)}


def bootstrap_delta(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray,
                    n_boot: int = 5000, seed: int = 20261127) -> dict:
    rng = np.random.default_rng(seed)
    base = ((p_base >= 0.5).astype(int) == y).astype(float)
    new = ((p_new >= 0.5).astype(int) == y).astype(float)
    delta = new - base
    n = len(delta)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = delta[idx].mean()
    return {
        "delta": float(delta.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


# ------------------------------------------------------------------------- models
class TextClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def encode(self, input_ids, attention_mask):
        b, k, length = input_ids.shape
        out = self.encoder(input_ids=input_ids.reshape(b * k, length),
                           attention_mask=attention_mask.reshape(b * k, length))
        h = out.last_hidden_state[:, 0].view(b, k, -1)
        valid = (attention_mask.sum(-1) > 0).to(h.dtype)
        weights = valid / valid.sum(1, keepdim=True).clamp(min=1.0)
        return (h * weights.unsqueeze(-1)).sum(1)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.dropout(self.encode(input_ids, attention_mask))), None


class ConcatClassifier(nn.Module):
    """Early fusion: project text/image embeddings, concatenate, MLP."""

    def __init__(self, model_name: str, image_dim: int, hidden: int = 256,
                 dropout: float = 0.2, gated: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        text_dim = self.encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim + 1, hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.image_norm = nn.LayerNorm(hidden)
        self.gated = gated
        if gated:
            self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode(self, input_ids, attention_mask):
        b, k, length = input_ids.shape
        out = self.encoder(input_ids=input_ids.reshape(b * k, length),
                           attention_mask=attention_mask.reshape(b * k, length))
        h = out.last_hidden_state[:, 0].view(b, k, -1)
        valid = (attention_mask.sum(-1) > 0).to(h.dtype)
        weights = valid / valid.sum(1, keepdim=True).clamp(min=1.0)
        return (h * weights.unsqueeze(-1)).sum(1)

    def forward(self, input_ids, attention_mask, image_emb, image_valid):
        h_text = self.encode(input_ids, attention_mask)
        img = torch.cat([image_emb, image_valid.unsqueeze(1)], dim=1)
        h_text = self.text_norm(self.text_proj(h_text))
        h_img = self.image_norm(self.image_proj(img))
        if self.gated:
            h_img = h_img * self.gate(torch.cat([h_text, h_img], dim=1))
        return self.head(torch.cat([h_text, h_img], dim=1))


class CrossAttentionClassifier(nn.Module):
    """Representation-level fusion: text tokens attend to per-image tokens.

    The text encoder is fine-tuned exactly like the concat baseline; the image
    encoder stays frozen and only its per-image CLS tokens are consumed here.
    """

    def __init__(self, model_name: str, image_dim: int, hidden: int = 256,
                 layers: int = 2, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        text_dim = self.encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim + 1, hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.image_norm = nn.LayerNorm(hidden)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.null_image = nn.Parameter(torch.zeros(1, 1, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode_tokens(self, input_ids, attention_mask):
        b, k, length = input_ids.shape
        out = self.encoder(input_ids=input_ids.reshape(b * k, length),
                           attention_mask=attention_mask.reshape(b * k, length))
        h = out.last_hidden_state.view(b, k * length, -1)
        mask = attention_mask.reshape(b, k * length).bool()
        return h, mask

    def forward(self, input_ids, attention_mask, image_tokens, image_mask):
        h_text, text_mask = self.encode_tokens(input_ids, attention_mask)
        h_text = self.text_norm(self.text_proj(h_text))

        valid = image_mask.to(image_tokens.dtype).unsqueeze(-1)
        h_img = self.image_norm(self.image_proj(torch.cat([image_tokens, valid], dim=-1)))
        # Always keep one valid key so authors without images do not produce NaNs.
        null = self.null_image.expand(h_img.shape[0], -1, -1)
        h_img = torch.cat([null, h_img], dim=1)
        img_keep = torch.cat(
            [torch.ones(image_mask.shape[0], 1, dtype=torch.bool, device=image_mask.device),
             image_mask.bool()], dim=1)

        h_text = self.decoder(
            h_text, h_img,
            tgt_key_padding_mask=~text_mask,
            memory_key_padding_mask=~img_keep,
        )
        weights = text_mask.to(h_text.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (h_text * weights.unsqueeze(-1)).sum(1)
        return self.head(pooled), None


@dataclass
class Tensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_emb: torch.Tensor
    image_valid: torch.Tensor
    y: torch.Tensor
    image_tokens: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None


def build_tensors(manifest, input_ids, attention_mask, images, idx, seed=None,
                  img_mean=None, img_std=None, per_image=None, per_image_mask=None):
    emb = images["emb"][idx].astype(np.float32)
    if img_mean is not None and img_std is not None:
        emb = (emb - img_mean) / img_std
    valid = images["valid"][idx].astype(np.float32)
    perm = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx))
        emb = emb[perm]
    image_tokens = image_mask = None
    if per_image is not None:
        image_tokens = per_image[idx].astype(np.float32)
        image_mask = per_image_mask[idx].astype(bool)
        if perm is not None:
            image_tokens = image_tokens[perm]
            image_mask = image_mask[perm]
    return Tensors(
        input_ids=torch.from_numpy(input_ids[idx].astype(np.int64)),
        attention_mask=torch.from_numpy(attention_mask[idx].astype(np.int64)),
        image_emb=torch.from_numpy(emb),
        image_valid=torch.from_numpy(valid),
        y=torch.from_numpy(labels_of(manifest)[idx]),
        image_tokens=(torch.from_numpy(image_tokens) if image_tokens is not None else None),
        image_mask=(torch.from_numpy(image_mask) if image_mask is not None else None),
    )


def iterate(t: Tensors, batch_size: int, shuffle: bool):
    n = len(t.y)
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch = {
            "input_ids": t.input_ids[idx],
            "attention_mask": t.attention_mask[idx],
            "image_emb": t.image_emb[idx],
            "image_valid": t.image_valid[idx],
            "y": t.y[idx],
        }
        if t.image_tokens is not None:
            batch["image_tokens"] = t.image_tokens[idx]
            batch["image_mask"] = t.image_mask[idx]
        yield batch


@torch.no_grad()
def predict(model, t: Tensors, batch_size: int = 64, use_images: bool = False,
            use_image_tokens: bool = False) -> np.ndarray:
    model.eval()
    probs = []
    for batch in iterate(t, batch_size, shuffle=False):
        ids = batch["input_ids"].cuda(non_blocking=True)
        mask = batch["attention_mask"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if use_image_tokens:
                logits, _ = model(ids, mask,
                                  batch["image_tokens"].cuda(non_blocking=True),
                                  batch["image_mask"].cuda(non_blocking=True))
            elif use_images:
                logits = model(ids, mask, batch["image_emb"].cuda(non_blocking=True),
                               batch["image_valid"].cuda(non_blocking=True))
            else:
                logits, _ = model(ids, mask)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def train_model(kind: str, data_dir: Path, model_dir: Path, out_dir: Path, epochs: int,
                batch_size: int, lr_encoder: float, lr_head: float, max_len: int, seed: int,
                shuffle_images: bool = False, limit: int | None = None,
                chunks: int = 1, image_dropout: float = 0.0, dropout: float = 0.2,
                image_pca: int = 0, image_prob_feature: bool = False,
                init_from: str | None = None, gated: bool = False) -> None:
    manifest, input_ids, attention_mask, images = load_data(data_dir, max_len, chunks)
    if input_ids is None:
        raise SystemExit(f"missing token cache for chunks={chunks}")
    per_image = per_image_mask = None
    if kind == "crossattn":
        token_path = data_dir / "image_tokens.npz"
        if not token_path.exists():
            raise SystemExit(f"missing per-image cache: {token_path}")
        tokens = np.load(token_path)
        per_image, per_image_mask = tokens["emb_images"], tokens["image_mask"]
    if image_prob_feature:
        images = {"emb": np.concatenate(
                      [images["emb"].astype(np.float32),
                       images["prob_female"].astype(np.float32)[:, None]], axis=1),
                  "valid": images["valid"]}
    fit_idx = split_index(manifest, "fit")
    calib_idx = split_index(manifest, "calib")
    if limit:
        fit_idx, calib_idx = fit_idx[:limit], calib_idx[:limit // 4]
    img_seed = seed if shuffle_images else None
    img_mean = img_std = None
    if kind == "concat":
        fit_emb = images["emb"][fit_idx].astype(np.float32)
        fit_valid = images["valid"][fit_idx].astype(bool)
        if fit_valid.any():
            img_mean = fit_emb[fit_valid].mean(0)
            img_std = fit_emb[fit_valid].std(0) + 1e-6
        if image_pca > 0:
            from sklearn.decomposition import PCA

            emb = (images["emb"].astype(np.float32) - img_mean) / img_std
            pca = PCA(n_components=image_pca, random_state=seed).fit(emb[fit_idx])
            images = {"emb": pca.transform(emb).astype(np.float32),
                      "valid": images["valid"]}
            img_mean = img_std = None
            model_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_dir / "concat_pca.npz",
                                components=pca.components_, mean=pca.mean_,
                                explained=pca.explained_variance_ratio_)
    t_fit = build_tensors(manifest, input_ids, attention_mask, images, fit_idx, img_seed,
                          img_mean, img_std, per_image, per_image_mask)
    t_cal = build_tensors(manifest, input_ids, attention_mask, images, calib_idx, img_seed,
                          img_mean, img_std, per_image, per_image_mask)

    torch.manual_seed(seed)
    np.random.seed(seed)
    image_dim = int(per_image.shape[2]) if kind == "crossattn" else images["emb"].shape[1]
    if kind == "concat":
        model = ConcatClassifier(TEXT_MODEL, image_dim, dropout=dropout, gated=gated).cuda()
    elif kind == "crossattn":
        model = CrossAttentionClassifier(TEXT_MODEL, image_dim, dropout=dropout).cuda()
    else:
        model = TextClassifier(TEXT_MODEL, dropout=dropout).cuda()
    if init_from and kind == "concat":
        ckpt = torch.load(init_from, map_location="cpu")
        enc = {k: v for k, v in ckpt["state"].items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        print(f"[concat] init encoder from {init_from}: "
              f"{len(enc)} tensors, missing={len(missing)}, unexpected={len(unexpected)}",
              flush=True)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)
    steps = math.ceil(len(fit_idx) / batch_size) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr_encoder, lr_head], total_steps=steps, pct_start=0.1)
    use_images = kind in {"concat", "crossattn"}
    use_image_tokens = kind == "crossattn"

    best = {"acc": -1.0, "ll": 1e9, "epoch": -1, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in iterate(t_fit, batch_size, shuffle=True):
            ids = batch["input_ids"].cuda(non_blocking=True)
            mask = batch["attention_mask"].cuda(non_blocking=True)
            y = batch["y"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_image_tokens:
                    logits, _ = model(
                        ids, mask,
                        batch["image_tokens"].cuda(non_blocking=True),
                        batch["image_mask"].cuda(non_blocking=True))
                elif use_images:
                    img = batch["image_emb"].cuda(non_blocking=True)
                    if image_dropout > 0:
                        keep = (torch.rand(img.shape[0], 1, device=img.device)
                                >= image_dropout).float()
                        img = img * keep
                    logits = model(ids, mask, img,
                                   batch["image_valid"].cuda(non_blocking=True))
                else:
                    logits, _ = model(ids, mask)
                loss = nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
        cal_p = predict(model, t_cal, use_images=use_images,
                        use_image_tokens=use_image_tokens)
        acc = float(accuracy_score(t_cal.y.numpy(), (cal_p >= 0.5).astype(int)))
        ll = float(log_loss(t_cal.y.numpy(), np.clip(cal_p, 1e-9, 1 - 1e-9), labels=[0, 1]))
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1),
                        "calib_accuracy": acc, "calib_log_loss": ll})
        print(f"[{kind}] epoch {epoch}/{epochs} loss {total / max(seen, 1):.4f} "
              f"calib_acc {acc:.4f} calib_ll {ll:.4f}", flush=True)
        if acc > best["acc"] + 1e-12 or (abs(acc - best["acc"]) <= 1e-12 and ll < best["ll"]):
            best = {"acc": acc, "ll": ll, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    name = f"{kind}{'_shuffled' if shuffle_images else ''}"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best["state"], "kind": kind, "image_dim": image_dim,
                "shuffle_images": shuffle_images, "epoch": best["epoch"]},
               model_dir / f"{name}.pt")

    all_idx = np.arange(len(manifest))
    t_all = build_tensors(manifest, input_ids, attention_mask, images, all_idx, img_seed,
                          img_mean, img_std, per_image, per_image_mask)
    p_all = predict(model, t_all, use_images=use_images,
                    use_image_tokens=use_image_tokens)
    probs = manifest[["author_id", "lang", "gender", "split"]].copy()
    probs["p_female"] = p_all
    probs["method"] = name
    probs.to_csv(out_dir / f"{name}_probs.csv", index=False)
    (out_dir / f"{name}_history.json").write_text(json.dumps({
        "config": {"epochs": epochs, "batch_size": batch_size, "lr_encoder": lr_encoder,
                   "lr_head": lr_head, "max_len": max_len, "chunks": chunks, "seed": seed,
                   "shuffle_images": shuffle_images, "limit": limit,
                   "per_image": kind == "crossattn"},
        "best_epoch": best["epoch"], "calib_accuracy": best["acc"], "history": history,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{kind}] saved {out_dir / f'{name}_probs.csv'} best_epoch={best['epoch']}",
          flush=True)


def text_tfidf(data_dir: Path, out_dir: Path, seed: int) -> None:
    """Char n-gram TF-IDF + LR on exactly the same fit/calib/test split."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    manifest = pd.read_csv(data_dir / "manifest_joint.csv")
    texts = pd.read_csv(data_dir / "texts.csv")
    mapping = {(r.author_id, r.lang): r.text for r in texts.itertuples(index=False)}
    all_text = [mapping.get((a, l), "") for a, l in zip(manifest["author_id"], manifest["lang"])]
    y = labels_of(manifest)
    fit_idx = split_index(manifest, "fit")
    p = np.full(len(manifest), 0.5, dtype=float)
    for lang in ["en", "es", "ar"]:
        lang_mask = (manifest["lang"] == lang).to_numpy()
        tr = np.intersect1d(fit_idx, np.flatnonzero(lang_mask))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=250000,
                              min_df=2, sublinear_tf=True, lowercase=True,
                              strip_accents="unicode")
        x_tr = vec.fit_transform([all_text[i] for i in tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0,
                                 random_state=seed)
        clf.fit(x_tr, y[tr])
        idx = np.flatnonzero(lang_mask)
        proba = clf.predict_proba(vec.transform([all_text[i] for i in idx]))
        female_col = list(clf.classes_).index(1)
        p[idx] = proba[:, female_col]
        print(f"[tfidf] {lang} fit={len(tr)} all={len(idx)}", flush=True)
    probs = manifest[["author_id", "lang", "gender", "split"]].copy()
    probs["p_female"] = p
    probs["method"] = "text_tfidf"
    probs.to_csv(out_dir / "text_tfidf_probs.csv", index=False)
    calib = (manifest["split"] == "calib").to_numpy()
    test = (manifest["split"] == "test").to_numpy()
    print("[tfidf] calib", json.dumps(binary_metrics(y[calib], p[calib])), flush=True)
    print("[tfidf] test", json.dumps(binary_metrics(y[test], p[test])), flush=True)


# ------------------------------------------------------------------- image / late
def image_only(data_dir: Path, out_dir: Path, seed: int) -> None:
    manifest, _, _, images = load_data(data_dir, 512)
    fit_idx = split_index(manifest, "fit")
    calib_idx = split_index(manifest, "calib")
    x = images["emb"].astype(np.float32)
    v = images["valid"].astype(np.float32)[:, None]
    x = np.concatenate([x, v], axis=1)
    mean, std = x[fit_idx].mean(0), x[fit_idx].std(0) + 1e-6
    xn = (x - mean) / std
    xn[~images["valid"]] = 0.0  # mean imputation for users without a usable image
    y = labels_of(manifest)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(xn[fit_idx], y[fit_idx])
    p = clf.predict_proba(xn)[:, 1]
    probs = manifest[["author_id", "lang", "gender", "split"]].copy()
    probs["p_female"] = p
    probs["method"] = "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    probs.to_csv(out_dir / "image_probs.csv", index=False)

    # Direct 10-image averaged ViT posterior, no learned head.
    p_vit = images["prob_female"].astype(np.float64)
    probs_vit = manifest[["author_id", "lang", "gender", "split"]].copy()
    probs_vit["p_female"] = p_vit
    probs_vit["method"] = "image_vit"
    probs_vit.to_csv(out_dir / "image_vit_probs.csv", index=False)

    cal = binary_metrics(y[calib_idx], p[calib_idx])
    cal_vit = binary_metrics(y[calib_idx], p_vit[calib_idx])
    test_idx = split_index(manifest, "test")
    print("[image] calib", json.dumps(cal), flush=True)
    print("[image] test", json.dumps(binary_metrics(y[test_idx], p[test_idx])), flush=True)
    print("[image_vit] calib", json.dumps(cal_vit), flush=True)
    print("[image_vit] test", json.dumps(binary_metrics(y[test_idx], p_vit[test_idx])),
          flush=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def late_fusion(data_dir: Path, out_dir: Path, seed: int) -> None:
    manifest, _, _, images = load_data(data_dir, 512)
    text = pd.read_csv(out_dir / "text_probs.csv")
    y = labels_of(manifest)
    calib = (manifest["split"] == "calib").to_numpy()
    # Pick the stronger image posterior on the calibration split.
    best_name, best_ll, image = None, 1e9, None
    for name in ["image", "image_vit"]:
        path = out_dir / f"{name}_probs.csv"
        if not path.exists():
            continue
        cand = pd.read_csv(path)
        ll = log_loss(y[calib], np.clip(cand["p_female"].to_numpy()[calib], 1e-9, 1 - 1e-9),
                      labels=[0, 1])
        if ll < best_ll:
            best_name, best_ll, image = name, ll, cand
    if image is None:
        raise SystemExit("no image predictions found")
    print(f"[late] using {best_name} (calib log loss {best_ll:.4f})", flush=True)
    if not (text["author_id"].equals(image["author_id"])):
        raise SystemExit("text/image prediction rows are not aligned")
    lt, li = _logit(text["p_female"].to_numpy()), _logit(image["p_female"].to_numpy())
    valid = images["valid"].astype(float)
    lang = pd.get_dummies(manifest["lang"]).to_numpy(dtype=float)
    feats = np.column_stack([lt, li, lt * li, valid, lang])
    stacker = LogisticRegression(max_iter=3000, C=0.5)
    stacker.fit(feats[calib], y[calib])
    p_stack = stacker.predict_proba(feats)[:, 1]
    # calibrated log-opinion pool: p ~ p_text^w * p_image^(1-w)
    grid = np.linspace(0.0, 1.0, 101)
    best_w, best_ll = 0.0, 1e9
    for w in grid:
        qf = np.exp(w * np.log(np.clip(text["p_female"], 1e-6, 1)) +
                    (1 - w) * np.log(np.clip(image["p_female"], 1e-6, 1)))
        qm = np.exp(w * np.log(np.clip(1 - text["p_female"], 1e-6, 1)) +
                    (1 - w) * np.log(np.clip(1 - image["p_female"], 1e-6, 1)))
        pf = qf / (qf + qm)
        ll = log_loss(y[calib], np.clip(pf[calib], 1e-9, 1 - 1e-9), labels=[0, 1])
        if ll < best_ll:
            best_w, best_ll = float(w), float(ll)
    qf = np.exp(best_w * np.log(np.clip(text["p_female"], 1e-6, 1)) +
                (1 - best_w) * np.log(np.clip(image["p_female"], 1e-6, 1)))
    qm = np.exp(best_w * np.log(np.clip(1 - text["p_female"], 1e-6, 1)) +
                (1 - best_w) * np.log(np.clip(1 - image["p_female"], 1e-6, 1)))
    p_pool = qf / (qf + qm)
    for name, p in [("late_stack", p_stack), ("late_logpool", p_pool)]:
        probs = manifest[["author_id", "lang", "gender", "split"]].copy()
        probs["p_female"] = p
        probs["method"] = name
        probs.to_csv(out_dir / f"{name}_probs.csv", index=False)
    print(f"[late] stacker coef {stacker.coef_.tolist()} intercept {stacker.intercept_.tolist()}")
    print(f"[late] logpool text weight {best_w:.2f}")


# --------------------------------------------------------------------- evaluate
METHODS = ["image", "image_vit", "text", "text_tfidf", "late_logpool", "late_stack",
           "concat", "concat_pure", "concat_init", "concat_prob", "concat_shuffled",
           "crossattn", "crossattn_ens"]


def add_tfidf_baseline(manifest: pd.DataFrame, out_dir: Path, preds: dict) -> None:
    """Attach the previously trained char n-gram TF-IDF text model as a reference."""
    base = out_dir.parent / "pan18"
    frames = []
    for name in ["text_predictions_calib.csv", "text_predictions_test.csv"]:
        path = base / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    mapping = {(r.author_id, r.lang): float(r.text_prob_female) for r in df.itertuples(index=False)}
    values = [mapping.get((a, l)) for a, l in zip(manifest["author_id"], manifest["lang"])]
    test_calib = (manifest["split"] != "fit").to_numpy()
    if all(v is not None for v, keep in zip(values, test_calib) if keep):
        preds["text_tfidf"] = np.array([0.5 if v is None else v for v in values])


def evaluate(data_dir: Path, out_dir: Path, n_boot: int) -> None:
    manifest, _, _, _ = load_data(data_dir, 512)
    y = labels_of(manifest)
    preds = {}
    for method in METHODS:
        path = out_dir / f"{method}_probs.csv"
        if path.exists():
            preds[method] = pd.read_csv(path)["p_female"].to_numpy()
    if "text" not in preds:
        raise SystemExit("text predictions are required")
    tfidf_path = out_dir / "text_tfidf_probs.csv"
    if tfidf_path.exists():
        preds["text_tfidf"] = pd.read_csv(tfidf_path)["p_female"].to_numpy()
    else:
        add_tfidf_baseline(manifest, out_dir, preds)
    test = (manifest["split"] == "test").to_numpy()
    calib = (manifest["split"] == "calib").to_numpy()

    results, temperatures = {}, {}
    for method, p in preds.items():
        t = fit_temperature(y[calib], p[calib])
        temperatures[method] = t
        pc = 1.0 / (1.0 + np.exp(-_logit(p) / t))
        results[method] = {"temperature": t, "test": binary_metrics(y[test], pc[test]),
                           "raw_test": binary_metrics(y[test], p[test]),
                           "calib": binary_metrics(y[calib], pc[calib])}
        results[method]["per_lang"] = {
            lang: binary_metrics(y[test & (manifest["lang"] == lang).to_numpy()],
                                 pc[test & (manifest["lang"] == lang).to_numpy()])
            for lang in ["en", "es", "ar"]
        }
        pred = (pc >= 0.5).astype(int)
        preds[method] = pc

    comparisons = {}
    base = preds["text"]
    for method in preds:
        if method == "text":
            continue
        row = {"vs_text": bootstrap_delta(y[test], base[test], preds[method][test], n_boot),
               "vs_text_mcnemar": mcnemar(y[test], base[test], preds[method][test])}
        if "text_tfidf" in preds:
            row["vs_text_tfidf"] = bootstrap_delta(y[test], preds["text_tfidf"][test],
                                                   preds[method][test], n_boot)
            row["vs_text_tfidf_mcnemar"] = mcnemar(y[test], preds["text_tfidf"][test],
                                                   preds[method][test])
        if "concat" in preds and method != "concat":
            row["vs_concat"] = bootstrap_delta(y[test], preds["concat"][test],
                                               preds[method][test], n_boot)
            row["vs_concat_mcnemar"] = mcnemar(y[test], preds["concat"][test],
                                               preds[method][test])
        comparisons[method] = row

    late_candidates = [k for k in ("late_logpool", "late_stack") if k in preds]
    best_late = max(late_candidates,
                    key=lambda k: results[k]["test"]["accuracy"]) if late_candidates else None
    if best_late:
        for method in preds:
            if method in (best_late, "text"):
                continue
            comparisons[method]["vs_best_late"] = bootstrap_delta(
                y[test], preds[best_late][test], preds[method][test], n_boot)
            comparisons[method]["vs_best_late_mcnemar"] = mcnemar(
                y[test], preds[best_late][test], preds[method][test])

    out = {"n": {"fit": int((manifest["split"] == "fit").sum()),
                 "calib": int(calib.sum()), "test": int(test.sum())},
           "best_late": best_late, "results": results, "comparisons": comparisons}
    (out_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    table = pd.DataFrame({
        m: {"acc": results[m]["test"]["accuracy"], "f1": results[m]["test"]["macro_f1"],
            "auc": results[m]["test"]["auc"], "ece": results[m]["test"]["ece"],
            "brier": results[m]["test"]["brier"]}
        for m in results}).T
    print(table.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(json.dumps({m: comparisons[m]["vs_text"] for m in comparisons}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["image", "text", "concat", "crossattn", "tfidf",
                                      "late", "evaluate"])
    ap.add_argument("--data", default="data/processed/pan18_joint")
    ap.add_argument("--out", default="outputs/pan18_joint")
    ap.add_argument("--model-dir", default="models/pan18_joint")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=4,
                    help="number of 512-token chunks per user")
    ap.add_argument("--image-dropout", type=float, default=0.0,
                    help="modality dropout for concat: zero image features with this probability")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--image-pca", type=int, default=0,
                    help="reduce image embeddings to this many PCA components (0 = off)")
    ap.add_argument("--image-prob-feature", action="store_true",
                    help="append the ViT image posterior as an extra image feature")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint whose encoder weights initialise the concat model")
    ap.add_argument("--gate", action="store_true",
                    help="learned per-user gate on the image branch")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--shuffle-images", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: cap fit rows (calib = limit/4)")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "image":
        image_only(data_dir, out_dir, args.seed)
    elif args.stage == "tfidf":
        text_tfidf(data_dir, out_dir, args.seed)
    elif args.stage in {"text", "concat", "crossattn"}:
        train_model(args.stage, data_dir, Path(args.model_dir), out_dir, args.epochs,
                    args.batch_size, args.lr_encoder, args.lr_head, args.max_len, args.seed,
                    args.shuffle_images, args.limit, args.chunks, args.image_dropout,
                    args.dropout, args.image_pca, args.image_prob_feature, args.init_from,
                    args.gate)
    elif args.stage == "late":
        late_fusion(data_dir, out_dir, args.seed)
    else:
        evaluate(data_dir, out_dir, args.n_boot)


if __name__ == "__main__":
    main()
