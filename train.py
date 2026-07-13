# -*- coding: utf-8 -*-
"""Train UAV/bird/clutter radar classifier."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import (
        DEFAULT_CLASS_NAMES,
        DEFAULT_META_FEATURE_NAMES,
        META_DIM,
        META_FEATURE_NAMES,
        RadarSample,
        RadarSampleDataset,
        build_label_mapping,
        canonical_label,
        collate_radar_batch,
        load_records,
        resolve_meta_feature_indices,
    )
    from .metrics import classification_metrics
    from .model import RadarThreeBranchNet
    from .target_types import format_target_type
except ImportError:
    from dataset import (  # type: ignore
        DEFAULT_CLASS_NAMES,
        DEFAULT_META_FEATURE_NAMES,
        META_DIM,
        META_FEATURE_NAMES,
        RadarSample,
        RadarSampleDataset,
        build_label_mapping,
        canonical_label,
        collate_radar_batch,
        load_records,
        resolve_meta_feature_indices,
    )
    from metrics import classification_metrics  # type: ignore
    from model import RadarThreeBranchNet  # type: ignore
    from target_types import format_target_type  # type: ignore


@dataclass
class TrainConfig:
    data: str = ""
    train_data: str = ""
    val_data: str = ""
    test_data: str = ""
    out_dir: str = "runs/radar_three_cls"
    class_names: str = "uav,bird,clutter"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    grad_clip: float = 5.0
    class_weight: bool = True
    seed: int = 42
    device: str = "auto"
    meta_features: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def stratified_split(
    records: Sequence[RadarSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[RadarSample], list[RadarSample], list[RadarSample]]:
    rng = random.Random(seed)
    groups: dict[object, list[RadarSample]] = {}
    for rec in records:
        groups.setdefault(canonical_label(rec.label), []).append(rec)

    train, val, test = [], [], []
    for _, items in sorted(groups.items(), key=lambda x: str(x[0])):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = max(1, min(n_test, n - 2))
            n_val = max(1, min(n_val, n - n_test - 1))
        else:
            n_test = 0
            n_val = 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def stratified_two_split(
    records: Sequence[RadarSample],
    first_ratio: float,
    seed: int,
) -> tuple[list[RadarSample], list[RadarSample]]:
    rng = random.Random(seed)
    groups: dict[object, list[RadarSample]] = {}
    for rec in records:
        groups.setdefault(canonical_label(rec.label), []).append(rec)
    first, second = [], []
    for _, items in sorted(groups.items(), key=lambda x: str(x[0])):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_first = int(round(n * first_ratio))
        if n >= 2:
            n_first = max(1, min(n_first, n - 1))
        first.extend(items[:n_first])
        second.extend(items[n_first:])
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def class_weight_tensor(records: Sequence[RadarSample], label_to_id: dict, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(label_to_id), dtype=np.float32)
    for rec in records:
        counts[int(label_to_id[canonical_label(rec.label)])] += 1.0
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true, y_pred = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch["label"]
        logits = model(batch)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * int(labels.numel())
        total_count += int(labels.numel())
        pred = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, len(class_names), class_names)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    class_names = tuple(x.strip() for x in cfg.class_names.split(",") if x.strip()) or DEFAULT_CLASS_NAMES
    meta_selection = cfg.meta_features.strip() if isinstance(cfg.meta_features, str) else cfg.meta_features
    meta_indices, meta_feature_names = resolve_meta_feature_indices(meta_selection)
    if cfg.train_data:
        train_records = load_records(cfg.train_data, require_label=True)
        val_records = load_records(cfg.val_data, require_label=True) if cfg.val_data else []
        test_records = load_records(cfg.test_data, require_label=True) if cfg.test_data else []
        if not val_records and not test_records:
            train_records, val_records, test_records = stratified_split(train_records, cfg.val_ratio, cfg.test_ratio, cfg.seed)
        elif not val_records:
            train_records, val_records = stratified_two_split(train_records, max(0.0, min(1.0, 1.0 - cfg.val_ratio)), cfg.seed)
        elif not test_records:
            val_records, test_records = stratified_two_split(val_records, 0.5, cfg.seed)
    else:
        if not cfg.data:
            raise ValueError("provide --data or --train-data")
        records = load_records(cfg.data, require_label=True)
        train_records, val_records, test_records = stratified_split(records, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    all_labels = [r.label for r in train_records + val_records + test_records if r.label is not None]
    label_to_id = build_label_mapping(all_labels, class_names=class_names)
    inverse_label = {int(v): k for k, v in label_to_id.items()}
    id_to_label = {}
    for i in range(len(inverse_label)):
        raw_label = inverse_label[i]
        if isinstance(raw_label, (int, np.integer)) and int(raw_label) == i and 0 <= i < len(class_names):
            id_to_label[i] = str(class_names[int(raw_label)])
        else:
            id_to_label[i] = format_target_type(raw_label, show_code=True)
    ordered_names = [id_to_label[i] for i in range(len(id_to_label))]

    train_ds = RadarSampleDataset(train_records, label_to_id, meta_features=meta_feature_names)
    val_ds = RadarSampleDataset(val_records, label_to_id, meta_features=meta_feature_names)
    test_ds = RadarSampleDataset(test_records, label_to_id, meta_features=meta_feature_names)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_radar_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_radar_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_radar_batch)

    model = RadarThreeBranchNet(num_classes=len(label_to_id), meta_dim=len(meta_feature_names), dropout=cfg.dropout).to(device)
    weight = class_weight_tensor(train_records, label_to_id, device) if cfg.class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"[Init] device={device} classes={ordered_names}", flush=True)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    print(f"[Meta] dim={len(meta_feature_names)} features={meta_feature_names}", flush=True)

    best_val_acc = -1.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total_count, correct = 0.0, 0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            labels = batch["label"]
            logits = model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * int(labels.numel())
            total_count += int(labels.numel())
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())

        train_loss = total_loss / max(total_count, 1)
        train_acc = correct / max(total_count, 1)
        val_metrics = evaluate(model, val_loader, criterion, device, ordered_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.5f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['macro_precision']:.4f} val_recall={val_metrics['macro_recall']:.4f}",
            flush=True,
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
                    "id_to_label": {str(k): str(v) for k, v in id_to_label.items()},
                    "meta_dim": len(meta_feature_names),
                    "meta_feature_names": list(meta_feature_names),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pth",
            )

    test_metrics = evaluate(model, test_loader, criterion, device, ordered_names)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
            "id_to_label": {str(k): str(v) for k, v in id_to_label.items()},
            "meta_dim": len(meta_feature_names),
            "meta_feature_names": list(meta_feature_names),
            "config": asdict(cfg),
            "epoch": cfg.epochs,
            "test_metrics": test_metrics,
        },
        out_dir / "last.pth",
    )
    summary = {
        "class_names": ordered_names,
        "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
        "meta_features": list(meta_feature_names),
        "best_val_accuracy": best_val_acc,
        "test": test_metrics,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Test]", json.dumps(test_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[Save] {out_dir / 'best.pth'}", flush=True)
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train radar UAV/bird/clutter classifier")
    parser.add_argument("--data", default="", help="single .npz/.db file or directory, auto split")
    parser.add_argument("--train-data", default="", help="train .npz/.db file or directory")
    parser.add_argument("--val-data", default="", help="val .npz/.db file or directory")
    parser.add_argument("--test-data", default="", help="test .npz/.db file or directory")
    parser.add_argument("--out-dir", default="runs/radar_three_cls")
    parser.add_argument("--class-names", default="uav,bird,clutter")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--class-weight", dest="class_weight", action="store_true", default=True)
    parser.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--meta-features",
        default="",
        help=(
            "comma separated meta feature names/indices; empty uses all. "
            f"Default: {','.join(DEFAULT_META_FEATURE_NAMES)}. Available: {','.join(META_FEATURE_NAMES)}"
        ),
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
