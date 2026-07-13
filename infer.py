# -*- coding: utf-8 -*-
"""Run inference with a trained radar classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import META_DIM, RadarSampleDataset, collate_radar_batch, load_records, resolve_meta_feature_indices
    from .model import RadarThreeBranchNet
    from .train import move_batch, resolve_device
except ImportError:
    from dataset import META_DIM, RadarSampleDataset, collate_radar_batch, load_records, resolve_meta_feature_indices  # type: ignore
    from model import RadarThreeBranchNet  # type: ignore
    from train import move_batch, resolve_device  # type: ignore


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt_path = str(Path(path).expanduser().resolve())
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    id_to_label_raw = ckpt.get("id_to_label", {})
    id_to_label = {int(k): str(v) for k, v in id_to_label_raw.items()}
    num_classes = len(id_to_label)
    meta_feature_names = ckpt.get("meta_feature_names")
    if meta_feature_names:
        _, meta_feature_names = resolve_meta_feature_indices(meta_feature_names)
    meta_dim = int(ckpt.get("meta_dim", len(meta_feature_names) if meta_feature_names else META_DIM))
    model = RadarThreeBranchNet(num_classes=num_classes, meta_dim=meta_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, id_to_label, meta_feature_names


@torch.no_grad()
def infer(checkpoint: str, input_path: str, device_name: str = "auto", batch_size: int = 32) -> list[dict]:
    device = resolve_device(device_name)
    model, id_to_label, meta_feature_names = load_checkpoint(checkpoint, device)
    records = load_records(input_path, require_label=False)
    for rec in records:
        rec.label = None
    dataset = RadarSampleDataset(records, label_to_id={}, meta_features=meta_feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_radar_batch)
    outputs: list[dict] = []
    for batch in loader:
        sample_ids = batch["sample_id"]
        batch = move_batch(batch, device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        scores, preds = torch.max(probs, dim=1)
        for i, sample_id in enumerate(sample_ids):
            prob_map = {
                id_to_label[j]: float(probs[i, j].detach().cpu().item())
                for j in range(len(id_to_label))
            }
            pred_id = int(preds[i].detach().cpu().item())
            outputs.append(
                {
                    "sample_id": sample_id,
                    "pred_id": pred_id,
                    "pred_label": id_to_label[pred_id],
                    "score": float(scores[i].detach().cpu().item()),
                    "probabilities": prob_map,
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer radar target class")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npz/.db file or directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="", help="optional json output path")
    args = parser.parse_args()
    result = infer(args.checkpoint, args.input, args.device, args.batch_size)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
