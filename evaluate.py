"""Evaluate a distilled student (plain ResNet-18, no adapters or teacher needed).

Usage:
    python evaluate.py --config configs/aptos.yaml --ckpt runs/aptos/cfcot_.../student_resnet18.pth
"""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from cfcot import StagedResNet, evaluate_predictions
from cfcot.data import CSVImageDataset

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--fold", type=int, default=None)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
fold = args.fold if args.fold is not None else cfg.get("fold", 0)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = StagedResNet(cfg["student_arch"], cfg["num_classes"], pretrained=False).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev)); model.eval()
loader = DataLoader(CSVImageDataset(cfg["val_csv"].format(fold=fold), cfg["image_root"], False),
                    batch_size=64, shuffle=False, num_workers=4)
yt, yp = [], []
with torch.no_grad():
    for x, y in loader:
        logits, _ = model(x.to(dev))
        yp += logits.argmax(1).cpu().tolist(); yt += y.tolist()
print(evaluate_predictions(yt, yp, cfg["num_classes"]))
print("params:", sum(p.numel() for p in model.parameters()))
