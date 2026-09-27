"""Train the ResNet-50 teacher with SORD cross-entropy, then freeze it for distillation.

Usage:
    python train_teacher.py --config configs/aptos.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfcot import StagedResNet, sord_ce, evaluate_predictions
from cfcot.data import CSVImageDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fold = args.fold if args.fold is not None else cfg.get("fold", 0)
    fmt = lambda s: s.format(fold=fold)

    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(CSVImageDataset(fmt(cfg["train_csv"]), cfg["image_root"], True),
                    batch_size=cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(CSVImageDataset(fmt(cfg["val_csv"]), cfg["image_root"], False),
                    batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    model = StagedResNet(cfg["teacher_arch"], cfg["num_classes"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best, ckpt = -1.0, fmt(cfg["teacher_ckpt"])
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in tqdm(tr, desc=f"teacher ep{ep}", leave=False):
            x, y = x.to(dev), y.to(dev)
            logits, _ = model(x)
            loss = sord_ce(logits, y, cfg["sord_lambda"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m = evaluate(model, va, dev, cfg["num_classes"])
        print(f"ep{ep} {m}")
        if m["OA"] > best:
            best = m["OA"]; torch.save(model.state_dict(), ckpt)
    print("best OA", best, "->", ckpt)


@torch.no_grad()
def evaluate(model, loader, dev, C):
    model.eval(); yt, yp = [], []
    for x, y in loader:
        logits, _ = model(x.to(dev))
        yp += logits.argmax(1).cpu().tolist(); yt += y.tolist()
    return evaluate_predictions(yt, yp, C)


if __name__ == "__main__":
    main()
