"""CF-CoT knowledge distillation: ResNet-50 (frozen) -> ResNet-18.

    L = L_CE^sord + L_KD + alpha * L_adapt + beta * L_CoT          (Eq. 1)

Usage:
    python train_student.py --config configs/aptos.yaml
    python train_student.py --config configs/aptos.yaml --beta 0      # w/o L_CoT
    python train_student.py --config configs/aptos.yaml --alpha 0     # w/o L_adapt
    python train_student.py --config configs/adience.yaml --fold 3

Adapters and the teacher are used only during training; the saved checkpoint
is a plain ResNet-18 (see evaluate.py).
"""
import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfcot import (Adapters, StagedResNet, adapt_loss, build_thoughts, cot_loss,
                   evaluate_predictions, freeze, gaussian_curriculum, kd_loss, sord_ce)
from cfcot.data import CSVImageDataset
from cfcot.metrics import coarse_accuracy


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="override L_adapt weight")
    ap.add_argument("--beta", type=float, default=None, help="override L_CoT weight")
    ap.add_argument("--no_kd", action="store_true", help="ablation: drop terminal L_KD")
    ap.add_argument("--no_sord", action="store_true", help="ablation: one-hot CE instead of SORD")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config))
    if args.alpha is not None: cfg["alpha"] = args.alpha
    if args.beta is not None: cfg["beta"] = args.beta
    if args.seed is not None: cfg["seed"] = args.seed
    fold = args.fold if args.fold is not None else cfg.get("fold", 0)
    fmt = lambda s: s.format(fold=fold)
    out = args.out or f"runs/{cfg['dataset']}/cfcot_a{cfg['alpha']}_b{cfg['beta']}_s{cfg['seed']}_f{fold}"
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = cfg["num_classes"]
    tr = DataLoader(CSVImageDataset(fmt(cfg["train_csv"]), cfg["image_root"], True),
                    batch_size=cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(CSVImageDataset(fmt(cfg["val_csv"]), cfg["image_root"], False),
                    batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    teacher = StagedResNet(cfg["teacher_arch"], C, pretrained=False).to(dev)
    teacher.load_state_dict(torch.load(fmt(cfg["teacher_ckpt"]), map_location=dev))
    freeze(teacher)
    student = StagedResNet(cfg["student_arch"], C).to(dev)
    adapters = Adapters(student.out_channels, teacher.out_channels).to(dev)

    params = list(student.parameters()) + list(adapters.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    bins, taus = cfg["bins"], cfg["taus"]
    total_steps = cfg["epochs"] * len(tr)
    step, best, log = 0, -1.0, []

    for ep in range(cfg["epochs"]):
        student.train(); adapters.train()
        for x, y in tqdm(tr, desc=f"cf-cot ep{ep}", leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                t_logits, t_feats = teacher(x)
            s_logits, s_feats = student(x)

            # label term
            if args.no_sord:
                l_ce = torch.nn.functional.cross_entropy(s_logits, y)
            else:
                l_ce = sord_ce(s_logits, y, cfg["sord_lambda"])
            # terminal KD
            l_kd = s_logits.new_zeros(()) if args.no_kd else kd_loss(s_logits, t_logits, cfg["kd_temperature"])

            # chain-of-thought: route student stages 1..3 through frozen teacher tail
            adapted = adapters(s_feats[:3])
            l_adapt = adapt_loss(adapted, t_feats[:3]) if cfg["alpha"] > 0 else s_logits.new_zeros(())
            if cfg["beta"] > 0:
                thoughts = build_thoughts(teacher, adapted)
                w = gaussian_curriculum(step / total_steps, cfg["curriculum_mu"], cfg["curriculum_sigma"])
                l_cot = cot_loss(thoughts, t_logits, bins, taus, w)
            else:
                l_cot = s_logits.new_zeros(())

            loss = l_ce + l_kd + cfg["alpha"] * l_adapt + cfg["beta"] * l_cot
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        sched.step()

        m = evaluate(student, va, dev, C)
        m["epoch"] = ep
        log.append(m); print(m)
        if m["OA"] > best:
            best = m["OA"]
            torch.save(student.state_dict(), os.path.join(out, "student_resnet18.pth"))
            torch.save(adapters.state_dict(), os.path.join(out, "adapters.pth"))
            # stage-wise coarse/fine accuracy of the thoughts (Fig. 3c)
            stage_acc = stage_dynamics(teacher, student, adapters, va, dev, bins)
            json.dump({"val": m, "stage_dynamics": stage_acc}, open(os.path.join(out, "best.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=2)
    print("best OA", best, "->", out)


@torch.no_grad()
def evaluate(model, loader, dev, C):
    model.eval(); yt, yp = [], []
    for x, y in loader:
        logits, _ = model(x.to(dev))
        yp += logits.argmax(1).cpu().tolist(); yt += y.tolist()
    return evaluate_predictions(yt, yp, C)


@torch.no_grad()
def stage_dynamics(teacher, student, adapters, loader, dev, bins):
    """Accuracy of z_1, z_2, z_3 and the final head under coarse (bins[0]) and fine evaluation."""
    student.eval(); adapters.eval()
    yt, preds = [], [[] for _ in range(4)]
    for x, y in loader:
        x = x.to(dev)
        s_logits, s_feats = student(x)
        thoughts = build_thoughts(teacher, adapters(s_feats[:3]))
        for k, z in enumerate(thoughts + [s_logits]):
            preds[k] += z.argmax(1).cpu().tolist()
        yt += y.tolist()
    coarse = bins[0]
    names = ["stage1", "stage2", "stage3", "final"]
    return {n: {"coarse": coarse_accuracy(yt, p, coarse),
                "fine": float((torch.tensor(yt) == torch.tensor(p)).float().mean() * 100)}
            for n, p in zip(names, preds)}


if __name__ == "__main__":
    main()
