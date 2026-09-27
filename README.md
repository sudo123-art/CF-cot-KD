# CF-CoT: Coarse-to-Fine Chain-of-Thought Knowledge Distillation for Visual Ordinal Rating

Official code for the paper *CF-CoT: Coarse-to-Fine Chain-of-Thought Knowledge Distillation for Visual Ordinal Rating* (submitted to ICASSP 2027).

CF-CoT distils the **decision process** of a frozen teacher, not only its final answer.
Lightweight 1×1 adapters route intermediate student features into the frozen teacher tail,
producing a chain of stage-wise predictions that are aligned to the teacher at progressively finer
label granularity (e.g. referable / non-referable → 4 bins → 5 DR grades) via symmetric
Jensen–Shannon divergence, under a Gaussian stage curriculum and ordinal-aware supervision.
All adapters and teacher weights are discarded at test time: the deployed model is a plain ResNet-18.

<p align="center"><img src="assets/framework.png" width="90%"></p>

## Results

**APTOS 2019** (ResNet-50 teacher → ResNet-18 student, grade-stratified 2,928/734 split, seed 42)

| Method | OA↑ | mAcc↑ | F1↑ | QWK↑ | MAE↓ |
|---|---|---|---|---|---|
| Teacher (ResNet-50) | 83.61 | 60.11 | 63.47 | 0.8861 | 0.2227 |
| Student (ResNet-18) | 81.83 | 60.26 | 62.88 | 0.8645 | 0.2500 |
| KD | 84.43 | 62.95 | 65.95 | 0.8980 | 0.2063 |
| SP | 85.25 | 64.04 | 66.72 | 0.8950 | 0.2022 |
| SDD | 85.25 | 64.57 | 68.05 | 0.8931 | 0.2049 |
| **CF-CoT (Ours)** | **85.52** | 65.27 | **68.70** | **0.9061** | **0.1940** |

**Adience** (five subject-exclusive folds, mean±std): CF-CoT reaches the best student OA, 47.23±4.12 (teacher 47.87±3.59).

Full tables and the component ablation are in the paper.

## Installation

```bash
git clone https://github.com/sudo123-art/CF-cot-KD.git
cd CF-cot-KD
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, a single GPU.

## Data

See [`data/README.md`](data/README.md) for the expected layout.

```bash
# APTOS 2019: build the grade-stratified 2,928 / 734 split used in the paper
python scripts/prepare_aptos.py --csv data/aptos/train.csv --out data/aptos --seed 42
```

For Adience, produce `fold{k}_train.csv` / `fold{k}_val.csv` (columns `path,label`, labels 0–7 for the
eight age groups) from the official subject-exclusive fold files.

## Training

```bash
# 1. teacher (ResNet-50, SORD cross-entropy), then frozen
python train_teacher.py --config configs/aptos.yaml

# 2. CF-CoT distillation (full model)
python train_student.py --config configs/aptos.yaml

# ablations (Table 3)
python train_student.py --config configs/aptos.yaml --beta 0            # w/o L_CoT
python train_student.py --config configs/aptos.yaml --alpha 0           # w/o L_adapt
python train_student.py --config configs/aptos.yaml --alpha 0 --beta 0  # KD + SORD
python train_student.py --config configs/aptos.yaml --no_kd             # w/o L_KD
python train_student.py --config configs/aptos.yaml --no_sord           # w/o SORD

# Adience, one fold at a time
python train_student.py --config configs/adience.yaml --fold 0
```

Each run writes `student_resnet18.pth`, `adapters.pth`, `log.json`, and `best.json` (which also
contains the coarse / fine accuracy of each intermediate thought, used for Fig. 3(c)) under `runs/`.

## Evaluation

```bash
python evaluate.py --config configs/aptos.yaml --ckpt runs/aptos/<run>/student_resnet18.pth
```

Inference uses the unmodified ResNet-18 only; no adapter or teacher is loaded.

## Hyper-parameters

| | APTOS | Adience |
|---|---|---|
| bins (coarse → fine) | 2 / 4 / 5 | 3 / 5 / 8 |
| stage temperatures τ | (2, 3, 4) | (2, 3, 4) |
| curriculum μ, σ | (0.1, 0.4, 0.8), 0.2 | same |
| α (L_adapt), β (L_CoT) | 0.5, 1.0 | 0.6, 0.9 |
| KD temperature, SORD λ | 4, 1 | 4, 1 |
| optimiser | AdamW, lr 1e-4, wd 1e-5, batch 32, cosine, 50 epochs | same |

All values are in `configs/*.yaml`.

## Code structure

```
cfcot/losses.py    SORD-CE, KD, pooled-softmax JS chain-of-thought loss, adapter loss, Gaussian curriculum
cfcot/models.py    StagedResNet (stage features + teacher-tail routing), 1x1 adapters
cfcot/metrics.py   OA / mAcc / macro-F1 / QWK / MAE, coarse-bin accuracy
cfcot/data.py      CSV-driven image datasets and augmentations
train_teacher.py   teacher training
train_student.py   CF-CoT distillation and ablation switches
evaluate.py        student-only evaluation
```

## Citation

```bibtex
@inproceedings{wang2027cfcot,
  title     = {{CF-CoT}: Coarse-to-Fine Chain-of-Thought Knowledge Distillation for Visual Ordinal Rating},
  author    = {Wang, Mingqian and Yao, Yingliang and Xiong, Wenxin and Wang, Bo and Ren, Jian and Xu, Meng},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027}
}
```

## License

MIT.
