# Ablation — YOLOv8n + SE (Squeeze-and-Excitation)

The main ablation adds **one** attention block to the Phase-1 winner (YOLOv8n) to measure the
effect of a single, lightweight change. **SE** is used (not CBAM) because it is lighter, barely
changes the architecture or speed, and isolates one variable. CBAM is kept for a **later
secondary experiment** only.

## What changes

- A single `SEResearch` block is inserted **right after SPPF** (`configs/yolov8n_se.yaml`).
  Everything else is identical to `models/yolov8n_baseline.yaml`.
- SE is **eager** (all params built in `__init__`) so they are in the optimizer.
- Adds ~8k params (`SE(256, r=16)`), negligible vs the 3.0M baseline.
- Trained on **XWOD** — the same data as the baseline — so the only variable is SE.
  (The final best model uses the merged Phase-2 dataset; the ablation stays on XWOD.)

## Flow

```bash
# 1. Sanity-check before the full run
python scripts/sanity_se.py \
  --cfg configs/yolov8n_se.yaml \
  --data configs/xwod.yaml \
  --weights /content/workspace/runs/stage1_bdd30k_yolov8n/weights/best.pt

# 2. Train YOLOv8n + SE on XWOD (partial-loads pretrained weights, skips the SE layer)
python scripts/train.py \
  --data configs/xwod.yaml \
  --cfg configs/yolov8n_se.yaml \
  --weights /content/workspace/runs/stage1_bdd30k_yolov8n/weights/best.pt \
  --name improved_yolov8n_se_xwod
```

Hyperparameters in `scripts/train.py` mirror the baseline XWOD run (AdamW, lr0=5e-4,
epochs=50, patience=15, imgsz=640, batch=16, seed=42) for a fair comparison.

### Weight transfer

`scripts/train.py` pairs the non-attention layers of the pretrained checkpoint with the
non-attention layers of the SE model in order, and skips the inserted SE block. This handles
the head index shift caused by inserting SE after SPPF (a plain name-based load would miss the
whole head). Verified: 355/357 tensors transfer, only the 2 SE tensors are left randomly
initialized.

## Comparison plan

| Model | Data | Purpose |
|---|---|---|
| Baseline YOLOv8n | XWOD | Reference (already trained: `stage2_xwod_yolov8n_from_bdd30k`) |
| **+SE** | XWOD | Main ablation (this doc) |
| +CBAM | XWOD | Secondary experiment — only if SE looks promising |

Evaluate all variants the same way: XWOD test, DAWN val, ACDC.
