# Phase 2 — Merged Dataset for the Final Best Model

Phase 1 selected the winner (**YOLOv8n** for realtime). Phase 2 fine-tunes that winner on a
**merged training set** to improve adverse-weather robustness while preventing catastrophic
forgetting of the earlier driving domain.

## What it merges

| Source | Split used | Amount | Purpose |
|---|---|---|---|
| XWOD | train | all | Main adverse-weather data |
| ACDC | train | all (optional) | More fog/rain/snow/night variety |
| BDD30K | train | 20–30% replay | Retain driving-domain knowledge (anti-forgetting) |

- **Validation** = XWOD val (main val set).
- **Test is NOT merged** — final evaluation is run separately on XWOD test, DAWN val, and
  ACDC val/test to avoid leakage. In `dataset.yaml`, `test` points at `images/val` only so
  the path is valid.
- All sources are already in the fixed 6-class order → labels copied as-is (no remap):
  `0 person · 1 bicycle · 2 car · 3 motorcycle · 4 bus · 5 truck`.
- Filenames are prefixed `xwod_ / acdc_ / bdd_` to avoid collisions.

## Run

```bash
python scripts/build_phase2_dataset.py \
  --xwod-root /content/workspace/datasets/xwod_6cls_yolo \
  --bdd-root  /content/workspace/datasets/bdd100k_6cls_30k_yolo \
  --acdc-root /content/workspace/datasets/acdc_6cls_yolo \
  --out-root  /content/workspace/datasets/phase2_merged_yolo \
  --bdd-replay-ratio 0.3 --seed 42 --mode symlink --clean
```

`--acdc-root` is optional: if the folder doesn't exist, it builds **XWOD + BDD replay** only.

### Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--xwod-root` | required | XWOD YOLO dataset root |
| `--bdd-root` | required | BDD30K YOLO dataset root |
| `--acdc-root` | none | ACDC YOLO dataset root (optional) |
| `--out-root` | required | Output merged dataset |
| `--bdd-replay-ratio` | 0.3 | Fraction of BDD30K train to replay |
| `--seed` | 42 | Sampling seed (protocol default) |
| `--mode` | symlink | `symlink` (saves disk) or `copy` |
| `--clean` | off | Remove `--out-root` first |

### Outputs

```
phase2_merged_yolo/
  images/{train,val}/     # prefixed images (symlink or copy)
  labels/{train,val}/
  dataset.yaml            # nc=6, test → images/val (placeholder)
  manifest.csv            # source_dataset, split, image_path, label_path, new_image, new_label
  stats.json              # images per source + boxes per class
```

The script prints an images-per-source and boxes-per-class table when it finishes.

## Then train the final model

```bash
# Final YOLOv8n (+SE ablation) on the merged dataset — override the placeholder dataset path
python scripts/train_ultralytics.py \
  --config configs/ultralytics/phase2_final_yolov8n_from_xwod.yaml \
  --data /content/workspace/datasets/phase2_merged_yolo/dataset.yaml
```

## Order of work

1. Run the **+SE ablation on XWOD** first (ablation must stay on XWOD for a fair comparison —
   do **not** train the SE-only ablation on the merged set).
2. Build the merged dataset with this script.
3. Download/prepare ACDC (`scripts/prepare_acdc.py`), then rebuild to include it.
4. Train the **final YOLOv8n + SE** on the merged dataset.
5. Evaluate the final model on **XWOD test + DAWN val + ACDC val/test**.
