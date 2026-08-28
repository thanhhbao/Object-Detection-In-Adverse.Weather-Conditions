#!/usr/bin/env python3
"""
Smoke test for P2-A1 RT-DETR multi-scale embedding.

Usage:
  python scripts/smoke_p2_a1_embedding.py \\
    --weights /workspace/runs/phase2_final_rtdetr/weights/best.pt \\
    --embedding-layers 21 24 27 \\
    --sample-image /path/to/one/train/image.jpg

Exits 0 on success, 1 on any error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke check: RT-DETR multi-scale embedding")
    p.add_argument("--weights", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--embedding-layers", nargs="+", type=int, default=[21, 24, 27],
                   help="Layer indices to hook (default: 21 24 27 for RT-DETR-L P3/P4/P5)")
    p.add_argument("--sample-image", type=Path, default=None,
                   help="Optional: path to a sample image. If omitted, uses a 640x640 dummy tensor.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    embedding_layers = args.embedding_layers

    print("=" * 60)
    print("P2-A1 Embedding Smoke Check")
    print(f"  weights:          {args.weights}")
    print(f"  embedding-layers: {embedding_layers}")
    print(f"  sample-image:     {args.sample_image or '(dummy 640x640)'}")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
        from ultralytics import YOLO
    except ImportError as e:
        print(f"SMOKE CHECK: FAIL — missing dependency: {e}")
        sys.exit(1)

    # Import MultiScaleHook from active_retrieval
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from active_retrieval import MultiScaleHook
    except ImportError as e:
        print(f"SMOKE CHECK: FAIL — cannot import MultiScaleHook: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    IMGSZ = 640

    try:
        # 1. Load model
        print(f"\n[1/5] Loading model from {args.weights}...")
        model = YOLO(str(args.weights))
        model.to(device)
        print("  Model loaded.")

        # 2. Check for RTDETRDecoder
        has_rtdetr = False
        try:
            from ultralytics.nn.modules.head import RTDETRDecoder  # type: ignore[import]
            has_rtdetr = any(isinstance(m, RTDETRDecoder) for m in model.model.model)
        except ImportError:
            pass

        if has_rtdetr:
            print("  RTDETRDecoder detected — RT-DETR architecture confirmed.")
        else:
            print("SMOKE CHECK: FAIL — RTDETRDecoder NOT found in loaded checkpoint.")
            print(f"  Expected RT-DETR model for P2-A1 embedding layers {embedding_layers}.")
            print("  Ensure you passed the correct Phase2 RT-DETR best.pt checkpoint.")
            sys.exit(1)

        # 3. Register MultiScaleHook
        print(f"\n[2/5] Registering MultiScaleHook on layers {embedding_layers}...")
        hook = MultiScaleHook(model, embedding_layers)
        print(f"  Hooks registered on {len(embedding_layers)} layers.")

        # 4. Load or create sample image
        print("\n[3/5] Preparing input tensor...")
        if args.sample_image and args.sample_image.exists():
            try:
                import cv2  # type: ignore[import]
                img = cv2.imread(str(args.sample_image))
                if img is None:
                    raise ValueError(f"cv2 could not read: {args.sample_image}")
                img = cv2.resize(img, (IMGSZ, IMGSZ))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)
                print(f"  Loaded from {args.sample_image}: shape={list(tensor.shape)}")
            except Exception as e:
                print(f"  WARNING: Could not load sample image ({e}). Using dummy tensor.")
                tensor = torch.rand(1, 3, IMGSZ, IMGSZ).to(device)
        else:
            if args.sample_image:
                print(f"  WARNING: {args.sample_image} not found. Using dummy tensor.")
            tensor = torch.rand(1, 3, IMGSZ, IMGSZ).to(device)
            print(f"  Created dummy tensor: shape={list(tensor.shape)}")

        # 5. Forward pass
        print("\n[4/5] Running forward pass...")
        with torch.no_grad():
            model.model(tensor)

        # Check hook outputs
        if not hook.feats:
            print("SMOKE CHECK: FAIL — hooks produced no output (feats dict is empty).")
            hook.remove()
            sys.exit(1)

        shapes = {idx: list(hook.feats[idx].shape) for idx in embedding_layers if idx in hook.feats}
        missing = [idx for idx in embedding_layers if idx not in hook.feats]
        if missing:
            print(f"SMOKE CHECK: FAIL — layers {missing} produced no output.")
            hook.remove()
            sys.exit(1)

        total_dim = sum(v[-1] for v in shapes.values())
        model_type_str = "RT-DETR (RTDETRDecoder)" if has_rtdetr else "YOLO/other"
        print(f"  Embedding model type: {model_type_str}")
        print(f"  Embedding layers: {embedding_layers}")
        print(f"  Layer output shapes (after GAP): {shapes}")
        print(f"  Final embedding dimension: {total_dim}")

        # Get embedding and verify
        emb = hook.get_embedding(embedding_layers)
        hook.clear()
        print(f"  Embedding shape: {list(emb.shape)}")

        # Verify finite
        if not torch.isfinite(emb).all():
            print("SMOKE CHECK: FAIL — embedding contains non-finite values (NaN or Inf).")
            hook.remove()
            sys.exit(1)
        print("  Embedding is finite: OK")

        # Verify L2-norm ≈ 1.0
        norms = emb.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            print(f"SMOKE CHECK: FAIL — L2 norms not ≈ 1.0: {norms.tolist()}")
            hook.remove()
            sys.exit(1)
        print(f"  L2 norm ≈ 1.0: OK (got {norms[0].item():.6f})")

        # 5b. Run model.predict and print prediction count
        print("\n[5/5] Running model.predict...")
        if args.sample_image and args.sample_image.exists():
            results = model.predict(str(args.sample_image), verbose=False, conf=0.25, imgsz=IMGSZ)
        else:
            # predict on dummy — create a temp file
            import tempfile
            import numpy as np  # type: ignore[import]
            dummy_arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            import cv2  # type: ignore[import]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
            cv2.imwrite(tmp_path, cv2.cvtColor(dummy_arr, cv2.COLOR_RGB2BGR))
            results = model.predict(tmp_path, verbose=False, conf=0.25, imgsz=IMGSZ)
            import os
            os.unlink(tmp_path)

        pred_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        print(f"  Predictions: {pred_count} boxes at conf >= 0.25")

        hook.remove()

    except SystemExit:
        raise
    except Exception as e:
        print(f"\nSMOKE CHECK: FAIL — unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\nSMOKE CHECK: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
