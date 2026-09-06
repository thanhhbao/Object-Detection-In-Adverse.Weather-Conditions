#!/usr/bin/env python3
"""Generate the F1 ablation report from an official_eval backup.

Reads the per-arm, per-dataset metrics JSON produced by the official test
evaluation and writes a Vietnamese markdown report plus a flat CSV.

The metrics JSON carries overall precision/recall (so overall F1 is exact) but
only mAP per class, not per-class precision/recall. Per-class F1 therefore
cannot be derived here — it needs a fresh validation pass, which is what
scripts/eval_f1_matrix.py does.

Usage:
  python scripts/report_f1_ablation.py \
    --metrics-root <backup>/metrics \
    --failure-summary <backup>/../a1_failure_analysis/summary.json \
    --out docs/f1_ablation_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = [("phase2", "Phase 2"), ("a0r", "A0R"), ("a1_dino", "A1-DINO")]
DATASETS = [("xwod", "XWOD", "cùng miền"), ("bdd", "BDD", "miền nguồn"),
            ("dawn", "DAWN", "ngoài miền"), ("acdc", "ACDC", "giữ lại")]
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
RARE = {"bicycle", "motorcycle", "bus"}


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def load(metrics_root: Path) -> dict:
    out = {}
    for arm, _ in ARMS:
        for ds, _, _ in DATASETS:
            p = metrics_root / arm / f"official_{ds}_test_metrics.json"
            if p.is_file():
                out[(arm, ds)] = json.loads(p.read_text(encoding="utf-8"))
    if not out:
        raise SystemExit(f"No metrics JSON found under {metrics_root}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", type=Path, required=True)
    ap.add_argument("--failure-summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    M = load(args.metrics_root)
    fail = (json.loads(args.failure_summary.read_text(encoding="utf-8"))
            if args.failure_summary and args.failure_summary.is_file() else None)

    L: list[str] = []
    a = L.append

    a("# Báo cáo F1 — so sánh Phase 2, A0R và A1-DINO")
    a("")
    a("Đánh giá trên tập test giữ lại của bốn bộ dữ liệu. Ba cấu hình dùng chung")
    a("kiến trúc RT-DETR-L và cùng số ảnh huấn luyện (55.985), chỉ khác ở cách")
    a("chọn 5.000 ảnh bổ sung: A0R lấy ngẫu nhiên trong nhóm ảnh chứa lớp hiếm,")
    a("A1-DINO truy hồi bằng độ tương đồng đặc trưng DINOv2.")
    a("")

    # ── 1. Overall ──────────────────────────────────────────────────────────
    a("## 1. F1 tổng thể")
    a("")
    a("F1 tính từ precision và recall trung bình theo lớp: `F1 = 2PR/(P+R)`.")
    a("")
    a("| Tập test | Vai trò | " + " | ".join(n for _, n in ARMS)
      + " | Tốt nhất | Biên độ |")
    a("|---|---|" + "---:|" * len(ARMS) + "---|---:|")
    rows = []
    for ds, dname, role in DATASETS:
        vals, cells = {}, []
        for arm, _ in ARMS:
            m = M.get((arm, ds))
            if not m:
                cells.append("—")
                continue
            v = f1(m["precision"], m["recall"])
            vals[arm] = v
            cells.append(f"{v:.4f}")
            rows.append({"dataset": ds, "arm": arm, "class": "ALL",
                         "precision": round(m["precision"], 6),
                         "recall": round(m["recall"], 6), "f1": round(v, 6),
                         "map50": round(m["map50"], 6),
                         "map50_95": round(m["map50_95"], 6)})
        if not vals:
            continue
        best = max(vals, key=vals.get)
        spread = max(vals.values()) - min(vals.values())
        bname = dict(ARMS)[best]
        cells = [f"**{c}**" if c == f"{vals.get(best, -1):.4f}" else c for c in cells]
        a(f"| {dname} | {role} | " + " | ".join(cells) + f" | {bname} | {spread:.4f} |")
    a("")

    a("| Tập test | A0R − Phase 2 | A1-DINO − Phase 2 | A1-DINO − A0R |")
    a("|---|---:|---:|---:|")
    for ds, dname, _ in DATASETS:
        if not all((arm, ds) in M for arm, _ in ARMS):
            continue
        v = {arm: f1(M[(arm, ds)]["precision"], M[(arm, ds)]["recall"])
             for arm, _ in ARMS}
        a(f"| {dname} | {v['a0r']-v['phase2']:+.4f} | "
          f"{v['a1_dino']-v['phase2']:+.4f} | {v['a1_dino']-v['a0r']:+.4f} |")
    a("")

    spreads = []
    for ds, _, _ in DATASETS:
        vs = [f1(M[(arm, ds)]["precision"], M[(arm, ds)]["recall"])
              for arm, _ in ARMS if (arm, ds) in M]
        if len(vs) == len(ARMS):
            spreads.append(max(vs) - min(vs))
    if spreads:
        a(f"Biên độ giữa ba cấu hình lớn nhất là {max(spreads):.4f} và nhỏ nhất là")
        a(f"{min(spreads):.4f}. Việc bổ sung 5.000 ảnh — khoảng 9% tập huấn luyện —")
        a("làm F1 thay đổi dưới một điểm phần trăm rưỡi ở mọi tập. A0R, tức lấy mẫu")
        a("ngẫu nhiên, đạt F1 cao hơn A1-DINO ở cả bốn tập.")
        a("")

    # ── 2. Per class ────────────────────────────────────────────────────────
    a("## 2. Kết quả theo lớp")
    a("")
    a("Bộ metrics lưu lại chỉ có precision và recall ở mức tổng thể, nên phần theo")
    a("lớp dùng mAP50 và mAP50-95. Lớp hiếm được in đậm.")
    a("")
    for ds, dname, _ in DATASETS:
        if not all((arm, ds) in M for arm, _ in ARMS):
            continue
        a(f"### {dname}")
        a("")
        a("| Lớp | " + " | ".join(f"{n} mAP50" for _, n in ARMS) + " | A1 − A0R |")
        a("|---|" + "---:|" * (len(ARMS) + 1))
        for c in CLASSES:
            v = {arm: M[(arm, ds)]["per_class"].get(c, {}).get("map50", float("nan"))
                 for arm, _ in ARMS}
            for arm, _ in ARMS:
                pc = M[(arm, ds)]["per_class"].get(c, {})
                rows.append({"dataset": ds, "arm": arm, "class": c,
                             "precision": "", "recall": "", "f1": "",
                             "map50": round(pc.get("map50", float("nan")), 6),
                             "map50_95": round(pc.get("map50_95", float("nan")), 6)})
            label = f"**{c}**" if c in RARE else c
            a(f"| {label} | " + " | ".join(f"{v[arm]:.4f}" for arm, _ in ARMS)
              + f" | {v['a1_dino']-v['a0r']:+.4f} |")
        a("")

    # weakest cells overall
    weak = []
    for ds, dname, _ in DATASETS:
        if ("phase2", ds) not in M:
            continue
        for c in CLASSES:
            m50 = M[("phase2", ds)]["per_class"].get(c, {}).get("map50")
            if m50 is not None:
                weak.append((m50, dname, c))
    weak.sort()
    a("### Các ô yếu nhất")
    a("")
    a("Xếp theo mAP50 của Phase 2, thấp nhất trước:")
    a("")
    a("| Tập | Lớp | mAP50 |")
    a("|---|---|---:|")
    for m50, dname, c in weak[:8]:
        a(f"| {dname} | {c} | {m50:.4f} |")
    a("")

    # ── 3. Failure analysis ─────────────────────────────────────────────────
    if fail:
        a("## 3. Phân tích lỗi của A1-DINO")
        a("")
        t = fail.get("transitions", {})
        a("Theo dõi từng đối tượng ground truth qua hai mô hình, trên "
          f"{fail.get('query_image_count', '?')} ảnh truy vấn XWOD và ACDC, "
          f"ngưỡng phát hiện {fail.get('hard_confidence', '?')}:")
        a("")
        a("| Chuyển trạng thái | Số lượng |")
        a("|---|---:|")
        a(f"| Phát hiện được ở cả hai | {t.get('detected_to_detected', 0)} |")
        a(f"| Sai ở cả hai | {t.get('fail_to_fail', 0)} |")
        a(f"| A1-DINO **sửa được** | {t.get('fail_to_fixed', 0)} |")
        a(f"| A1-DINO **làm hỏng** | {t.get('detected_to_fail', 0)} |")
        a("")
        net = t.get("fail_to_fixed", 0) - t.get("detected_to_fail", 0)
        a(f"Ròng: **{net:+d}** đối tượng. A1-DINO làm hỏng nhiều hơn số nó sửa được.")
        a("")

        pcr = fail.get("per_class_failure_recovery", [])
        if pcr:
            a("| Lớp | GT | Phase 2 bỏ sót | A1 sửa | A1 làm hỏng | Ròng | Tỉ lệ phục hồi |")
            a("|---|---:|---:|---:|---:|---:|---:|")
            for r in pcr:
                n = r["a1_fixed_gt"] - r["a1_regressed_gt"]
                a(f"| {r['class']} | {r['all_gt']} | {r['phase2_failed_gt']} | "
                  f"{r['a1_fixed_gt']} | {r['a1_regressed_gt']} | {n:+d} | "
                  f"{r['failure_recovery_rate']*100:.1f}% |")
            a("")
            a("Cả ba lớp hiếm đều hỏng nhiều hơn sửa.")
            a("")

            a("### Độ tin cậy trên chính những đối tượng Phase 2 bỏ sót")
            a("")
            a("| Lớp | Phase 2 | A1-DINO | Thay đổi | Ngưỡng |")
            a("|---|---:|---:|---:|---:|")
            thr = fail.get("hard_confidence", 0.25)
            for r in pcr:
                p2 = r["mean_p2_conf_on_phase2_failures"]
                a1 = r["mean_a1_conf_on_same_failures"]
                a(f"| {r['class']} | {p2:.3f} | {a1:.3f} | {a1-p2:+.3f} | {thr} |")
            a("")
            a("Mô hình **có** nhận ra các đối tượng này, nhưng ở mức tin cậy dưới")
            a("ngưỡng. Dữ liệu bổ sung đẩy độ tin cậy lên — rõ nhất ở motorcycle —")
            a("song vẫn chưa đủ vượt ngưỡng. Nguyên nhân hỏng vì vậy không phải là")
            a("mô hình chưa từng học lớp đó, mà là học rồi nhưng không đủ chắc chắn.")
            a("")
            if fail.get("confidence_note"):
                a("Các giá trị trên là điểm tin cậy sau NMS, không phải xác suất "
                  "đã hiệu chuẩn.")
            a("")

    # ── 4. Conclusions ──────────────────────────────────────────────────────
    a("## 4. Nhận định")
    a("")
    a("1. Khác biệt giữa ba cấu hình nằm trong khoảng nhiễu. Truy hồi bằng DINOv2")
    a("   không vượt được lấy mẫu ngẫu nhiên; trên F1 tổng thể nó còn thấp hơn ở")
    a("   cả bốn tập test.")
    a("2. Phân tích chuyển trạng thái cho thấy A1-DINO tác động gần như ngẫu nhiên")
    a("   lên chính những lỗi mà nó nhắm tới, với xu hướng lệch nhẹ về phía xấu.")
    a("3. Nguồn dữ liệu bổ sung là BDD100K — ảnh đô thị điều kiện quang. Bổ sung")
    a("   thêm ảnh trời quang không thu hẹp được khoảng cách ở điều kiện sương mù,")
    a("   ban đêm và tuyết, vốn là chỗ ACDC yếu nhất.")
    a("4. Khoảng trống lớn nhất không nằm giữa các cấu hình mà nằm giữa các miền:")
    if ("phase2", "acdc") in M and ("phase2", "xwod") in M:
        fa = f1(M[("phase2", "acdc")]["precision"], M[("phase2", "acdc")]["recall"])
        fx = f1(M[("phase2", "xwod")]["precision"], M[("phase2", "xwod")]["recall"])
        a(f"   ACDC đạt F1 {fa:.4f} với recall {M[('phase2','acdc')]['recall']:.3f},")
        a(f"   trong khi XWOD đạt {fx:.4f}. Chênh lệch {fx-fa:.4f} lớn hơn nhiều lần")
        a("   mọi khác biệt giữa ba cấu hình.")
    a("")

    a("## 5. Hạn chế của báo cáo")
    a("")
    a("Bộ metrics đã lưu chỉ chứa precision và recall ở mức tổng thể, nên **F1 theo")
    a("từng lớp chưa tính được** — phần theo lớp trong báo cáo này dùng mAP thay thế.")
    a("Muốn có F1 theo lớp, cùng phép kiểm định val ↔ test, cần chạy lại đánh giá")
    a("bằng `scripts/eval_f1_matrix.py` với ba checkpoint đã lưu.")
    a("")
    a("Ngoài ra mọi số ở đây đo trên một lần huấn luyện cho mỗi cấu hình, không có")
    a("nhiều seed, nên chưa có khoảng tin cậy. Với biên độ nhỏ như đã thấy, đây là")
    a("lý do chính để không diễn giải thứ hạng giữa ba cấu hình như một kết luận.")
    a("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "arm", "class", "precision",
                                           "recall", "f1", "map50", "map50_95"])
        w.writeheader()
        w.writerows(rows)

    print(f"Report → {args.out}")
    print(f"CSV    → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
