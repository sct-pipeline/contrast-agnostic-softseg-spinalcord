"""
QC report on the hand-picked difficult cases (difficult_cases_for_qualitative_testing.yml).

Compares two models per volume, both via sct_deepseg:
  - seg_v4_crop : NEW model -> sct_deepseg spinalcord -fast  (sc-crop pipeline, v4 weights)
  - seg_v3_full : OLD model -> sct_deepseg spinalcord        (full-volume, v3 weights)

sc-crop (detect + crop + infer + uncrop) is handled internally by sct_deepseg -fast.
All outputs are in the original image space. Timing covers the full pipeline per volume.

Outputs:
  - results.csv  : one row per volume (dice_v4_crop, dice_v3_full, time_v4_crop_s, time_v3_full_s, ...)
  - metrics.json : aggregates on test/unseen volumes
  - qc/index.html: three overlay groups per volume (labels, seg_v3_full, seg_v4_crop)
  - run.log

Usage:
    python scripts/generate_qc_report_difficult.py
    python scripts/generate_qc_report_difficult.py --n-subjects 5

Author: Quentin Revillon
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

# ── Binaries ─────────────────────────────────────────────────────────────────
SCT_V4_BIN = "/home/quentinr/spinalcordtoolbox/bin/sct_deepseg"       # fork, -fast = sc-crop v4
SCT_V3_BIN = "/home/quentinr/spinalcordtoolbox-gpu/bin/sct_deepseg"   # master, full-volume v3

# ── GPU / CPU ─────────────────────────────────────────────────────────────────
V4_USE_GPU    = False   # True → GPU for sc-crop v4 model
V4_GPU_DEVICE = "0"
V3_USE_GPU    = True    # True → GPU for v3 model
V3_GPU_DEVICE = "0"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/quentinr/qc_results_difficult_fast"
# ─────────────────────────────────────────────────────────────────────────────

_CONTRAST_PATTERN = (
    r'.*(T1w|acq-sagthor_T2w|acq-sagcerv_T2w|acq-sagstir_T2w|acq-ax_T2w|acq-axial_T2w|T2w'
    r'|T2star|PSIR|STIR|UNIT1|flip-1_mt-on_MTS|flip-2_mt-off_MTS'
    r'|acq-MTon_MTR|acq-dwiMean_dwi|rec-average_dwi|acq-T1w_MTR|dwi).*'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yml", default=str(REPO / "difficult_cases_for_qualitative_testing.yml"))
    p.add_argument("--data-base", default="/home/quentinr/datasets_contrast_agnostic_retraining")
    p.add_argument("--datasplits", default=str(REPO / "datasplits"))
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--n-subjects", type=int, default=None)
    p.add_argument("--v3-cpu", action="store_true", default=False,
                   help="Run v3 inference on CPU (default: GPU)")
    return p.parse_args()


class Logger:
    def __init__(self, path: Path):
        self._f = open(path, "w")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def run(cmd: list, logger: Logger, env: dict | None = None) -> None:
    logger.log(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def parse_difficult_yml(path: Path) -> list[tuple]:
    cases, dataset, in_images = [], None, False
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s == "FILES_IMAGES:":
            in_images = True
            continue
        if not in_images:
            continue
        if s.startswith("- "):
            if dataset:
                cases.append((dataset, s[2:].strip()))
        elif s.endswith(":"):
            dataset = s[:-1].strip()
    return cases


def load_split_map(datasplits: Path, dataset: str) -> dict:
    f = datasplits / f"datasplit_{dataset}_seed50.yaml"
    if not f.exists():
        return {}
    d = yaml.safe_load(f.read_text())
    out = {}
    for split in ("test", "train", "val"):
        for subj in (d.get(split) or []):
            out[subj] = split
    return out


def contrast_name(path: str) -> str:
    m = re.search(_CONTRAST_PATTERN, path)
    return m.group(1) if m else "unknown"


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    g, p = gt > 0, pred > 0
    denom = g.sum() + p.sum()
    return float(2 * np.logical_and(g, p).sum() / denom) if denom > 0 else 0.0


def is_valid_nifti(path) -> bool:
    try:
        nib.load(str(path)).header
        return True
    except Exception:
        return False


GT_CONVENTION = {
    "basel-mp2rage":                ("labels_softseg_bin", "desc-softseg_label-SC_seg"),
    "canproco":                     ("labels", "seg-manual"),
    "data-multi-subject":           ("labels_softseg_bin", "desc-softseg_label-SC_seg"),
    "dcm-brno":                     ("labels", "seg"),
    "dcm-zurich":                   ("labels", "label-SC_mask-manual"),
    "dcm-zurich-lesions":           ("labels", "label-SC_mask-manual"),
    "dcm-zurich-lesions-20231115":  ("labels", "label-SC_mask-manual"),
    "lumbar-epfl":                  ("labels", "seg-manual"),
    "lumbar-vanderbilt":            ("labels", "label-SC_seg"),
    "sci-colorado":                 ("labels", "seg-manual"),
    "sci-paris":                    ("labels", "seg-manual"),
    "sci-zurich":                   ("labels", "seg-manual"),
    "sct-testing-large":            ("labels", "seg-manual"),
    "site_006":                     ("labels", "label-SC_seg"),
    "site_007":                     ("labels", "label-SC_seg"),
}


def find_gt(data_base: Path, dataset: str, rel: str, stem: str):
    if dataset not in GT_CONVENTION:
        return None
    folder, suffix = GT_CONVENTION[dataset]
    p = data_base / dataset / "derivatives" / folder / Path(rel).parent / f"{stem}_{suffix}.nii.gz"
    return p if is_valid_nifti(p) else None


def main():
    args = parse_args()
    data_base = Path(args.data_base)
    datasplits = Path(args.datasplits)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir + f"_{timestamp}")
    qc_dir = out / "qc"
    sct_qc = str(Path(SCT_V4_BIN).parent / "sct_qc")

    if qc_dir.exists():
        shutil.rmtree(qc_dir)
    for d in [qc_dir, out / "seg_v4_crop", out / "seg_v3_full"]:
        d.mkdir(parents=True, exist_ok=True)

    logger = Logger(out / "run.log")
    cases = parse_difficult_yml(Path(args.yml))
    if args.n_subjects:
        cases = cases[:args.n_subjects]
    logger.log(f"Parsed {len(cases)} difficult-case volumes from {args.yml}")

    split_cache: dict[str, dict] = {}
    rows = []

    for i, (dataset, rel) in enumerate(cases, 1):
        subject = rel.split("/")[0]
        session = next((p for p in rel.split("/") if p.startswith("ses-")), "")
        contrast = contrast_name(rel)
        stem = Path(rel).name.replace(".nii.gz", "")
        image = data_base / dataset / rel
        gt = find_gt(data_base, dataset, rel, stem)

        if dataset not in split_cache:
            split_cache[dataset] = load_split_map(datasplits, dataset)
        split = split_cache[dataset].get(subject, "not_in_split")

        logger.log(f"\n[{i}/{len(cases)}] {dataset}/{stem}  (split={split})")
        base = {"dataset": dataset, "subject": subject, "session": session,
                "contrast": contrast, "split": split}
        skip = {**base, "status": "missing_data", "dice_v4_crop": "", "dice_v3_full": "",
                "time_v4_crop_s": "", "time_v3_full_s": "", "has_gt": False,
                "image": str(image), "gt": "", "seg_v4_crop": "", "seg_v3_full": ""}

        if not is_valid_nifti(image):
            logger.log(f"  !! image missing / not a valid NIfTI, skipping: {image}")
            rows.append(skip)
            continue

        seg_v4_crop = out / "seg_v4_crop" / f"{stem}_seg_v4_crop.nii.gz"
        seg_v3_full = out / "seg_v3_full" / f"{stem}_seg_v3_full.nii.gz"

        try:
            # v4: sct_deepseg spinalcord -fast (sc-crop + v4 model, full pipeline in one call)
            if is_valid_nifti(seg_v4_crop):
                time_v4_crop = ""; logger.log("  reuse existing seg_v4_crop")
            else:
                v4_env = {"SCT_USE_GPU": "1", "CUDA_VISIBLE_DEVICES": V4_GPU_DEVICE,
                          "TORCHDYNAMO_DISABLE": "1"} if V4_USE_GPU else {}
                t = time.perf_counter()
                run([SCT_V4_BIN, "spinalcord", "-fast", "-i", str(image), "-o", str(seg_v4_crop)],
                    logger, env=v4_env)
                time_v4_crop = round(time.perf_counter() - t, 1)

            # v3: sct_deepseg spinalcord (full-volume, v3 model)
            if is_valid_nifti(seg_v3_full):
                time_v3_full = ""; logger.log("  reuse existing seg_v3_full")
            else:
                v3_env = {"SCT_USE_GPU": "1", "CUDA_VISIBLE_DEVICES": V3_GPU_DEVICE,
                          "TORCHDYNAMO_DISABLE": "1"} if not args.v3_cpu else {}
                t = time.perf_counter()
                run([SCT_V3_BIN, "spinalcord", "-i", str(image), "-o", str(seg_v3_full)],
                    logger, env=v3_env)
                time_v3_full = round(time.perf_counter() - t, 1)

            logger.log(f"  time_v4_crop={time_v4_crop}s  time_v3_full={time_v3_full}s")

            has_gt = gt is not None
            d_v4_crop = d_v3_full = None
            if has_gt:
                g = np.asarray(nib.load(gt).dataobj)
                d_v4_crop = dice(g, np.asarray(nib.load(seg_v4_crop).dataobj))
                d_v3_full = dice(g, np.asarray(nib.load(seg_v3_full).dataobj))
                logger.log(f"  dice_v4_crop={d_v4_crop:.4f}  dice_v3_full={d_v3_full:.4f}")
            else:
                logger.log(f"  !! GT missing/invalid (no Dice): {gt}")

            qid = f"{dataset}/{stem}"
            if has_gt:
                run([sct_qc, "-i", str(image), "-s", str(gt), "-p", "sct_deepseg_sc",
                     "-qc", str(qc_dir), "-qc-subject", qid, "-qc-dataset", "labels"], logger)
            run([sct_qc, "-i", str(image), "-s", str(seg_v3_full), "-p", "sct_deepseg_sc",
                 "-qc", str(qc_dir), "-qc-subject", qid, "-qc-dataset", "seg_v3_full"], logger)
            run([sct_qc, "-i", str(image), "-s", str(seg_v4_crop), "-p", "sct_deepseg_sc",
                 "-qc", str(qc_dir), "-qc-subject", qid, "-qc-dataset", "seg_v4_crop"], logger)

            rows.append({**base, "status": "ok",
                         "dice_v4_crop": round(d_v4_crop, 4) if d_v4_crop is not None else "",
                         "dice_v3_full": round(d_v3_full, 4) if d_v3_full is not None else "",
                         "time_v4_crop_s": time_v4_crop, "time_v3_full_s": time_v3_full,
                         "has_gt": has_gt,
                         "image": str(image), "gt": str(gt) if has_gt else "",
                         "seg_v4_crop": str(seg_v4_crop), "seg_v3_full": str(seg_v3_full)})
        except Exception as e:
            logger.log(f"  !! ERROR: {e}")
            rows.append({**skip, "status": "error"})
            continue

    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def _stats(vals):
        a = np.array([v for v in vals if v != ""], dtype=float)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "n": int(len(a))} if len(a) else None

    def agg(subset):
        return {"dice_v4_crop": _stats([r["dice_v4_crop"] for r in subset if r["has_gt"]]),
                "dice_v3_full": _stats([r["dice_v3_full"] for r in subset if r["has_gt"]])}

    by_split = {s: agg([r for r in rows if r["split"] == s]) for s in ("test", "not_in_split", "train", "val")}
    unseen = [r for r in rows if r["split"] in ("test", "not_in_split") and r["has_gt"]]
    t_v4 = [r["time_v4_crop_s"] for r in rows if isinstance(r["time_v4_crop_s"], float)]
    t_v3 = [r["time_v3_full_s"] for r in rows if isinstance(r["time_v3_full_s"], float)]

    metrics = {
        "n_volumes": len(rows),
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("ok", "missing_data", "error")},
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("test", "not_in_split", "train", "val")},
        "aggregate_unseen": agg(unseen),
        "by_split": by_split,
        "mean_time_v4_crop_s": round(float(np.mean(t_v4)), 1) if t_v4 else None,
        "mean_time_v3_full_s": round(float(np.mean(t_v3)), 1) if t_v3 else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    u = metrics["aggregate_unseen"]
    lines = ["Difficult-cases QC — summary", "=" * 40,
             f"Volumes: {len(rows)}  status: {metrics['status_counts']}",
             f"Splits : {metrics['split_counts']}", ""]
    if u["dice_v4_crop"]:
        lines += [f"UNSEEN (test+not_in_split), n={u['dice_v4_crop']['n']}:",
                  f"  Dice v4_crop : {u['dice_v4_crop']['mean']} ± {u['dice_v4_crop']['std']}",
                  f"  Dice v3_full : {u['dice_v3_full']['mean']} ± {u['dice_v3_full']['std']}", ""]
    if t_v4:
        lines.append(f"Mean time: v4_crop {metrics['mean_time_v4_crop_s']}s  "
                     f"v3_full {metrics['mean_time_v3_full_s']}s")
    lines += [f"CSV: {csv_path}", f"QC : {qc_dir}/index.html"]
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    logger.log("\n" + summary)
    logger.close()


if __name__ == "__main__":
    main()
