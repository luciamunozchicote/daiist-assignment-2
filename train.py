"""
Training pipeline: fracture triage on emergency X-rays.

Run with:  uv run python main.py train

1. Download the Roboflow Universe dataset (veda/bone-fracture-detection, v3).
2. Describe it: instances per class, split sizes, empty images, box sizes.
3. Finetune YOLO11s from the COCO checkpoint; copy the best weights to weights/.
4. Evaluate on the test split: per-class precision, recall, AP50, mAP50-95,
   and the confusion matrix. Compare against two baselines: the COCO model
   zero-shot and a predict-nothing model.
5. Score every validation and test image (highest fracture confidence).
6. Choose thresholds on the VALIDATION split: the single cost-minimising
   threshold, and the cheapest pass / review / fail band that fits the
   radiologist time available. Report them on the TEST split, alongside
   the three alternatives (do nothing, review everything, model alone).
7. Save everything app.py needs, so the app never trains.
"""

import json
import random
import shutil
import statistics
from collections import Counter

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

import config as C
import costing
from api_key import load_api_key

random.seed(C.SEED)
np.random.seed(C.SEED)
C.ARTIFACTS.mkdir(exist_ok=True)
C.WEIGHTS.parent.mkdir(exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
section("1. Download dataset")
if not (C.DATASET_DIR / "data.yaml").exists():
    from roboflow import Roboflow
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(C.WORKSPACE).project(C.PROJECT).version(C.VERSION).download(
        "yolov11", location=str(C.DATASET_DIR), overwrite=True)

# Roboflow writes relative split paths; make them absolute so YOLO finds them
# from any working directory.
data_yaml = C.DATASET_DIR / "data.yaml"
cfg = yaml.safe_load(data_yaml.read_text())
cfg.update({"path": str(C.DATASET_DIR), "train": "train/images", "val": "valid/images", "test": "test/images"})
data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
NAMES = cfg["names"]
print(f"{C.WORKSPACE}/{C.PROJECT} v{C.VERSION}: {len(NAMES)} classes -> {NAMES}")


def label_rows(label_path):
    """YOLO label rows. This dataset stores 4-corner polygons; YOLO turns them
    into the enclosing box, so the stats below do the same."""
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, coords = int(parts[0]), list(map(float, parts[1:]))
        if len(coords) == 4:
            w, h = coords[2], coords[3]
        else:
            xs, ys = coords[0::2], coords[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        rows.append((cls, w * h))
    return rows


def image_paths(split):
    return sorted(p for p in (C.DATASET_DIR / SPLITS[split] / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def has_fracture(image_path):
    label = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    return label.exists() and bool(label_rows(label))


# ---------------------------------------------------------------------------
# 2. Describe the dataset before training
# ---------------------------------------------------------------------------
section("2. Dataset composition")
stats = {}
for split in SPLITS:
    imgs = image_paths(split)
    per_class, areas, empty = Counter(), [], 0
    for img in imgs:
        rows = label_rows(img.parent.parent / "labels" / (img.stem + ".txt"))
        empty += not rows
        for cls, area in rows:
            per_class[NAMES[cls]] += 1
            areas.append(area)
    stats[split] = {
        "images": len(imgs), "images_without_fracture": empty, "boxes": sum(per_class.values()),
        "per_class": {n: per_class.get(n, 0) for n in NAMES},
        "box_area_share_median": statistics.median(areas) if areas else None,
        "box_area_share_p10": float(np.percentile(areas, 10)) if areas else None,
        "box_area_share_p90": float(np.percentile(areas, 90)) if areas else None,
    }
    s = stats[split]
    print(f"{split:5s} images {s['images']:5d} | no fracture {empty:4d} ({empty / len(imgs):.0%}) | boxes {s['boxes']:4d} | "
          f"median box = {s['box_area_share_median']:.1%} of image")
    print(f"      per class: {s['per_class']}")

# ---------------------------------------------------------------------------
# 3. Finetune
# ---------------------------------------------------------------------------
section(f"3. Finetune {C.BASE_MODEL} on {DEVICE}")
run_dir = C.ROOT / "runs" / "fracture"
last = run_dir / "weights" / "last.pt"
best = run_dir / "weights" / "best.pt"
resumed = False
if last.exists():
    # An interrupted run (laptop closed) continues from its last epoch instead
    # of starting over. A run that already finished raises, and we move on.
    try:
        YOLO(str(last)).train(resume=True, device=DEVICE)
        resumed = True
    except Exception as error:  # noqa: BLE001
        print(f"not resuming ({type(error).__name__}: {error})")
if not resumed and not best.exists():
    model = YOLO(C.BASE_MODEL)
    model.train(data=str(data_yaml), epochs=C.EPOCHS, patience=C.PATIENCE, imgsz=C.IMGSZ, batch=C.BATCH,
                seed=C.SEED, deterministic=True, device=DEVICE, project=str(C.ROOT / "runs"),
                name="fracture", exist_ok=True, plots=True, verbose=False)
shutil.copy(best, C.WEIGHTS)
print(f"best weights -> {C.WEIGHTS}")
model = YOLO(str(C.WEIGHTS))

# ---------------------------------------------------------------------------
# 4. Box-level evaluation on the test split
# ---------------------------------------------------------------------------
section("4. Test-split evaluation (box level)")
res = model.val(data=str(data_yaml), split="test", imgsz=C.IMGSZ, batch=C.BATCH, device=DEVICE,
                conf=0.001, iou=0.6, plots=True, project=str(C.ROOT / "runs"), name="test", exist_ok=True,
                verbose=False)
box = res.box
per_class = []
for i, name in enumerate(NAMES):
    per_class.append({
        "class": name, "test_instances": stats["test"]["per_class"][name],
        "precision": float(box.p[i]), "recall": float(box.r[i]),
        "ap50": float(box.ap50[i]), "map50_95": float(box.ap[i]),
    })
    print(f"{name:18s} n={stats['test']['per_class'][name]:3d}  P {box.p[i]:.3f}  R {box.r[i]:.3f}  "
          f"AP50 {box.ap50[i]:.3f}  mAP50-95 {box.ap[i]:.3f}")
overall = {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map)}
print(f"{'all':18s}        P {box.mp:.3f}  R {box.mr:.3f}  mAP50 {box.map50:.3f}  mAP50-95 {box.map:.3f}")
# Rows = predicted class, columns = true class; the last row/column is "background".
confusion = res.confusion_matrix.matrix.tolist()

# ---------------------------------------------------------------------------
# 5. Image-level scores: "does this X-ray show any fracture?"
# ---------------------------------------------------------------------------
section("5. Image-level scores")


def image_scores(yolo, split, any_class=True):
    scores, labels, names = [], [], []
    for img in image_paths(split):
        r = yolo.predict(str(img), conf=0.001, imgsz=C.IMGSZ, device=DEVICE, verbose=False)[0]
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
        scores.append(float(conf.max()) if conf.size else 0.0)
        labels.append(int(has_fracture(img)))
        names.append(img.name)
    return scores, labels, names


val_scores, val_labels, _ = image_scores(model, "val")
test_scores, test_labels, test_names = image_scores(model, "test")
auc_model = roc_auc_score(test_labels, test_scores)
print(f"finetuned model: image-level ROC-AUC on test = {auc_model:.3f}")

# Baseline 1: COCO model zero-shot. COCO has no fracture class, so it cannot
# produce a single correct box (AP = 0). At image level we let ANY detection
# count as "fracture" to give it its best possible chance.
coco = YOLO(C.BASE_MODEL)
coco_scores, _, _ = image_scores(coco, "test")
auc_coco = roc_auc_score(test_labels, coco_scores)
print(f"COCO zero-shot: AP50 = 0 (no fracture class); image-level ROC-AUC = {auc_coco:.3f}")

# Baseline 2: predict nothing. Every image passes as "no fracture".
nothing_scores = [0.0] * len(test_labels)

# ---------------------------------------------------------------------------
# 6. Costing and thresholds (chosen on validation, reported on test)
# ---------------------------------------------------------------------------
section("6. Cost sweep and operating points")
val_sweep = costing.sweep_single(val_scores, val_labels)
t_single = min(val_sweep, key=lambda x: x[1])[0]
max_share = costing.max_review_share()
low, high, _, _ = costing.best_band(val_scores, val_labels, max_share=max_share)
low_free, high_free, _, _ = costing.best_band(val_scores, val_labels, max_share=1.0)
print(f"single threshold (val minimum): {t_single:.2f}")
print(f"review capacity: {C.REVIEW_MINUTES_AVAILABLE} min / {C.REVIEW_MINUTES} min per read = "
      f"{C.REVIEW_MINUTES_AVAILABLE / C.REVIEW_MINUTES:.0f} reads for {C.CASES_PER_NIGHT} cases -> max {max_share:.0%} reviewed")
print(f"band within capacity (val): pass < {low:.2f} <= review < {high:.2f} <= fail")
print(f"band ignoring capacity (val): pass < {low_free:.2f} <= review < {high_free:.2f} <= fail")


def strategy(name, scores, lo, hi, note):
    rates = costing.outcome_rates(scores, test_labels, lo, hi)
    share = costing.review_share(rates)
    return {
        "strategy": name, "low": lo, "high": hi, "note": note,
        "cost_per_1000": costing.cost_per_1000(rates),
        "review_share": share,
        "reviews_per_night": share * C.CASES_PER_NIGHT,
        "radiologist_minutes_per_night": share * C.CASES_PER_NIGHT * C.REVIEW_MINUTES,
        "missed_fractures_per_1000": 1000 * C.BASE_RATE * (rates["pos_pass"] + rates["pos_review"] * (1 - C.REVIEWER_SENSITIVITY)),
        "unneeded_splints_per_1000": 1000 * (1 - C.BASE_RATE) * (rates["neg_fail"] + rates["neg_review"] * (1 - C.REVIEWER_SPECIFICITY)),
        "counts": costing.counts(scores, test_labels, lo, hi),
    }


strategies = [
    strategy("Do nothing (everyone passes)", nothing_scores, 1.01, 1.01, "predict-nothing baseline"),
    strategy("Radiologist reads everything", test_scores, -1.0, 1.01, "every case to review"),
    strategy("COCO zero-shot alone", coco_scores, *([min(costing.sweep_single(coco_scores, test_labels), key=lambda x: x[1])[0]] * 2),
             "threshold set at its own best on test (generous)"),
    strategy("Model alone", test_scores, t_single, t_single, "single threshold chosen on validation"),
    strategy("Model + review (fits capacity)", test_scores, low, high, "chosen on validation"),
    strategy("Model + review (unlimited capacity)", test_scores, low_free, high_free, "chosen on validation"),
]
for s in strategies:
    print(f"{s['strategy']:38s} €{s['cost_per_1000']:9,.0f} /1000 | review {s['review_share']:5.1%} "
          f"({s['reviews_per_night']:.1f}/night, {s['radiologist_minutes_per_night']:.0f} min) | "
          f"missed {s['missed_fractures_per_1000']:.0f}/1000 | splints {s['unneeded_splints_per_1000']:.0f}/1000")

# ---------------------------------------------------------------------------
# 7. Save artifacts for app.py
# ---------------------------------------------------------------------------
section("7. Saving artifacts")
(C.ARTIFACTS / "results.json").write_text(json.dumps({
    "dataset": {"workspace": C.WORKSPACE, "project": C.PROJECT, "version": C.VERSION, "names": NAMES, "stats": stats},
    "training": {"base_model": C.BASE_MODEL, "epochs": C.EPOCHS, "patience": C.PATIENCE, "imgsz": C.IMGSZ,
                 "batch": C.BATCH, "seed": C.SEED, "device": str(DEVICE)},
    "test_box": {"per_class": per_class, "overall": overall, "confusion_matrix": confusion,
                 "confusion_labels": NAMES + ["background"]},
    "baselines": {"coco_zero_shot": {"map50": 0.0, "image_roc_auc": auc_coco},
                  "predict_nothing": {"map50": 0.0, "image_roc_auc": 0.5},
                  "finetuned": {"map50": overall["map50"], "image_roc_auc": auc_model}},
    "thresholds": {"single": t_single, "low": low, "high": high, "low_unlimited": low_free,
                   "high_unlimited": high_free, "max_review_share": max_share},
    "val_sweep": val_sweep,
    "test_sweep": costing.sweep_single(test_scores, test_labels),
    "strategies": strategies,
}, indent=2))
(C.ARTIFACTS / "image_scores.json").write_text(json.dumps({
    "val": {"scores": val_scores, "labels": val_labels},
    "test": {"scores": test_scores, "labels": test_labels, "files": test_names, "coco_scores": coco_scores},
}))
print(f"saved to {C.ARTIFACTS}/ and {C.WEIGHTS}")
