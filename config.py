"""
Every number the pipeline depends on, in one place.

Shared by train.py and app.py so the dashboard prices decisions with exactly
the figures the training stage used. Where each figure comes from is in
REPORT.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "bone-fracture-v3"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

# --- Roboflow Universe dataset (used as published, not forked) -------------
WORKSPACE = "veda"
PROJECT = "bone-fracture-detection-daoon"
VERSION = 3          # the version with no augmentation: every image is a real X-ray

# --- Training ---------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"
EPOCHS = 80
PATIENCE = 20        # stop early if validation mAP has not improved for 20 epochs
IMGSZ = 512        # the source X-rays are at most 512 px: upscaling adds no detail
BATCH = 16
SEED = 0

# --- Business figures (EUR) -------------------------------------------------
# Setting: the trauma emergency department of Hospital Universitario La Paz
# (public, Comunidad de Madrid, Spain), at night (20:00-08:00), with no
# radiologist on site. Each
# extremity X-ray is either passed (no fracture), failed (treated as a
# fracture) or sent to a remote radiologist. Prices: Madrid public price list
# (Orden 1975/2023); breakdown in REPORT.md.
COST_FN = 1000       # missed fracture: ER return 175 + trauma consult 115 + X-ray 30
                     # + 5% delayed surgery (~300) + 1.5% claim x 22,900 (~340)
COST_FP = 150        # healthy patient splinted: trauma consult 115 + splint ~35
COST_TP = 0          # correct calls carry no extra cost: the treatment happens anyway
COST_TN = 0
COST_REVIEW = 20     # one remote radiologist read, out of hours (weakest figure)

# How a radiologist's read turns out when a case is sent for review.
REVIEWER_SENSITIVITY = 0.963   # 3.7% of fractures missed in emergency radiology
REVIEWER_SPECIFICITY = 0.98

# How often the positive class actually occurs in production.
BASE_RATE = 0.30     # share of trauma extremity X-rays that show a fracture

# Workload.
CASES_PER_NIGHT = 30         # extremity X-rays between 20:00 and 08:00. La Paz 2023:
                             # 252,397 ER visits (~690/day); ~25% trauma (~170/day);
                             # ~25% of those at night (~43); ~70% X-rayed -> ~30
REVIEW_MINUTES = 3           # one plain-film read
REVIEW_MINUTES_AVAILABLE = 60  # radiologist time per night for these reads
