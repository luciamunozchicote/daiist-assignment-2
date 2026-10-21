"""
Turning image scores into decisions and decisions into euros.

Shared by train.py (to pick the thresholds) and app.py (to reprice them live
when the sliders move), so both use exactly the same arithmetic.

An image's score is the highest confidence of any fracture box the model drew
on it (0 if it drew none). With two thresholds low <= high:

    score <  low          -> PASS    (no fracture, normal pathway)
    low <= score < high   -> REVIEW  (remote radiologist reads it)
    score >= high         -> FAIL    (treated as a fracture)

low == high means the model decides alone.
"""

import numpy as np

import config as C


def outcome_rates(scores, labels, low, high):
    """Share of positives and of negatives that end up in each outcome."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    passed = scores < low
    failed = scores >= high
    review = ~passed & ~failed
    pos, neg = labels, ~labels
    n_pos, n_neg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "pos_pass": (passed & pos).sum() / n_pos,    # missed fractures
        "pos_review": (review & pos).sum() / n_pos,
        "pos_fail": (failed & pos).sum() / n_pos,    # caught by the model
        "neg_pass": (passed & neg).sum() / n_neg,    # correctly cleared
        "neg_review": (review & neg).sum() / n_neg,
        "neg_fail": (failed & neg).sum() / n_neg,    # unnecessary splint
    }


def cost_per_1000(rates, base_rate=C.BASE_RATE, cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                  cost_review=C.COST_REVIEW, cost_tp=C.COST_TP, cost_tn=C.COST_TN):
    """Expected cost of 1,000 production cases.

    Rates are measured per class, then reweighted by the production base rate,
    because the dataset (~49% fractures) is far richer in fractures than a
    real emergency department (~30%). Reviewed cases cost a read, plus the
    radiologist's own misses and false alarms.
    """
    p, n = base_rate, 1 - base_rate
    review_pos = cost_review + (1 - C.REVIEWER_SENSITIVITY) * cost_fn + C.REVIEWER_SENSITIVITY * cost_tp
    review_neg = cost_review + (1 - C.REVIEWER_SPECIFICITY) * cost_fp + C.REVIEWER_SPECIFICITY * cost_tn
    per_case = (
        p * (rates["pos_pass"] * cost_fn + rates["pos_fail"] * cost_tp + rates["pos_review"] * review_pos)
        + n * (rates["neg_fail"] * cost_fp + rates["neg_pass"] * cost_tn + rates["neg_review"] * review_neg)
    )
    return 1000 * per_case


def review_share(rates, base_rate=C.BASE_RATE):
    """Share of production cases sent to a person."""
    return base_rate * rates["pos_review"] + (1 - base_rate) * rates["neg_review"]


def counts(scores, labels, low, high):
    """Raw confusion counts on the evaluated split (3 decisions x 2 truths)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    out = {}
    for name, mask in [("pass", scores < low), ("fail", scores >= high),
                       ("review", (scores >= low) & (scores < high))]:
        out[f"{name}_fracture"] = int((mask & labels).sum())
        out[f"{name}_no_fracture"] = int((mask & ~labels).sum())
    return out


THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def sweep_single(scores, labels, **costs):
    """Model decides alone: one threshold, cost at every value."""
    return [(float(t), cost_per_1000(outcome_rates(scores, labels, t, t), **costs)) for t in THRESHOLD_GRID]


def max_review_share():
    """The share of cases the available radiologist time can absorb."""
    reads_per_night = C.REVIEW_MINUTES_AVAILABLE / C.REVIEW_MINUTES
    return min(1.0, reads_per_night / C.CASES_PER_NIGHT)


def best_band(scores, labels, max_share=None, **costs):
    """Cheapest (low, high) pair whose review load fits the people available."""
    max_share = max_review_share() if max_share is None else max_share
    best = None
    for i, low in enumerate(THRESHOLD_GRID):
        for high in THRESHOLD_GRID[i:]:
            rates = outcome_rates(scores, labels, low, high)
            share = review_share(rates, costs.get("base_rate", C.BASE_RATE))
            if share > max_share + 1e-9:
                continue
            cost = cost_per_1000(rates, **costs)
            if best is None or cost < best[2] - 1e-9:
                best = (float(low), float(high), cost, share)
    return best
