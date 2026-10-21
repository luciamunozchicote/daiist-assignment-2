"""
Gradio dashboard: fracture triage at Hospital Universitario La Paz (night ER).

Run with:  uv run python main.py app      (after: uv run python main.py train)

Loads only what train.py produced: the committed weights in weights/ and the
results in artifacts/. Nothing is trained here. Moving the sliders reprices
the saved test-set scores with the same arithmetic train.py used (costing.py).
"""

import json
import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config as C
import costing

if not (C.ARTIFACTS / "results.json").exists() or not C.WEIGHTS.exists():
    raise SystemExit("Missing weights/ or artifacts/. Run `uv run python main.py train` first.")

R = json.loads((C.ARTIFACTS / "results.json").read_text())
S = json.loads((C.ARTIFACTS / "image_scores.json").read_text())
MODEL = YOLO(str(C.WEIGHTS))
NAMES = R["dataset"]["names"]
TEST_SCORES = np.array(S["test"]["scores"])
TEST_LABELS = np.array(S["test"]["labels"])
TH = R["thresholds"]

INK, ACCENT, WARN, OK, MUTED = "#1f2a44", "#2a6fdb", "#d1495b", "#2ca58d", "#8a94a6"


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def predict_image(image, low, high):
    if image is None:
        return None, "Upload an X-ray to see the model's call."
    low, high = min(low, high), max(low, high)
    r = MODEL.predict(image, conf=max(low, 0.001), imgsz=C.IMGSZ, verbose=False)[0]
    score = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
    if score >= high:
        decision = f"**FAIL: treat as a fracture** (score {score:.2f} ≥ {high:.2f})"
    elif score >= low:
        decision = f"**REVIEW: send to the remote radiologist** ({low:.2f} ≤ score {score:.2f} < {high:.2f})"
    else:
        decision = f"**PASS: no fracture found** (score {score:.2f} < {low:.2f})"
    boxes = [f"- {NAMES[int(c)]}: {float(p):.2f}" for c, p in zip(r.boxes.cls, r.boxes.conf)]
    detail = "\n".join(boxes) if boxes else "- no boxes above the review threshold"
    return r.plot()[:, :, ::-1], f"{decision}\n\nBoxes shown (confidence ≥ {low:.2f}):\n{detail}"


def operating_view(low, high, cost_fn, cost_fp, cost_review, base_rate):
    low, high = min(low, high), max(low, high)
    costs = dict(base_rate=base_rate, cost_fn=cost_fn, cost_fp=cost_fp, cost_review=cost_review)
    rates = costing.outcome_rates(TEST_SCORES, TEST_LABELS, low, high)
    cost = costing.cost_per_1000(rates, **costs)
    share = costing.review_share(rates, base_rate)
    k = costing.counts(TEST_SCORES, TEST_LABELS, low, high)

    z = [[k["pass_fracture"], k["review_fracture"], k["fail_fracture"]],
         [k["pass_no_fracture"], k["review_no_fracture"], k["fail_no_fracture"]]]
    text = [[f"missed<br>{z[0][0]}", f"to radiologist<br>{z[0][1]}", f"caught<br>{z[0][2]}"],
            [f"cleared<br>{z[1][0]}", f"to radiologist<br>{z[1][1]}", f"unneeded splint<br>{z[1][2]}"]]
    cm = go.Figure(go.Heatmap(z=z, x=["PASS", "REVIEW", "FAIL"], y=["Fracture", "No fracture"],
                              text=text, texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Test split ({len(TEST_LABELS)} X-rays): decision vs truth",
                     xaxis_title="Decision", yaxis_title="Truth", height=360)

    reviews = share * C.CASES_PER_NIGHT
    minutes = reviews * C.REVIEW_MINUTES
    fits = minutes <= C.REVIEW_MINUTES_AVAILABLE + 1e-9
    alt = {
        "Do nothing (everyone passes)": costing.cost_per_1000(costing.outcome_rates(TEST_SCORES, TEST_LABELS, 1.01, 1.01), **costs),
        "Radiologist reads everything": costing.cost_per_1000(costing.outcome_rates(TEST_SCORES, TEST_LABELS, -1, 1.01), **costs),
        f"Model alone at {TH['single']:.2f}": costing.cost_per_1000(costing.outcome_rates(TEST_SCORES, TEST_LABELS, TH["single"], TH["single"]), **costs),
    }
    alt_rows = "\n".join(f"| {name} | €{v:,.0f} |" for name, v in alt.items())
    kpis = f"""
### €{cost:,.0f} per 1,000 X-rays · {share:.1%} sent for review

| | |
|---|---|
| Reviews per night ({C.CASES_PER_NIGHT} X-rays) | **{reviews:.1f}** → {minutes:.0f} radiologist minutes |
| Radiologist time available | {C.REVIEW_MINUTES_AVAILABLE} min/night → {"✅ fits" if fits else "⚠️ **over capacity**"} |
| Fractures missed by the model (pass) | {rates['pos_pass']:.1%} of fractures |
| Healthy patients splinted (fail) | {rates['neg_fail']:.1%} of healthy patients |

**Compare (same costs and base rate):**

| Alternative | Cost per 1,000 |
|---|---|
| **This band ({low:.2f}–{high:.2f})** | **€{cost:,.0f}** |
{alt_rows}

Rates are measured on the test split, then reweighted to a production base rate of {base_rate:.0%}
(the dataset is ~49% fractures). Reviewed cases cost €{cost_review:.0f} plus the radiologist's own
{1 - C.REVIEWER_SENSITIVITY:.1%} miss rate.
"""
    return cm, kpis, cost_curve(low, high, costs)


def cost_curve(low=None, high=None, costs=None):
    costs = costs or {}
    val = costing.sweep_single(S["val"]["scores"], S["val"]["labels"], **costs)
    test = costing.sweep_single(TEST_SCORES, TEST_LABELS, **costs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[t for t, _ in val], y=[c for _, c in val], name="validation (chooses)",
                             line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=[t for t, _ in test], y=[c for _, c in test], name="test (reports)",
                             line=dict(color=WARN, dash="dot")))
    t_best = min(val, key=lambda x: x[1])[0]
    fig.add_vline(x=t_best, line_color=OK, line_dash="dash",
                  annotation_text=f"chosen single threshold {t_best:.2f}", annotation_position="top right")
    if low is not None:
        fig.add_vrect(x0=low, x1=max(high, low + 0.002), fillcolor=ACCENT, opacity=0.12, line_width=0,
                      annotation_text="review band", annotation_position="top left")
    fig.update_layout(title="Expected cost per 1,000 X-rays vs threshold (model decides alone)",
                      xaxis_title="Confidence threshold", yaxis_title="€ per 1,000 X-rays", height=420)
    return fig


def per_class_table():
    df = pd.DataFrame(R["test_box"]["per_class"])
    df = df.rename(columns={"class": "Class", "test_instances": "Test instances", "precision": "Precision",
                            "recall": "Recall", "ap50": "AP50", "map50_95": "mAP50-95"})
    o = R["test_box"]["overall"]
    df.loc[len(df)] = ["ALL (mean)", int(df["Test instances"].sum()), o["precision"], o["recall"], o["map50"], o["map50_95"]]
    return df.round(3)


def confusion_plot():
    m = np.array(R["test_box"]["confusion_matrix"])
    labels = R["test_box"]["confusion_labels"]
    fig = go.Figure(go.Heatmap(z=m, x=labels, y=labels, text=m.astype(int), texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Box-level confusion matrix (test, rows = predicted, columns = true)",
                      xaxis_title="True", yaxis_title="Predicted", height=520)
    return fig


def baselines_table():
    b = R["baselines"]
    return pd.DataFrame([
        {"Model": "Predict nothing", "mAP50 (boxes)": 0.0, "Image-level ROC-AUC": b["predict_nothing"]["image_roc_auc"]},
        {"Model": "COCO yolo11n zero-shot", "mAP50 (boxes)": 0.0, "Image-level ROC-AUC": round(b["coco_zero_shot"]["image_roc_auc"], 3)},
        {"Model": "Finetuned yolo11n", "mAP50 (boxes)": round(b["finetuned"]["map50"], 3), "Image-level ROC-AUC": round(b["finetuned"]["image_roc_auc"], 3)},
    ])


def strategies_table():
    rows = []
    for s in R["strategies"]:
        rows.append({"Strategy": s["strategy"], "Thresholds": f"{max(s['low'], 0):.2f} / {min(s['high'], 1):.2f}",
                     "€ per 1,000": round(s["cost_per_1000"]), "Reviewed": f"{s['review_share']:.0%}",
                     "Reviews/night": round(s["reviews_per_night"], 1),
                     "Missed fractures /1,000": round(s["missed_fractures_per_1000"], 1),
                     "Unneeded splints /1,000": round(s["unneeded_splints_per_1000"], 1)})
    return pd.DataFrame(rows)


def dataset_table():
    st = R["dataset"]["stats"]
    rows = []
    for split, s in st.items():
        rows.append({"Split": split, "Images": s["images"], "No fracture": s["images_without_fracture"],
                     "Boxes": s["boxes"], **s["per_class"],
                     "Median box (% of image)": round(100 * s["box_area_share_median"], 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
INTRO = f"""
# Fracture triage: Hospital Universitario La Paz, night emergency department

Each extremity X-ray taken between 20:00 and 08:00 is **passed** (no fracture), **failed** (treated as a
fracture) or **sent to a remote radiologist**. Finetuned **{R['training']['base_model']}** on
`{C.WORKSPACE}/{C.PROJECT}` v{C.VERSION}. This app only loads `weights/best.pt` and saved results; nothing is retrained.
Costs: missed fracture €{C.COST_FN}, unneeded splint €{C.COST_FP}, radiologist read €{C.COST_REVIEW}. Base rate {C.BASE_RATE:.0%}.
"""

with gr.Blocks(title="Fracture triage") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        low = gr.Slider(0, 1, value=TH["low"], step=0.01, label="Review threshold (below → PASS)")
        high = gr.Slider(0, 1, value=TH["high"], step=0.01, label="Fail threshold (at or above → FAIL)")

    with gr.Tab("Operating point"):
        with gr.Row():
            c_fn = gr.Number(value=C.COST_FN, label="Missed fracture (€)")
            c_fp = gr.Number(value=C.COST_FP, label="Unneeded splint (€)")
            c_rv = gr.Number(value=C.COST_REVIEW, label="Radiologist read (€)")
            br = gr.Slider(0.05, 0.6, value=C.BASE_RATE, step=0.01, label="Base rate (share of X-rays with a fracture)")
        kpi = gr.Markdown()
        with gr.Row():
            cm = gr.Plot()
            curve = gr.Plot()
        inputs = [low, high, c_fn, c_fp, c_rv, br]
        for comp in inputs:
            comp.change(operating_view, inputs, [cm, kpi, curve])
        demo.load(operating_view, inputs, [cm, kpi, curve])
        gr.Markdown("### Chosen strategies on the test split (thresholds chosen on validation)")
        gr.Dataframe(strategies_table(), interactive=False)

    with gr.Tab("Try an X-ray"):
        with gr.Row():
            img = gr.Image(type="numpy", label="Upload an X-ray")
            out = gr.Image(label="Model output")
        verdict = gr.Markdown()
        for comp in [img, low, high]:
            comp.change(predict_image, [img, low, high], [out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown("Precision and recall per class at the confidence that maximises F1; AP50 and mAP50-95 "
                    "over all confidences. Test instances per class are small, so per-class figures are noisy.")
        gr.Dataframe(per_class_table(), interactive=False)
        gr.Plot(confusion_plot())
        gr.Markdown("### Baselines on the same test split")
        gr.Dataframe(baselines_table(), interactive=False)

    with gr.Tab("Dataset"):
        gr.Dataframe(dataset_table(), interactive=False)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("APP_PORT", 7860)),
                inbrowser=not os.environ.get("CI"))
