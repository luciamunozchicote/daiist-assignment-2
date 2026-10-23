# Assignment 2 Report

- **Name**: Lucía Muñoz Chicote
- **Student ID**: 19445
- **Email**: lmunoz.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** At night (20:00 to 08:00) the trauma emergency department of Hospital Universitario La Paz (public, Comunidad de Madrid) has no radiologist on site. Every extremity X-ray taken in that window gets one of three outcomes from the model:

- **PASS**: no fracture, the patient follows the normal pathway.
- **FAIL**: treated as a fracture (splint and fracture-clinic appointment).
- **REVIEW**: sent to a radiologist, who reads it remotely.

The emergency doctor acts on PASS and FAIL; the radiologist reads the REVIEW cases. I chose the night because that is when the problem is real: in European EDs most missed fractures happen between 20:00 and 02:00, and in Spain the on-call radiologist usually reports plain films only until about midnight or 1:00. The rest wait until morning.

**The cost of each mistake.**

| Outcome | What happens | Cost |
|---|---|---|
| False negative (fracture passed) | Patient goes home, returns, may heal badly, need surgery or claim | €1,000 |
| False positive (healthy patient failed) | Unneeded splint and fracture-clinic visit | €150 |
| Review | One remote radiologist read (about 3 min) | €20 |
| True positive / true negative | The right care happens anyway | €0 |

A missed fracture costs about **7 times** a false alarm. That ratio drives everything downstream: the threshold has to sit low, the model should almost never clear a patient on its own, and the metric I care about is cost per 1,000 X-rays (which in practice rewards recall), not accuracy or mAP.

**Where the euros come from** (Madrid public prices, Orden 1975/2023, plus stated assumptions):

- **Missed fracture ≈ €960, rounded to €1,000**: return to trauma ER €175 + first traumatology consultation €115 + repeat X-ray ~€30 + 5% needing surgery because of the delay (5% × €6,000 ≈ €300) + 1.5% ending in an upheld claim (1.5% × median upheld claim of €22,900 in Madrid 2013 to 2022, *Revista Española de Medicina Legal* ≈ €340). The 5% and 1.5% are my assumptions.
- **False alarm ≈ €150**: traumatology consultation €115 + splint and materials ~€35 (assumed).
- **Remote read €20**: the weakest figure. Spanish data is thin; a private report is about €10, and I doubled it for night/urgent work.

## Dataset

- Roboflow Universe, workspace `veda`, project `bone-fracture-detection-daoon`, **version 3** (CC BY 4.0), used as published: <https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/3>
- Task: object detection. One image is one upper-extremity X-ray (most ≤ 512 px). Boxes mark fractures, labelled by location in **6 classes**: elbow, fingers, forearm, humerus, shoulder, wrist.
- I used version 3 because versions 2 and 4 include augmented copies (rotated, brightness-changed). Those inflate the counts and can leak near-duplicates into the test split. Version 3 contains only real images.

**Why it serves the problem.** About half of the images have no fracture at all (no box). The question at the ER is "does this X-ray show a fracture?", and the dataset contains both answers, so I can measure misses and false alarms at image level. A dataset with only fractured images would look similar but could not answer that question.

## What the dataset's composition implies

| Split | Images | No fracture | Boxes | Elbow | Fingers | Forearm | Humerus | Shoulder | Wrist |
|---|---|---|---|---|---|---|---|---|---|
| Train | 1,211 | 607 (50%) | 698 | 113 | 178 | 107 | 104 | 120 | 76 |
| Val | 348 | 175 (50%) | 204 | 29 | 48 | 43 | 36 | 20 | 28 |
| Test | 169 | 86 (51%) | 96 | 17 | 27 | 14 | 15 | 17 | 6 |

The median box covers about **2% of the image** (10th to 90th percentile: 0.6% to 6%). Fractures are tiny objects.

What this means:

- **Per-class test numbers are too thin to trust.** Each class has 6 to 27 test fractures, all below the ~40 the brief warns about. Wrist has 6: one hit more or less moves its recall by about 17 points.
- **So I make the business decision at image level** (any fracture, yes/no). The test set has 83 positive and 86 negative images, which is more than any single class.
- **The base rate is wrong for production.** About 49% of test images have a fracture; in a real ER about 30% of trauma extremity X-rays are positive. I reweight every cost to 30%. Without that, false alarms would look cheaper and the model better than it would be at La Paz.
- **Tiny objects** are hard for detection, so I expect low recall, especially on small joints.
- **The test split may not look like La Paz's images.** The hospitals and patients are unknown, the images are already cropped, and there are only upper-limb views. I cannot conclude anything about children or about lower-limb X-rays (ankles, knees), because they are not in the data.

## Finetuning and baselines

**Model and task.** Detection, because the labels are boxes and a box shows the doctor or radiologist *where* to look. For the business decision I collapse it to an image score: the highest box confidence in the image.

I finetuned **YOLO11n** and **YOLO11s** from their COCO checkpoints: up to 80 epochs, early stopping after 20 epochs without improvement, image size 512 (the source images are ≤ 512 px, so upscaling adds nothing), batch 16, seed 0, trained on a MacBook Air M2 (Apple MPS). YOLO11n ran all 80 epochs; YOLO11s stopped at 68. The kept weights are in `weights/best.pt`.

I kept the one that was cheaper on **validation**. The test set never chooses anything.

| Model | Val mAP50 | Val image ROC-AUC | Val € per 1,000 |
|---|---|---|---|
| **YOLO11n (kept)** | **0.255** | **0.87** | **31,422** |
| YOLO11s | 0.224 | 0.83 | 33,835 |

The bigger model lost because there is too little data (1,211 images, 76 to 178 boxes per class): more parameters overfit sooner and generalise worse. The bottleneck is data, not model size.

**Results on the same test split:**

| Model | mAP50 (boxes) | Image-level ROC-AUC |
|---|---|---|
| Trivial baseline: predict nothing | 0 | 0.50 |
| COCO YOLO11n zero-shot | 0 | 0.47 |
| **Finetuned YOLO11n** | **0.237** | **0.81** |

COCO's 80 classes include no fracture, so the zero-shot model cannot produce a single correct box. At image level I gave it its best chance by counting *any* detection as "fracture": it scores 0.47, no better than chance. The gap (from nothing to 0.81 AUC) is as large as I expected, since the pretrained model has never seen the concept. The absolute level (0.24 mAP50) is modest, which matches the data limits above.

## Per-class evaluation and failure modes

Test set, precision and recall at the confidence that maximises F1. The positive class for the business decision is "fracture present" (any of the 6).

| Class | Test boxes | Precision | Recall | AP50 |
|---|---|---|---|---|
| **Forearm** | 14 | 0.67 | 0.43 | **0.45** (carries the headline) |
| Humerus | 15 | 0.48 | 0.33 | 0.29 |
| Wrist | 6 | 0.20 | 0.33 | 0.22 (6 cases: noise) |
| Shoulder | 17 | 0.28 | 0.41 | 0.19 |
| Fingers | 27 | 0.26 | 0.26 | 0.19 |
| **Elbow** | 17 | 0.16 | 0.12 | **0.08** (drags) |
| All (mean) | 96 | 0.34 | 0.31 | 0.24 (mAP50-95: 0.08) |

**Where it breaks:**

- **Long bones work, small joints don't.** Forearm and humerus fractures are long, visible lines. Elbow and fingers are small joints with overlapping bones, where a fracture is a few pixels.
- **It confuses neighbours.** Humerus and shoulder are mixed up 9 times (the upper humerus *is* part of the shoulder), and fingers are predicted as wrist 8 times. These are partly labelling-boundary problems, not only model errors. For the triage decision they don't matter: any class means "fracture".
- **Fractures missed outright** (predicted as background): forearm 5, fingers 5, elbow 3. These are the expensive errors.
- **Loose boxes.** mAP50 is 0.24 but mAP50-95 is 0.08: the model finds roughly where the fracture is, but its boxes don't fit tightly. Acceptable for pointing a radiologist at a region, not for anything that needs precise localisation.

## Cost matrix and operating point

**Costs:** FN €1,000, FP €150, review €20, correct €0 (sources in the first section).

**Base rate: 30%.** Trauma extremity X-rays in the ED are positive in 30.6% of cases (PubMed 25913167). The test set is 49%, so I measure the rates separately on fractures and on healthy images and reweight them to 30%.

**Cost formula** (per 1,000 X-rays):

1,000 × [0.30 × (share of fractures passed × €1,000 + share reviewed × (€20 + 3.7% × €1,000)) + 0.70 × (share of healthy failed × €150 + share reviewed × (€20 + 2% × €150))]

The radiologist isn't perfect either: I assume they miss 3.7% of fractures (PubMed 16950710) and have 98% specificity (assumed).

**The sweep (model decides alone).** I swept the threshold from 0.00 to 1.00 in steps of 0.01 on validation and took the minimum cost: **0.03**. On test that costs **€87,472 per 1,000 X-rays**, missing 33 fractures and splinting 366 healthy patients per 1,000. The threshold is so low because a miss costs about 7 times a false alarm, so it pays to call "fracture" at very low confidence. 0.5 would only be right if both errors cost the same.

**The assumption that would most change the answer: the cost of a missed fracture.** At €1,000 the best single threshold is 0.03 and 52% of healthy patients get splinted. At €500 or less it rises to 0.08 and only 30% are splinted; at €150 (equal to a false alarm) it reaches 0.26. Above €1,000 nothing changes, because the model already flags almost everything. The base rate matters too: at 15% instead of 30% the threshold rises from 0.03 to 0.08. The dashboard lets you change both and see the answer move.

## What a person still has to check

**Two thresholds** (chosen on validation as the cheapest band whose review load fits the radiologist's time):

- score < 0.01 → **PASS**
- 0.01 to 0.32 → **REVIEW**
- score ≥ 0.32 → **FAIL**

On test this sends **61% of X-rays to review**. Of the cases the model decides itself, it passes 5% of all fractures and splints 6% of all healthy patients. The PASS threshold is almost zero because the model is never confident enough to clear a patient when a miss costs €1,000. Its job is to settle the obvious fractures and pass the uncertain ones to a person.

**Workload.**

- Volume: about **30 extremity X-rays per night**. La Paz had 252,397 ER visits in 2023 (about 690 a day, hospital annual report) × ~25% trauma (Spanish studies: 24 to 27%) × ~25% at night × ~70% X-rayed. The last two shares are my assumptions.
- 61% reviewed → **18.4 reads per night × 3 min = 55 minutes** (3 min per plain film: Royal College of Radiologists, about 20 reports an hour).
- Available time: I assume **60 radiologist minutes per night** for these reads, because the on-call radiologist also covers La Paz's urgent CTs (57,693 CTs a year).
- Because Spanish on-call radiologists usually stop reporting plain films around midnight, the reads that arrive after midnight have to go to a **remote (teleradiology) radiologist**. That is the "remote read" priced at €20, and it is what my recommendation below proposes to buy.
- **It needs about one radiologist-hour a night. That is realistic**, though tight.

**Priced against the alternatives** (test, per 1,000 X-rays, reweighted to 30%):

| Strategy | € per 1,000 | Reviewed | Missed / 1,000 | Splinted / 1,000 | Feasible? |
|---|---|---|---|---|---|
| Do nothing (everyone passes) | 300,000 | 0% | 300 | 0 | Yes, but terrible |
| Model decides alone (0.03) | 87,472 | 0% | 33 | 366 | Yes |
| **Model + review (my design)** | **40,561** | **61%** | **21** | **50** | **Yes, 55 min** |
| Radiologist reads everything | 33,200 | 100% | 11 | 14 | No: needs 90 min, only 60 available |

Reading everything is cheaper per X-ray, but it needs 90 minutes a night that the radiologist doesn't have. Among the feasible options, my design is the cheapest: it costs 7.4 times less than doing nothing and half as much as letting the model decide alone.

**The main finding: radiologist time is worth buying.** For each amount of radiologist time I re-chose the two thresholds on validation and priced them on test:

| Radiologist min / night | Best € per 1,000 | Missed / 1,000 |
|---|---|---|
| 0 | 87,472 | 33 |
| 15 | 72,787 | 34 |
| 30 | 60,787 | 35 |
| 45 | 54,982 | 38 |
| **60 (assumed)** | **40,561** | **21** |
| 75 | 31,779 | 6 |
| 90+ | 31,982 | 7 |

Going from 0 to 75 minutes saves €55,693 per 1,000 X-rays: on average about **€11,000 per extra 15 minutes**, though the steps are uneven (between about €6,000 and €15,000, because each step re-chooses the thresholds on a small validation set). 1,000 X-rays is about 33 nights at 30 a night, so that is roughly **€1,300 per radiologist hour**, far more than an hour of radiologist time costs. My recommendation is to buy night teleradiology up to about 75 minutes; beyond that it stops helping. The model's real value is triage: it lets a scarce radiologist spend their time on the uncertain cases.

**The dashboard** (`uv run python main.py app`) only loads `weights/best.pt` and the saved results in `artifacts/`; it never trains. It lets you upload an X-ray and see the boxes and the PASS / REVIEW / FAIL call, move the two thresholds and the costs and watch the confusion matrix, the cost per 1,000 and the review share update, see the cost-against-threshold curve with the chosen threshold marked, the cost-against-radiologist-time curve, and the per-class results and baselines.

## Limitations & next steps

- **Small test set** (169 images, 6 to 27 fractures per class), so per-class numbers are noise. *Fix:* pool other public fracture datasets (for example FracAtlas, GRAZPEDWRI-DX) and use repeated cross-validation to get confidence intervals.
- **Data, not model size, is the bottleneck** (YOLO11s lost to YOLO11n). *Fix:* more labelled X-rays, ideally La Paz's own archive with the radiologist's report as the label.
- **No pediatric or lower-limb X-rays.** *Fix:* the system must not be used on children or ankles/knees until it has been tested on them; collect and evaluate those images first.
- **Small joints fail** (elbow AP50 0.08). *Fix:* a class-specific threshold, or simply send every elbow view to review.
- **Costs are partly assumed** (5% surgery, 1.5% claims, €20 per read). *Fix:* get La Paz's own figures from finance and the legal department; meanwhile the dashboard shows how the answer moves with each one.
- **Night volume and radiologist time are estimated.** *Fix:* measure them from La Paz's RIS logs for one month before deciding the thresholds.
- **The test images may not resemble La Paz's.** *Fix:* a silent pilot: run the model in the background for a few weeks, compare it to the radiologist's next-morning reports, and only then let it act.

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant throughout this assignment:

- **Dataset and framing:** after I chose the medical area and the fracture X-ray idea, Claude checked several Roboflow Universe datasets for class counts and splits and proposed this one. I asked for the case to be set in Spain and in a specific hospital; Claude proposed La Paz.
- **Figures and sources:** Claude searched for and proposed the costs, the base rate, the radiologist reading time and accuracy, and the La Paz volumes, with their sources (Madrid public prices, La Paz's 2023 annual report, PubMed, SERAM, the Royal College of Radiologists). I reviewed and accepted them.
- **Code and training:** Claude wrote the code (`config.py`, `costing.py`, `train.py`, `app.py`), ran the training on my laptop, proposed the radiologist-time sweep and the YOLO11n vs YOLO11s comparison, and fixed bugs.
- **Report:** Claude prepared a guide explaining every decision and number. I wrote this report from it; Claude corrected two figures and helped with the final formatting.

I studied every decision and number in this report so that I can explain and defend them without the code in front of me.

## How to run it

```bash
uv run python main.py train   # downloads the data, trains, evaluates, saves weights/ and artifacts/
uv run python main.py app     # opens the dashboard (it never trains)
```
