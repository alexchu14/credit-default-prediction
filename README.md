# Credit Default Prediction


This project began as a Homework  for *Deep Learning in Fintech* at NTU (
Spring 2026) and has been restructured and extended afterward: moved
from a single notebook into a reusable `src/` pipeline, with
additional diagnostics and a cost-sensitive decision layer not
required by the original assignment.



## Problem & Motivation

Lenders need to flag likely defaulters before the fact, but the two
error types are not equally costly: missing an actual defaulter
(false negative) typically costs far more than over-scrutinizing a
good customer (false positive). A model report that only shows
accuracy hides this entirely, this project tries to show the
trade-off explicitly instead.



## Data

Default of Credit Card Clients dataset, 30,000 rows, 23 features
([UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
mirrored on [OpenML #42477](https://www.openml.org/d/42477)).

OpenML exposes the columns under anonymized names (`x1`...`x23`)
rather than the original human-readable ones. Mapping, for reference:

| Column | Meaning |
|---|---|
| `x1` | `LIMIT_BAL` — amount of given credit (NT dollar) |
| `x2` | `SEX` (1 = male, 2 = female) |
| `x3` | `EDUCATION` (1 = graduate school, 2 = university, 3 = high school, 4 = other) |
| `x4` | `MARRIAGE` (1 = married, 2 = single, 3 = other) |
| `x5` | `AGE` (years) |
| `x6`–`x11` | `PAY_0, PAY_2...PAY_6` — repayment status, Sept. to Apr. 2005 |
| `x12`–`x17` | `BILL_AMT1...6` — bill statement amount, Sept. to Apr. 2005 |
| `x18`–`x23` | `PAY_AMT1...6` — previous payment amount, Sept. to Apr. 2005 |

Target: `1` = default on next month's payment, `0` = no default.
78% / 22% class split on the training set (16,355 / 4,645 of 21,000).



## Methodology

- Stratified 70/15/15 train/validation/test split.
- Preprocessing: `OneHotEncoder` for categorical columns, `RobustScaler`
  for numeric ones — fit on the training split only.
- Four models (Logistic Regression, SVM, Random Forest, MLP), each
  tuned via manual grid search against the validation set (macro-F1).
- SMOTE applied to the training split only, models retuned, compared
  against the non-resampled baseline.
- Cost-sensitive threshold search (`src/cost_analysis.py`): rather than
  accepting scikit-learn's default 0.5 cutoff, the threshold that
  minimizes expected cost (false negatives weighted 5x false
  positives, by assumption ) is found on the
  validation set and evaluated on the test set.



## Results (initial baseline run)

Test set, default 0.5 threshold:


| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.775 | 0.492 | 0.574 | 0.691 | 0.760 | 0.537 |
| Support Vector Machine | 0.424 | 0.249 | 0.796 | 0.421 | 0.623 | 0.322 |
| Random Forest | 0.814 | 0.637 | 0.365 | 0.676 | 0.760 | 0.530 |
| Multi-Layer Perceptron | 0.815 | 0.651 | 0.354 | 0.674 | 0.776 | 0.548 |


After SMOTE (training set resampled, test distribution unchanged; SVM row
after the `tol` fix and `max_iter=10000`):


| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.768 | 0.481 | 0.585 | 0.687 | 0.759 | 0.535 |
| Support Vector Machine | 0.612 | 0.336 | 0.772 | 0.582 | 0.753 | 0.488 |
| Random Forest | 0.800 | 0.556 | 0.473 | 0.693 | 0.762 | 0.536 |
| Multi-Layer Perceptron | 0.739 | 0.435 | 0.598 | 0.663 | 0.747 | 0.498 |




## Critical Analysis


**Accuracy alone is misleading here.** A trivial classifier that
always predicts "no default" would score 3,504 / 4,500 ≈ **77.9%
accuracy** on this test set, without learning anything. Random Forest
and MLP's ~81.4–81.5% is a real but modest improvement over that
floor; the SVM's 42% is worse than doing nothing. This is why
the comparison tables above lead with Macro-F1 and AUPRC rather than
accuracy.

**The original SVM result was a bug confirmed by re-running after the fix.**

| | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| SVM — before fix (`tol=0.1`) | 0.424 | 0.249 | 0.796 | 0.421 | 0.623 | 0.322 |
| SVM — after fix (`tol=1e-3`, default) | 0.817 | 0.692 | 0.313 | 0.661 | 0.705 | 0.493 |


0.42 accuracy combined with 0.80 recall and 0.25 precision was the
signature of a classifier predicting "default" for most inputs.
After the fix (default `tol=1e-3`, `max_iter` raised from 3000 to 10000 to
compensate for the stricter criterion needing more iterations), the
model is no longer degenerate ; but it is now, notably, the weakest
of the four models on Macro-F1 (0.661), AUROC (0.705), and AUPRC
(0.493), despite having the highest raw accuracy and precision. It
reaches that accuracy by being conservative, it flags a default only when
very confident. This is a second, independent confirmation of the
point above: accuracy alone would have ranked this SVM as
competitive, when in fact it is the least useful
classifier of the four by every metric that accounts for class
imbalance.


**Convergence warnings still appear after the fix, but this is a
different and far less severe issue than before.** With `tol=0.1`,
the solver stopped far from the optimum.
With `tol=1e-3`, hitting `max_iter` means the solver stops very
close to the optimum but hasn't formally certified it: an
efficiency problem, not a correctness one, and the clean, interpretable
results above are consistent with that reading. 


**SMOTE's effect is not uniform across models**, which argues against
treating it as a default best practice:

- Random Forest: Macro-F1 improves (0.676 → 0.693) and recall rises
  substantially (0.365 → 0.473), it catches more real defaulters, at
  the cost of precision (0.637 → 0.556).
- Multi-Layer Perceptron: every metric gets worse (AUROC 0.776 →
  0.747, accuracy 0.815 → 0.739). SMOTE actively hurts this model on
  this dataset.
- Logistic Regression: essentially unchanged (Macro-F1 0.691 → 0.687).

Given that missing a defaulter is assumed to be costlier than
over-flagging a good customer, Random Forest with SMOTE is the more
defensible candidate among the four for this reason alone, the
recall gain is a real trade favoring that cost structure. The
cost-sensitive threshold analysis (`src/cost_analysis.py`) makes this
trade-off explicit and tunable rather than leaving it implicit in a
0.5 cutoff.




## Reproducing updated results


**Cost-sensitive threshold, Random Forest + SMOTE** (`cost_fn=5,
cost_fp=1`), validation-selected threshold applied to the test set:

| Threshold | Precision (Class 1) | Recall (Class 1) | Macro-F1 | FN | FP | Expected cost |
|---|---|---|---|---|---|---|
| Default (0.5) | 0.556 | 0.473 | 0.693 | 525 | 376 | 3,001 |
| Cost-optimal (0.25) | 0.344 | 0.769 | 0.592 | 230 | 1,458 | 2,608 |


Moving to the cost-optimal threshold cuts expected cost by ~13% and
nearly doubles recall (catching 766/996 defaulters vs. 471/996), at
the cost of precision collapsing (0.556 → 0.344) and false positives
nearly quadrupling (376 → 1,458). The marginal trade is about 3.67
extra false positives per false negative avoided ; below the assumed
5:1 budget, so the move is cost-justified under that assumption.


Notably, **Macro-F1 drops** at the cost-optimal threshold (0.693 →
0.592). This is expected, not a bug: Macro-F1 treats both classes
symmetrically and has no notion of asymmetric business cost, so
minimizing expected dollar cost and maximizing Macro-F1 are different
objectives that can  and here do  point to different thresholds.
This is the central reason this module exists rather than just
picking the model with the best F1.


One caveat worth being explicit about: at the cost-optimal threshold,
1,458 of 3,504 genuinely creditworthy test customers (42%) are
flagged. The cost formula only captures the direct dollar cost of
`5×FN + 1×FP`, it does not price in the operational burden of
manually reviewing that many false alarms, customer dissatisfaction,
or reputational cost. The `cost_fn:cost_fp = 5:1` assumption drives
this result and has not been validated against a real institution's
figures. The sensitivity analysis below shows how much it matters.


**Sensitivity to the cost ratio** (Random Forest + SMOTE, threshold
chosen on validation, everything below measured on the test set;
996 defaulters and 3,504 good customers). Costs are in units of one
false positive and are only comparable within a row:

| FN:FP | Optimal threshold | Recall | Precision | Cost: approve all | Cost: flag all | Cost at 0.5 | Cost at optimal |
|---|---|---|---|---|---|---|---|
| 1:1 | 0.66 | 0.320 | 0.659 | 996 | 3,504 | 901 | 842 |
| 2:1 | 0.50 | 0.473 | 0.556 | 1,992 | 3,504 | 1,426 | 1,426 |
| 5:1 | 0.25 | 0.769 | 0.344 | 4,980 | 3,504 | 3,001 | 2,608 |
| 10:1 | 0.14 | 0.934 | 0.264 | 9,960 | 3,504 | 5,626 | 3,258 |
| 20:1 | 0.06 | 0.988 | 0.232 | 19,920 | 3,504 | 10,876 | 3,504 |




## Limitations & Future Work

- **The SVM is the weakest of the four models even after the fix**
  (Macro-F1 0.656, AUPRC 0.511 vs. 0.674–0.691 and 0.530–0.548 for
  the others), and it is the model most sensitive to training-set
  size: convergence needed a higher `max_iter` both for the baseline
  and, more so, for the SMOTE-enlarged training set. 
- **Probability calibration has not been checked.** The cost analysis
  relies on the forest's predicted probabilities, and the gap between
  the optimal thresholds and the textbook ones suggests they are
  miscalibrated (possibly because of SMOTE). A calibration curve, and
  recalibration, would make the
  threshold results more trustworthy.
- **No engineered features yet.** 
- **The 5:1 false-negative-to-false-positive cost ratio is an
  assumption**, not a figure sourced from a real lender, the
  threshold analysis is only as good as this input, and should be
  treated as a demonstration of the method rather than a production
  figure.
- **No experiment tracking** (e.g. MLflow) : each tuning run currently
  overwrites the previous one rather than being logged for
  comparison.












