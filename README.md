# Fairlearn vs. AIF360 — a functional comparison

A reproducible, end-to-end comparison of the two main open-source fairness
toolkits, run on synthetic credit-approval data whose bias was **deliberately
injected with known parameters**.

The point of generating the data rather than using Adult or COMPAS is that
nobody knows how much discrimination those datasets actually contain, so a
fairness number computed on them can never be checked against a reference.
Here it can: the generator records both the biased labels and the
counterfactually fair ones.

---

## Quick start

Everything is already executed and committed with outputs — open any notebook
to read the results without running anything. To reproduce from scratch:

```bash
python -m pip install "fairlearn" "aif360[DisparateImpactRemover]"
```

Then run the notebooks in order, selecting the **Python 3.10.5** kernel:

1. `01_data_generation/01_generate_biased_credit_data.ipynb`
2. `02_fairlearn/02a_fairlearn_detection.ipynb`
3. `02_fairlearn/02b_fairlearn_mitigation.ipynb`
4. `03_aif360/03a_aif360_detection.ipynb`
5. `03_aif360/03b_aif360_mitigation.ipynb`
6. `comparison/04_fairlearn_vs_aif360.ipynb`

Or headlessly, from the project root:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=3.10.5 --ExecutePreprocessor.timeout=5400 01_data_generation/01_generate_biased_credit_data.ipynb 02_fairlearn/02a_fairlearn_detection.ipynb 02_fairlearn/02b_fairlearn_mitigation.ipynb 03_aif360/03a_aif360_detection.ipynb 03_aif360/03b_aif360_mitigation.ipynb comparison/04_fairlearn_vs_aif360.ipynb
```

Total runtime is roughly 8 minutes.

### Environment this was produced with

| component | version |
|-----------|---------|
| Python | 3.10.5 (pyenv-win) |
| fairlearn | 0.14.0 |
| aif360 | 0.6.1 |
| BlackBoxAuditing | 0.1.54 |
| numpy / pandas / scikit-learn | 2.2.5 / 2.2.3 / 1.6.1 |

Every notebook prints its own environment table as its first output, so a
stored result can always be tied back to the stack that produced it.

---

## Layout

```
common.py                     shared config: seed, paths, schema, the ONE baseline model
01_data_generation/           synthetic data with four injected bias mechanisms
    data/                     credit_train.csv, credit_test.csv, ground_truth.json, schema.md
    figures/                  one figure per injected mechanism
02_fairlearn/                 detection (02a) and mitigation (02b)
    results/ figures/
03_aif360/                    detection (03a) and mitigation (03b)
    results/ figures/
comparison/                   capability matrices, head-to-head, combined leaderboard
    results/ figures/
```

`common.py` is what keeps the comparison honest. Both libraries are handed the
same split and the same `make_baseline_model()`, and notebooks 03a/03b assert
that their predictions are byte-identical to the ones 02a audited. Any
difference in the reported numbers is therefore a difference between the
libraries, never between the models.

---

## The dataset

14,677 synthetic loan applications (10,273 train / 4,404 test), seed 42.
Sensitive attributes: `sex`, `age_group`, `race`. The model sees 11 features
and **none of the sensitive attributes** — the "fairness through unawareness"
setting most teams actually deploy.

### Four injected bias mechanisms

| ID | Mechanism | Injection | Breaks |
|----|-----------|-----------|--------|
| **B1** | Historical / label bias | qualified applicants in unprivileged groups flipped 1→0 with group-specific probability (composed multiplicatively) | demographic parity, equal opportunity |
| **B2** | Measurement bias | `credit_score` observed with 3–4× more noise for women and non-majority race groups | equalized odds, per-group AUC |
| **B3** | Proxy leakage | `career_gap_months`, `employment_type` track `sex`; `region` tracks `race` | the idea that dropping the sensitive column helps |
| **B4** | Representation bias | high-income women and `GroupC` undersampled | minority-group precision, statistical power |

Sensitive attributes are recoverable from the non-sensitive features alone at
AUC **0.877** (sex), **0.988** (age_group) and **0.711** (race).

### The oracle columns

`oracle_y_fair` (the label before B1) and `oracle_credit_score_true` (the score
before B2) are stored alongside the data and are **never model inputs**. They
make it possible to ask the question no real dataset can answer: *did this
mitigation algorithm actually recover the fair outcome, or just move a metric?*

### A caveat the notebooks take seriously

`oracle_y_fair` still contains a selection-rate gap of about **−0.14**, because
the generator also applies an upstream pay penalty to women. That residual is
not a "legitimate" difference — it is labour-market discrimination the lender
did not cause but would pass through. So there are two defensible targets, and
notebook 02b reports against both rather than picking one:

* **target A** (gap ≈ −0.14): remove only the lender's own discrimination;
* **target B** (gap = 0): also neutralise the inherited inequality.

Which one you want is a policy choice. Neither library makes it for you, and
neither warns you that an API argument just made it on your behalf.

---

## What was found

### The two libraries agree where they overlap

Demographic parity difference, parity ratio (AIF360 calls it
`disparate_impact`) and equal opportunity difference produce identical values
in both libraries, up to AIF360's signed vs. Fairlearn's unsigned convention.
Choosing between them is never about metric correctness.

More strikingly, AIF360's `ExponentiatedGradientReduction` **imports and calls
Fairlearn internally**, and reproduces its predictions exactly. At the
in-processing reductions stage the two libraries are the same code.

### Capability scores, as actually exercised

| | Fairlearn | AIF360 |
|---|---|---|
| Detection capabilities (of 20) | 12.5 | 15.0 |
| Mitigation: worked | 6 | 10 |
| Mitigation: absent | 8 | 3 |
| Mitigation: broken or degenerate | 0 | 2 |

### Fairlearn's distinctive strengths

* **`MetricFrame` accepts any callable.** A custom business cost function
  becomes a group metric with `make_derived_metric`. AIF360's catalogue is
  closed; extending it means subclassing.
* **Bootstrap confidence intervals** on every quantity. AIF360 reports point
  estimates only, everywhere — which matters because B4 made some subgroups
  small.
* **`control_features`** conditions *any* metric on *any* stratifier. AIF360
  has one fixed `conditional_demographic_disparity`.
* **`GridSearch`** returns the whole Pareto frontier as deterministic models,
  rather than one randomised ensemble.
* Everything ran first time.

### AIF360's distinctive strengths

* **Dataset-level metrics** (`BinaryLabelDatasetMetric`) audit the data before
  a model exists. This is how you decide whether pre-processing is even the
  right stage. Fairlearn has no such entry point.
* **Individual fairness**: `consistency` needs the feature matrix, so it is
  structurally out of `MetricFrame`'s reach.
* **Distributional inequality**: generalized entropy / Theil, with a
  between-vs-within decomposition.
* **Automated bias discovery** via MDSS `bias_scan`.
* **`Reweighing`** — free, interpretable, estimator-agnostic, no Fairlearn
  equivalent.
* **Post-processing depth**: three algorithms with explicit control over which
  error cost is equalised.

### Three findings that needed both libraries

**1. Unawareness bought almost nothing.** Fairlearn measured a demographic
parity difference of **0.204** on `sex` — about **95%** of the 0.214 gap
present in the training labels — from a model that never saw `sex`. The parity
ratio of 0.61 would fail the US four-fifths rule.

**2. Group fairness and individual fairness disagree here.** AIF360's entropy
decomposition shows group membership explains **under 0.01%** of the
individual-level inequality in who receives a wrong decision (between-group GEI
0.000012 against a total of 0.1249). Both readings are true; a team running
only one library would have believed only half the story.

**3. A mitigation can "improve fairness" while approving zero women.**
`CalibratedEqOddsPostprocessing(cost_constraint="fpr")` equalised its target
almost perfectly — generalized FPR gap from −0.051 to −0.004 — by driving the
female selection rate to **0.000**. Accuracy barely moved (0.755 → 0.736). The
metric it was told to optimise improved. Only a *panel* of metrics, including a
per-group degeneracy check, catches this.

Across all 39 mitigation runs, **3 were degenerate** — some group approved
almost always or almost never — and every one of them posts an excellent score
on the fairness metric it was optimising.

### The best methods on this dataset

Only three runs improved accuracy against the *oracle fair* labels, i.e. got
better at the real task rather than just at the metric:

| library | method | acc. vs oracle | gain | gap |
|---------|--------|---------------|------|-----|
| aif360 | `RejectOptionClassification` (avg odds) | 0.7881 | +0.0098 | −0.112 |
| aif360 | `RejectOptionClassification` (SPD) | 0.7834 | +0.0050 | −0.031 |
| fairlearn | `ThresholdOptimizer` (equalized odds) | 0.7797 | +0.0014 | −0.107 |

`RejectOptionClassification` wins because its mechanism matches the shape of
the injected bias: it flips only decisions inside a band of uncertainty around
the boundary, and B1 put many wrongly-denied applicants exactly there. The
lesson is that the best algorithm depends on the *shape* of the bias, not on
the stage it operates at.

---

## Not covered, and why

| algorithm | library | reason |
|-----------|---------|--------|
| `AdversarialFairnessClassifier` | Fairlearn | requires `torch`; not installed |
| `AdversarialDebiasing` | AIF360 | requires `tensorflow`; not installed |
| `OptimPreproc` | AIF360 | requires `cvxpy`; not installed |

Neither library is advantaged: both lose their adversarial in-processor.

Two AIF360 algorithms were run and **did not work**, which is recorded rather
than omitted:

* **`MetaFairClassifier`** raises `TypeError: cannot unpack non-iterable
  NoneType` for every `tau > 0` — i.e. for every setting that actually requests
  fairness. The solver in `celisMeta/StatisticalRate.py` returns `None`. Only
  `tau=0` runs, which switches the constraint off and produces a *worse* gap
  (−0.320) than the plain baseline.
* **`GerryFairClassifier`** collapses to a constant classifier under its
  documented defaults. Its base learner is a `LinearRegression` whose outputs on
  standardised features never cross the 0.5 threshold, so the ensemble is
  degenerate before fairness enters the picture.

A third is a silent trap rather than a failure: AIF360's post-processors need a
**held-out calibration split**. Fitted on the same data the model was trained
on they mis-calibrate badly, and nothing warns you. Notebook 03b uses a 70/30
split within the training data and says so.

---

## Reproducibility

* One seed (`common.SEED = 42`) threads through data generation, splitting and
  every estimator that accepts a `random_state`.
* `ExponentiatedGradient` is randomised by design; every `predict` call passes
  the seed explicitly.
* Notebooks are stored **with their outputs**, and every table is also written
  to a CSV under `results/` so the numbers survive independently of the
  notebook JSON.
* `01_data_generation/data/ground_truth.json` records the injected parameters
  *and* the disparities they actually produced, including the calibration
  constants used to keep the age axis interpretable.

One design note worth flagging: the experience term in the generator is capped
at 10 years. Without that cap `employment_years` grows monotonically with age,
older applicants look genuinely lower-risk, and the nominally unprivileged
`55_plus` group ends up with the *higher* approval rate — which silently
inverts the sign of every age fairness metric downstream. The cap and a small
repayment-horizon penalty were calibrated so the fair label is close to
age-neutral. Both constants are recorded in `ground_truth.json`.

---

## Which library should you use

They are complements more often than substitutes.

**Fairlearn** if your fairness question is about a metric you already have, if
you need to know whether a disparity is statistically real, if your model is
not a logistic regression, or if the code has to be maintained by people who
have not read the fairness literature.

**AIF360** if you need to audit data before training, if individual fairness
matters, if you do not know which subgroup is harmed, or if you want
pre-processing and post-processing options that Fairlearn simply does not have.

**Both** if the work is real. Each of the three findings above came from a
different library, and the third one — the mitigation that approves zero women
while reporting improved fairness — is the one that would otherwise have
reached production.
