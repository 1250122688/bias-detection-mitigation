"""
Shared configuration for the Fairlearn vs. AIF360 functional comparison.

Every notebook in this project imports from this module so that the two
libraries are compared on *exactly* the same data, the same feature set and the
same baseline estimator. If the baseline differed between the two sides, any
difference in the reported fairness numbers could not be attributed to the
libraries themselves.

Import this from a notebook that lives in a sub-folder with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parent))
    import common
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A single seed is threaded through data generation, train/test splitting and
# every estimator that accepts a random_state. Notebooks must not create their
# own ad-hoc seeds.
SEED = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def find_project_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Walk upwards from `start` until the directory containing common.py."""
    here = (start or pathlib.Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "common.py").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the project root (no common.py found in any parent "
        f"of {here})."
    )


PROJECT_ROOT = find_project_root(pathlib.Path(__file__).parent)

DATA_DIR = PROJECT_ROOT / "01_data_generation" / "data"
DATA_FIG_DIR = PROJECT_ROOT / "01_data_generation" / "figures"
FAIRLEARN_RESULTS = PROJECT_ROOT / "02_fairlearn" / "results"
FAIRLEARN_FIGURES = PROJECT_ROOT / "02_fairlearn" / "figures"
AIF360_RESULTS = PROJECT_ROOT / "03_aif360" / "results"
AIF360_FIGURES = PROJECT_ROOT / "03_aif360" / "figures"
COMPARISON_DIR = PROJECT_ROOT / "comparison"

TRAIN_CSV = DATA_DIR / "credit_train.csv"
TEST_CSV = DATA_DIR / "credit_test.csv"
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
# The target is the *observed* (historically biased) approval decision, which
# is what a real institution would have in its records.
TARGET = "loan_approved"
FAVORABLE_LABEL = 1
UNFAVORABLE_LABEL = 0

# Oracle columns exist only because the data is synthetic. They record the
# counterfactual "what the label would have been without discrimination" and
# the noise-free credit score. They are NEVER model inputs; they are used only
# to score how much of the injected bias a mitigation actually recovered.
ORACLE_COLUMNS = ["oracle_y_fair", "oracle_credit_score_true"]

SENSITIVE_COLUMNS = ["sex", "age_group", "race"]

# `age` is deliberately kept out of the model: it is the raw sensitive
# attribute behind age_group. Both libraries are evaluated in the common
# "fairness through unawareness" setting, where sensitive attributes are
# withheld from the estimator but bias still leaks through proxies.
NUMERIC_FEATURES = [
    "income",
    "education_years",
    "employment_years",
    "credit_score",
    "debt_to_income",
    "num_prior_defaults",
    "loan_amount",
    "career_gap_months",
]

CATEGORICAL_FEATURES = [
    "region",
    "employment_type",
    "has_cosigner",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------
# Sensitive attribute specifications
# --------------------------------------------------------------------------
# AIF360 requires numeric privileged / unprivileged group definitions, while
# Fairlearn works directly with the string labels. Both views are declared here
# so that the two notebooks cannot drift apart on who counts as privileged.
SENSITIVE_SPECS = {
    "sex": {
        "privileged": "Male",
        "unprivileged": "Female",
        "binary_map": {"Male": 1, "Female": 0},
    },
    "age_group": {
        "privileged": "under_55",
        "unprivileged": "55_plus",
        "binary_map": {"under_55": 1, "55_plus": 0},
    },
    # race is multi-class; GroupA is the majority/privileged group and the two
    # remaining groups are pooled as unprivileged for the binary AIF360 view.
    "race": {
        "privileged": "GroupA",
        "unprivileged": ["GroupB", "GroupC"],
        "binary_map": {"GroupA": 1, "GroupB": 0, "GroupC": 0},
    },
}

# The attribute used whenever a single-axis analysis is needed.
PRIMARY_SENSITIVE = "sex"


def binarize_sensitive(frame: pd.DataFrame, attribute: str) -> pd.Series:
    """Map a sensitive column onto the 1=privileged / 0=unprivileged encoding."""
    mapping = SENSITIVE_SPECS[attribute]["binary_map"]
    return frame[attribute].map(mapping).astype(int)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the (train, test) frames written by the data generation notebook."""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{TRAIN_CSV} not found. Run "
            "01_data_generation/01_generate_biased_credit_data.ipynb first."
        )
    dtypes = {"has_cosigner": "int64"}
    train = pd.read_csv(TRAIN_CSV, dtype=dtypes)
    test = pd.read_csv(TEST_CSV, dtype=dtypes)
    return train, test


def load_ground_truth() -> dict:
    """Return the injected-bias parameters recorded by the generator."""
    with open(GROUND_TRUTH_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the model input matrix and the observed target."""
    return frame[FEATURES].copy(), frame[TARGET].copy()


# --------------------------------------------------------------------------
# Baseline model
# --------------------------------------------------------------------------


def make_baseline_model():
    """The single unaware baseline classifier shared by both notebooks.

    Logistic regression is chosen on purpose: several AIF360 in-processing
    algorithms are themselves linear, and the Fairlearn reductions refit the
    estimator once per iteration, so it has to be cheap.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, solver="lbfgs", random_state=SEED
                ),
            ),
        ]
    )


def encode_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dense numeric feature frames for both splits.

    AIF360 operates on fully numeric arrays rather than sklearn pipelines, so
    its notebooks need the design matrix materialised up front. The encoder is
    fitted on train only, and both splits are returned with identical columns.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    encoder.fit(train[FEATURES])
    names = list(encoder.get_feature_names_out())
    train_enc = pd.DataFrame(
        encoder.transform(train[FEATURES]), columns=names, index=train.index
    )
    test_enc = pd.DataFrame(
        encoder.transform(test[FEATURES]), columns=names, index=test.index
    )
    return train_enc, test_enc


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

# One shared palette so that the same group is the same colour in every figure
# across all three folders.
GROUP_PALETTE = {
    "Male": "#4C72B0",
    "Female": "#DD8452",
    "under_55": "#4C72B0",
    "55_plus": "#DD8452",
    "GroupA": "#4C72B0",
    "GroupB": "#DD8452",
    "GroupC": "#55A868",
}

PRIVILEGED_COLOR = "#4C72B0"
UNPRIVILEGED_COLOR = "#DD8452"


def set_plot_style() -> None:
    """Apply a consistent matplotlib style to every notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def savefig(fig, path: pathlib.Path, name: str) -> pathlib.Path:
    """Persist a figure so results survive outside the notebook outputs."""
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    fig.savefig(target)
    return target


# --------------------------------------------------------------------------
# Environment reporting
# --------------------------------------------------------------------------


def environment_report() -> pd.DataFrame:
    """Return the library versions this run was produced with.

    Printed at the top of every notebook so a stored result can always be tied
    back to the exact stack that produced it.
    """
    rows = [
        ("python", platform.python_version()),
        ("platform", f"{platform.system()} {platform.release()}"),
        ("executable", sys.executable),
        ("seed", str(SEED)),
    ]
    for module_name in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "fairlearn",
        "aif360",
    ]:
        try:
            module = __import__(module_name)
            rows.append((module_name, getattr(module, "__version__", "unknown")))
        except Exception:
            rows.append((module_name, "not installed"))
    return pd.DataFrame(rows, columns=["component", "version"])


def new_rng() -> np.random.Generator:
    """A fresh generator seeded from SEED, for any stochastic step."""
    return np.random.default_rng(SEED)
