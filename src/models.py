"""Model factory for the phishing detection project."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_logistic_pipeline() -> Pipeline:
    """Return a Logistic Regression pipeline with StandardScaler preprocessing.

    Scaling is essential for Logistic Regression because it puts all features
    on comparable ranges, stabilises gradient-based optimisation, and ensures
    that L2 regularisation affects each coefficient proportionally to its
    predictive contribution rather than its raw numeric scale.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def make_models() -> dict:
    """Return a fresh dictionary of untrained classifiers.

    Tree-based models (Decision Tree, Random Forest) do not require feature
    scaling because their split decisions are threshold-based and invariant
    to monotone transformations of the feature scale.  Logistic Regression
    uses an internal StandardScaler pipeline for the reason described above.

    Returns
    -------
    dict[str, estimator]
        Mapping from model name to an unfitted scikit-learn estimator.
    """
    return {
        "Logistic Regression": make_logistic_pipeline(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
