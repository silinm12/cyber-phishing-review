"""Model training and evaluation utilities."""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import make_models


def get_positive_class_scores(model, X_data: pd.DataFrame):
    """Return positive-class probability scores for *X_data*.

    Uses ``predict_proba`` when available, falls back to ``decision_function``,
    and finally to hard ``predict`` labels.  This allows the same scoring
    function to work with both probabilistic and non-probabilistic estimators.

    Parameters
    ----------
    model:
        A fitted scikit-learn estimator.
    X_data:
        Feature matrix for which to produce scores.

    Returns
    -------
    np.ndarray
        1-D array of positive-class scores.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_data)
    return model.predict(X_data)


def evaluate_model_suite(
    X_train_data: pd.DataFrame,
    X_test_data: pd.DataFrame,
    y_train_data: pd.Series,
    y_test_data: pd.Series,
    model_dict=None,
):
    """Train and evaluate every model in *model_dict* using a consistent metric set.

    Metrics reported:
    - Accuracy       - overall fraction of correct predictions.
    - Precision      - fraction of predicted phishing sites that are truly phishing.
    - Recall         - fraction of actual phishing sites that were detected.
    - F1 Score       - harmonic mean of precision and recall.
    - F2 Score       - recall-weighted F-beta (beta=2); emphasizes reducing false negatives
                       because missing a phishing site is more harmful than a false alert.
    - Matthews Correlation Coefficient (MCC) - balanced metric using all four confusion
                       matrix cells; robust to class imbalance.
    - ROC-AUC        - area under the ROC curve; measures ranking quality.
    - PR-AUC         - area under the precision-recall curve; especially informative
                       for the positive (phishing) class.

    Parameters
    ----------
    X_train_data, X_test_data:
        Feature matrices for training and evaluation.
    y_train_data, y_test_data:
        Target labels for training and evaluation.
    model_dict:
        Optional mapping of model name to unfitted estimator.
        Defaults to :func:`src.models.make_models`.

    Returns
    -------
    results : pd.DataFrame
        Per-model metric table sorted by F1 Score (descending).
    fitted_models : dict
        Mapping of model name to fitted estimator.
    predictions : dict
        Mapping of model name to hard label predictions on the test set.
    scores : dict
        Mapping of model name to positive-class probability scores on the test set.
    """
    if model_dict is None:
        model_dict = make_models()

    evaluation_rows = []
    fitted_models: dict = {}
    predictions: dict = {}
    scores: dict = {}

    for model_name, model in model_dict.items():
        fitted_model = clone(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fitted_model.fit(X_train_data, y_train_data)
            y_pred = fitted_model.predict(X_test_data)
            y_score = get_positive_class_scores(fitted_model, X_test_data)

        fitted_models[model_name] = fitted_model
        predictions[model_name] = y_pred
        scores[model_name] = y_score

        evaluation_rows.append({
            "Model": model_name,
            "Accuracy":  accuracy_score(y_test_data, y_pred),
            "Precision": precision_score(y_test_data, y_pred, zero_division=0),
            "Recall":    recall_score(y_test_data, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test_data, y_pred, zero_division=0),
            "F2 Score":  fbeta_score(y_test_data, y_pred, beta=2, zero_division=0),
            "Matthews Correlation Coefficient": matthews_corrcoef(y_test_data, y_pred),
            "ROC-AUC":   roc_auc_score(y_test_data, y_score),
            "PR-AUC":    average_precision_score(y_test_data, y_score),
        })

    results = pd.DataFrame(evaluation_rows).sort_values("F1 Score", ascending=False)
    return results, fitted_models, predictions, scores
