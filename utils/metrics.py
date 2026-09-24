"""Regression metrics on log and dollar scale + classification metrics."""

import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def regression_metrics(y_log, p_log, smearing: float = 1.0) -> dict:
    """y_log, p_log = log(revenue). Dollar predictions = exp(p_log) * smearing (Duan)."""
    y_log, p_log = np.asarray(y_log), np.asarray(p_log)
    y_usd, p_usd = np.exp(y_log), np.exp(p_log) * smearing
    ape = np.abs(p_usd - y_usd) / y_usd
    return {
        "RMSE_log": rmse(y_log, p_log),
        "MAE_log": float(mean_absolute_error(y_log, p_log)),
        "R2_log": float(r2_score(y_log, p_log)),
        "RMSE_usd": rmse(y_usd, p_usd),
        "MAE_usd": float(mean_absolute_error(y_usd, p_usd)),
        "MedianAE_usd": float(np.median(np.abs(p_usd - y_usd))),
        "MedianAPE_pct": float(np.median(ape) * 100),
        "within_2x_pct": float(np.mean((p_usd / y_usd <= 2) & (p_usd / y_usd >= 0.5)) * 100),
    }


def classification_metrics(y, prob, threshold: float = 0.5) -> dict:
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": float(threshold), "AUC": float(roc_auc_score(y, prob)),
        "AP": float(average_precision_score(y, prob)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "sensitivity_recall": float(sens), "specificity": float(spec),
        "precision": float(prec),
        "F1": float(2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0,
        "accuracy": float((tp + tn) / len(y)),
    }

# ----------------------------------------------------------------------------- model helpers
import statsmodels.api as sm


def fit_ols(y, X, cov_type: str = "nonrobust"):
    return sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X, has_constant="add")).fit(
        cov_type=cov_type)


def predict_ols(res, X) -> np.ndarray:
    return np.asarray(res.predict(sm.add_constant(X, has_constant="add")))


def fit_glm_gamma(y_millions, X):
    """Gamma GLM with log link on revenue in USD millions (Lecture 3)."""
    fam = sm.families.Gamma(link=sm.families.links.Log())
    return sm.GLM(np.asarray(y_millions, dtype=float),
                  sm.add_constant(X, has_constant="add"), family=fam).fit()


def predict_glm_log(res, X) -> np.ndarray:
    """Returns log(revenue USD) so it is comparable with the OLS on log revenue."""
    return np.log(np.asarray(res.predict(sm.add_constant(X, has_constant="add"))) * 1e6)


def drop_rare_columns(X, min_count: int = 10) -> list:
    """Drop zero-variance columns and 0/1 columns with fewer than min_count ones (or zeros)."""
    keep = []
    for c in X.columns:
        s = X[c].astype(float)
        if s.std() == 0:
            continue
        if set(np.unique(s.values)) <= {0.0, 1.0} and min(s.sum(), len(s) - s.sum()) < min_count:
            continue
        keep.append(c)
    return keep


def expected_cost(y, prob, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Average misclassification cost per film given a loss matrix (Lecture 2, slide 40)."""
    y, pred = np.asarray(y), (np.asarray(prob) >= threshold).astype(int)
    fn = np.sum((y == 1) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    return float((cost_fn * fn + cost_fp * fp) / len(y))