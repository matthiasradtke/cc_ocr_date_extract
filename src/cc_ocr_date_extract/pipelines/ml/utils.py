from typing import Dict, Any
import pandas as pd
import numpy as np
import re

from sklearn.metrics import average_precision_score, confusion_matrix


def evaluate_ocr_result(df: pd.DataFrame) -> Dict[str, Any]:
    n_docs = len(pd.unique(df['file_number']))
    n_no_belegdatum = sum(df.dropna().groupby('file_number').apply(lambda x: 'Belegdatum' not in x['label'].values))
    result = {'n_docs': n_docs,
              'n_found_dates': len(df),
              'n_docs_no_date_found': sum(pd.isna(df['match_date'])),
              # how often does no date matches the given belegdatum
              'n_docs_no_belegdatum_found': n_no_belegdatum}

    return result


def custom_avg_precision_score(y_true, y_score):
    # use 1-y_score because make_scorer only yields predict_probab[:,1] for binary case
    return average_precision_score(y_true, 1 - y_score, pos_label='Belegdatum')


def remove_numbers_from_string(s: str) -> str:
    return re.sub(r'[\d\W]+', '', s).lower()


def apply_threshold(label, predict_proba, pos_label, threshold):
    prediction = pd.Series([pos_label if v >= threshold else 'other_date' for v in predict_proba])
    n_pred_pos = sum(prediction == pos_label)
    tn, fp, fn, tp = confusion_matrix(label, prediction, labels=['other_date', 'Belegdatum']).ravel()
    p = tp / (tp + fp)
    n_docs_manual = len(label) - n_pred_pos
    return tn, fp, fn, tp, n_pred_pos, n_docs_manual, threshold, p


def find_threshold(label, predict_proba, pos_label, target_precision):
    # Find prediction threshold so get target precision
    assert (len(label) == len(predict_proba))
    for t in np.arange(0, 1.01, 0.01):
        tn, fp, fn, tp, n_pred_pos, n_docs_manual, t, p = apply_threshold(label, predict_proba, pos_label, t)
        if p >= target_precision:
            return tn, fp, fn, tp, n_pred_pos, n_docs_manual, t, p
    return apply_threshold(label, predict_proba, pos_label, threshold=0.5)


def round_floats_in_dict(d):
    if isinstance(d, dict):
        return {k: np.round(v, 2) if isinstance(v, float) else round_floats_in_dict(v) for (k, v) in d.items()}
    return d