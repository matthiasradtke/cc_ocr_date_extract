from typing import Dict, Any
import pandas as pd
import numpy as np
import re

from sklearn.metrics import average_precision_score


def evaluate_ocr_result(df: pd.DataFrame) -> Dict[str, Any]:
    result = {'n_docs': len(pd.unique(df['file_number'])),
              'n_found_dates': len(df),
              'n_docs_no_date_found': sum(pd.isna(df['match_date'])),
              # how often does no date matches the given belegdatum
              'n_docs_no_belegdatum_found': (
                      len(pd.unique(df['file_number'])) -
                      len(pd.unique(df.query("label == 'Belegdatum'")['file_number'])))}

    return result


def custom_avg_precision_score(y_true, y_score):
    # use 1-y_score because make_scorer only yields predict_probab[:,1] for binary case
    return average_precision_score(y_true, 1 - y_score, pos_label='Belegdatum')


def remove_numbers_from_string(s: str) -> str:
    return re.sub(r'[\d\W]+', '', s).lower()


def get_single_date_for_doc(g):
    # first only consider predicted belegdatum, as this is what we want to have..
    res = g.query("prediction == 'Belegdatum'")
    if len(res) == 0:
        # if no belegdatum predicted, use dates where the label is belegdatum
        res = g.query("label == 'Belegdatum'")
        if len(res) == 0:
            res = g
    # use result with highest prediction probability
    res = res.loc[res['predict_proba_predicted_class'].idxmax()]
    return res


def find_threshold(label, predict_proba, pos_label='Belegdatum', target_precision=0.95):
    # Find prediction threshold so get target precision
    assert (len(label) == len(predict_proba))
    df = pd.DataFrame({'label': label})
    for t in np.arange(0, 1.01, 0.01):
        df['prediction_t'] = [pos_label if v >= t else 'other_date' for v in predict_proba]
        tp = len(df[(df['prediction_t'] == pos_label) & (df['label'] == pos_label)])
        fp = len(df[(df['prediction_t'] == pos_label) & (df['label'] != pos_label)])
        n_pred_pos = sum(df['prediction_t'] == pos_label)
        p = tp / (tp + fp)
        if p >= target_precision:
            return tp, n_pred_pos, t, target_precision
    return 0, 0, 1, target_precision


def round_floats_in_dict(d):
    if isinstance(d, dict):
        return {k: np.round(v, 2) if isinstance(v, float) else round_floats_in_dict(v) for (k, v) in d.items()}
    return d
