from typing import Dict, Any, List
import logging

import json

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn_pandas import DataFrameMapper


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> List[pd.DataFrame]:
    """Splits data into training and test sets.
        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.
    """
    df_train = data.sample(frac=parameters['train_size'], random_state=parameters['random_state'])
    df_test = data.drop(df_train.index)

    return [df_train, df_test]


def transform_into_training_format(df: pd.DataFrame) -> pd.DataFrame:
    # load matches as list of dicts
    df['matches'] = df['matches'].apply(lambda x: json.loads(x))
    # explode dataframe so that every match is in a row
    df = df.explode('matches', ignore_index=True)
    # make match details to columns
    df = pd.concat([df, df.apply(lambda x: x['matches'], axis=1, result_type='expand')], axis=1)

    # create label
    df['label'] = df[['Belegdatum', 'match_date']].apply(
        lambda x: 'Belegdatum' if x['Belegdatum'] == x['match_date'] else 'other_date', axis=1
    )
    # create text feature
    df = df.drop('pdf_text', axis=1)
    df['text'] = df['text_left'].fillna('').str.cat(df['text_right'].fillna(''), sep=' ')

    evaluate_ocr_result(df)

    # drop documents were no dates have been found
    df = df[['file_number', 'match_date', 'text', 'label']].dropna()

    return df


def evaluate_ocr_result(df: pd.DataFrame) -> Dict[str, Any]:
    result = {'n_docs': len(pd.unique(df['file_number'])),
              'n_found_dates': len(df),
              'n_docs_no_date_found': sum(pd.isna(df['match_date'])),
              # how often does no date matches the given belegdatum
              'n_docs_no_belegdatum_found': (
                      len(pd.unique(df['file_number'])) -
                      len(pd.unique(df.query("label == 'Belegdatum'")['file_number'])))}

    log = logging.getLogger(__name__)
    log.info("OCR results: " + str(result))

    return result


def train_model(df: pd.DataFrame, parameters: Dict[str, Any]) -> (Pipeline, pd.Series):
    vectorizer_text = TfidfVectorizer(**parameters['vectorizer'])

    feature_selection = None
    classifier = RandomForestClassifier(**parameters['classifier'])

    pipe = Pipeline([
        ('features', DataFrameMapper([
            ('text', vectorizer_text)
        ])),
        ('feature_selection', feature_selection),
        ('clf', classifier),
    ])

    pipe.fit(df, df['label'])

    feature_names = pd.Series(vectorizer_text.get_feature_names())

    return pipe, feature_names


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


def find_threshold(df_, predict_proba, prec=0.95):
    # Find prediction threshold so get target precision
    for t in np.arange(0, 1.01, 0.01):
        df_['prediction_t'] = ['Belegdatum' if v >= t else 'other_date' for v in predict_proba]
        tp = len(df_.query("prediction_t == 'Belegdatum' & label == 'Belegdatum'"))
        fp = len(df_.query("prediction_t == 'Belegdatum' & label != 'Belegdatum'"))
        n_pred_pos = sum(df_['prediction_t'] == 'Belegdatum')
        p = tp / (tp + fp)
        if p >= prec:
            return tp, n_pred_pos, t, prec
    return 0, 0, 1


def evaluate_model(pipe: Pipeline, df: pd.DataFrame) -> pd.Series:
    df_eval = df.reset_index(drop=True).copy()
    df_eval['prediction'] = pipe.predict(df)
    predict_proba = pipe.predict_proba(df)

    result = {'n_docs': len(pd.unique(df_eval['file_number'])),
              'n_dates': len(df_eval),
              'n_label': dict(df_eval['label'].value_counts()),
              'n_prediction': dict(df_eval['prediction'].value_counts()),
              'metrics': {}}

    # Consider all dates
    # result['metrics']['acc_dates'] = accuracy_score(df_eval['label'], df_eval['prediction'])
    # result['metrics']['f1_dates'] = f1_score(df_eval['label'], df_eval['prediction'], pos_label='Belegdatum')

    # Consider single date for each document
    # # Get prediction probability for the predicted class
    classes = list(pipe.classes_)
    df_eval['prediction_int'] = df_eval['prediction'].apply(lambda l: classes.index(l))
    df_eval = pd.concat([df_eval, pd.DataFrame(predict_proba)], axis=1)
    df_eval['predict_proba_predicted_class'] = df_eval.apply(lambda r: r[r['prediction_int']], axis=1)

    df_docs = df_eval.groupby('file_number').apply(get_single_date_for_doc)

    tn, fp, fn, tp = confusion_matrix(df_docs['label'], df_docs['prediction'],
                                      labels=['other_date', 'Belegdatum']).ravel()
    result['metrics']['tn'] = tn
    result['metrics']['fp'] = fp
    result['metrics']['fn'] = fn
    result['metrics']['tp'] = tp

    result['metrics']['acc_docs'] = accuracy_score(df_docs['label'], df_docs['prediction'])
    result['metrics']['f1_docs'] = f1_score(df_docs['label'], df_docs['prediction'], pos_label='Belegdatum')
    result['metrics']['ratio_correct_docs'] = result['metrics']['tp'] / result['n_docs']

    log = logging.getLogger(__name__)

    predict_proba_belegdatum = pipe.predict_proba(df_docs)[:, 0]
    tp, n_pred_pos, t, prec = find_threshold(df_docs, predict_proba_belegdatum)
    log.info(f"Tuned threshold: {t} (prec: {prec})")

    result['metrics']['tuned_tp'] = tp
    result['metrics']['tuned_prec'] = prec
    result['metrics']['tuned_n_docs_we_trust'] = n_pred_pos
    result['metrics']['tuned_n_docs_manual'] = len(df_docs) - n_pred_pos
    result['metrics']['tuned_threshold'] = t

    result['metrics'] = {k: np.round(v, 2) for (k, v) in result['metrics'].items()}

    log.info(f"Model results: {result}")

    return pd.Series(result)