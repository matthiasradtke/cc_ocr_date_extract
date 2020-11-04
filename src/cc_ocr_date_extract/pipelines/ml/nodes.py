from typing import Dict, Any, List
import logging
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, f1_score, average_precision_score, accuracy_score, confusion_matrix

from .utils import evaluate_ocr_result, remove_numbers_from_string, custom_avg_precision_score
from .utils import get_single_date_for_doc, find_threshold, round_floats_in_dict


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

    ocr_result = evaluate_ocr_result(df)
    log = logging.getLogger(__name__)
    log.info(f"OCR results: {ocr_result}")

    # drop documents were no dates have been found
    df = df[['file_number', 'match_date', 'text', 'label']].dropna()

    return df


def train_model(df: pd.DataFrame, parameters: Dict[str, Any]) -> (Pipeline, str, pd.Series):
    vectorizer = TfidfVectorizer(preprocessor=remove_numbers_from_string, **parameters['vectorizer'])
    classifier = RandomForestClassifier(n_jobs=parameters['n_jobs'], random_state=parameters['random_state'],
                                        **parameters['classifier'])
    pipe = Pipeline([
        ('features', make_column_transformer((vectorizer, 'text'))),
        ('clf', classifier),
    ])

    log = logging.getLogger(__name__)

    scoring = {
        'f1': make_scorer(f1_score, pos_label='Belegdatum'),
        'avg_prec': make_scorer(custom_avg_precision_score, needs_proba=True),
    }
    cv = GroupShuffleSplit(random_state=parameters['random_state'])

    if not parameters['perform_grid_search']:
        best_pipeline = pipe.fit(df, df['label'])
        best_params = str(best_pipeline.get_params())
    else:
        param_grid = {
            'features__tfidfvectorizer__analyzer': ['char_wb'],
            # 'features__tfidfvectorizer__preprocessor': [None, remove_numbers_from_string],
            # 'features__tfidfvectorizer__ngram_range': [(1, 4), (1, 5), (1, 6)],
            'features__tfidfvectorizer__max_df': [0.7, 0.8, 0.9],
            # 'features__tfidfvectorizer__max_features': [1000, None],
            'features__tfidfvectorizer__min_df': [1],
            # 'features__tfidfvectorizer__use_idf': [True, False],
            # 'features__tfidfvectorizer__binary': [True, False],
            'features__tfidfvectorizer__binary': [False],
            'clf__n_estimators': [100, 200, 300],
            # 'clf__max_features': [None, 'sqrt', 100],
        }
        grid_search = GridSearchCV(pipe, param_grid,
                                   refit='avg_prec',
                                   scoring=scoring, cv=cv, verbose=1, n_jobs=parameters['n_jobs'])
        grid_search.fit(df, df['label'], groups=df['file_number'])

        best_pipeline = grid_search.best_estimator_
        best_params = str(grid_search.best_params_)

        log.info(f"Grid Search: Best parameters: {best_params})")

    # final cv scores
    cv_scores = cross_validate(best_pipeline, df, y=df['label'], groups=df['file_number'],
                               scoring=scoring, cv=cv, n_jobs=parameters['n_jobs'], return_train_score=True)
    cv_scores = {k: np.round(np.mean(v), 2) for (k, v) in cv_scores.items()}

    log.info(f"F1 test score: {cv_scores['test_f1']}")
    log.info(f"Avg precision test score: {cv_scores['test_avg_prec']}")
    log.info(f"F1 train score: {cv_scores['train_f1']}")
    log.info(f"Avg precision train score: {cv_scores['train_avg_prec']}")

    feature_names = pd.Series(best_pipeline.named_steps['features'].get_feature_names())

    return best_pipeline, best_params, feature_names


def evaluate_model(pipe: Pipeline, df: pd.DataFrame) -> pd.Series:
    df_eval = df.reset_index(drop=True).copy()
    df_eval['prediction'] = pipe.predict(df_eval)
    predict_proba = pipe.predict_proba(df_eval)

    result = {'n_docs': len(pd.unique(df_eval['file_number'])),
              'n_dates': len(df_eval),
              'n_label': dict(df_eval['label'].value_counts()),
              'n_prediction': dict(df_eval['prediction'].value_counts()),
              'metrics': {}}

    # Consider all dates
    result['metrics']['f1_date'] = f1_score(df_eval['label'], df_eval['prediction'], pos_label='Belegdatum')
    result['metrics']['avg_prec_dates'] = average_precision_score(df_eval['label'], predict_proba[:, 0],
                                                                  pos_label='Belegdatum')

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

    predict_proba_belegdatum = pipe.predict_proba(df_docs)[:, 0]
    result['metrics']['avg_prec_docs'] = average_precision_score(df_docs['label'], predict_proba_belegdatum,
                                                                 pos_label='Belegdatum')

    # tune threshoold
    tn, fp, fn, tp, n_pred_pos, n_docs_manual, t, precision = find_threshold(
        df_docs['label'], predict_proba_belegdatum, pos_label='Belegdatum'
    )
    result['tune_threshold_metrics'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'n_pred_pos': n_pred_pos,
        'n_docs_manual': n_docs_manual,
        'threshold': t,
        'precision': precision,
    }

    result['metrics'] = round_floats_in_dict(result['metrics'])
    result['tune_threshold_metrics'] = round_floats_in_dict(result['tune_threshold_metrics'])

    log = logging.getLogger(__name__)
    log.info(f"Model results: {result}")

    return pd.Series(result)
