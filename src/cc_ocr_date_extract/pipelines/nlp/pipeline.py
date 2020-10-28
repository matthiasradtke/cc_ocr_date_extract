from kedro.pipeline import Pipeline, node

from .nodes import create_spacy_nlp_object, clean_text, get_date_matches


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_spacy_nlp_object,
                inputs='parameters',
                outputs='nlp_object',
                name='load spacy nlp object',
            ),
            node(
                func=clean_text,
                inputs='data_raw',
                outputs='data_clean',
                name='roughly clean text',
            ),
            node(
                func=get_date_matches,
                inputs=['nlp_object', 'data_clean', 'parameters'],
                outputs='data_matches',
                name='get searched matches from text',
            )
        ]
    )
