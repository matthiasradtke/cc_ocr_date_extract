from kedro.pipeline import Pipeline, node

from .nodes import split_data, transform_into_training_format, train_model, evaluate_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['data_matches', 'parameters'],
                outputs=['train', 'test'],
                name='split data: train test',
                tags=['train', 'test']
            ),
            node(
                func=transform_into_training_format,
                inputs='train',
                outputs='train_formatted',
                name='bring training data into model format',
                tags=['train', 'test']
            ),
            node(
                func=transform_into_training_format,
                inputs='test',
                outputs='test_formatted',
                name='bring test data into model format',
                tags=['test']
            ),
            node(
                func=train_model,
                inputs=['train_formatted', 'parameters'],
                outputs=['date_model', 'date_model_parameters', 'feature_names'],
                name='train model',
                tags=['train', 'test']
            ),
            node(
                func=evaluate_model,
                inputs=['date_model', 'test_formatted'],
                outputs=['date_model_test_evaluation', 'test_docs'],
                name='evaluate model on testing data',
                tags=['test']
            ),
        ]
    )
