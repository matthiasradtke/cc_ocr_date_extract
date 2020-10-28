from kedro.pipeline import Pipeline, node

from .nodes import split_data, transform_into_training_format, train_model, evaluate_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['data_matches', 'parameters'],
                outputs=['train_validate', 'test'],
                name='split data: train_validate test',
                tags=['train', 'validate', 'test']
            ),
            node(
                func=split_data,
                inputs=['train_validate', 'parameters'],
                outputs=['train', 'validate'],
                name='split data: train validate',
                tags=['train', 'validate', 'test']
            ),
            node(
                func=transform_into_training_format,
                inputs='train',
                outputs='train_formatted',
                name='bring training data into model format',
                tags=['train', 'validate', 'test']
            ),
            node(
                func=transform_into_training_format,
                inputs='validate',
                outputs='validate_formatted',
                name='bring validation data into model format',
                tags=['validate']
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
                outputs=['date_model', 'feature_names'],
                name='train model',
                tags=['train', 'validate', 'test']
            ),
            node(
                func=evaluate_model,
                inputs=['date_model', 'train_formatted'],
                outputs='date_model_train_evaluation',
                name='evaluate model on training data',
                tags=['train']
            ),
            node(
                func=evaluate_model,
                inputs=['date_model', 'validate_formatted'],
                outputs='date_model_validate_evaluation',
                name='evaluate model on validation data',
                tags=['validate']
            ),
            node(
                func=evaluate_model,
                inputs=['date_model', 'test_formatted'],
                outputs='date_model_test_evaluation',
                name='evaluate model on testing data',
                tags=['test']
            ),
        ]
    )
