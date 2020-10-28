from typing import Any, Dict

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together.
    """

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Hook implementation to start an MLflow run
        with the same run_id as the Kedro pipeline run.
        """
        mlflow.start_run(run_name=run_params["run_id"])
        mlflow.log_params(run_params)

    @hook_impl
    def after_node_run(self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        if node._func_name == "train_model":
            mlflow.sklearn.log_model(outputs["date_model"], "model")
            mlflow.log_params(inputs["parameters"])

        elif node.name == "evaluate model on training data":
            metrics = outputs["date_model_train_evaluation"]["metrics"]
            metrics = {f"train_{k}": v for (k, v) in metrics.items()}
            mlflow.log_metrics(metrics)

        elif node.name == "evaluate model on testing data":
            mlflow.log_metrics(outputs["date_model_test_evaluation"]["metrics"])

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run
        after the Kedro pipeline finishes.
        """
        mlflow.end_run()


model_tracking_hooks = ModelTrackingHooks()
