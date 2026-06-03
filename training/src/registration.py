"""Register the best model of a given experiment type in the MLflow Model Registry."""

from __future__ import annotations

import argparse

import mlflow
import yaml


def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def register_model(
    tracking_uri: str,
    experiment_name: str,
    experiment_type: str,
    alias: str,
) -> None:
    """Register the best run of a given experiment type under the given alias."""
    mlflow.set_tracking_uri(tracking_uri)

    #TODO:
    # - Use MLflowClient to query MLflow for the best run of this
    #   experiment matching the given experiment_type (by F1 score)
    # - Register the model of the best run under the given alias
    # - Set a model version tag on the aliased model with as name
    #   `vllm_model_path` and as value the value of this parameter
    #   stored in the run
    #
    #Hint: Normally only runs that have a logged MLflow model (via mlflow.transformers.log_model
    # or similar) can be registered. However, due to the resource restrictions of the VMs we are
    # using for this assignment, we do not upload any actual files, we only store the model weights
    # file path for later retrieval

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model in MLflow Model Registry")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--experiment-type", required=True, help="Value of params.experiment_type to filter on")
    parser.add_argument("--alias", required=True, help="Model alias to assign (e.g. champion, challenger)")
    args = parser.parse_args()

    register_model(args.tracking_uri, args.experiment_name, args.experiment_type, args.alias)
