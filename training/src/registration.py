"""Register the best model of a given experiment type in the MLflow Model Registry."""
from __future__ import annotations
from mlflow.tracking import MlflowClient


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

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.experiment_type = '{experiment_type}'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No runs found for experiment_type='{experiment_type}'"
        )

    best_run = runs[0]
    run_id = best_run.info.run_id
    vllm_model_path = best_run.data.params.get("vllm_model_path")
    f1 = best_run.data.metrics.get("f1_macro")
    print(f"Best {experiment_type} run: {run_id} (f1_macro={f1}), "
          f"vllm_model_path={vllm_model_path}")

    model_registry_name = "sentiment-qwen2.5"

    # Ensure the registered model exists
    try:
        client.create_registered_model(model_registry_name)
    except Exception:
        pass  # already exists

    # Create a model version pointing to the run
    model_version = client.create_model_version(
        name=model_registry_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    # Assign alias
    client.set_registered_model_alias(
        name=model_registry_name,
        alias=alias,
        version=model_version.version,
    )

    # Store vllm_model_path as a version tag for Part B retrieval
    client.set_model_version_tag(
        name=model_registry_name,
        version=model_version.version,
        key="vllm_model_path",
        value=vllm_model_path or "",
    )

    print(f"Registered '{model_registry_name}' version {model_version.version} "
          f"with alias '{alias}' and vllm_model_path='{vllm_model_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model in MLflow Model Registry")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--experiment-type", required=True, help="Value of params.experiment_type to filter on")
    parser.add_argument("--alias", required=True, help="Model alias to assign (e.g. champion, challenger)")
    args = parser.parse_args()

    register_model(args.tracking_uri, args.experiment_name, args.experiment_type, args.alias)
