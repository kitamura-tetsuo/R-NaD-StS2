import mlflow
import os
from typing import Any, Dict
import logging

class ExperimentManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints", run_id: str | None = None, log_checkpoints: bool = False):
        # Ensure logs go to the project root regardless of CWD
        mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
        mlflow.set_experiment(experiment_name=experiment_name)
        self.log_checkpoints = log_checkpoints

        # Force run_id to None if it's an empty string or just whitespace to trigger latest run search
        if run_id is not None and not run_id.strip():
            run_id = None

        # If no specific run_id is provided, try to resume the latest run from this experiment
        if not run_id and not mlflow.active_run():
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                try:
                    # Search for the most recently started run in this experiment
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
                    if not runs.empty:
                        run_id = runs.iloc[0].run_id
                        logging.info(f"ExperimentManager: No run_id provided. Auto-detected latest MLflow run to resume: {run_id}")
                except Exception as e:
                    logging.warning(f"ExperimentManager: Failed to search for existing runs: {e}")

        # Start a run explicitly if not already active
        if run_id:
            if mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
                logging.info(f"ExperimentManager: Ending current active run ({mlflow.active_run().info.run_id}) to switch to {run_id}")
                mlflow.end_run()
            
            if not mlflow.active_run():
                try:
                    # Attempt to resume existing run
                    mlflow.start_run(run_id=run_id)
                    logging.info(f"ExperimentManager: Successfully resumed MLflow run: {run_id}")
                except Exception as e:
                    logging.warning(f"ExperimentManager: Failed to resume run {run_id}: {e}. Starting a NEW run instead.")
                    mlflow.start_run()
        elif not mlflow.active_run():
            logging.info("ExperimentManager: No run_id provided or found. Starting a NEW MLflow run.")
            mlflow.start_run()

        self.run_id = mlflow.active_run().info.run_id


        # We save checkpoints using run_id to avoid conflicts
        self.checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, self.run_id))
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logging.info(f"Initialized ExperimentManager with checkpoint dir: {self.checkpoint_dir}, log_checkpoints: {self.log_checkpoints}")

    def log_params(self, config: Any):
        """Logs configuration parameters to MLflow."""
        if hasattr(config, '_asdict'):
            params = dict(config._asdict())
        elif isinstance(config, dict):
            params = dict(config)
        else:
            try:
                params = dict(vars(config))
            except TypeError:
                params = {"config": str(config)}

        # Define keys to exclude from parameters and which ones should be tags
        # Adapted from StS2 context
        save_as_tags = ["batch_size", "unroll_length", "model_type"]

        # Set tags
        for key in save_as_tags:
            if key in params:
                mlflow.set_tag(key, str(params[key]))

        # Filter params and ensure values are loggable (int, float, string, bool)
        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                filtered_params[k] = v
            else:
                filtered_params[k] = str(v)

        if filtered_params:
            mlflow.log_params(filtered_params)

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Logs metrics to MLflow."""
        flat_metrics = {}

        def flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, prefix + k + '.')
                else:
                    try:
                        flat_metrics[prefix + k] = float(v)
                    except (ValueError, TypeError):
                        pass # Skip non-numeric metrics

        flatten(metrics)
        mlflow.log_metrics(flat_metrics, step=step)

    def log_checkpoint_artifact(self, step: int, ckpt_path: str):
        """Logs the .pkl checkpoint as an MLflow artifact."""
        if self.log_checkpoints and os.path.exists(ckpt_path):
            # Log the directory as an artifact in a 'checkpoints' folder in MLflow
            mlflow.log_artifact(ckpt_path, artifact_path=f"checkpoints/step_{step}")
            logging.info(f"Saved checkpoint for step {step} to MLflow.")
        elif not self.log_checkpoints:
            pass # logging.info(f"Skipping checkpoint upload to MLflow for step {step} (disabled).")
        else:
            logging.warning(f"Checkpoint path {ckpt_path} does not exist.")
