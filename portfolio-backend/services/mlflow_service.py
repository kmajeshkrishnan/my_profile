import mlflow
import mlflow.pytorch
import os
import tempfile
import time
from typing import Dict, Any, Optional
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

class MLFlowService:
    def __init__(self):
        self.setup_mlflow()
        self.experiment_name = settings.mlflow_experiment_name
        
    def setup_mlflow(self):
        """Setup MLFlow tracking and registry."""
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_registry_uri(settings.mlflow_registry_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLFlow setup completed", tracking_uri=settings.mlflow_tracking_uri)
            
        except Exception as e:
            logger.error("Failed to setup MLFlow", error=str(e))
            raise
    
    def log_model_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log model metrics to MLFlow."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info("Metrics logged successfully", metrics=metrics)
        except Exception as e:
            logger.error("Failed to log metrics", error=str(e))
    
    def log_model_params(self, params: Dict[str, Any]):
        """Log model parameters to MLFlow."""
        try:
            mlflow.log_params(params)
            logger.info("Parameters logged successfully", params=params)
        except Exception as e:
            logger.error("Failed to log parameters", error=str(e))
    
    def log_model_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log model artifacts to MLFlow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info("Artifact logged successfully", local_path=local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.error("Failed to log artifact", error=str(e))
    
    def register_model(self, model_name: str, model_path: str, description: str = ""):
        """Register a model in MLFlow model registry."""
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                description=description
            )
            logger.info("Model registered successfully", model_name=model_name, model_uri=model_uri)
            return registered_model
        except Exception as e:
            logger.error("Failed to register model", error=str(e))
            raise
    
    def load_model(self, model_name: str, version: Optional[int] = None):
        """Load a model from MLFlow model registry."""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pytorch.load_model(model_uri)
            logger.info("Model loaded successfully", model_name=model_name, version=version)
            return model
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise
    
    def start_run(self, run_name: Optional[str] = None):
        """Start an MLFlow run."""
        try:
            mlflow.start_run(run_name=run_name)
            logger.info("MLFlow run started", run_name=run_name)
        except Exception as e:
            logger.error("Failed to start MLFlow run", error=str(e))
            raise
    
    def end_run(self):
        """End the current MLFlow run."""
        try:
            mlflow.end_run()
            logger.info("MLFlow run ended")
        except Exception as e:
            logger.error("Failed to end MLFlow run", error=str(e))
    
    def log_prediction_metrics(self, prediction_time: float, image_size: int, num_instances: int):
        """Log prediction-specific metrics."""
        metrics = {
            "prediction_time_ms": prediction_time * 1000,
            "image_size_bytes": image_size,
            "num_instances_detected": num_instances,
            "throughput_instances_per_second": num_instances / prediction_time if prediction_time > 0 else 0
        }
        self.log_model_metrics(metrics)
        return metrics

mlflow_service = MLFlowService() 