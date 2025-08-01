import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Application settings
    app_name: str = "ML Portfolio Backend"
    version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "4"))
    
    # MLFlow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_registry_uri: str = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "cutler_model")
    
    # Database settings
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./mlflow.db")
    
    # Redis settings
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Sentry settings
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    
    # Model settings
    model_path: str = os.getenv("MODEL_PATH", "CutLER/cutler/model_zoo/cutler_cascade_final.pth")
    config_path: str = os.getenv("CONFIG_PATH", "CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml")
    
    # File upload settings
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "10 * 1024 * 1024"))  # 10MB
    allowed_file_types: list = ["image/jpeg", "image/png", "image/jpg"]
    
    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    class Config:
        env_file = ".env"

settings = Settings() 