from celery import Celery
from typing import Dict, Any
import time
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# Initialize Celery
celery_app = Celery(
    'ml_portfolio',
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=['services.celery_service']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

@celery_app.task(bind=True)
def process_image_async(self, image_data: bytes, image_name: str):
    """Process image asynchronously using Celery."""
    try:
        logger.info("Starting async image processing", task_id=self.request.id, image_name=image_name)
        
        # Import here to avoid circular imports
        from services.cutler_service import CutlerService
        from services.mlflow_service import mlflow_service
        from services.monitoring_service import monitoring_service
        
        cutler_service = CutlerService()
        
        # Start MLFlow run
        mlflow_service.start_run(run_name=f"async_prediction_{self.request.id}")
        
        start_time = time.time()
        
        # Process the image
        result = cutler_service.process_image(image_data)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        if result["success"]:
            # Log metrics
            metrics = mlflow_service.log_prediction_metrics(
                prediction_time=prediction_time,
                image_size=len(image_data),
                num_instances=result.get("num_instances", 0)
            )
            
            # Record monitoring metrics
            monitoring_service.record_prediction(
                model_name="cutler",
                status="success",
                duration=prediction_time,
                image_size=len(image_data),
                num_instances=result.get("num_instances", 0)
            )
            
            logger.info("Async image processing completed successfully", 
                       task_id=self.request.id,
                       prediction_time=prediction_time,
                       num_instances=result.get("num_instances", 0))
            
        else:
            # Record failure
            monitoring_service.record_prediction(
                model_name="cutler",
                status="error",
                duration=prediction_time,
                image_size=len(image_data),
                num_instances=0
            )
            
            logger.error("Async image processing failed", 
                        task_id=self.request.id,
                        error=result.get("error", "Unknown error"))
        
        # End MLFlow run
        mlflow_service.end_run()
        
        return result
        
    except Exception as e:
        logger.error("Async image processing failed with exception", 
                    task_id=self.request.id,
                    error=str(e))
        
        # Record failure
        monitoring_service.record_prediction(
            model_name="cutler",
            status="error",
            duration=0,
            image_size=len(image_data),
            num_instances=0
        )
        
        raise

@celery_app.task
def cleanup_old_files():
    """Clean up old temporary files."""
    try:
        import os
        import tempfile
        from datetime import datetime, timedelta
        
        # Clean up files older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        temp_dir = tempfile.gettempdir()
        cleaned_count = 0
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time and filename.startswith('ml_portfolio_'):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning("Failed to remove old file", file_path=file_path, error=str(e))
        
        logger.info("Cleanup completed", cleaned_count=cleaned_count)
        return {"cleaned_count": cleaned_count}
        
    except Exception as e:
        logger.error("Cleanup failed", error=str(e))
        raise

@celery_app.task
def health_check():
    """Health check task for Celery workers."""
    try:
        # Import services to check their health
        from services.cutler_service import CutlerService
        from services.mlflow_service import mlflow_service
        
        # Check if CutLER service is working
        cutler_service = CutlerService()
        
        # Check MLFlow connection
        mlflow_service.setup_mlflow()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "cutler": "available",
                "mlflow": "available"
            }
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        } 