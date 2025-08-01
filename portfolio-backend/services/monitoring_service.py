from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
from logging_config import get_logger

logger = get_logger(__name__)

class MonitoringService:
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Model prediction metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'status']
        )
        
        self.prediction_duration = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction duration in seconds',
            ['model_name']
        )
        
        # File upload metrics
        self.file_upload_size = Histogram(
            'file_upload_size_bytes',
            'File upload size in bytes',
            ['file_type']
        )
        
        self.file_upload_counter = Counter(
            'file_uploads_total',
            'Total number of file uploads',
            ['file_type', 'status']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of the model in bytes',
            ['model_name']
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        logger.info("Request recorded", 
                   method=method, 
                   endpoint=endpoint, 
                   status=status, 
                   duration=duration)
    
    def record_prediction(self, model_name: str, status: str, duration: float, 
                         image_size: int, num_instances: int):
        """Record model prediction metrics."""
        self.prediction_counter.labels(model_name=model_name, status=status).inc()
        self.prediction_duration.labels(model_name=model_name).observe(duration)
        
        # Record file size if available
        if image_size > 0:
            self.file_upload_size.labels(file_type="image").observe(image_size)
        
        logger.info("Prediction recorded", 
                   model_name=model_name, 
                   status=status, 
                   duration=duration,
                   image_size=image_size,
                   num_instances=num_instances)
    
    def record_file_upload(self, file_type: str, status: str, file_size: int):
        """Record file upload metrics."""
        self.file_upload_counter.labels(file_type=file_type, status=status).inc()
        if file_size > 0:
            self.file_upload_size.labels(file_type=file_type).observe(file_size)
        
        logger.info("File upload recorded", 
                   file_type=file_type, 
                   status=status, 
                   file_size=file_size)
    
    def set_active_connections(self, count: int):
        """Set the number of active connections."""
        self.active_connections.set(count)
    
    def set_model_memory_usage(self, model_name: str, memory_bytes: int):
        """Set the memory usage of a model."""
        self.model_memory_usage.labels(model_name=model_name).set(memory_bytes)
    
    def get_metrics(self):
        """Get Prometheus metrics."""
        return generate_latest()
    
    def get_metrics_content_type(self):
        """Get the content type for metrics."""
        return CONTENT_TYPE_LATEST

monitoring_service = MonitoringService() 