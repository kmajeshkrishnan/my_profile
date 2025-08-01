import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import services
from services.cutler_service import CutlerService
from services.mlflow_service import mlflow_service
from services.monitoring_service import monitoring_service
from services.celery_service import celery_app, process_image_async
from config import settings
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready ML model serving API with MLFlow integration",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize services
cutler_service = CutlerService()

# Custom middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record request metrics
    monitoring_service.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=process_time
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check MLFlow connection
        mlflow_service.setup_mlflow()
        
        # Check model info
        model_info = cutler_service.get_model_info()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.version,
            "environment": settings.environment,
            "services": {
                "mlflow": "available",
                "cutler": "available" if "error" not in model_info else "error"
            },
            "model_info": model_info
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=monitoring_service.get_metrics(),
        media_type=monitoring_service.get_metrics_content_type()
    )

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    return cutler_service.get_model_info()

# MLFlow runs endpoint
@app.get("/mlflow/runs")
async def get_mlflow_runs():
    """Get recent MLFlow runs."""
    try:
        import mlflow
        runs = mlflow.search_runs(
            experiment_names=[settings.mlflow_experiment_name],
            max_results=10
        )
        return {"runs": runs.to_dict('records')}
    except Exception as e:
        logger.error("Failed to get MLFlow runs", error=str(e))
        return {"error": str(e)}

# Process image endpoint (synchronous)
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Process image synchronously."""
    try:
        # Validate file type
        if file.content_type not in settings.allowed_file_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Validate file size
        if len(contents) > settings.max_file_size:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Record file upload metrics
        monitoring_service.record_file_upload(
            file_type=file.content_type,
            status="success",
            file_size=len(contents)
        )
        
        # Process the image using CutLER service
        result = cutler_service.process_image(contents)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing image", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Process image endpoint (asynchronous)
@app.post("/process-image/async")
async def process_image_async_endpoint(file: UploadFile = File(...)):
    """Process image asynchronously using Celery."""
    try:
        # Validate file type
        if file.content_type not in settings.allowed_file_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Validate file size
        if len(contents) > settings.max_file_size:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Record file upload metrics
        monitoring_service.record_file_upload(
            file_type=file.content_type,
            status="success",
            file_size=len(contents)
        )
        
        # Submit async task
        task = process_image_async.delay(contents, file.filename)
        
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Image processing started asynchronously"
        }
        
    except Exception as e:
        logger.error("Error starting async image processing", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Get task status endpoint
@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an async task."""
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.ready():
            if task.successful():
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task.result
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(task.info)
                }
        else:
            return {
                "task_id": task_id,
                "status": "processing"
            }
            
    except Exception as e:
        logger.error("Error getting task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "process_image": "/process-image",
            "process_image_async": "/process-image/async",
            "task_status": "/task/{task_id}",
            "mlflow_runs": "/mlflow/runs",
            "docs": "/docs" if settings.debug else None
        }
    }

if __name__ == "__main__":
    # Configure Uvicorn for production
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        limit_request_line=settings.max_file_size,
        limit_request_field_size=settings.max_file_size,
        limit_concurrency=100,
        limit_max_requests=1000,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    ) 