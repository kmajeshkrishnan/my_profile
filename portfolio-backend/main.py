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
from services.rag_service import rag_service
from services.mlflow_service import mlflow_service
from services.monitoring_service import monitoring_service
from services.celery_service import celery_app, process_image_async
from config import settings
from logging_config import setup_logging, get_logger
from models import RagQueryRequest, RagQueryResponse, ServiceInfoResponse, HealthCheckResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready ML model serving API with MLFlow integration and RAG capabilities",
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

# Startup event to initialize RAG service
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Set the resume path to the data folder
        resume_path = "data/resume.txt"
        rag_service.resume_path = resume_path
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize RAG service", error=str(e))

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
        
        # Check RAG service health
        rag_health = await rag_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.version,
            "environment": settings.environment,
            "services": {
                "mlflow": "available",
                "cutler": "available" if "error" not in model_info else "error",
                "rag": rag_health["status"]
            },
            "model_info": model_info,
            "rag_health": rag_health
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

# RAG service info endpoint
@app.get("/rag/info", response_model=ServiceInfoResponse)
async def get_rag_info():
    """Get information about the RAG service."""
    return rag_service.get_service_info()

# RAG health check endpoint
@app.get("/rag/health", response_model=HealthCheckResponse)
async def rag_health_check():
    """Health check for RAG service."""
    return await rag_service.health_check()

# RAG query endpoint
@app.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(request: RagQueryRequest):
    """Query the RAG system with a question about the resume."""
    try:
        # Record file upload metrics (for consistency, even though it's not a file upload)
        monitoring_service.record_file_upload(
            file_type="text_query",
            status="success",
            file_size=len(request.query)
        )
        
        # Query the RAG system
        result = await rag_service.query(request.query)
        
        # Create response
        response = RagQueryResponse(
            success=result["success"],
            response=result.get("response"),
            error=result.get("error"),
            processing_time=result["processing_time"],
            query=result["query"],
            metadata=result.get("metadata") if request.include_metadata else None
        )
        
        return response
        
    except Exception as e:
        logger.error("RAG query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

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
            "rag_info": "/rag/info",
            "rag_health": "/rag/health",
            "rag_query": "/rag/query",
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