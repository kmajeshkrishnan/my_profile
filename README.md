# ML Portfolio Backend - Production Ready

A production-ready ML model serving application with MLFlow integration, monitoring, async processing, and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **MLFlow Integration**: Model tracking, versioning, and experiment management
- **RAG System**: Resume-based question answering using LightRAG
- **Async Processing**: Celery-based background task processing
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Production Ready**: Structured logging, health checks, and security best practices
- **Model Serving**: CutLER object detection model with REST API
- **Scalable**: Docker containerization with proper resource management
- **Optimized Builds**: Split requirements for better Docker layer caching

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Nginx         │    │   Backend API   │
│   (Angular)     │◄──►│   (Reverse      │◄──►│   (FastAPI)     │
│                 │    │    Proxy)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLFlow        │    │   Redis         │    │   Celery        │
│   (Tracking)    │    │   (Message      │    │   (Background   │
│                 │    │    Broker)      │    │    Tasks)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Grafana       │    │   RAG + CutLER  │
│   (Metrics)     │    │   (Dashboards)  │    │   (Inference)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kmajeshkrishnan/my_profile.git
   cd my_profile
   ```

2. **Start development environment**:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Access the application**:
   - Frontend: http://localhost:4200
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Production

1. **Set environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your production settings
   ```

2. **Start production environment**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Access production services**:
   - Application: http://localhost
   - MLFlow UI: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /model/info` - Model information

### RAG (Resume Question Answering)

- `GET /rag/info` - RAG service information
- `GET /rag/health` - RAG service health check
- `POST /rag/query` - Query resume with questions

### Image Processing

- `POST /process-image` - Synchronous image processing
- `POST /process-image/async` - Asynchronous image processing
- `GET /task/{task_id}` - Get async task status

### MLFlow Integration

- `GET /mlflow/runs` - Get recent MLFlow runs

## RAG System

The application includes a powerful RAG (Retrieval-Augmented Generation) system that allows users to ask questions about your resume and get intelligent, context-aware answers.

### Features

- **Resume-Based Q&A**: Ask questions about your experience, skills, and background
- **Intelligent Responses**: Uses LightRAG for context-aware answers
- **Production Monitoring**: Full integration with MLFlow and Prometheus
- **Health Checks**: Comprehensive health monitoring
- **Async Processing**: Non-blocking query processing

### Example Usage

```bash
# Ask about your experience
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your experience with machine learning?"}'

# Ask about your skills
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What programming languages do you know?"}'

# Get detailed response with metadata
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about your projects", "include_metadata": true}'
```

### Sample Questions

- "What is your educational background?"
- "What are your technical skills?"
- "Tell me about your work experience"
- "What projects have you worked on?"
- "What are your areas of expertise?"
- "What technologies do you use?"

### Response Format

```json
{
  "success": true,
  "response": "I have extensive experience in machine learning...",
  "processing_time": 1.234,
  "query": "What is your experience with machine learning?",
  "metadata": {
    "total_queries": 15,
    "successful_queries": 14,
    "success_rate": 0.933
  }
}
```

## MLFlow Integration

The application integrates MLFlow for:

- **Experiment Tracking**: All model predictions are logged as experiments
- **Model Versioning**: Models can be registered and versioned
- **Metrics Logging**: Prediction times, accuracy, and other metrics
- **Artifact Storage**: Processed images and model artifacts

### MLFlow UI

Access the MLFlow UI at `http://localhost:5000` to:
- View experiments and runs
- Compare model performance
- Download model artifacts
- Register model versions

## Monitoring

### Prometheus Metrics

The application exposes Prometheus metrics at `/metrics`:

- **HTTP Requests**: Total requests, duration, status codes
- **Model Predictions**: Prediction count, duration, success rate
- **RAG Queries**: Query count, response time, success rate
- **File Uploads**: Upload count, file sizes, success rate
- **System Metrics**: Memory usage, active connections

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin) to view:
- Request rate and latency
- Model prediction performance
- RAG query performance
- System resource usage
- Error rates and trends

## Background Processing

### Celery Tasks

The application uses Celery for background processing:

- **Image Processing**: Async image processing for large files
- **RAG Queries**: Async processing for complex queries
- **Cleanup**: Automatic cleanup of temporary files
- **Health Checks**: Periodic health checks of services

### Task Management

- Submit async tasks: `POST /process-image/async`
- Check task status: `GET /task/{task_id}`
- Monitor workers: Celery Flower (optional)

## Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# MLFlow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=cutler_model

# Redis
REDIS_URL=redis://localhost:6379

# Model
MODEL_PATH=CutLER/cutler/model_zoo/cutler_cascade_final.pth
CONFIG_PATH=CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml

# RAG
RAG_RESUME_PATH=data/ajeshs_resume.txt
RAG_WORKING_DIR=./rag_data

# File Upload
MAX_FILE_SIZE=10485760  # 10MB

# Monitoring
SENTRY_DSN=your_sentry_dsn  # Optional
```

### Docker Compose Services

- **backend**: FastAPI application
- **frontend**: Angular application
- **nginx**: Reverse proxy
- **mlflow**: MLFlow tracking server
- **redis**: Message broker for Celery
- **celery-worker**: Background task workers
- **celery-beat**: Task scheduler
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization

### Requirements Structure

The project uses split requirements files for optimized Docker builds:

- **`requirements_heavy.txt`**: Slow-changing ML dependencies (PyTorch, OpenCV, Detectron2, LightRAG)
- **`requirements_base.txt`**: Fast-changing application dependencies (FastAPI, Celery, MLFlow)

This structure allows for better Docker layer caching and faster rebuilds during development.

## Security

### Production Security Features

- **Non-root user**: Application runs as non-root user
- **Resource limits**: CPU and memory limits configured
- **Health checks**: All services have health checks
- **Structured logging**: JSON logging for production
- **Error tracking**: Sentry integration (optional)

### Security Best Practices

1. **Environment Variables**: Use `.env` files for secrets
2. **Network Security**: Use internal Docker networks
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Regular Updates**: Keep dependencies updated
5. **Monitoring**: Monitor for security issues

## Scaling

### Horizontal Scaling

- **Backend**: Scale with multiple workers
- **Celery**: Add more worker processes
- **Redis**: Use Redis Cluster for high availability
- **MLFlow**: Use external database (PostgreSQL)

### Vertical Scaling

- **Resource Limits**: Adjust CPU/memory limits
- **Model Optimization**: Use GPU acceleration
- **Caching**: Implement Redis caching
- **CDN**: Use CDN for static assets

## Troubleshooting

### Common Issues

1. **MLFlow Connection**: Check MLFlow server is running
2. **Redis Connection**: Verify Redis is accessible
3. **Model Loading**: Check model files exist
4. **RAG Service**: Verify resume file exists and is readable
5. **File Upload**: Verify file size limits

### Logs

- **Application logs**: `docker-compose logs backend`
- **Celery logs**: `docker-compose logs celery-worker`
- **MLFlow logs**: `docker-compose logs mlflow`
- **RAG logs**: Check backend logs for RAG service messages

### Health Checks

- **Application**: `curl http://localhost:8000/health`
- **MLFlow**: `curl http://localhost:5000/health`
- **RAG Service**: `curl http://localhost:8000/rag/health`
- **Redis**: `docker-compose exec redis redis-cli ping`

## Development

### Adding New Models

1. **Create model service**: Add new service in `services/`
2. **Update MLFlow**: Log model parameters and metrics
3. **Add endpoints**: Create FastAPI endpoints
4. **Update monitoring**: Add Prometheus metrics

### Adding New Metrics

1. **Update monitoring service**: Add new metrics
2. **Update endpoints**: Record metrics in endpoints
3. **Update Grafana**: Create new dashboards

### Adding New Dependencies

When adding new dependencies:

1. **ML/Heavy dependencies**: Add to `requirements_heavy.txt`
2. **Application dependencies**: Add to `requirements_base.txt`
3. **Rebuild containers**: `docker-compose build --no-cache`

### Customizing RAG

1. **Update resume**: Replace `data/ajeshs_resume.txt` with your resume
2. **Modify prompts**: Update query formatting in `rag_service.py`
3. **Add context**: Include additional documents in the RAG system
4. **Fine-tune responses**: Adjust response generation parameters

## License

This project is licensed under the MIT License. 