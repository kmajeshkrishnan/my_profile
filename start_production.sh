#!/bin/bash

# Production startup script for ML Portfolio Backend

set -e

echo "🚀 Starting ML Portfolio Backend in production mode..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file from example. Please edit it with your settings."
    else
        echo "❌ .env.example not found. Please create a .env file manually."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p monitoring
mkdir -p logs
mkdir -p data

# Build and start services
echo "🔨 Building and starting production services..."
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
fi

# Check MLFlow
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLFlow is healthy"
else
    echo "❌ MLFlow is not responding"
fi

# Check Redis
if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

echo ""
echo "🎉 Production environment is ready!"
echo ""
echo "📊 Access your services:"
echo "   • Application: http://localhost"
echo "   • MLFlow UI: http://localhost:5000"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "   • View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "   • Stop services: docker-compose -f docker-compose.prod.yml down"
echo "   • Restart services: docker-compose -f docker-compose.prod.yml restart"
echo ""
echo "🔍 Monitor your application:"
echo "   • Health check: curl http://localhost:8000/health"
echo "   • Metrics: curl http://localhost:8000/metrics"
echo "   • Model info: curl http://localhost:8000/model/info" 