import os
import asyncio
import time
from typing import Dict, Any, Optional
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import nest_asyncio
from config import settings
from logging_config import get_logger
from services.mlflow_service import mlflow_service
from services.monitoring_service import monitoring_service

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

logger = get_logger(__name__)

class RagService:
    def __init__(self, working_dir=".", resume_path=None):
        self.working_dir = working_dir
        self.resume_path = resume_path
        self.rag = None
        self.initialized = False
        self.initialization_time = None
        self.total_queries = 0
        self.successful_queries = 0

    async def initialize(self):
        """
        Initialize the RAG service with the specified working directory and resume data.
        """
        if self.initialized:
            logger.info("RAG service already initialized")
            return

        start_time = time.time()
        logger.info("Initializing RAG service", working_dir=self.working_dir, resume_path=self.resume_path)

        try:
            # Create working directory if it doesn't exist
            if not os.path.exists(self.working_dir):
                os.makedirs(self.working_dir)
                logger.info("Created working directory", path=self.working_dir)

            # Initialize RAG instance
            self.rag = LightRAG(
                working_dir=self.working_dir,
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
            )

            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            logger.info("RAG storages initialized successfully")

            # Load resume data if provided
            if self.resume_path and os.path.exists(self.resume_path):
                logger.info("Loading resume data", path=self.resume_path)
                
                # Load single resume file
                if os.path.isfile(self.resume_path):
                    try:
                        with open(self.resume_path, "r", encoding="utf-8") as f:
                            resume_content = f.read()
                            # Use a thread pool to avoid blocking the event loop
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, lambda: self.rag.insert(resume_content))
                        
                        # Log resume data metrics
                        resume_size = len(resume_content)
                        logger.info("Resume data loaded successfully", 
                                   size_bytes=resume_size, 
                                   characters=len(resume_content))
                        
                        # Log to MLFlow
                        mlflow_service.log_model_params({
                            "resume_path": self.resume_path,
                            "resume_size_bytes": resume_size,
                            "resume_characters": len(resume_content),
                            "rag_working_dir": self.working_dir
                        })
                    except Exception as e:
                        logger.error("Failed to read resume file", file=self.resume_path, error=str(e))
                        raise RuntimeError(f"Failed to read resume file: {str(e)}")
                else:
                    logger.warning("Resume path is not a file", path=self.resume_path)
            else:
                logger.warning("Resume path not provided or does not exist", 
                              resume_path=self.resume_path)
        
        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            raise RuntimeError(f"RAG service initialization failed: {str(e)}")

        self.initialized = True
        self.initialization_time = time.time() - start_time
        
        logger.info("RAG service initialized successfully", 
                   initialization_time=self.initialization_time)

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with the given question.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing response and metadata
        """
        if not self.initialized:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
        
        start_time = time.time()
        self.total_queries += 1
        
        logger.info("Processing RAG query", question=question[:100] + "..." if len(question) > 100 else question)
        
        try:
            # Format the question
            formatted_question = "Give concise answer for the following question (Do not provide references): " + question
            
            # Query the RAG system using a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.rag.query(
                    formatted_question, 
                    param=QueryParam(mode="local")
                )
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.successful_queries += 1
            
            # Log metrics
            mlflow_service.log_prediction_metrics(
                prediction_time=processing_time,
                image_size=0,  # Not applicable for RAG
                num_instances=len(response) if isinstance(response, str) else 1
            )
            
            # Record monitoring metrics
            monitoring_service.record_prediction(
                model_name="rag",
                status="success",
                duration=processing_time,
                image_size=0,
                num_instances=1
            )
            
            logger.info("RAG query completed successfully", 
                       processing_time=processing_time,
                       response_length=len(response))
            
            return {
                "success": True,
                "response": response,
                "processing_time": processing_time,
                "query": question,
                "metadata": {
                    "total_queries": self.total_queries,
                    "successful_queries": self.successful_queries,
                    "success_rate": self.successful_queries / self.total_queries
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            monitoring_service.record_prediction(
                model_name="rag",
                status="error",
                duration=processing_time,
                image_size=0,
                num_instances=0
            )
            
            logger.error("RAG query failed", 
                        error=str(e), 
                        processing_time=processing_time)
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "query": question
            }

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the RAG service."""
        return {
            "service_name": "RAG Service",
            "initialized": self.initialized,
            "working_dir": self.working_dir,
            "resume_path": self.resume_path,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0,
            "initialization_time": self.initialization_time
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the RAG service."""
        try:
            if not self.initialized:
                return {
                    "status": "unhealthy",
                    "error": "RAG service not initialized"
                }
            
            # Test with a simple query
            test_response = await self.query("What is your name?")
            
            return {
                "status": "healthy" if test_response["success"] else "unhealthy",
                "initialized": self.initialized,
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries
            }
            
        except Exception as e:
            logger.error("RAG health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Create a singleton instance
rag_service = RagService() 