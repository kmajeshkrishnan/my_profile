import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class RagService:
    def __init__(self, working_dir=".", resume_path=None):
        self.working_dir = working_dir
        self.resume_path = resume_path
        self.rag = None
        self.initialized = False

    async def initialize(self):
        """
        Initialize the RAG service with the specified working directory and resume data.
        """
        if self.initialized:
            return

        # Create working directory if it doesn't exist
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Initialize RAG instance
        self.rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

        # Load resume data if provided
        if self.resume_path and os.path.exists(self.resume_path):
            with open(self.resume_path, "r", encoding="utf-8") as f:
                # Use a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.rag.insert(f.read()))
        
        self.initialized = True

    async def query(self, question):
        """
        Query the RAG system with the given question.
        
        Args:
            question: The question to answer
            
        Returns:
            response: The generated response
        """
        if not self.initialized:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
        
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
        
        return response

# Create a singleton instance
rag_service = RagService() 