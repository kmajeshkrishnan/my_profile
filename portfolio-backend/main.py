from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import RagQueryRequest

from services.cutler_service import CutlerService
from services.rag_service import rag_service

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Portfolio Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cutler_service = CutlerService()

# Initialize RAG service
@app.on_event("startup")
async def startup_event():
    # Set the resume path
    resume_path = "data/ajeshs_resume.txt"
    rag_service.resume_path = resume_path
    await rag_service.initialize()


@app.get("/")
async def test():
    return "Hello World"

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Process the image using CutLER service
        result = cutler_service.process_image(contents)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-query")
async def rag_query(request: RagQueryRequest):
    try:
        # Query the RAG system
        response = await rag_service.query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 