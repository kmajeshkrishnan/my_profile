from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from PIL import Image
import io
import os
from dotenv import load_dotenv
import requests
from requests_toolbelt import MultipartEncoder
import torch
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

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

# Initialize CutLER model
def init_cutler_model():
    cfg = get_cfg()
    cfg.merge_from_file("model_zoo/configs/COCO-Semisupervised/cascade_mask_rcnn_R_50_FPN_100perc.yaml")
    cfg.MODEL.WEIGHTS = "model_zoo/checkpoints/cutler_fully_100perc.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

# Initialize model at startup
predictor = init_cutler_model()

# Pydantic models for request validation
class TextRequest(BaseModel):
    text: str

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Run CutLER model
        outputs = predictor(image_np)
        
        # Visualize predictions
        v = Visualizer(image_np[:, :, ::-1], MetadataCatalog.get("coco_2017_val"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Convert back to PIL Image
        result_image = Image.fromarray(out.get_image()[:, :, ::-1])
        
        # Save processed image to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-text")
async def process_text(request: TextRequest):
    try:
        # Example: Using a simple text transformation
        # In a real application, you would integrate with an LLM service
        processed_text = request.text.upper()
        return {"processed_text": processed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-contact-email")
async def send_contact_email(request: ContactRequest):
    try:
        # Get Mailgun configuration from environment variables
        mailgun_api_key = os.getenv("MAILGUN_API_KEY")
        mailgun_domain = os.getenv("MAILGUN_DOMAIN")
        admin_email = os.getenv("ADMIN_EMAIL")
        
        if not all([mailgun_api_key, mailgun_domain, admin_email]):
            raise HTTPException(status_code=500, detail="Mailgun configuration is incomplete")
        
        # Prepare email content
        subject = f"New Contact Form Submission from {request.name}"
        
        # Create HTML email body
        html_body = f"""
        <h2>New Contact Form Submission</h2>
        <p><strong>Name:</strong> {request.name}</p>
        <p><strong>Email:</strong> {request.email}</p>
        <p><strong>Message:</strong></p>
        <p>{request.message}</p>
        """
        
        # Create plain text email body
        text_body = f"""
        New Contact Form Submission
        
        Name: {request.name}
        Email: {request.email}
        Message:
        {request.message}
        """
        
        # Prepare the data for Mailgun API
        data = {
            'from': f'Portfolio Contact Form <noreply@{mailgun_domain}>',
            'to': admin_email,
            'subject': subject,
            'text': text_body,
            'html': html_body,
            'h:Reply-To': request.email
        }
        
        # Send email using Mailgun API
        response = requests.post(
            f'https://api.mailgun.net/v3/{mailgun_domain}/messages',
            auth=('api', mailgun_api_key),
            data=data
        )
        
        if response.status_code == 200:
            return {"message": "Contact form submission received and forwarded successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send email: {response.text}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 