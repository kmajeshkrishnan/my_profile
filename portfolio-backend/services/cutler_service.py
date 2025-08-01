import os
import tempfile
import base64
import cv2
import numpy as np
from PIL import Image
import io
import time
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
import logging

# Add CutLER to Python path
sys.path.append('CutLER')
sys.path.append('CutLER/cutler')
sys.path.append('CutLER/cutler/demo')

from config import add_cutler_config
from predictor import VisualizationDemo
from config import settings
from logging_config import get_logger
from services.mlflow_service import mlflow_service
from services.monitoring_service import monitoring_service

logger = get_logger(__name__)

class CutlerService:
    def __init__(self):
        self.cfg = self._setup_cfg()
        self.demo = VisualizationDemo(self.cfg)
        setup_logger(name="fvcore")
        self.logger = setup_logger()
        
        # Log model parameters to MLFlow
        self._log_model_params()
        
    def _setup_cfg(self):
        cfg = get_cfg()
        add_cutler_config(cfg)
        
        # Load config file
        cfg.merge_from_file(settings.config_path)
        
        # Set model weights
        cfg.MODEL.WEIGHTS = settings.model_path
        
        # Set device to CPU
        cfg.MODEL.DEVICE = "cpu"
        
        # Disable SyncBN for CPU
        if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
            cfg.MODEL.RESNETS.NORM = "BN"
            cfg.MODEL.FPN.NORM = "BN"
        
        # Set confidence threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        
        cfg.freeze()
        return cfg
    
    def _log_model_params(self):
        """Log model parameters to MLFlow."""
        try:
            params = {
                "model_path": settings.model_path,
                "config_path": settings.config_path,
                "device": self.cfg.MODEL.DEVICE,
                "confidence_threshold": self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                "model_type": "CutLER",
                "version": settings.version
            }
            mlflow_service.log_model_params(params)
        except Exception as e:
            logger.warning("Failed to log model parameters", error=str(e))

    def process_image(self, image_bytes):
        try:
            # Log the size of the image being processed
            logger.info(f"Processing image of size: {len(image_bytes)} bytes")
            
            # Start timing
            start_time = time.time()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run inference
            predictions, visualized_output = self.demo.run_on_image(img)
            
            # Convert the visualized output to bytes
            img_array = visualized_output.get_image()
            img_pil = Image.fromarray(img_array)
            
            # Save to bytes buffer
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            num_instances = len(predictions["instances"]) if "instances" in predictions else 0
            
            # Log metrics
            mlflow_service.log_prediction_metrics(
                prediction_time=processing_time,
                image_size=len(image_bytes),
                num_instances=num_instances
            )
            
            # Record monitoring metrics
            monitoring_service.record_prediction(
                model_name="cutler",
                status="success",
                duration=processing_time,
                image_size=len(image_bytes),
                num_instances=num_instances
            )
            
            logger.info("Image processing completed successfully", 
                       processing_time=processing_time,
                       num_instances=num_instances,
                       image_size=len(image_bytes))
            
            return {
                "success": True,
                "image": image_base64,
                "num_instances": num_instances,
                "processing_time": processing_time
            }
            
        except Exception as e:
            # Calculate processing time even for errors
            end_time = time.time()
            processing_time = end_time - start_time if 'start_time' in locals() else 0
            
            # Record error metrics
            monitoring_service.record_prediction(
                model_name="cutler",
                status="error",
                duration=processing_time,
                image_size=len(image_bytes),
                num_instances=0
            )
            
            logger.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def get_model_info(self):
        """Get information about the loaded model."""
        try:
            return {
                "model_name": "CutLER",
                "model_path": settings.model_path,
                "config_path": settings.config_path,
                "device": self.cfg.MODEL.DEVICE,
                "confidence_threshold": self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                "version": settings.version
            }
        except Exception as e:
            logger.error("Failed to get model info", error=str(e))
            return {"error": str(e)} 