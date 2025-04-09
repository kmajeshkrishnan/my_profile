import os
import tempfile
import base64
import cv2
import numpy as np
from PIL import Image
import io
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys

# Add CutLER to Python path
sys.path.append('CutLER')
sys.path.append('CutLER/cutler')
sys.path.append('CutLER/cutler/demo')

from config import add_cutler_config
from predictor import VisualizationDemo

class CutlerService:
    def __init__(self):
        self.cfg = self._setup_cfg()
        self.demo = VisualizationDemo(self.cfg)
        setup_logger(name="fvcore")
        self.logger = setup_logger()

    def _setup_cfg(self):
        cfg = get_cfg()
        add_cutler_config(cfg)
        
        # Load config file
        cfg.merge_from_file("CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml")
        
        # Set model weights
        cfg.MODEL.WEIGHTS = "CutLER/cutler/model_zoo/cutler_cascade_final.pth"
        
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

    def process_image(self, image_bytes):
        try:
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
            
            return {
                "success": True,
                "image": image_base64,  # Convert bytes to string for JSON
                "num_instances": len(predictions["instances"]) if "instances" in predictions else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 