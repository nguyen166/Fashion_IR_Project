"""
SigLIP2 Model wrapper for Fashion Image Retrieval.
Provides image and text embedding extraction using google/siglip2-so400m-patch16-naflex.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, List
from transformers import AutoProcessor, AutoModel
from pathlib import Path
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class SigLIPModel:
    """
    Wrapper class for SigLIP2 model to extract image and text embeddings.
    The model is loaded once and reused for all embedding extractions.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize SigLIP2 model and processor."""
        if SigLIPModel._initialized:
            return
        
        logger.info(f"Loading SigLIP2 model: {settings.model_name}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor (trust_remote_code for SigLIP2)
        self.processor = AutoProcessor.from_pretrained(settings.model_name, trust_remote_code=True,use_fast=True)
        self.model = AutoModel.from_pretrained(settings.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = settings.embedding_dim
        
        SigLIPModel._initialized = True
        logger.info("SigLIP2 model loaded successfully")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            L2 normalized embeddings
        """
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / (norm + 1e-8)
    
    def get_image_embedding(
        self, 
        image: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]
    ) -> np.ndarray:
        """
        Extract embedding from image(s).
        
        Args:
            image: Single image or list of images. Can be:
                   - File path (str or Path)
                   - PIL Image object
                   
        Returns:
            Normalized embedding vector(s) of shape (embedding_dim,) or (N, embedding_dim)
        """
        # Handle single image vs batch
        if not isinstance(image, list):
            images = [image]
            single_input = True
        else:
            images = image
            single_input = False
        
        # Load images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            pil_images.append(pil_img)
        
        # Process images
        with torch.no_grad():
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            image_features = self.model.get_image_features(**inputs)
            embeddings = image_features.cpu().numpy()
        
        # Normalize embeddings
        embeddings = self._normalize(embeddings)
        
        # Return single embedding if single input
        if single_input:
            return embeddings[0]
        return embeddings
    
    def get_text_embedding(
        self, 
        text: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Extract embedding from text(s).
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Normalized embedding vector(s) of shape (embedding_dim,) or (N, embedding_dim)
        """
        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Process texts
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text features
            text_features = self.model.get_text_features(**inputs)
            embeddings = text_features.cpu().numpy()
        
        # Normalize embeddings
        embeddings = self._normalize(embeddings)
        
        # Return single embedding if single input
        if single_input:
            return embeddings[0]
        return embeddings


# Global model instance
_model_instance: SigLIPModel = None


def get_model() -> SigLIPModel:
    """
    Get the global model instance.
    Creates the instance if it doesn't exist.
    
    Returns:
        SigLIPModel instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = SigLIPModel()
    return _model_instance
