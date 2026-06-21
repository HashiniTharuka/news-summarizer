"""
Model loader for T5 News Summarizer
Handles loading the fine-tuned model and tokenizer
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path="./"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def get_model_info(self):
        """Get model information"""
        if not self.loaded:
            return {"status": "Not loaded"}
        
        return {
            "status": "Loaded",
            "device": str(self.device),
            "model_type": "T5-small",
            "vocab_size": self.tokenizer.vocab_size,
            "model_path": self.model_path
        }

# Singleton instance
model_loader = ModelLoader()