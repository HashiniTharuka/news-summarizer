"""
Model loader for T5 News Summarizer
Handles loading the fine-tuned model and tokenizer
"""

import logging
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path=None):
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = self._resolve_model_path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def _resolve_model_path(self, model_path):
        """Find the local model directory regardless of the current working directory."""
        candidates = []

        if model_path:
            path = Path(model_path)
            candidates.append(path if path.is_absolute() else self.base_dir / path)
            if not path.is_absolute():
                candidates.append(self.base_dir / "models" / path.name)
                candidates.append(self.base_dir / path.name)

        candidates.extend([
            self.base_dir / "t5-news-summarizer",
            self.base_dir / "models" / "t5-news-summarizer",
            self.base_dir,
        ])

        for candidate in candidates:
            if (candidate / "spiece.model").exists() or (candidate / "pytorch_model.bin").exists() or (candidate / "model.safetensors").exists():
                return str(candidate)

        return str(self.base_dir / "t5-news-summarizer")
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            model_dir = Path(self.model_path)

            # Load tokenizer from the saved SentencePiece model when available.
            spiece_path = model_dir / "spiece.model"
            if spiece_path.exists():
                self.tokenizer = T5Tokenizer(vocab_file=str(spiece_path), extra_ids=100)
            else:
                self.tokenizer = T5Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(str(model_dir), local_files_only=True)
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