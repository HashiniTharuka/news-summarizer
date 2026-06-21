"""
Summarization functions with prompt engineering
"""

import torch
import time
from typing import List, Dict, Optional

class NewsSummarizer:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        
        # Prompt templates
        self.prompts = {
            "zero_shot": lambda text: f"summarize: {text}",
            "few_shot": lambda text: f"""summarize: 
Example: UK economy shows strong growth in Q3
Summary: UK economy rebounds strongly in third quarter

Text: {text}
Summary:""",
            "chain_of_thought": lambda text: f"""summarize: 
Read the article carefully:
{text}

Key points:
1. Main topic:
2. Key events:
3. Important details:

Summary:"""
        }
    
    def summarize(self, text: str, max_length: int = 150, 
                  min_length: int = 50, num_beams: int = 4,
                  prompt_type: str = "zero_shot") -> Dict:
        """Generate summary with selected prompt strategy"""
        
        if not self.model:
            return {"error": "Model not loaded", "summary": ""}
        
        try:
            start_time = time.time()
            
            # Prepare prompt
            prompt_func = self.prompts.get(prompt_type, self.prompts["zero_shot"])
            prompt = prompt_func(text)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean output
            summary = self._clean_summary(summary)
            
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "processing_time": f"{processing_time:.2f}s",
                "prompt_type": prompt_type,
                "input_length": len(text.split()),
                "output_length": len(summary.split())
            }
            
        except Exception as e:
            return {"error": str(e), "summary": ""}
    
    def _clean_summary(self, text: str) -> str:
        """Clean generated summary"""
        # Remove common prefixes
        prefixes = ["Input:", "Output:", "Summary:", "Now:", 
                   "key points:", "UK economy", "strong growth"]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove repetition
        words = text.split()
        if len(words) > 100:
            text = " ".join(words[:80])
        
        return text.strip()
    
    def compare_prompts(self, text: str) -> Dict:
        """Compare all prompt strategies"""
        results = {}
        for prompt_type in self.prompts.keys():
            result = self.summarize(text, prompt_type=prompt_type)
            results[prompt_type] = result
        return results

    def batch_summarize(self, texts: List[str]) -> List[Dict]:
        """Summarize multiple texts"""
        return [self.summarize(text) for text in texts]