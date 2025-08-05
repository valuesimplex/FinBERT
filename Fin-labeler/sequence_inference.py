"""
Sentiment analysis inference module for sequence classification.

This module provides a professional framework for performing batch inference
on sentiment analysis tasks using pre-trained BERT models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, BertTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_LENGTH = 512
DEFAULT_CLS_INDEX = 0
DEFAULT_HIDDEN_LAYER = 12
DEFAULT_SENTIMENT_LABELS = {0: '负面', 1: '正面'}


class SentimentInferenceEngine:
    """
    Professional sentiment analysis inference engine.
    
    This class provides batch inference capabilities for sentiment classification
    with support for extracting both predictions and hidden representations.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        cls_index: int = DEFAULT_CLS_INDEX,
        hidden_layer: int = DEFAULT_HIDDEN_LAYER,
        id2label: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the pre-trained model
            device: Device to run inference on (auto-detected if None)
            max_length: Maximum sequence length for tokenization
            cls_index: Index of the CLS token (default: 0)
            hidden_layer: Layer index for hidden state extraction (default: 12)
            id2label: Label mapping dictionary (default: sentiment labels)
        
        Raises:
            ValueError: If model_path is empty or invalid
            FileNotFoundError: If model_path doesn't exist
        """
        if not model_path:
            raise ValueError("Model path cannot be empty")
            
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.cls_index = cls_index
        self.hidden_layer = hidden_layer
        self.id2label = id2label or DEFAULT_SENTIMENT_LABELS.copy()
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self) -> None:
        """Load pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            ).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def infer_batch_sequencecls(
        self, 
        texts: List[str]
    ) -> Tuple[List[str], List[List[float]], List[List[float]]]:
        """
        Perform batch inference using the class-based approach.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            Tuple containing (predicted_classes, softmax_probs, cls_vectors)
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
            
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings")
            
        # Use global function with temporary global variable setup
        global tokenizer, model, device
        original_tokenizer, original_model, original_device = tokenizer, model, device
        
        try:
            tokenizer, model, device = self.tokenizer, self.model, self.device
            return infer_batch_sequencecls(texts)
        finally:
            tokenizer, model, device = original_tokenizer, original_model, original_device
    
    def infer_single(self, text: str) -> Tuple[str, List[float], List[float]]:
        """
        Perform inference on a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple containing (predicted_class, probabilities, cls_vector)
        """
        results = self.infer_batch_sequencecls([text])
        return results[0][0], results[1][0], results[2][0]
    
    def get_device_info(self) -> str:
        """Get information about the device being used."""
        return str(self.device)
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get model configuration information."""
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'max_length': self.max_length,
            'cls_index': self.cls_index,
            'hidden_layer': self.hidden_layer,
            'num_labels': len(self.id2label),
            'labels': list(self.id2label.values())
        }


def create_inference_engine(
    model_path: str,
    device: Optional[str] = None,
    **kwargs
) -> SentimentInferenceEngine:
    """
    Factory function to create a sentiment inference engine.
    
    Args:
        model_path: Path to the pre-trained model
        device: Device to run inference on
        **kwargs: Additional arguments for SentimentInferenceEngine
        
    Returns:
        Configured SentimentInferenceEngine instance
    """
    return SentimentInferenceEngine(
        model_path=model_path,
        device=device,
        **kwargs
    )


# Global variables for backward compatibility (same as original script)
tokenizer = None
model = None
device = None


def infer_batch_sequencecls(texts):
    """
    Global function maintaining exact original logic for backward compatibility.
    
    This function preserves the exact same processing logic as the original code.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        softmax_probs = F.softmax(logits, dim=-1)
        # BERT中的第一个记（[CLS]）的索引为0
        cls_index = 0
        # 获取BERT模型的输出
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 获取第n层的隐藏状态
        n=12
        nlayers_states = outputs.hidden_states[n]
        # 提取CLS标记的向量
        cls_vectors = nlayers_states[:, cls_index, :].detach().cpu().numpy()    
        predicted_class_ids = logits.argmax(dim=-1).tolist()
    # 情感分类分类
    id2label = {0: '负面', 1: '正面'}
    predicted_classes = [id2label[pred] for pred in predicted_class_ids]
    return predicted_classes, softmax_probs.tolist(), cls_vectors.tolist()


def main():
    """
    Main function preserving original script behavior exactly.
    """
    global tokenizer, model, device
    
    modelpath = ""  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    model = AutoModelForSequenceClassification.from_pretrained(modelpath).to(device)
    texts = ["这是一个正面的评论", "这是一个负面的评论"]
    predicted_classes, softmax_probs, cls_vectors = infer_batch_sequencecls(texts)
    print(predicted_classes)


if __name__ == "__main__":
    main()