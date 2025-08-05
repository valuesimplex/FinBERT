
"""
Downstream dataset utilities for sentiment analysis tasks.

This module provides data loading and processing utilities for sentiment analysis,
supporting both CSV and text file formats with proper tokenization and label mapping.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SENTIMENT_LABELS = {0: "负面", 1: "正面"}
DEFAULT_TEXT_SEPARATOR = "    "
DEFAULT_ENCODING = "utf-8"


@dataclass
class TextLabelRecord:
    """Data structure for text-label pairs."""
    text: str
    label: Union[str, int]
    
    def __post_init__(self):
        """Validate input data after initialization."""
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Text must be a non-empty string")
        if self.label is None:
            raise ValueError("Label cannot be None")


def load_sentiment_data_from_txt(
    file_path: Union[str, Path], 
    separator: str = DEFAULT_TEXT_SEPARATOR,
    encoding: str = DEFAULT_ENCODING
) -> List[TextLabelRecord]:
    """
    Load sentiment analysis data from a text file.
    
    Args:
        file_path: Path to the text file
        separator: Separator between text and label (default: 4 spaces)
        encoding: File encoding (default: utf-8)
        
    Returns:
        List of TextLabelRecord objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    records = []
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = [part for part in line.split(separator) if part]
                
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line {line_num}: insufficient parts")
                    continue
                    
                text, label = parts[0], parts[1]
                
                try:
                    record = TextLabelRecord(text=text, label=label)
                    records.append(record)
                except ValueError as e:
                    logger.warning(f"Skipping invalid record at line {line_num}: {e}")
                    
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode file with encoding {encoding}: {e}")
        
    if not records:
        raise ValueError(f"No valid records found in {file_path}")
        
    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


class SentimentDatasetFromTxt(Dataset):
    """
    Dataset class for loading sentiment analysis data from text files.
    
    The text file should have one sample per line with the format:
    text{separator}label
    """
    
    def __init__(
        self, 
        file_path: Union[str, Path], 
        shuffle: bool = True,
        separator: str = DEFAULT_TEXT_SEPARATOR,
        encoding: str = DEFAULT_ENCODING
    ):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the text file
            shuffle: Whether to shuffle the data
            separator: Separator between text and label
            encoding: File encoding
        """
        self.file_path = Path(file_path)
        self.separator = separator
        self.encoding = encoding
        
        self.records = self._load_data()
        
        if shuffle:
            random.shuffle(self.records)
            
        logger.info(f"Initialized dataset with {len(self.records)} samples")
            
    def _load_data(self) -> List[TextLabelRecord]:
        """Load and parse data from the file."""
        return load_sentiment_data_from_txt(
            self.file_path, 
            self.separator, 
            self.encoding
        )

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> TextLabelRecord:
        """Get a single sample by index."""
        if idx >= len(self.records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.records)}")
        return self.records[idx]
        
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get label-to-ID and ID-to-label mappings.
        
        Returns:
            Tuple of (label2id, id2label) dictionaries
        """
        label_encoder = LabelEncoder()
        labels = [record.label for record in self.records]
        
        int_labels = label_encoder.fit_transform(labels)
        
        # Create mappings
        unique_labels = label_encoder.classes_
        id2label = {int(i): label for i, label in enumerate(unique_labels)}
        label2id = {label: int(i) for i, label in enumerate(unique_labels)}
        
        logger.info(f"Label mappings: {id2label}")
        return label2id, id2label


class SentimentDatasetFromCSV(Dataset):
    """
    Dataset class for loading sentiment analysis data from CSV files.
    
    The CSV file should have 'text' and 'label' columns.
    """
    
    def __init__(self, csv_path: Union[str, Path]):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing
        """
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
            
        # Validate required columns
        required_columns = {'text', 'label'}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove rows with missing values
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=['text', 'label'])
        final_size = len(self.df)
        
        if initial_size != final_size:
            logger.warning(f"Removed {initial_size - final_size} rows with missing values")
        
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.records = [
            TextLabelRecord(text, label) 
            for text, label in zip(self.texts, self.labels)
        ]
        
        logger.info(f"Loaded {len(self.records)} samples from CSV")
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> TextLabelRecord:
        """Get a single sample by index."""
        if idx >= len(self.records):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.records)}")
        return self.records[idx]
    
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get label-to-ID and ID-to-label mappings for sentiment analysis.
        
        Returns:
            Tuple of (label2id, id2label) dictionaries
        """
        id2label = DEFAULT_SENTIMENT_LABELS.copy()
        label2id = {label: id_ for id_, label in id2label.items()}
        
        logger.info(f"Using default sentiment label mappings: {id2label}")
        return label2id, id2label


class SentimentDataCollator:
    """
    Data collator for sentiment analysis tasks.
    
    Handles tokenization and batching of text data with labels.
    """
    
    def __init__(
        self, 
        tokenizer, 
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True
    ) -> None:
        """
        Initialize the data collator.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length (defaults to tokenizer's max length)
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length or getattr(tokenizer, 'model_max_length', 512)
        self.padding = padding
        self.truncation = truncation

    def __call__(self, records: List[TextLabelRecord]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of records.
        
        Args:
            records: List of TextLabelRecord objects
            
        Returns:
            Dictionary containing input_ids and labels tensors
            
        Raises:
            ValueError: If records list is empty or contains invalid data
        """
        if not records:
            raise ValueError("Cannot process empty batch")
            
        texts = []
        labels = []
        
        for record in records:
            if not isinstance(record, TextLabelRecord):
                raise ValueError(f"Expected TextLabelRecord, got {type(record)}")
                
            texts.append(record.text)
            
            try:
                label_int = int(record.label)
                labels.append(label_int)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid label '{record.label}': {e}")

        # Tokenize texts
        try:
            inputs = self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors='pt',
            )
        except Exception as e:
            raise ValueError(f"Tokenization failed: {e}")
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
       
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'labels': labels_tensor
        }


# Backward compatibility aliases
SentimentDataset2 = SentimentDatasetFromCSV
sentiment2_collator = SentimentDataCollator

__all__ = [
    'TextLabelRecord',
    'load_sentiment_data_from_txt',
    'SentimentDatasetFromTxt',
    'SentimentDatasetFromCSV', 
    'SentimentDataCollator',
    'SentimentDataset2',  # Deprecated
    'sentiment2_collator',  # Deprecated
]
