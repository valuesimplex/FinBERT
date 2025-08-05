"""
Fine-tuning script for sentiment classification using BERT-based models.

This script provides a professional framework for fine-tuning transformer models
on sentiment analysis tasks with comprehensive evaluation and logging capabilities.
"""

import argparse
import csv
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from downstream_dataset import SentimentDataset2, sentiment2_collator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "valuesimplex-ai-lab/FinBERT2-base"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 510


@dataclass
class FineTuningConfig:
    """Configuration class for fine-tuning parameters."""
    
    # Model and data paths
    model_name: str = DEFAULT_MODEL_NAME
    train_data_path: str = "SC_2/train_SC_2.csv"
    test_data_path: str = "SC_2/test_SC_2.csv"
    output_base_dir: str = "classifier_models"
    results_csv_path: str = "sentiment2.csv"
    
    # Training parameters
    num_epochs: int = 1
    train_batch_size: int = 5
    eval_batch_size: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = DEFAULT_MAX_LENGTH
    
    # Evaluation and logging
    logging_steps: int = 1
    eval_steps: int = 10
    save_steps: int = 100000
    metric_for_best_model: str = "f1"
    
    # Other settings
    seed: int = DEFAULT_SEED
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if not (0 < self.learning_rate < 1):
            raise ValueError("Learning rate must be between 0 and 1")


class SentimentFinetuner:
    """Professional sentiment analysis fine-tuning framework."""
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.eval_counter = 0
        
        # Set up paths
        self.experiment_name = self._generate_experiment_name()
        self.output_dir = Path(config.output_base_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized fine-tuner for experiment: {self.experiment_name}")
        
    def _generate_experiment_name(self) -> str:
        """Generate experiment name based on model name."""
        model_name = self.config.model_name
        if model_name.split("/")[-1] == "encoder_model":
            return f"{model_name.split('/')[-2]}_SC_2"
        else:
            return f"{model_name.split('/')[-1]}_SC_2"
    
    def setup_reproducibility(self) -> None:
        """Set up reproducible training environment."""
        seed = self.config.seed
        
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        
        logger.info(f"Set random seed to {seed} for reproducibility")
    
    def load_datasets(self) -> Tuple[SentimentDataset2, SentimentDataset2]:
        """
        Load training and testing datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset)
            
        Raises:
            FileNotFoundError: If dataset files don't exist
        """
        train_path = Path(self.config.train_data_path)
        test_path = Path(self.config.test_data_path)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        logger.info(f"Loading training dataset from {train_path}")
        train_dataset = SentimentDataset2(train_path)
        
        logger.info(f"Loading test dataset from {test_path}")
        test_dataset = SentimentDataset2(test_path)
        
        logger.info(f"Loaded {len(train_dataset)} training and {len(test_dataset)} test samples")
        return train_dataset, test_dataset
    
    def setup_model_and_tokenizer(self, num_classes: int, label_mappings: Tuple[Dict, Dict]):
        """
        Set up model and tokenizer.
        
        Args:
            num_classes: Number of classification classes
            label_mappings: Tuple of (label2id, id2label) mappings
            
        Returns:
            Tuple of (model, tokenizer, data_collator)
        """
        label2id, id2label = label_mappings
        
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        
        logger.info(f"Loading model from {self.config.model_name}")
        model = BertForSequenceClassification.from_pretrained(
            self.config.model_name,
            problem_type="single_label_classification",
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id
        )
        
        data_collator = sentiment2_collator(
            tokenizer=tokenizer,
            max_length=self.config.max_length
        )
        
        logger.info(f"Model setup complete with {num_classes} classes")
        return model, tokenizer, data_collator
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments configuration."""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_strategy="steps",
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            seed=self.config.seed
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics and log results.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        self.eval_counter += 1
        
        # Convert predictions to class labels (same as original code)
        predictions = torch.argmax(torch.tensor(predictions), dim=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        current_steps = self.eval_counter * self.config.eval_steps
        
        metrics = {
            'experimentname': self.experiment_name,
            'currentsteps': current_steps,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Original logic: only write CSV on main process in distributed training
        if self.training_args.local_rank in (0, -1):
            # CSV writing logic (same as original)
            csv_path = Path(self.config.results_csv_path)
            file_exists = csv_path.exists()
            fieldnames = ['experimentname', 'currentsteps', 'accuracy', 'precision', 'recall', 'f1']
            
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)
        
        return metrics
    

    def train(self) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting sentiment analysis fine-tuning pipeline")
        
        # Setup reproducibility
        self.setup_reproducibility()
        
        # Load datasets
        train_dataset, test_dataset = self.load_datasets()
        label2id, id2label = train_dataset.get_label_mappings()
        num_classes = len(label2id)
        
        # Setup model and tokenizer
        model, tokenizer, data_collator = self.setup_model_and_tokenizer(
            num_classes, (label2id, id2label)
        )
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Store training_args for use in compute_metrics
        self.training_args = training_args
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        logger.info(f"Saving model and tokenizer to {self.output_dir}")
        tokenizer.save_pretrained(self.output_dir)
        model.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT models for sentiment classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        help='Name or path of the pretrained model'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        help='Path to test data CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Base directory for output models'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create configuration
    config = FineTuningConfig()
    
    # Override config with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.train_data:
        config.train_data_path = args.train_data
    if args.test_data:
        config.test_data_path = args.test_data
    if args.output_dir:
        config.output_base_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed
    
    logger.info(f"Using model: {config.model_name}")
    
    # Initialize and run fine-tuner
    try:
        finetuner = SentimentFinetuner(config)
        finetuner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
