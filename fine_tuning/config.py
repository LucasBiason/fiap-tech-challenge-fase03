"""
Fine-Tuning Configuration

Centralized module for fine-tuning process configurations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .model_detector import ModelConfig


@dataclass
class TrainingConfig:
    """Training configurations."""
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Data configurations."""
    train_file: str
    eval_file: Optional[str] = None
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = True


@dataclass
class FineTuningConfig:
    """Complete fine-tuning configuration."""
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig
    output_dir: str
    project_name: str = "fiap-finetuning"
    
    @classmethod
    def create_default(cls, train_file: str, output_dir: str) -> 'FineTuningConfig':
        """
        Create default configuration with automatic detection.
        
        Args:
            train_file: Path to training file
            output_dir: Output directory
            
        Returns:
            FineTuningConfig: Created configuration
        """
        from .model_detector import get_optimal_config
        
        model_config = get_optimal_config()
        
        training_config = TrainingConfig(
            num_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            gradient_accumulation_steps=1,
            fp16=model_config.use_unsloth,  # FP16 only with Unsloth
            dataloader_num_workers=4
        )
        
        data_config = DataConfig(
            train_file=train_file,
            max_seq_length=model_config.max_length,
            preprocessing_num_workers=4,
            overwrite_cache=True
        )
        
        return cls(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            output_dir=output_dir
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_config.model_name,
            'framework': self.model_config.framework,
            'use_unsloth': self.model_config.use_unsloth,
            'max_length': self.model_config.max_length,
            'batch_size': self.model_config.batch_size,
            'quantization': self.model_config.quantization,
            'device_map': self.model_config.device_map,
            'num_epochs': self.training_config.num_epochs,
            'learning_rate': self.training_config.learning_rate,
            'output_dir': self.output_dir,
            'train_file': self.data_config.train_file,
            'max_seq_length': self.data_config.max_seq_length
        }
    
    def print_summary(self):
        """Print configuration summary."""
        print("=== FINE-TUNING CONFIGURATION ===")
        print(f"Model: {self.model_config.model_name}")
        print(f"Framework: {self.model_config.framework}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print(f"Learning Rate: {self.training_config.learning_rate}")
        print(f"Batch Size: {self.model_config.batch_size}")
        print(f"Max Length: {self.model_config.max_length}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Training File: {self.data_config.train_file}")
        print("=" * 35)


class ConfigDataPreparation:
    """Simple configuration class for data processing."""

    def __init__(self, **kwargs):
        """
        Initialize configuration with custom parameters.

        Args:
            **kwargs: Any configuration parameters
        """
        # Set defaults
        self.required_fields = ['title', 'content']
        self.min_title_length = 5
        self.min_content_length = 10
        self.chunk_size = 300
        self.generate_synthetic_content = True

        # Override with any provided parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary for easy passing to other classes."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}

    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """String representation of configuration."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"Config({', '.join(items)})"
