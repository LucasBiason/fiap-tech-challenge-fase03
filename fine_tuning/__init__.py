"""
Pacote de Fine-Tuning

Módulos para fine-tuning de modelos de linguagem com detecção automática
de compatibilidade e suporte a Unsloth e Transformers.
"""

from .model_detector import ModelDetector, ModelConfig, get_optimal_config
from .config import FineTuningConfig, TrainingConfig, DataConfig
from .trainer import FineTuningTrainer

__version__ = "1.0.0"
__author__ = "FIAP Tech Challenge"

__all__ = [
    "ModelDetector",
    "ModelConfig",
    "get_optimal_config",
    "FineTuningConfig",
    "TrainingConfig",
    "DataConfig",
    "FineTuningTrainer"
]
