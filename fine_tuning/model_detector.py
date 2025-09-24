"""
Compatible Model Detector

This module automatically detects which model and framework to use
based on the available environment (Colab vs Local) and system resources.
"""

import platform
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuração do modelo detectado."""
    model_name: str
    framework: str
    use_unsloth: bool
    max_length: int
    batch_size: int
    quantization: Optional[str]
    device_map: str
    reason: str


class ModelDetector:
    """Detector automático de modelo compatível."""
    
    def __init__(self):
        self.is_colab = self._detect_colab()
        self.system_info = self._get_system_info()
    
    def _detect_colab(self) -> bool:
        """Detecta se está rodando no Google Colab."""
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Coleta informações do sistema."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Detectar GPU
        gpu_available = False
        gpu_memory = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        return {
            'memory_gb': memory_gb,
            'cpu_count': cpu_count,
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory,
            'platform': platform.system(),
            'is_colab': self.is_colab
        }
    
    def detect_best_model(self) -> ModelConfig:
        """
        Detecta o melhor modelo baseado nos recursos disponíveis.
        
        Returns:
            ModelConfig: Configuração otimizada do modelo
        """
        info = self.system_info
        
        # Configurações por ambiente
        if self.is_colab:
            return self._get_colab_config(info)
        else:
            return self._get_local_config(info)
    
    def _get_colab_config(self, info: Dict[str, Any]) -> ModelConfig:
        """Configuração para Google Colab."""
        
        # Colab T4 (15GB VRAM)
        if info['gpu_memory_gb'] >= 14:
            return ModelConfig(
                model_name="microsoft/DialoGPT-medium",
                framework="unsloth",
                use_unsloth=True,
                max_length=512,
                batch_size=4,
                quantization="4bit",
                device_map="auto",
                reason="Colab T4 GPU - Unsloth eficiente"
            )
        
        # Colab sem GPU ou GPU limitada
        return ModelConfig(
            model_name="distilgpt2",
            framework="transformers",
            use_unsloth=False,
            max_length=256,
            batch_size=2,
            quantization=None,
            device_map="cpu",
            reason="Colab sem GPU adequada - usando modelo menor"
        )
    
    def _get_local_config(self, info: Dict[str, Any]) -> ModelConfig:
        """Configuração para ambiente local."""
        
        # Sistema com GPU potente (RTX 3080+, RTX 4080+)
        if info['gpu_available'] and info['gpu_memory_gb'] >= 10:
            return ModelConfig(
                model_name="microsoft/DialoGPT-medium",
                framework="unsloth",
                use_unsloth=True,
                max_length=512,
                batch_size=6,
                quantization="4bit",
                device_map="auto",
                reason="GPU local potente - usando Unsloth otimizado"
            )
        
        # Sistema com GPU média (GTX 1060, RTX 3060)
        if info['gpu_available'] and info['gpu_memory_gb'] >= 6:
            return ModelConfig(
                model_name="distilgpt2",
                framework="unsloth",
                use_unsloth=True,
                max_length=256,
                batch_size=4,
                quantization="4bit",
                device_map="auto",
                reason="GPU local média - usando Unsloth com modelo menor"
            )
        
        # Sistema apenas CPU com muita RAM
        if info['memory_gb'] >= 16:
            return ModelConfig(
                model_name="distilgpt2",
                framework="transformers",
                use_unsloth=False,
                max_length=128,
                batch_size=1,
                quantization=None,
                device_map="cpu",
                reason="Sistema CPU com muita RAM - modelo pequeno"
            )
        
        # Sistema limitado
        return ModelConfig(
            model_name="distilgpt2",
            framework="transformers",
            use_unsloth=False,
            max_length=64,
            batch_size=1,
            quantization=None,
            device_map="cpu",
            reason="Sistema limitado - configuração mínima"
        )
    
    def print_detection_summary(self, config: ModelConfig):
        """Imprime resumo da detecção."""
        print("=== DETECÇÃO DE MODELO COMPATÍVEL ===")
        print(f"Ambiente: {'Google Colab' if self.is_colab else 'Local'}")
        print(f"RAM: {self.system_info['memory_gb']:.1f} GB")
        print(f"GPU: {'Sim' if self.system_info['gpu_available'] else 'Não'}")
        
        if self.system_info['gpu_available']:
            print(f"VRAM: {self.system_info['gpu_memory_gb']:.1f} GB")
        
        print(f"\nModelo Selecionado: {config.model_name}")
        print(f"Framework: {config.framework}")
        print(f"Unsloth: {'Sim' if config.use_unsloth else 'Não'}")
        print(f"Quantização: {config.quantization or 'Nenhuma'}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Max Length: {config.max_length}")
        print(f"Razão: {config.reason}")
        print("=" * 40)


def get_optimal_config() -> ModelConfig:
    """
    Função principal para obter configuração otimizada.
    
    Returns:
        ModelConfig: Configuração detectada automaticamente
    """
    detector = ModelDetector()
    config = detector.detect_best_model()
    detector.print_detection_summary(config)
    return config


if __name__ == "__main__":
    # Teste da detecção
    config = get_optimal_config()
    print(f"\nConfiguração final: {config}")
