"""
Trainer de Fine-Tuning

Módulo principal para execução do fine-tuning com suporte a Unsloth e fallbacks.
"""

import os
import json
import logging
from typing import Optional, Any

from .config import FineTuningConfig


class FineTuningTrainer:
    """Trainer principal para fine-tuning."""
    
    def __init__(self, config: FineTuningConfig):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração do fine-tuning
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Criar diretório de saída
        os.makedirs(config.output_dir, exist_ok=True)
    
    def setup_model(self) -> bool:
        """
        Configura modelo e tokenizer baseado na configuração detectada.
        
        Returns:
            bool: True se setup foi bem-sucedido
        """
        try:
            if self.config.model_config.use_unsloth:
                return self._setup_unsloth_model()
            else:
                return self._setup_transformers_model()
        except Exception as e:
            self.logger.error(f"Erro ao configurar modelo: {e}")
            return False
    
    def _setup_unsloth_model(self) -> bool:
        """Configura modelo usando Unsloth."""
        try:
            from unsloth import FastLanguageModel  # noqa: F401
            
            self.logger.info("Configurando modelo com Unsloth...")
            
            # Carregar modelo
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_config.model_name,
                max_seq_length=self.config.model_config.max_length,
                dtype=None,
                load_in_4bit=self.config.model_config.quantization == "4bit"
            )
            
            # Configurar LoRA
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None
            )
            
            self.logger.info("Modelo Unsloth configurado com sucesso!")
            return True
            
        except ImportError:
            self.logger.warning("Unsloth não disponível, usando Transformers...")
            return self._setup_transformers_model()
        except Exception as e:
            self.logger.error(f"Erro no setup Unsloth: {e}")
            return False
    
    def _setup_transformers_model(self) -> bool:
        """Configura modelo usando Transformers padrão."""
        try:
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, 
                BitsAndBytesConfig
            )
            
            self.logger.info("Configurando modelo com Transformers...")
            
            # Configurar quantização se necessário
            quantization_config = None
            if self.config.model_config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16"
                )
            
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_config.model_name,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_config.model_name,
                quantization_config=quantization_config,
                device_map=self.config.model_config.device_map,
                torch_dtype="auto"
            )
            
            self.logger.info("Modelo Transformers configurado com sucesso!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no setup Transformers: {e}")
            return False
    
    def prepare_dataset(self) -> Optional[Any]:
        """
        Prepara dataset para treinamento.
        
        Returns:
            Dataset preparado ou None se erro
        """
        try:
            from datasets import Dataset
            
            self.logger.info("Preparando dataset...")
            
            # Carregar dados
            data = []
            with open(self.config.data_config.train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            self.logger.info(f"Carregados {len(data)} exemplos")
            
            # Criar dataset
            dataset = Dataset.from_list(data)
            
            # Tokenizar
            def tokenize_function(examples):
                inputs = []
                for i in range(len(examples['instruction'])):
                    text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Input:\n{examples['input'][i]}\n\n### Response:\n{examples['output'][i]}"
                    inputs.append(text)
                
                return self.tokenizer(
                    inputs,
                    truncation=True,
                    padding=True,
                    max_length=self.config.model_config.max_length,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            self.logger.info("Dataset preparado com sucesso!")
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar dataset: {e}")
            return None
    
    def setup_trainer(self, dataset) -> bool:
        """
        Configura trainer para treinamento.
        
        Args:
            dataset: Dataset preparado
            
        Returns:
            bool: True se setup foi bem-sucedido
        """
        try:
            if self.config.model_config.use_unsloth:
                return self._setup_unsloth_trainer(dataset)
            else:
                return self._setup_transformers_trainer(dataset)
        except Exception as e:
            self.logger.error(f"Erro ao configurar trainer: {e}")
            return False
    
    def _setup_unsloth_trainer(self, dataset) -> bool:
        """Configura trainer Unsloth."""
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            
            training_args = TrainingArguments(
                per_device_train_batch_size=self.config.model_config.batch_size,
                gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
                warmup_steps=self.config.training_config.warmup_steps,
                max_steps=-1,
                learning_rate=self.config.training_config.learning_rate,
                fp16=not self.config.training_config.fp16,
                bf16=self.config.training_config.fp16,
                logging_steps=self.config.training_config.logging_steps,
                optim="adamw_8bit",
                weight_decay=self.config.training_config.weight_decay,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.training_config.num_epochs,
                save_steps=self.config.training_config.save_steps,
                save_total_limit=3,
                remove_unused_columns=False
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.model_config.max_length,
                dataset_num_proc=2,
                args=training_args
            )
            
            self.logger.info("Trainer Unsloth configurado!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no trainer Unsloth: {e}")
            return False
    
    def _setup_transformers_trainer(self, dataset) -> bool:
        """Configura trainer Transformers."""
        try:
            from transformers import (
                Trainer, TrainingArguments, DataCollatorForLanguageModeling
            )
            
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.training_config.num_epochs,
                per_device_train_batch_size=self.config.model_config.batch_size,
                gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
                warmup_steps=self.config.training_config.warmup_steps,
                learning_rate=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay,
                logging_steps=self.config.training_config.logging_steps,
                save_steps=self.config.training_config.save_steps,
                eval_steps=self.config.training_config.eval_steps,
                fp16=self.config.training_config.fp16,
                dataloader_num_workers=self.config.training_config.dataloader_num_workers,
                remove_unused_columns=False
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            self.logger.info("Trainer Transformers configurado!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no trainer Transformers: {e}")
            return False
    
    def train(self) -> bool:
        """
        Executa o treinamento.
        
        Returns:
            bool: True se treinamento foi bem-sucedido
        """
        try:
            self.logger.info("Iniciando treinamento...")
            self.trainer.train()
            
            self.logger.info("Treinamento concluído!")
            
            # Salvar modelo
            self.save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {e}")
            return False
    
    def save_model(self):
        """Salva modelo treinado."""
        try:
            self.logger.info("Salvando modelo...")
            
            if self.config.model_config.use_unsloth:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(self.model)
            
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Salvar configuração
            config_path = os.path.join(self.config.output_dir, "training_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Modelo salvo em: {self.config.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
    
    def run_full_pipeline(self) -> bool:
        """
        Executa pipeline completo de fine-tuning.
        
        Returns:
            bool: True se pipeline foi bem-sucedido
        """
        self.logger.info("Iniciando pipeline completo de fine-tuning...")
        
        # Setup modelo
        if not self.setup_model():
            return False
        
        # Preparar dataset
        dataset = self.prepare_dataset()
        if dataset is None:
            return False
        
        # Setup trainer
        if not self.setup_trainer(dataset):
            return False
        
        # Treinar
        if not self.train():
            return False
        
        self.logger.info("Pipeline completo finalizado com sucesso!")
        return True
