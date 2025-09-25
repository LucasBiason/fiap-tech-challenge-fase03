# FIAP Tech Challenge - Fase 3

## Visão Geral

Este projeto implementa um sistema completo de fine-tuning de modelos de linguagem para o FIAP Tech Challenge Fase 3. O sistema foi projetado para ser **simples, eficiente e compatível** com diferentes ambientes (Google Colab e local).

## Características Principais

- **Detecção Automática**: Sistema detecta automaticamente o melhor modelo e configurações baseado no hardware disponível
- **Compatibilidade Unsloth**: Suporte completo ao Unsloth para treinamento otimizado quando disponível
- **Fallback Inteligente**: Se Unsloth não estiver disponível, usa Transformers padrão automaticamente
- **Processamento em Chunks**: Processa datasets grandes sem problemas de memória
- **Formato Alpaca**: Converte dados para formato padrão de fine-tuning

## Estrutura do Projeto

```
fiap-tech-challenge-fase03/
├── fine_tuning/                 # Módulos Python organizados
│   ├── __init__.py
│   ├── model_detector.py       # Detecção automática de modelo
│   ├── config.py               # Configurações
│   ├── trainer.py              # Trainer principal
│   ├── dataset_downloader.py   # Download de datasets
│   ├── dataset_analyzer.py     # Análise de datasets
│   ├── data_processor.py       # Processamento de dados
├── preparacao_dados.ipynb      # Notebook de preparação (NOVO)
├── fine_tuning_otimizado.ipynb # Notebook de fine-tuning (ATUALIZADO)
├── data/                       # Dados processados
└── requirements.txt            # Dependências
```

## Uso Rápido

### 1. Preparação de Dados

```python
# Execute o notebook preparacao_dados.ipynb
# Ele baixa, analisa e processa o dataset automaticamente
```

### 2. Teste Funcional Rápido (RECOMENDADO)

Se o treinamento está demorando muito (20+ horas), use o teste funcional:

```bash
# Parar treinamento atual e executar teste rápido
python stop_and_test.py
```

**O que o teste faz:**
- Para qualquer treinamento em andamento
- Cria amostra de apenas 1000 registros
- Configura para 1 época apenas
- Tempo estimado: 5-15 minutos
- Valida se o sistema está funcionando

### 3. Fine-Tuning Completo

```python
from fine_tuning.config import FineTuningConfig
from fine_tuning.trainer import FineTuningTrainer

# Configuração automática
config = FineTuningConfig.create_default(
    train_file="./data/trn_finetune.jsonl",
    output_dir="./output"
)

# Executar treinamento
trainer = FineTuningTrainer(config)
success = trainer.run_full_pipeline()
```

### 4. Criação Manual de Amostra

```bash
# Criar amostra pequena manualmente
python create_sample.py -i ./data/trn_finetune.jsonl -o ./data/trn_finetune_sample.jsonl -s 1000
```

## Detecção Automática de Modelo

O sistema detecta automaticamente:

### Google Colab
- **T4 GPU (15GB)**: DialoGPT-medium + Unsloth + 4bit quantization
- **Sem GPU**: DistilGPT2 + Transformers padrão

### Ambiente Local
- **GPU Potente (10GB+)**: DialoGPT-medium + Unsloth + 4bit quantization
- **GPU Média (6GB+)**: DistilGPT2 + Unsloth + 4bit quantization
- **Apenas CPU**: DistilGPT2 + Transformers + configuração mínima

## Exemplo de Saída da Detecção

```
=== DETECÇÃO DE MODELO COMPATÍVEL ===
Ambiente: Local
RAM: 30.5 GB
GPU: Não

Modelo Selecionado: distilgpt2
Framework: transformers
Unsloth: Não
Quantização: Nenhuma
Batch Size: 1
Max Length: 128
Razão: Sistema CPU com muita RAM - modelo pequeno
```

## Instalação

```bash
# Instalar dependências básicas
pip install ijson tqdm psutil gdown

# Para fine-tuning completo
pip install torch transformers datasets accelerate peft trl

# Para Unsloth (opcional - será detectado automaticamente)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Notebooks

### preparacao_dados.ipynb
- Download automático do dataset
- Análise de estrutura
- Processamento em chunks
- Conversão para formato Alpaca

### fine_tuning_otimizado.ipynb
- Detecção automática de modelo
- Configuração otimizada
- Pipeline completo de treinamento

## Vantagens da Nova Arquitetura

1. **Simplicidade**: Código organizado e fácil de entender
2. **Robustez**: Fallbacks automáticos para diferentes ambientes
3. **Eficiência**: Otimizações automáticas baseadas no hardware
4. **Compatibilidade**: Funciona em Colab e ambiente local
5. **Manutenibilidade**: Código modular e bem documentado

## Troubleshooting

### Erro de Memória
- O sistema detecta automaticamente e ajusta batch_size
- Usa processamento em chunks para datasets grandes

### Unsloth Não Disponível
- Sistema usa Transformers padrão automaticamente
- Não é necessário instalar Unsloth

### GPU Não Detectada
- Sistema usa configuração CPU otimizada
- Ajusta automaticamente parâmetros de treinamento

## Contribuição

Este projeto foi reorganizado para ser mais simples e eficiente. Todos os módulos estão na pasta `fine_tuning/` com documentação em português e código limpo.