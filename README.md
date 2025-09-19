# FIAP Tech Challenge - Phase 03


## Overview

This repository contains a complete pipeline for data preparation and language model fine-tuning, developed for the FIAP Tech Challenge Phase 3.
Each stage is implemented in a dedicated Jupyter Notebook, with clear, reproducible steps and code. This README provides a comprehensive guide to the workflow, requirements, and usage.

1. **Data Preparation** (`data_preparation.ipynb`)
2. **Fine-Tuning** (`fine_tuning.ipynb`)

---

## 1. Data Preparation (`data_preparation.ipynb`)

### Purpose
Prepare and clean the raw dataset for use in language model fine-tuning. This includes filtering, cleaning, and formatting the data into prompt/completion pairs.

### Steps

1. **Setup and Imports**
   - Import required libraries: `pandas`, `numpy`, `os`, `json`, and `ijson` (for efficient JSON processing).
   - Mount Google Drive (if running in Colab) to access data files.

2. **File Paths**
   - Define paths for the raw JSON data and output files (filtered and processed datasets).

3. **Install Dependencies**
   - Install the `ijson` library for efficient, chunked JSON reading (especially useful for large files).

4. **Filter and Save Data in Chunks**
   - Read the raw JSON data in chunks to handle large files efficiently.
   - Select only the necessary columns (`title`, `content`).
   - Save the filtered data in JSONL format, appending each chunk to a single file.

5. **Load and Clean Filtered Data**
   - Load the filtered data.
   - Remove rows where `title` or `content` is empty or null.
   - Save the cleaned data for further processing.

6. **Data Inspection**
   - Display the head of the processed DataFrame.
   - Show the shape of the data and count null/empty values for quality assurance.

7. **Create Prompt/Completion Pairs**
   - Construct prompt/completion pairs in English:
     - Prompt: `Question: <title>\nAnswer:`
     - Completion: `<content>`
   - Save the final dataset in JSONL format, ready for fine-tuning.

### Output
- `trn_finetune.jsonl`: Cleaned and formatted dataset with prompt/completion pairs for model training.

---

## 2. Fine-Tuning (`fine_tuning.ipynb`)

### Purpose
Use the prepared dataset to fine-tune a language model for question-answering or similar NLP tasks.

### Steps

1. **Setup and Imports**
   - Import necessary libraries for model training (e.g., Hugging Face Transformers, PyTorch, or other frameworks as required).

2. **Load Prepared Data**
   - Load the `trn_finetune.jsonl` file generated in the data preparation step.

3. **Model Selection and Configuration**
   - Choose a pre-trained language model suitable for fine-tuning (e.g., GPT, BERT, etc.).
   - Configure training parameters (batch size, learning rate, epochs, etc.).

4. **Fine-Tuning Process**
   - Tokenize the prompt/completion pairs.
   - Train the model on the dataset.
   - Monitor training metrics and adjust parameters as needed.

5. **Evaluation and Saving**
   - Evaluate the fine-tuned model on validation data (if available).
   - Save the trained model and any relevant artifacts.

### Output
- Fine-tuned model weights and configuration files.
- Training logs and evaluation metrics.

---

## Environment Setup

### Prerequisites
- Python 3.8+ (recommended: 3.11)
- Google Colab access (recommended) or local GPU
- Jupyter Notebook
- pandas, numpy
- ijson
- (For fine-tuning) Hugging Face Transformers, PyTorch or TensorFlow

### Local Setup (Optional)

If you want to run locally, follow these steps:

#### 1. Clone Repository
```bash
git clone https://github.com/your-username/fiap-tech-challenge-fase03.git
cd fiap-tech-challenge-fase03
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# OR install only essentials to start:
pip install pandas numpy ijson tqdm psutil gdown
```

#### 4. Configure Jupyter
```bash
# Install Jupyter if not available
pip install jupyter notebook

# Start Jupyter
jupyter notebook
```

#### 5. Verify Installation
```python
# Quick test in Python
import pandas as pd
import numpy as np
import ijson
import tqdm
import psutil

print("Environment configured successfully!")
```

### Google Colab Setup (Recommended)

To use with Google Colab (easier and with free GPU):

#### 1. Open in Colab
- Access [Google Colab](https://colab.research.google.com/)
- Upload notebooks or connect with GitHub

#### 2. Install Dependencies
```python
# In the first notebook cell
!pip install ijson tqdm psutil gdown sentence-transformers

# For fine-tuning (use in separate notebook)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

#### 3. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Important Tips

1. **For large datasets (1GB+):** ALWAYS use Google Colab
2. **For local development:** Test with small samples first
3. **Monitoring:** Always monitor memory usage
4. **Backup:** Save checkpoints frequently

---

## Project Structure

```
fiap-tech-challenge-fase03/
├── data_preparation.ipynb    # Data cleaning and preparation notebook
├── fine_tuning.ipynb         # Model fine-tuning notebook
├── requirements.txt          # Complete dependencies
├── README.md                 # This documentation
```

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key libraries include:

### Core ML/AI
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers
- `unsloth` - Memory optimization
- `peft` - Parameter-efficient fine-tuning
- `trl` - Transformer reinforcement learning

### Data Processing
- `pandas` - Data manipulation
- `ijson` - Streaming JSON processing
- `psutil` - Memory monitoring
- `tqdm` - Progress bars

### Evaluation
- `sentence-transformers` - Semantic similarity
- `scikit-learn` - Machine learning metrics
- `rouge-score` - Text evaluation metrics

---

## License

This project is for educational purposes as part of the FIAP Tech Challenge. Please refer to the institution for licensing details.

---

## Contributing

This is an educational project. For improvements or suggestions, please follow the course guidelines and submission procedures.