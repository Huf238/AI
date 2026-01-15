# 🤖 Document Training Pipeline (Mistral 7B)

A production-ready, beginner-friendly pipeline for fine-tuning language models on your document corpus using QLoRA (Quantized Low-Rank Adaptation).

**✨ No login required! Uses completely FREE open-source models!**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Project Structure](#project-structure)
- [Step-by-Step Guide](#step-by-step-guide)
- [Available Models](#available-models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## 🎯 Overview

This pipeline allows you to:
1. **Extract text** from PDFs, Word documents, and text files
2. **Process documents** into training chunks with proper formatting
3. **Fine-tune a language model** on your data using memory-efficient QLoRA
4. **Query the trained model** about your documents

### What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained model and teaching it new information from your documents. After fine-tuning, the model "understands" your document content and can answer questions about it.

### What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) is a technique that makes fine-tuning possible on consumer hardware:
- **Quantization**: Compresses the model from 16GB+ to ~4GB
- **LoRA**: Only trains small adapter layers instead of the full model
- **Result**: Train a 7B parameter model on an 8GB GPU!

## ✨ Features

- 📄 **Multi-format document support**: PDF, DOCX, TXT, MD
- 🧩 **Smart chunking**: Splits documents with overlap to preserve context
- 📊 **Automatic dataset creation**: Generates training data in instruction format
- 🚀 **QLoRA fine-tuning**: Train on consumer GPUs (8GB+ VRAM)
- 💾 **Checkpoint saving**: Resume training if interrupted
- 📈 **Progress tracking**: TensorBoard integration for monitoring
- 💬 **Interactive inference**: Chat with your trained model
- 🆓 **100% FREE**: Uses open-source models, no accounts needed!

## 💻 Hardware Requirements

### For Mistral 7B (Default - Best Quality)

| Component | Requirement |
|-----------|------------|
| GPU | NVIDIA with 8GB+ VRAM (RTX 3070, RTX 4070, etc.) |
| RAM | 16GB system memory |
| Storage | 20GB free space |
| OS | Windows 10/11, Linux |

### For Phi-2 (Smaller - 2.7B params)

| Component | Requirement |
|-----------|------------|
| GPU | NVIDIA with 6GB+ VRAM (RTX 3060, etc.) |
| RAM | 12GB system memory |
| Storage | 10GB free space |

### For TinyLlama (Smallest - 1.1B params)

| Component | Requirement |
|-----------|------------|
| GPU | NVIDIA with 4GB+ VRAM |
| RAM | 8GB system memory |
| Storage | 5GB free space |

## 🚀 Quick Start

```bash
# 1. Navigate to the project
cd llama2_document_trainer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your documents to the 'documents' folder

# 5. Run the complete pipeline
python run_pipeline.py

# 6. Chat with your trained model
python scripts/inference.py --model_path models/mistral-document-finetuned_*
```

That's it! **No login, no API keys, no license acceptance required!**

## 📦 Detailed Setup

### Step 1: Install Python

Make sure you have Python 3.9 or later installed:

```bash
python --version  # Should show 3.9+
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install PyTorch with CUDA

For GPU acceleration, install PyTorch with CUDA support:

```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch (adjust cuda version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
llama2_document_trainer/
├── documents/              # 📂 Place your PDF/DOCX/TXT files here
├── data/
│   ├── extracted/          # 📄 Extracted text from documents
│   └── training/           # 📊 Processed training dataset
├── models/
│   └── checkpoints/        # 💾 Training checkpoints
├── scripts/
│   ├── extract_text.py     # 📝 Extract text from documents
│   ├── create_dataset.py   # 🔧 Create training dataset
│   ├── train.py            # 🚂 Fine-tune the model
│   └── inference.py        # 💬 Chat with trained model
├── config.py               # ⚙️ Configuration settings
├── requirements.txt        # 📋 Python dependencies
├── run_pipeline.py         # 🎯 Run entire pipeline at once
└── README.md               # 📖 This file
```

## 📚 Step-by-Step Guide

### Step 1: Prepare Your Documents

1. The `documents` folder should already exist
2. Add your PDF, DOCX, or TXT files
3. Organize into subfolders if desired (the script recursively finds all files)

**Supported formats:**
- `.pdf` - PDF documents (including scanned with OCR!)
- `.docx` - Microsoft Word documents
- `.txt` - Plain text files
- `.md` - Markdown files
- `.png`, `.jpg`, `.jpeg` - Images (with OCR enabled)

**Tips:**
- For scanned documents, use the `--ocr` flag
- Ensure documents are in readable condition

### Step 2: Extract Text

```bash
python scripts/extract_text.py
```

This script:
- Reads all documents from `documents/`
- Extracts clean text
- Saves to `data/extracted/`

**For scanned documents or images, enable OCR:**
```bash
python scripts/extract_text.py --ocr
```

**Output:**
```
Found 50 documents to process
Extracting text: 100%|██████████| 50/50 [00:30<00:00]
✓ Successfully extracted: 50
  (including 10 with OCR)
Total words extracted: 250,000
```

### OCR Setup (for scanned documents)

To use OCR, install Tesseract:

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and note the installation path
3. Add to PATH or set `TESSDATA_PREFIX` environment variable

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### Step 3: Create Training Dataset

```bash
python scripts/create_dataset.py
```

This script:
- Chunks text into 384-token segments
- Creates Q&A pairs for training
- Splits into train/validation sets
- Saves in Hugging Face format

**Options:**
```bash
# Custom chunk size
python scripts/create_dataset.py --chunk_size 256 --overlap 32

# More Q&A pairs per chunk
python scripts/create_dataset.py --qa_pairs 5
```

### Step 4: Fine-Tune the Model

```bash
python scripts/train.py
```

This script:
- Downloads Mistral 7B automatically (first run only)
- Applies 4-bit quantization
- Applies LoRA adapters
- Trains on your dataset
- Saves checkpoints every 100 steps

**Training Options:**
```bash
# More epochs (better learning, longer training)
python scripts/train.py --epochs 5

# Lower batch size (if running out of memory)
python scripts/train.py --batch_size 2

# Use a smaller model (less VRAM required)
python scripts/train.py --model microsoft/phi-2
```

**Monitoring Training:**

Training logs are saved to TensorBoard. View them with:
```bash
tensorboard --logdir models/
```

### Step 5: Test Your Model

```bash
# Interactive chat (replace * with actual timestamp)
python scripts/inference.py --model_path models/mistral-document-finetuned_*

# Single question
python scripts/inference.py --model_path models/mistral-document-finetuned_* \
    --query "What is the main topic?"

# With automatic context retrieval
python scripts/inference.py --model_path models/mistral-document-finetuned_* \
    --use_context
```

## 🤖 Available Models

All these models are **FREE** and don't require any login:

| Model | Size | VRAM Needed | Quality | Best For |
|-------|------|-------------|---------|----------|
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ~8GB | ⭐⭐⭐⭐⭐ | Best quality, general use |
| `microsoft/phi-2` | 2.7B | ~6GB | ⭐⭐⭐⭐ | Good quality, less resources |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | ~4GB | ⭐⭐⭐ | Low-resource machines |
| `openlm-research/open_llama_7b` | 7B | ~8GB | ⭐⭐⭐⭐ | Alternative to Mistral |

To use a different model:
```bash
python scripts/train.py --model microsoft/phi-2
```

Or edit `config.py`:
```python
model_name: str = "microsoft/phi-2"  # Change this
```

## ⚙️ Configuration

All settings are in `config.py`. Key parameters:

### Model Settings

```python
# Default: Mistral 7B (best quality)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Maximum sequence length
max_seq_length = 2048
```

### Training Settings

```python
# Number of training epochs
num_train_epochs = 3

# Batch size (reduce if out of memory)
per_device_train_batch_size = 4

# Learning rate (2e-4 works well for most cases)
learning_rate = 2e-4
```

### LoRA Settings

```python
# LoRA rank (higher = more capacity, more memory)
r = 16

# LoRA alpha (scaling factor)
lora_alpha = 32
```

## 🔧 Troubleshooting

### CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```bash
   python scripts/train.py --batch_size 1
   ```
2. Use a smaller model:
   ```bash
   python scripts/train.py --model microsoft/phi-2
   ```
3. Close other GPU applications

### bitsandbytes Issues on Windows

**Symptoms:** `ImportError: bitsandbytes` error

**Solutions:**
1. Install the Windows-compatible version:
   ```bash
   pip uninstall bitsandbytes
   pip install bitsandbytes-windows
   ```
2. Or use WSL2 (Windows Subsystem for Linux)

### Slow Training

**Symptoms:** Training is very slow

**Solutions:**
1. Ensure CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
2. Make sure you have a GPU with enough VRAM
3. Increase batch size if you have memory available

### Model Download Issues

**Symptoms:** Model fails to download

**Solutions:**
1. Check your internet connection
2. Try again (Hugging Face servers can be slow sometimes)
3. The models are large (several GB) - be patient!

### Poor Model Responses

**Symptoms:** Model gives generic or irrelevant answers

**Solutions:**
1. Train for more epochs (3-5)
2. Add more training documents
3. Use `--use_context` during inference
4. Increase LoRA rank in config.py

## ❓ FAQ

### How long does training take?

Depends on your hardware and data:
- **100 documents, RTX 3080, Mistral 7B**: ~1-2 hours
- **100 documents, RTX 3060, Phi-2**: ~30-45 minutes
- **CPU only**: Not recommended (very slow)

### How much data do I need?

- **Minimum**: 10-20 documents for basic results
- **Recommended**: 50-100+ documents for good results
- **Quality matters**: Well-written, relevant documents work better

### Which model should I use?

- **Have 8GB+ VRAM**: Use Mistral 7B (default) - best quality
- **Have 6GB VRAM**: Use Phi-2 - still very good
- **Have 4GB VRAM**: Use TinyLlama - decent quality

### Can I run this on CPU only?

Technically yes, but it's extremely slow (days instead of hours). We strongly recommend having an NVIDIA GPU.

### Do I need to pay for anything?

**No!** Everything is free:
- Python: Free
- PyTorch: Free
- All models: Free (Apache 2.0 / MIT licenses)
- No API keys needed
- No accounts needed

### How do I improve quality?

1. **Better documents**: Clean, well-formatted text
2. **More training**: Increase epochs (3-5)
3. **Higher LoRA rank**: Use r=32 or r=64 in config.py
4. **Manual Q&A pairs**: Edit the training data to add your own

## 📄 License

This project is MIT licensed. The models have their own licenses:
- Mistral 7B: Apache 2.0 (free for commercial use)
- Phi-2: MIT (free for commercial use)
- TinyLlama: Apache 2.0 (free for commercial use)

## 🙏 Acknowledgments

- Mistral AI for the excellent Mistral 7B model
- Microsoft for Phi-2
- Hugging Face for transformers and PEFT
- Tim Dettmers for bitsandbytes

---

**Happy Training! 🚀✨**

If you have questions, check the troubleshooting section above.
