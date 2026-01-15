# 🤖 AI Trading Model - Complete Guide

## 📑 Table of Contents
1. [Quick Start](#-quick-start)
2. [First Time Setup](#-first-time-setup-do-once)
3. [Training Your Model](#-training-your-model)
4. [Using Your Trained Model](#-using-your-trained-model)
5. [Trading Assistant](#-trading-assistant-real-time-stocks)
6. [Adding More Documents](#-adding-more-documents)
7. [Troubleshooting](#-troubleshooting)
8. [All Commands Reference](#-all-commands-reference)

---

## 🚀 Quick Start

```bash
# 1. Open terminal and navigate to project
cd "C:\Users\camer\OneDrive\Desktop\AI Test\ai_model"

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Train the model (if not done yet)
python run_pipeline.py --skip_extraction --skip_dataset

# 4. Use the trading assistant
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_*
```

---

## 📦 First Time Setup (Do Once)

### Step 1: Open Terminal
Open PowerShell or Command Prompt

### Step 2: Navigate to Project
```bash
cd "C:\Users\camer\OneDrive\Desktop\AI Test\ai_model"
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment
```bash
venv\Scripts\activate
```
You should see `(venv)` at the start of your command line.

### Step 5: Install All Dependencies
```bash
pip install -r requirements.txt
pip install yfinance ta pandas
```

### Step 6: (Optional) Speed Up Downloads
```bash
pip install hf_xet
```

---

## 🎯 Training Your Model

### Option A: One Command (Recommended)
```bash
python run_pipeline.py
```
This does everything automatically:
1. ✅ Extracts text from your PDFs
2. ✅ Creates training dataset
3. ✅ Downloads Mistral 7B model (~15GB, first time only)
4. ✅ Trains the model on your documents

### Option B: Skip Already Completed Steps
If you've already extracted text and created the dataset:
```bash
python run_pipeline.py --skip_extraction --skip_dataset
```

### Option C: Step by Step
```bash
# Step 1: Extract text from documents
python scripts/extract_text.py

# Step 2: Create training dataset
python scripts/create_dataset.py

# Step 3: Train the model
python scripts/train.py
```

### Training Options
```bash
# More training epochs (better learning)
python scripts/train.py --epochs 5

# Lower batch size (if out of memory)
python scripts/train.py --batch_size 1

# Use smaller model (needs less GPU memory)
python scripts/train.py --model microsoft/phi-2
```

### How Long Does Training Take?
| Phase | Time |
|-------|------|
| Model download (first time) | 1-3 hours |
| Training (3 epochs) | 1-2 hours |
| **Total first time** | **2-5 hours** |
| **Subsequent runs** | **1-2 hours** |

---

## 💬 Using Your Trained Model

### Chat Mode (Interactive)
```bash
python scripts/inference.py --model_path models/mistral-document-finetuned_*
```
Then just type your questions:
```
You: What is RSI?
Assistant: [Answer based on your documents]

You: How do I identify a bullish chart pattern?
Assistant: [Answer based on your documents]

You: quit
```

### Single Question Mode
```bash
python scripts/inference.py --model_path models/mistral-document-finetuned_* --query "What is RSI?"
```

### With Document Context
```bash
python scripts/inference.py --model_path models/mistral-document-finetuned_* --use_context
```

---

## 📈 Trading Assistant (Real-Time Stocks)

The Trading Assistant fetches **live stock data** and uses your AI to analyze it!

### Launch Trading Assistant
```bash
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_*
```

### Quick Stock Analysis
```bash
# Analyze Apple
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker AAPL

# Analyze Meta with options data
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker META --options

# Just see technical data (no AI)
python scripts/trading_assistant.py --ticker TSLA --no_ai
```

### In Interactive Mode, Type:
| Command | What It Does |
|---------|--------------|
| `AAPL` | Analyze Apple stock |
| `META call option` | Meta with options data |
| `Should I buy TSLA?` | Tesla recommendation |
| `NVDA options` | Nvidia options breakdown |
| `SPY` | S&P 500 ETF analysis |
| `quit` | Exit the assistant |

### What You Get:
- 📊 **Price Action**: Current price, daily change, 52-week range
- 📈 **RSI**: With oversold/overbought signals
- 📉 **MACD**: Bullish/bearish crossovers
- 📊 **Moving Averages**: 20, 50 SMA and 9, 21 EMA
- 📈 **Bollinger Bands**: Upper, middle, lower bands
- 📊 **Volume Analysis**: Compared to average
- 🎯 **Support/Resistance**: Predicted bottoms & tops
- 📐 **Fibonacci Levels**: 23.6%, 38.2%, 50%, 61.8%
- 📊 **Options Chain**: Full analysis with Put/Call ratio
- 🔥 **Unusual Activity**: High volume options alerts
- 💰 **Entry Levels**: Suggested stop loss & take profit
- 🤖 **AI Recommendation**: Based on your trading documents

### New Features:
- **Bottom/Top Prediction**: Uses pivot points, Fibonacci, and swing highs/lows
- **Options Flow**: Put/Call ratio, max pain levels, unusual activity detection
- **Key Strike Levels**: Shows where big money is positioned

---

## 📁 Adding More Documents

### Step 1: Add Files
Put your new PDFs, DOCX, or TXT files in:
```
C:\Users\camer\OneDrive\Desktop\AI Test\ai_model\documents\
```

### Step 2: Retrain
```bash
python run_pipeline.py
```

### Supported File Types
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.txt` - Text files
- `.md` - Markdown files
- `.png`, `.jpg` - Images (with `--ocr` flag)

### For Scanned Documents
```bash
python run_pipeline.py --ocr
```

---

## ❓ Troubleshooting

### "CUDA out of memory"
Your GPU doesn't have enough memory. Try:
```bash
python scripts/train.py --batch_size 1
```
Or use a smaller model:
```bash
python scripts/train.py --model microsoft/phi-2
```

### Model Download is Slow
Install the fast downloader:
```bash
pip install hf_xet
```

### "Module not found" Errors
Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Training Interrupted
Resume from checkpoint:
```bash
python scripts/train.py --resume_from models/mistral-document-finetuned_*/checkpoint-*
```

### Check if GPU is Working
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 📋 All Commands Reference

### Setup Commands
```bash
cd "C:\Users\camer\OneDrive\Desktop\AI Test\ai_model"
venv\Scripts\activate
pip install -r requirements.txt
```

### Training Commands
```bash
python run_pipeline.py                              # Full pipeline
python run_pipeline.py --skip_extraction --skip_dataset  # Skip completed steps
python scripts/extract_text.py                      # Extract text only
python scripts/create_dataset.py                    # Create dataset only
python scripts/train.py                             # Train only
python scripts/train.py --epochs 5                  # More epochs
python scripts/train.py --batch_size 1              # Less memory
python scripts/train.py --model microsoft/phi-2    # Smaller model
```

### Chat Commands
```bash
python scripts/inference.py --model_path models/mistral-document-finetuned_*
python scripts/inference.py --model_path models/mistral-document-finetuned_* --query "Your question"
python scripts/inference.py --model_path models/mistral-document-finetuned_* --use_context
```

### Trading Assistant Commands
```bash
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_*
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker AAPL
python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker META --options
python scripts/trading_assistant.py --ticker TSLA --no_ai
```

---

## 📊 Your Current Documents

Your model is being trained on:
1. `Algorithmic_Trading_Bot.pdf`
2. `daytradingrsi.txt`
3. `Idenitfying-Chart-Patterns.pdf`
4. `RSI_PDF.pdf`
5. `Rsi& Volume Correlation.pdf`

---

## 🎯 Example Questions to Ask

After training completes, try:
- "What is RSI and how do I use it?"
- "How do I identify chart patterns?"
- "When should I buy based on RSI?"
- "Explain the algorithmic trading strategy"
- "What does RSI volume correlation tell us?"
- "What RSI value indicates oversold?"

---

**Happy Trading! 🚀📈💰**
