"""
============================================================================
Configuration Settings for Document Training Pipeline
============================================================================
This file contains all configurable parameters for the training pipeline.
Adjust these settings based on your hardware capabilities and requirements.

BEGINNER TIP: Start with the default values, then adjust based on:
- Your GPU memory (VRAM)
- Number of documents you have
- How much time you want to spend training

NOTE: This pipeline uses Mistral 7B by default - a completely FREE model
that doesn't require any login or license acceptance!
============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# These paths define where your files are stored

# Base directory (automatically detects the project root)
BASE_DIR = Path(__file__).parent.absolute()

# Directory containing your PDF and document files
DOCUMENTS_DIR = BASE_DIR / "documents"

# Directory for processed training data
DATA_DIR = BASE_DIR / "data"

# Directory for saved models and checkpoints
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class ModelConfig:
    """
    Settings for the model.
    
    AVAILABLE FREE MODELS (no login required):
    - mistralai/Mistral-7B-Instruct-v0.2: Best quality, 7B params (~8GB VRAM)
    - microsoft/phi-2: Smaller but capable, 2.7B params (~6GB VRAM)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0: Very small, 1.1B params (~4GB VRAM)
    - openlm-research/open_llama_7b: Open recreation of LLaMA (~8GB VRAM)
    
    CHOOSING A MODEL SIZE:
    - 7B: Requires ~8GB VRAM with 4-bit quantization (RTX 3070/4070 or better)
    - 2-3B: Requires ~6GB VRAM (RTX 3060 or better)
    - 1B: Requires ~4GB VRAM (most modern GPUs)
    
    For consumer hardware, we recommend Mistral 7B or Phi-2.
    """
    
    # Model name from Hugging Face Hub (NO LOGIN REQUIRED for these!)
    # Options: 
    #   "mistralai/Mistral-7B-Instruct-v0.2" (recommended - best quality)
    #   "microsoft/phi-2" (smaller, still good)
    #   "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (very small, fast)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Set to False since we're specifying the exact model name
    use_chat_model: bool = False
    
    # Maximum sequence length (context window)
    # Mistral supports up to 8192 tokens (we use 2048 for efficiency)
    max_seq_length: int = 2048
    
    # Data type for model weights
    # Options: "float16", "bfloat16", "float32"
    # bfloat16 is recommended for newer GPUs (Ampere and later)
    torch_dtype: str = "float16"


# ============================================================================
# QUANTIZATION CONFIGURATION (for running on consumer hardware)
# ============================================================================
@dataclass
class QuantizationConfig:
    """
    Settings for model quantization to reduce memory usage.
    
    WHAT IS QUANTIZATION?
    Quantization reduces the precision of model weights (e.g., from 32-bit to 4-bit),
    dramatically reducing memory requirements while maintaining most of the quality.
    
    4-bit quantization: ~75% memory reduction, minimal quality loss
    8-bit quantization: ~50% memory reduction, nearly no quality loss
    """
    
    # Enable 4-bit quantization (highly recommended for consumer GPUs)
    load_in_4bit: bool = True
    
    # Enable 8-bit quantization (use if 4-bit causes issues)
    load_in_8bit: bool = False
    
    # Quantization type for 4-bit
    # "nf4" (NormalFloat4) is generally best for LLMs
    bnb_4bit_quant_type: str = "nf4"
    
    # Use double quantization for additional memory savings
    bnb_4bit_use_double_quant: bool = True
    
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype: str = "float16"


# ============================================================================
# LoRA CONFIGURATION (Parameter-Efficient Fine-Tuning)
# ============================================================================
@dataclass
class LoRAConfig:
    """
    Settings for Low-Rank Adaptation (LoRA).
    
    WHAT IS LoRA?
    Instead of fine-tuning all model parameters (billions!), LoRA only trains
    small "adapter" matrices. This reduces:
    - Training time by 10-100x
    - Memory requirements significantly
    - Risk of catastrophic forgetting
    
    The trained LoRA adapter can be merged with the base model or used separately.
    """
    
    # Rank of the low-rank matrices
    # Higher = more capacity but more memory/compute
    # Typical values: 8, 16, 32, 64
    # For document training, 16-32 is usually sufficient
    r: int = 16
    
    # Alpha parameter for LoRA scaling
    # The actual scaling factor is alpha/r
    # Typical: same as r, or 2x r
    lora_alpha: int = 32
    
    # Dropout for LoRA layers (regularization)
    # 0.05-0.1 is typical
    lora_dropout: float = 0.05
    
    # Which modules to apply LoRA to
    # For Llama 2, these are the attention projection layers
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",   # Query projection
        "k_proj",   # Key projection  
        "v_proj",   # Value projection
        "o_proj",   # Output projection
        "gate_proj",  # MLP gate projection
        "up_proj",    # MLP up projection
        "down_proj",  # MLP down projection
    ])
    
    # Bias training strategy
    # "none": Don't train biases (most common)
    # "all": Train all biases
    # "lora_only": Only train biases in LoRA layers
    bias: str = "none"
    
    # Task type for PEFT
    task_type: str = "CAUSAL_LM"


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    """
    Settings for the training process.
    
    ADJUSTING FOR YOUR HARDWARE:
    - Less VRAM? Reduce batch_size, increase gradient_accumulation_steps
    - More VRAM? Increase batch_size for faster training
    - Slow training? Increase learning_rate slightly
    - Poor results? Decrease learning_rate, increase num_train_epochs
    """
    
    # Output directory for checkpoints and final model
    output_dir: str = str(MODELS_DIR / "mistral-document-finetuned")
    
    # Number of training epochs
    # 1-3 epochs is usually sufficient for document fine-tuning
    num_train_epochs: int = 3
    
    # Batch size per GPU
    # Reduce if you run out of memory
    per_device_train_batch_size: int = 4
    
    # Gradient accumulation steps
    # Effective batch size = batch_size * gradient_accumulation_steps
    # Increase this if you reduce batch_size
    gradient_accumulation_steps: int = 4
    
    # Learning rate
    # For LoRA fine-tuning, 1e-4 to 3e-4 works well
    learning_rate: float = 2e-4
    
    # Weight decay (L2 regularization)
    weight_decay: float = 0.01
    
    # Warmup ratio (fraction of training for learning rate warmup)
    warmup_ratio: float = 0.03
    
    # Learning rate scheduler
    # "cosine" provides smooth decay, "linear" is simpler
    lr_scheduler_type: str = "cosine"
    
    # Maximum gradient norm (for gradient clipping)
    max_grad_norm: float = 0.3
    
    # Save checkpoint every N steps
    save_steps: int = 100
    
    # Keep only the N most recent checkpoints
    save_total_limit: int = 3
    
    # Log training metrics every N steps
    logging_steps: int = 10
    
    # Evaluation strategy
    # "steps": evaluate every N steps
    # "epoch": evaluate at the end of each epoch
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    
    # Use mixed precision training (recommended for speed)
    fp16: bool = True
    bf16: bool = False  # Set to True if using Ampere GPU or newer
    
    # Optimizer settings
    optim: str = "paged_adamw_32bit"
    
    # Gradient checkpointing (trades compute for memory)
    # Enable if running out of memory
    gradient_checkpointing: bool = True
    
    # Group sequences by length for efficient batching
    group_by_length: bool = True
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Report training metrics to (set to "none" to disable)
    # Options: "none", "wandb", "tensorboard"
    report_to: str = "tensorboard"


# ============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# ============================================================================
@dataclass
class DocumentConfig:
    """
    Settings for document processing and chunking.
    
    CHUNK SIZE CONSIDERATIONS:
    - Smaller chunks: More training samples, but less context per sample
    - Larger chunks: Better context, but fewer samples and more memory
    
    For Llama 2 with 2048 max sequence length:
    - Chunk size of 256-512 tokens is recommended
    - This leaves room for instruction prompts and responses
    """
    
    # Chunk size in tokens
    # Should be less than max_seq_length to leave room for prompts
    chunk_size: int = 384
    
    # Overlap between chunks (prevents cutting off important context)
    chunk_overlap: int = 64
    
    # Minimum chunk size (discard chunks smaller than this)
    min_chunk_size: int = 50
    
    # Supported file extensions
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf",
        ".txt", 
        ".docx",
        ".doc",
        ".md",
    ])
    
    # Clean extracted text (remove extra whitespace, special chars)
    clean_text: bool = True
    
    # Split on sentence boundaries when possible
    respect_sentence_boundaries: bool = True


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
@dataclass
class InferenceConfig:
    """
    Settings for model inference (generating responses).
    """
    
    # Maximum new tokens to generate
    max_new_tokens: int = 512
    
    # Temperature (randomness)
    # 0.0 = deterministic, 1.0 = more random
    # For factual Q&A, use 0.1-0.3
    temperature: float = 0.2
    
    # Top-p (nucleus) sampling
    # Only consider tokens with cumulative probability >= top_p
    top_p: float = 0.9
    
    # Top-k sampling
    # Only consider the k most likely tokens
    top_k: int = 40
    
    # Repetition penalty
    # >1.0 reduces repetition
    repetition_penalty: float = 1.1
    
    # Number of beams for beam search (1 = greedy)
    num_beams: int = 1
    
    # Enable sampling (set False for deterministic output)
    do_sample: bool = True


# ============================================================================
# INSTRUCTION TEMPLATE
# ============================================================================
# This template formats training data for instruction-following

INSTRUCTION_TEMPLATE = """### Instruction:
You are a helpful AI assistant that answers questions based on the provided document context.
Answer the following question using only the information from the context below.
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

### Context:
{context}

### Question:
{question}

### Response:
{answer}"""

INFERENCE_TEMPLATE = """### Instruction:
You are a helpful AI assistant that answers questions based on the provided document context.
Answer the following question using only the information from the context below.
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

### Context:
{context}

### Question:
{question}

### Response:
"""


# ============================================================================
# HELPER FUNCTION: Get actual model name
# ============================================================================
def get_model_name(config: ModelConfig) -> str:
    """Get the correct model name based on configuration."""
    # For Mistral and other models, just return the model name directly
    return config.model_name


# ============================================================================
# CREATE DEFAULT CONFIGURATIONS
# ============================================================================
model_config = ModelConfig()
quantization_config = QuantizationConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
document_config = DocumentConfig()
inference_config = InferenceConfig()


# ============================================================================
# PRINT CONFIGURATION (for debugging)
# ============================================================================
def print_config():
    """Print all configuration settings."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    configs = [
        ("Model", model_config),
        ("Quantization", quantization_config),
        ("LoRA", lora_config),
        ("Training", training_config),
        ("Document", document_config),
        ("Inference", inference_config),
    ]
    
    for name, cfg in configs:
        table = Table(title=f"{name} Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in vars(cfg).items():
            table.add_row(key, str(value))
        
        console.print(table)
        console.print()


if __name__ == "__main__":
    print_config()

