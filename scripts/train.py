"""
============================================================================
Model Fine-Tuning Script with QLoRA/LoRA
============================================================================
This script fine-tunes a language model on your document dataset using 
parameter-efficient methods (LoRA/QLoRA) that can run on consumer hardware.

DEFAULT MODEL: Mistral 7B (FREE, no login required!)

WHAT THIS SCRIPT DOES:
1. Loads the model with 4-bit quantization (QLoRA)
2. Adds LoRA adapters to the model
3. Loads your training dataset
4. Trains the model with progress tracking
5. Saves checkpoints and final model

HARDWARE REQUIREMENTS:
- GPU with 8GB+ VRAM for 7B model (RTX 3070/4070 or better)
- GPU with 6GB+ VRAM for 2-3B model (RTX 3060 or better)
- 16GB+ system RAM

USAGE:
    python scripts/train.py
    
    Or with custom settings:
    python scripts/train.py --epochs 5 --batch_size 2 --learning_rate 1e-4

BEFORE RUNNING:
1. Run extract_text.py and create_dataset.py first
2. That's it! No login or license acceptance required!

REQUIREMENTS:
    pip install transformers torch peft bitsandbytes accelerate datasets tqdm rich
============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADING WITH QUANTIZATION
# ============================================================================
def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    torch_dtype: str = "float16"
):
    """
    Load language model with optional quantization for memory efficiency.
    
    WHAT IS QUANTIZATION?
    - Normal models use 32-bit or 16-bit numbers for weights
    - Quantization uses 4-bit or 8-bit numbers
    - This reduces memory by 4-8x with minimal quality loss
    
    Args:
        model_name: Hugging Face model name (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
        load_in_4bit: Use 4-bit quantization (recommended)
        load_in_8bit: Use 8-bit quantization
        torch_dtype: Data type for computation
        
    Returns:
        model, tokenizer tuple
    """
    console.print(f"[bold blue]Loading model: {model_name}[/bold blue]")
    
    # Determine data type
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(torch_dtype, torch.float16)
    
    # Configure quantization
    bnb_config = None
    if load_in_4bit:
        console.print("[green]Using 4-bit quantization (QLoRA)[/green]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        console.print("[green]Using 8-bit quantization[/green]")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load tokenizer
    console.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set padding token (Llama doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # Load model
    console.print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype if not (load_in_4bit or load_in_8bit) else None,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    console.print("[green]✓ Model loaded successfully![/green]")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total parameters: {total_params:,}")
    console.print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


# ============================================================================
# LoRA CONFIGURATION
# ============================================================================
def apply_lora(model, lora_config_dict: dict = None):
    """
    Apply LoRA (Low-Rank Adaptation) to the model.
    
    WHAT IS LoRA?
    - Instead of training all billions of parameters, LoRA adds small
      trainable matrices to specific layers
    - These matrices learn to adapt the model to your data
    - Training is 10-100x faster and uses much less memory
    
    Args:
        model: The base model
        lora_config_dict: LoRA configuration dictionary
        
    Returns:
        Model with LoRA adapters
    """
    from config import lora_config as default_lora
    
    if lora_config_dict is None:
        lora_config_dict = {}
    
    # Create LoRA configuration
    config = LoraConfig(
        r=lora_config_dict.get('r', default_lora.r),
        lora_alpha=lora_config_dict.get('lora_alpha', default_lora.lora_alpha),
        target_modules=lora_config_dict.get('target_modules', default_lora.target_modules),
        lora_dropout=lora_config_dict.get('lora_dropout', default_lora.lora_dropout),
        bias=lora_config_dict.get('bias', default_lora.bias),
        task_type=TaskType.CAUSAL_LM,
    )
    
    console.print("[bold blue]Applying LoRA adapters...[/bold blue]")
    console.print(f"  Rank (r): {config.r}")
    console.print(f"  Alpha: {config.lora_alpha}")
    console.print(f"  Target modules: {config.target_modules}")
    console.print(f"  Dropout: {config.lora_dropout}")
    
    # Apply LoRA
    model = get_peft_model(model, config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


# ============================================================================
# DATASET LOADING AND PREPROCESSING
# ============================================================================
def load_training_data(data_dir: Path, tokenizer, max_length: int = 2048):
    """
    Load and preprocess the training dataset.
    
    Args:
        data_dir: Directory containing train.json and val.json
        tokenizer: The model's tokenizer
        max_length: Maximum sequence length
        
    Returns:
        train_dataset, val_dataset tuple
    """
    data_dir = Path(data_dir)
    
    # Try to load from Hugging Face format first
    hf_path = data_dir / "hf_dataset"
    if hf_path.exists():
        console.print(f"Loading dataset from {hf_path}")
        dataset_dict = load_from_disk(str(hf_path))
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]
    else:
        # Load from JSON files
        train_path = data_dir / "train.json"
        val_path = data_dir / "val.json"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Please run create_dataset.py first."
            )
        
        console.print(f"Loading dataset from JSON files...")
        
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
    
    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Validation samples: {len(val_dataset)}")
    
    # Function to convert different formats to unified "text" field
    def normalize_example(example):
        """Convert different data formats to a unified text format."""
        # If already has "text" field, use it
        if "text" in example and example["text"]:
            return {"text": str(example["text"])}
        
        # If has instruction/input/output format, combine them
        if "instruction" in example:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            return {"text": text}
        
        # If has question/answer/context format
        if "question" in example and "answer" in example:
            question = example.get("question", "")
            answer = example.get("answer", "")
            context = example.get("context", "")
            
            if context:
                text = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n{answer}"
            else:
                text = f"### Question:\n{question}\n\n### Answer:\n{answer}"
            return {"text": text}
        
        # Fallback: concatenate all string values
        text_parts = []
        for key, value in example.items():
            if isinstance(value, str) and value:
                text_parts.append(value)
        return {"text": " ".join(text_parts) if text_parts else ""}
    
    console.print("Normalizing data format...")
    train_dataset = train_dataset.map(normalize_example, desc="Normalizing train set")
    val_dataset = val_dataset.map(normalize_example, desc="Normalizing validation set")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        """Tokenize examples and create labels for causal LM."""
        # Get the text field and ensure all items are strings
        texts = examples["text"]
        
        # Handle both batched and single examples
        if isinstance(texts, list):
            # Ensure all texts are strings
            texts = [str(t) if t is not None else "" for t in texts]
        else:
            texts = str(texts) if texts is not None else ""
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    console.print("Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train set"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    return train_dataset, val_dataset


# ============================================================================
# TRAINING SETUP
# ============================================================================
def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    training_args_dict: dict = None
):
    """
    Create a Hugging Face Trainer with the specified configuration.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args_dict: Override training arguments
        
    Returns:
        Configured Trainer object
    """
    from config import training_config as default_training
    
    if training_args_dict is None:
        training_args_dict = {}
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = training_args_dict.get(
        'output_dir',
        default_training.output_dir
    ) + f"_{timestamp}"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args_dict.get('num_train_epochs', default_training.num_train_epochs),
        per_device_train_batch_size=training_args_dict.get('per_device_train_batch_size', default_training.per_device_train_batch_size),
        per_device_eval_batch_size=training_args_dict.get('per_device_train_batch_size', default_training.per_device_train_batch_size),
        gradient_accumulation_steps=training_args_dict.get('gradient_accumulation_steps', default_training.gradient_accumulation_steps),
        learning_rate=training_args_dict.get('learning_rate', default_training.learning_rate),
        weight_decay=training_args_dict.get('weight_decay', default_training.weight_decay),
        warmup_ratio=training_args_dict.get('warmup_ratio', default_training.warmup_ratio),
        lr_scheduler_type=training_args_dict.get('lr_scheduler_type', default_training.lr_scheduler_type),
        max_grad_norm=training_args_dict.get('max_grad_norm', default_training.max_grad_norm),
        save_steps=training_args_dict.get('save_steps', default_training.save_steps),
        save_total_limit=training_args_dict.get('save_total_limit', default_training.save_total_limit),
        logging_steps=training_args_dict.get('logging_steps', default_training.logging_steps),
        eval_strategy=training_args_dict.get('evaluation_strategy', default_training.evaluation_strategy),
        eval_steps=training_args_dict.get('eval_steps', default_training.eval_steps),
        fp16=training_args_dict.get('fp16', default_training.fp16),
        bf16=training_args_dict.get('bf16', default_training.bf16),
        optim=training_args_dict.get('optim', default_training.optim),
        gradient_checkpointing=training_args_dict.get('gradient_checkpointing', default_training.gradient_checkpointing),
        group_by_length=training_args_dict.get('group_by_length', default_training.group_by_length),
        report_to=training_args_dict.get('report_to', default_training.report_to),
        seed=training_args_dict.get('seed', default_training.seed),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    console.print(f"[green]✓ Trainer configured[/green]")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Epochs: {training_args.num_train_epochs}")
    console.print(f"  Batch size: {training_args.per_device_train_batch_size}")
    console.print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    console.print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    console.print(f"  Learning rate: {training_args.learning_rate}")
    
    return trainer


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(
    model_name: str = None,
    data_dir: Path = None,
    output_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    use_4bit: bool = True,
    lora_r: int = None,
    resume_from_checkpoint: str = None,
):
    """
    Main training function that orchestrates the entire fine-tuning process.
    
    Args:
        model_name: Hugging Face model name
        data_dir: Directory containing training data
        output_dir: Where to save the model
        epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        use_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    from config import (
        model_config, training_config, DATA_DIR, get_model_name
    )
    
    # Set defaults from config
    if model_name is None:
        model_name = get_model_name(model_config)
    if data_dir is None:
        data_dir = DATA_DIR / "training"
    
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    console.print("[bold magenta]   Document Fine-Tuning with QLoRA[/bold magenta]")
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    console.print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        console.print(f"[green]✓ CUDA available: {torch.cuda.get_device_name(0)}[/green]")
        console.print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        console.print("[yellow]⚠ CUDA not available. Training will be slow on CPU.[/yellow]")
    
    console.print()
    
    # Step 1: Load model and tokenizer
    console.print("[bold]Step 1: Loading model and tokenizer[/bold]")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=use_4bit,
        load_in_8bit=not use_4bit,
    )
    console.print()
    
    # Step 2: Apply LoRA
    console.print("[bold]Step 2: Applying LoRA adapters[/bold]")
    lora_config_dict = {}
    if lora_r is not None:
        lora_config_dict['r'] = lora_r
    model = apply_lora(model, lora_config_dict)
    console.print()
    
    # Step 3: Load training data
    console.print("[bold]Step 3: Loading training data[/bold]")
    train_dataset, val_dataset = load_training_data(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=model_config.max_seq_length,
    )
    console.print()
    
    # Step 4: Create trainer
    console.print("[bold]Step 4: Setting up trainer[/bold]")
    training_args_dict = {}
    if output_dir:
        training_args_dict['output_dir'] = output_dir
    if epochs:
        training_args_dict['num_train_epochs'] = epochs
    if batch_size:
        training_args_dict['per_device_train_batch_size'] = batch_size
    if learning_rate:
        training_args_dict['learning_rate'] = learning_rate
    
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args_dict=training_args_dict,
    )
    console.print()
    
    # Step 5: Train!
    console.print("[bold]Step 5: Starting training[/bold]")
    console.print("[yellow]This may take a while depending on your data and hardware...[/yellow]")
    console.print()
    
    try:
        # Start training
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Print results
        console.print()
        console.print("[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold green]   Training Complete![/bold green]")
        console.print("[bold green]" + "="*60 + "[/bold green]")
        console.print()
        console.print(f"Training loss: {train_result.training_loss:.4f}")
        console.print(f"Total steps: {train_result.global_step}")
        
        # Save the final model
        console.print()
        console.print("[bold]Saving final model...[/bold]")
        trainer.save_model()
        tokenizer.save_pretrained(trainer.args.output_dir)
        
        console.print(f"[green]✓ Model saved to: {trainer.args.output_dir}[/green]")
        
        # Save training info
        info_path = Path(trainer.args.output_dir) / "training_info.json"
        training_info = {
            "model_name": model_name,
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "epochs": training_args_dict.get('num_train_epochs', training_config.num_train_epochs),
            "batch_size": training_args_dict.get('per_device_train_batch_size', training_config.per_device_train_batch_size),
            "learning_rate": training_args_dict.get('learning_rate', training_config.learning_rate),
            "lora_r": lora_r or 16,
            "use_4bit": use_4bit,
        }
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        return trainer.args.output_dir
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        console.print("Checkpoints have been saved.")
        raise
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 2 on document data with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default settings
    python scripts/train.py
    
    # Custom settings
    python scripts/train.py --epochs 5 --batch_size 2 --learning_rate 1e-4
    
    # Use a smaller model (for less VRAM)
    python scripts/train.py --model microsoft/phi-2
    
    # Resume from checkpoint
    python scripts/train.py --resume_from checkpoint-500
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: mistralai/Mistral-7B-Instruct-v0.2)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the trained model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per device (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use 8-bit)"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        resume_from_checkpoint=args.resume_from,
    )


if __name__ == "__main__":
    main()

