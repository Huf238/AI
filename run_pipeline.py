"""
============================================================================
Complete Training Pipeline Runner
============================================================================
This script runs the entire training pipeline from start to finish:
1. Extract text from documents
2. Create training dataset
3. Fine-tune the model
4. (Optionally) Test with sample questions

USAGE:
    python run_pipeline.py
    
    Or with options:
    python run_pipeline.py --skip_extraction --epochs 5

This is a convenience script that calls the individual scripts in order.
============================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_step(step_num: int, step_name: str, command: list, skip: bool = False):
    """Run a pipeline step."""
    console.print()
    console.print(Panel(
        f"[bold]Step {step_num}: {step_name}[/bold]",
        style="blue"
    ))
    
    if skip:
        console.print("[yellow]⏭ Skipping...[/yellow]")
        return True
    
    console.print(f"[dim]Running: {' '.join(command)}[/dim]")
    console.print()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).parent
        )
        console.print(f"[green]✓ {step_name} complete[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ {step_name} failed with error code {e.returncode}[/red]")
        return False
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Llama 2 document training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip text extraction (use if already done)"
    )
    
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned documents and images"
    )
    
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help="Skip dataset creation (use if already done)"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to fine-tune (default: Llama-2-7b-chat-hf)"
    )
    
    args = parser.parse_args()
    
    # Header
    console.print()
    console.print(Panel(
        "[bold magenta]🤖 Document Training Pipeline[/bold magenta]\n\n"
        "This will run the complete training pipeline.\n"
        "Make sure you have:\n"
        "1. Added documents to the 'documents/' folder\n"
        "2. A GPU with 8GB+ VRAM (or patience for CPU training)\n\n"
        "[green]✨ No login required! Uses free open-source models![/green]",
        title="Welcome",
        border_style="magenta"
    ))
    console.print()
    
    # Check for documents
    docs_dir = Path(__file__).parent / "documents"
    if not args.skip_extraction:
        doc_count = sum(1 for f in docs_dir.rglob("*") 
                       if f.suffix.lower() in ['.pdf', '.docx', '.txt', '.md'])
        if doc_count == 0:
            console.print("[red]Error: No documents found in 'documents/' folder[/red]")
            console.print("[yellow]Please add your PDF, DOCX, or TXT files first.[/yellow]")
            return
        console.print(f"[green]Found {doc_count} documents[/green]")
    
    python_exe = sys.executable
    
    # Step 1: Extract text
    extract_cmd = [python_exe, "scripts/extract_text.py"]
    if args.ocr:
        extract_cmd.append("--ocr")
    
    success = run_step(
        1, 
        "Extract Text from Documents",
        extract_cmd,
        skip=args.skip_extraction
    )
    if not success and not args.skip_extraction:
        return
    
    # Step 2: Create dataset
    success = run_step(
        2,
        "Create Training Dataset",
        [python_exe, "scripts/create_dataset.py"],
        skip=args.skip_dataset
    )
    if not success and not args.skip_dataset:
        return
    
    # Step 3: Train model
    train_cmd = [python_exe, "scripts/train.py", 
                 "--epochs", str(args.epochs),
                 "--batch_size", str(args.batch_size)]
    if args.model:
        train_cmd.extend(["--model", args.model])
    
    success = run_step(
        3,
        "Fine-Tune Model",
        train_cmd,
        skip=args.skip_training
    )
    
    # Final summary
    console.print()
    console.print(Panel(
        "[bold green]Pipeline Complete![/bold green]\n\n"
        "Your fine-tuned model is saved in the 'models/' folder.\n\n"
        "To use your model:\n"
        "[cyan]python scripts/inference.py --model_path models/mistral-document-finetuned_*[/cyan]\n\n"
        "To run with automatic context:\n"
        "[cyan]python scripts/inference.py --model_path models/mistral-document-finetuned_* --use_context[/cyan]",
        title="🎉 Success",
        border_style="green"
    ))


if __name__ == "__main__":
    main()

