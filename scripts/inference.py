"""
============================================================================
Inference Script for Fine-Tuned Model
============================================================================
This script loads your fine-tuned model and allows you to ask questions
about your documents.

WHAT THIS SCRIPT DOES:
1. Loads the fine-tuned model (base model + LoRA adapters)
2. Provides an interactive chat interface
3. Optionally includes relevant document context in prompts
4. Generates answers based on trained knowledge

USAGE:
    # Interactive mode
    python scripts/inference.py --model_path models/mistral-document-finetuned_*
    
    # Single query
    python scripts/inference.py --model_path models/mistral-document-finetuned_* --query "What is..."
    
    # With context from documents
    python scripts/inference.py --model_path models/mistral-document-finetuned_* --use_context

REQUIREMENTS:
    pip install transformers torch peft bitsandbytes accelerate rich
============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler

# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADING
# ============================================================================
class DocumentQA:
    """
    A class to handle document Q&A with the fine-tuned model.
    
    This class encapsulates:
    - Model loading (with optional 4-bit quantization)
    - LoRA adapter loading
    - Text generation with configurable parameters
    - Context retrieval from documents
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = None,
        load_in_4bit: bool = True,
        device: str = None
    ):
        """
        Initialize the DocumentQA model.
        
        Args:
            model_path: Path to the fine-tuned model (LoRA adapter)
            base_model_name: Name of the base model (auto-detected if not provided)
            load_in_4bit: Use 4-bit quantization for memory efficiency
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_path = Path(model_path)
        self.load_in_4bit = load_in_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load training info to get base model name
        if base_model_name is None:
            info_path = self.model_path / "training_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    training_info = json.load(f)
                    base_model_name = training_info.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
            else:
                base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        self.base_model_name = base_model_name
        
        # Load model and tokenizer
        self._load_model()
        
        # Load document chunks for context (if available)
        self.chunks = self._load_chunks()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        console.print(f"[bold blue]Loading model from: {self.model_path}[/bold blue]")
        console.print(f"Base model: {self.base_model_name}")
        
        # Configure quantization
        bnb_config = None
        if self.load_in_4bit and self.device == "cuda":
            console.print("[green]Using 4-bit quantization[/green]")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        console.print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        console.print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Load LoRA adapters
        console.print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.model_path),
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        console.print("[green]✓ Model loaded successfully![/green]")
    
    def _load_chunks(self) -> List[Dict]:
        """Load document chunks for context retrieval."""
        # Look for chunks.json in the data directory
        from config import DATA_DIR
        chunks_path = DATA_DIR / "training" / "chunks.json"
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            console.print(f"[green]✓ Loaded {len(chunks)} document chunks for context[/green]")
            return chunks
        
        console.print("[yellow]⚠ No document chunks found for context[/yellow]")
        return []
    
    def find_relevant_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_length: int = 1000
    ) -> str:
        """
        Find relevant context from document chunks.
        
        This is a simple keyword-based search. For better results, consider
        using a vector database or semantic search.
        
        Args:
            query: The user's question
            top_k: Number of chunks to include
            max_context_length: Maximum context length in characters
            
        Returns:
            Concatenated relevant context
        """
        if not self.chunks:
            return ""
        
        # Simple keyword matching (could be improved with embeddings)
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.chunks:
            chunk_text = chunk.get("text", "").lower()
            chunk_words = set(chunk_text.split())
            
            # Score based on word overlap
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored_chunks.append((overlap, chunk))
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Get top-k chunks
        context_parts = []
        total_length = 0
        
        for score, chunk in scored_chunks[:top_k]:
            text = chunk.get("text", "")
            if total_length + len(text) <= max_context_length:
                context_parts.append(text)
                total_length += len(text)
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        context: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        use_context: bool = False,
        raw_prompt: bool = False,
    ) -> str:
        """
        Generate an answer to the question.
        
        Args:
            question: The user's question (or full prompt if raw_prompt=True)
            context: Optional context from documents
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Use sampling (False = greedy decoding)
            use_context: Automatically find relevant context
            raw_prompt: If True, use the question as-is without reformatting
            
        Returns:
            Generated answer
        """
        from config import INFERENCE_TEMPLATE
        
        # Check if the question already contains instruction markers (structured prompt)
        is_structured = any(marker in question for marker in ['### Instruction:', '### Question:', '### Answer:'])
        
        if is_structured or raw_prompt:
            # Use the prompt as-is (it's already formatted)
            prompt = question
        else:
            # Find context if requested
            if use_context and context is None:
                context = self.find_relevant_context(question)
            
            # Format prompt
            if context:
                prompt = INFERENCE_TEMPLATE.format(
                    context=context,
                    question=question
                )
            else:
                # Simple prompt without context
                prompt = f"### Question:\n{question}\n\n### Answer:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract just the answer (remove the prompt)
        # Try multiple common separators
        answer = generated_text
        
        for separator in ["### Answer:", "### Response:", "Based on the", "Answer:"]:
            if separator in generated_text:
                parts = generated_text.split(separator)
                if len(parts) > 1:
                    # Get the last part after the separator
                    answer = separator + parts[-1] if separator.startswith("Based") else parts[-1]
                    answer = answer.strip()
                    break
        else:
            # Fallback: remove the prompt from the beginning
            if len(generated_text) > len(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()
        
        # Clean up common artifacts
        answer = answer.replace("</s>", "").replace("<s>", "").strip()
        
        return answer
    
    def chat(self, use_context: bool = False):
        """
        Start an interactive chat session.
        
        Args:
            use_context: Automatically include relevant document context
        """
        console.print()
        console.print("[bold green]Document Q&A Chat[/bold green]")
        console.print("Type your questions about the documents.")
        console.print("Type 'quit' or 'exit' to end the session.")
        console.print("Type 'context on/off' to toggle automatic context.")
        console.print()
        
        while True:
            try:
                # Get user input
                question = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                # Check for exit
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                # Check for context toggle
                if question.lower() == 'context on':
                    use_context = True
                    console.print("[green]Context retrieval enabled[/green]")
                    continue
                elif question.lower() == 'context off':
                    use_context = False
                    console.print("[yellow]Context retrieval disabled[/yellow]")
                    continue
                
                # Generate answer
                console.print("[dim]Thinking...[/dim]")
                
                answer = self.generate(
                    question=question,
                    use_context=use_context
                )
                
                # Display answer
                console.print()
                console.print(Panel(
                    Markdown(answer),
                    title="[bold green]Assistant[/bold green]",
                    border_style="green"
                ))
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


# ============================================================================
# BATCH INFERENCE
# ============================================================================
def batch_inference(
    model: DocumentQA,
    questions: List[str],
    use_context: bool = False
) -> List[Dict]:
    """
    Run inference on a batch of questions.
    
    Args:
        model: The DocumentQA model
        questions: List of questions
        use_context: Include document context
        
    Returns:
        List of dictionaries with questions and answers
    """
    from tqdm import tqdm
    
    results = []
    
    for question in tqdm(questions, desc="Processing questions"):
        answer = model.generate(
            question=question,
            use_context=use_context
        )
        
        results.append({
            "question": question,
            "answer": answer
        })
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Llama 2 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive chat
    python scripts/inference.py --model_path models/mistral-document-finetuned_YYYYMMDD_HHMMSS
    
    # Single query
    python scripts/inference.py --model_path models/mistral-document-finetuned_YYYYMMDD_HHMMSS \\
        --query "What is the main topic of the documents?"
    
    # With automatic context
    python scripts/inference.py --model_path models/mistral-document-finetuned_YYYYMMDD_HHMMSS \\
        --use_context
    
    # Run on a file of questions
    python scripts/inference.py --model_path models/mistral-document-finetuned_YYYYMMDD_HHMMSS \\
        --questions_file questions.txt --output_file answers.json
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to answer (otherwise starts interactive mode)"
    )
    
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Provide specific context for the query"
    )
    
    parser.add_argument(
        "--use_context",
        action="store_true",
        help="Automatically find and include relevant document context"
    )
    
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="File with questions (one per line)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for batch results (JSON)"
    )
    
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        console.print(f"[red]Error: Model path does not exist: {model_path}[/red]")
        console.print("[yellow]Please train a model first using train.py[/yellow]")
        return
    
    # Load model
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    console.print("[bold magenta]   Document Q&A Assistant[/bold magenta]")
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    console.print()
    
    model = DocumentQA(
        model_path=str(model_path),
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit
    )
    
    # Handle different modes
    if args.questions_file:
        # Batch mode
        with open(args.questions_file) as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = batch_inference(model, questions, args.use_context)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {args.output_file}[/green]")
        else:
            for result in results:
                console.print(f"\n[cyan]Q: {result['question']}[/cyan]")
                console.print(f"[green]A: {result['answer']}[/green]")
    
    elif args.query:
        # Single query mode
        answer = model.generate(
            question=args.query,
            context=args.context,
            use_context=args.use_context,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        
        console.print()
        console.print(Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green"
        ))
    
    else:
        # Interactive mode
        model.chat(use_context=args.use_context)


if __name__ == "__main__":
    main()

