"""
============================================================================
Dataset Creation Script for Model Fine-Tuning
============================================================================
This script processes extracted text and creates a training dataset in the
instruction-following format that Llama 2 expects.

WHAT THIS SCRIPT DOES:
1. Loads extracted text files
2. Splits text into chunks (256-512 tokens with overlap)
3. Creates Q&A pairs from chunks (synthetic data generation)
4. Formats data for instruction fine-tuning
5. Saves dataset in Hugging Face format

USAGE:
    python scripts/create_dataset.py
    
    Or with options:
    python scripts/create_dataset.py --chunk_size 384 --overlap 64

REQUIREMENTS:
    pip install transformers datasets tiktoken nltk tqdm rich
============================================================================
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXT CHUNKING
# ============================================================================
class TextChunker:
    """
    Splits text into overlapping chunks suitable for training.
    
    WHY CHUNK TEXT?
    Large documents need to be split into smaller pieces because:
    1. Models have limited context windows
    2. Smaller chunks = more training examples
    3. Overlap ensures no information is lost at chunk boundaries
    """
    
    def __init__(
        self,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        respect_sentences: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            min_chunk_size: Minimum chunk size (smaller chunks are discarded)
            respect_sentences: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        
        # Approximate: 1 token ≈ 4 characters (works well for English)
        self.chars_per_token = 4
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return len(text) // self.chars_per_token
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            source: Source document name (for metadata)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # Target chunk size in characters
        target_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.chunk_overlap * self.chars_per_token
        min_chars = self.min_chunk_size * self.chars_per_token
        
        # Split into paragraphs first for cleaner chunks
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed target, save current chunk
            if current_chunk and len(current_chunk) + len(para) > target_chars:
                if len(current_chunk) >= min_chars:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source,
                        "chunk_index": chunk_index,
                        "token_count": self.count_tokens(current_chunk)
                    })
                    chunk_index += 1
                
                # Keep some overlap from the end of current chunk
                if len(current_chunk) > overlap_chars:
                    # Find a sentence boundary for overlap
                    overlap_text = current_chunk[-overlap_chars:]
                    sentence_end = overlap_text.find('. ')
                    if sentence_end > 0:
                        current_chunk = overlap_text[sentence_end + 2:]
                    else:
                        current_chunk = overlap_text
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            
            # If single paragraph is too long, split it by sentences
            while len(current_chunk) > target_chars * 1.5:
                # Find a good split point (sentence boundary)
                split_point = target_chars
                search_area = current_chunk[split_point - 200:split_point + 200] if split_point > 200 else current_chunk[:split_point + 200]
                
                # Look for sentence ending
                for ending in ['. ', '! ', '? ', '.\n']:
                    idx = search_area.rfind(ending)
                    if idx > 0:
                        actual_split = (split_point - 200 + idx + len(ending)) if split_point > 200 else (idx + len(ending))
                        if actual_split > min_chars:
                            split_point = actual_split
                            break
                
                # Save the chunk
                chunk_text = current_chunk[:split_point].strip()
                if len(chunk_text) >= min_chars:
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "chunk_index": chunk_index,
                        "token_count": self.count_tokens(chunk_text)
                    })
                    chunk_index += 1
                
                # Keep remainder with overlap
                current_chunk = current_chunk[split_point - overlap_chars:].strip()
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= min_chars:
            chunks.append({
                "text": current_chunk.strip(),
                "source": source,
                "chunk_index": chunk_index,
                "token_count": self.count_tokens(current_chunk)
            })
        
        return chunks


# ============================================================================
# Q&A PAIR GENERATION
# ============================================================================
@dataclass
class QAPair:
    """A question-answer pair for training."""
    question: str
    answer: str
    context: str
    source: str


def generate_qa_pairs_from_chunk(chunk: Dict, num_pairs: int = 3) -> List[QAPair]:
    """
    Generate Q&A pairs from a text chunk.
    
    This creates synthetic training data by generating questions and answers
    from the chunk text. For better results, you can manually create Q&A pairs
    or use a larger model to generate them.
    
    Args:
        chunk: Chunk dictionary with text and metadata
        num_pairs: Number of Q&A pairs to generate per chunk
        
    Returns:
        List of QAPair objects
    """
    text = chunk["text"]
    source = chunk.get("source", "unknown")
    
    pairs = []
    
    # Split into sentences for Q&A generation
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 20]
    
    if not sentences:
        return pairs
    
    # Question templates
    question_templates = [
        # Factual questions
        "What does this document say about {topic}?",
        "Can you explain {topic} based on the document?",
        "What information is provided about {topic}?",
        "Summarize what the document says about {topic}.",
        "According to the document, what is {topic}?",
        # General questions
        "What are the key points in this section?",
        "Can you summarize this information?",
        "What is the main idea of this passage?",
    ]
    
    # Generate pairs based on available sentences
    for i in range(min(num_pairs, len(sentences))):
        sentence = sentences[i % len(sentences)]
        
        # Extract potential topics (nouns/noun phrases)
        # Simple extraction: get capitalized words or phrases
        words = sentence.split()
        topics = [w.strip('.,!?()[]') for w in words if len(w) > 4]
        
        if topics:
            topic = random.choice(topics)
            template = random.choice(question_templates[:5])  # Use factual templates
            question = template.format(topic=topic)
        else:
            question = random.choice(question_templates[5:])  # Use general templates
        
        # The answer is based on the chunk content
        # In production, you'd want more sophisticated answer generation
        answer_sentences = sentences[:min(3, len(sentences))]
        answer = ". ".join(answer_sentences) + "."
        
        pairs.append(QAPair(
            question=question,
            answer=answer,
            context=text,
            source=source
        ))
    
    return pairs


def create_completion_examples(chunks: List[Dict]) -> List[Dict]:
    """
    Create completion-style training examples.
    
    Instead of Q&A, these examples train the model to continue/complete text
    based on the document content.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of completion training examples
    """
    examples = []
    
    for chunk in chunks:
        text = chunk["text"]
        
        # Create a completion example
        # Format: Given the start of the text, complete it
        if len(text) > 100:
            split_point = len(text) // 3
            prompt_part = text[:split_point]
            completion_part = text[split_point:]
            
            examples.append({
                "instruction": "Continue the following document text:",
                "input": prompt_part.strip(),
                "output": completion_part.strip(),
                "source": chunk.get("source", ""),
                "type": "completion"
            })
    
    return examples


# ============================================================================
# INSTRUCTION FORMAT CREATION
# ============================================================================
def format_for_instruction_tuning(
    qa_pairs: List[QAPair],
    instruction_template: str = None
) -> List[Dict]:
    """
    Format Q&A pairs for instruction fine-tuning.
    
    Creates training examples in the format expected by Llama 2:
    ### Instruction: ...
    ### Context: ...
    ### Question: ...
    ### Response: ...
    
    Args:
        qa_pairs: List of QAPair objects
        instruction_template: Optional custom template
        
    Returns:
        List of formatted training examples
    """
    from config import INSTRUCTION_TEMPLATE
    
    if instruction_template is None:
        instruction_template = INSTRUCTION_TEMPLATE
    
    examples = []
    
    for pair in qa_pairs:
        # Format the full training text
        formatted_text = instruction_template.format(
            context=pair.context,
            question=pair.question,
            answer=pair.answer
        )
        
        examples.append({
            "text": formatted_text,
            "question": pair.question,
            "answer": pair.answer,
            "context": pair.context,
            "source": pair.source,
            "type": "qa"
        })
    
    return examples


# ============================================================================
# DATASET CREATION
# ============================================================================
def create_dataset(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    qa_pairs_per_chunk: int = 2,
    include_completions: bool = True,
    train_split: float = 0.9
) -> Dict:
    """
    Create a complete training dataset from extracted text files.
    
    Args:
        input_dir: Directory containing extracted text files
        output_dir: Directory to save the dataset
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        qa_pairs_per_chunk: Number of Q&A pairs per chunk
        include_completions: Include completion-style examples
        train_split: Fraction of data for training (rest is validation)
        
    Returns:
        Statistics dictionary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    text_files = list(input_dir.glob("*.txt"))
    # Exclude metadata files
    text_files = [f for f in text_files if not f.name.endswith("_metadata.txt")]
    
    if not text_files:
        logger.error(f"No text files found in {input_dir}")
        logger.info("Please run extract_text.py first to extract text from documents.")
        return {}
    
    # =====================================================
    # DEBUG: Show all documents that will be processed
    # =====================================================
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                    📚 DOCUMENT PROCESSING DEBUG                    [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"[bold green]✓ Found {len(text_files)} documents to process:[/bold green]\n")
    
    for i, tf in enumerate(text_files, 1):
        file_size = tf.stat().st_size
        size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
        console.print(f"  [cyan]{i}.[/cyan] {tf.name} [dim]({size_str})[/dim]")
    
    console.print()
    
    # Initialize chunker
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    all_qa_pairs = []
    file_stats = []  # Track stats per file
    
    # =====================================================
    # Process each file with detailed logging
    # =====================================================
    console.print("[bold yellow]Processing documents...[/bold yellow]\n")
    
    for text_file in tqdm(text_files, desc="Processing files"):
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk the text
        chunks = chunker.chunk_text(text, source=text_file.stem)
        num_chunks = len(chunks)
        all_chunks.extend(chunks)
        
        # Generate Q&A pairs
        file_qa_count = 0
        for chunk in chunks:
            qa_pairs = generate_qa_pairs_from_chunk(chunk, num_pairs=qa_pairs_per_chunk)
            file_qa_count += len(qa_pairs)
            all_qa_pairs.extend(qa_pairs)
        
        # Track stats for this file
        file_stats.append({
            "file": text_file.name,
            "chars": len(text),
            "chunks": num_chunks,
            "qa_pairs": file_qa_count
        })
        
        # Debug output for each file
        console.print(f"  [green]✓[/green] {text_file.name}: {len(text):,} chars → {num_chunks} chunks → {file_qa_count} Q&A pairs")
    
    # =====================================================
    # Summary table
    # =====================================================
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                    📊 PROCESSING SUMMARY                         [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
    
    doc_table = Table(title="Documents Processed", show_header=True, header_style="bold magenta")
    doc_table.add_column("Document", style="cyan")
    doc_table.add_column("Characters", justify="right", style="green")
    doc_table.add_column("Chunks", justify="right", style="yellow")
    doc_table.add_column("Q&A Pairs", justify="right", style="blue")
    
    for fs in file_stats:
        doc_table.add_row(
            fs["file"][:40] + "..." if len(fs["file"]) > 40 else fs["file"],
            f"{fs['chars']:,}",
            str(fs["chunks"]),
            str(fs["qa_pairs"])
        )
    
    # Add totals row
    doc_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{sum(fs['chars'] for fs in file_stats):,}[/bold]",
        f"[bold]{len(all_chunks)}[/bold]",
        f"[bold]{len(all_qa_pairs)}[/bold]"
    )
    
    console.print(doc_table)
    console.print()
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(text_files)} documents")
    logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs")
    
    # Format for instruction tuning
    instruction_examples = format_for_instruction_tuning(all_qa_pairs)
    
    # Add completion examples
    if include_completions:
        completion_examples = create_completion_examples(all_chunks)
        logger.info(f"Created {len(completion_examples)} completion examples")
        instruction_examples.extend(completion_examples)
    
    # Shuffle data
    random.shuffle(instruction_examples)
    
    # Split into train/validation
    split_idx = int(len(instruction_examples) * train_split)
    train_data = instruction_examples[:split_idx]
    val_data = instruction_examples[split_idx:]
    
    # Save datasets
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(train_data)} training examples to {train_path}")
    logger.info(f"Saved {len(val_data)} validation examples to {val_path}")
    
    # Save chunks for reference (useful for inference)
    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Create Hugging Face dataset format
    try:
        from datasets import Dataset, DatasetDict
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # Save in Hugging Face format
        hf_path = output_dir / "hf_dataset"
        dataset_dict.save_to_disk(str(hf_path))
        logger.info(f"Saved Hugging Face dataset to {hf_path}")
        
    except ImportError:
        logger.warning("datasets library not available, skipping HF format")
    
    # Statistics
    stats = {
        "num_files": len(text_files),
        "num_chunks": len(all_chunks),
        "num_qa_pairs": len(all_qa_pairs),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "avg_chunk_tokens": sum(c["token_count"] for c in all_chunks) / len(all_chunks) if all_chunks else 0
    }
    
    # Print statistics
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.1f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)
    
    return stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create training dataset from extracted text",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    from config import DATA_DIR, document_config
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DATA_DIR / "extracted",
        help="Directory containing extracted text files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "training",
        help="Directory to save the dataset"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=document_config.chunk_size,
        help="Chunk size in tokens"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=document_config.chunk_overlap,
        help="Overlap between chunks in tokens"
    )
    
    parser.add_argument(
        "--qa_pairs",
        type=int,
        default=2,
        help="Number of Q&A pairs per chunk"
    )
    
    parser.add_argument(
        "--no_completions",
        action="store_true",
        help="Don't include completion-style examples"
    )
    
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data for training"
    )
    
    args = parser.parse_args()
    
    console.print("[bold blue]Dataset Creation for Model Fine-Tuning[/bold blue]")
    console.print(f"Input directory: {args.input_dir}")
    console.print(f"Output directory: {args.output_dir}")
    console.print(f"Chunk size: {args.chunk_size} tokens")
    console.print(f"Overlap: {args.overlap} tokens")
    console.print()
    
    # Check input directory
    if not args.input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {args.input_dir}[/red]")
        console.print("[yellow]Please run extract_text.py first.[/yellow]")
        return
    
    # Create dataset
    create_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        qa_pairs_per_chunk=args.qa_pairs,
        include_completions=not args.no_completions,
        train_split=args.train_split
    )


if __name__ == "__main__":
    main()

