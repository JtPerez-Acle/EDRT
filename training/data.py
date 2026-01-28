"""Data loading and preprocessing for TinyStories and other datasets."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 256
    batch_size: int = 32
    num_workers: int = 4
    pack_sequences: bool = True
    val_split_size: int = 1000  # Number of validation examples
    max_stories: Optional[int] = None  # Limit total stories (for quick testing)


def load_tokenizer(tokenizer_name: str = "gpt2") -> PreTrainedTokenizer:
    """
    Load tokenizer from HuggingFace.

    Args:
        tokenizer_name: Name of the tokenizer (default: gpt2 with 50257 vocab)

    Returns:
        Configured tokenizer with pad token set
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # GPT-2 doesn't have a pad token by default, use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class PackedSequenceDataset(Dataset):
    """
    Dataset that packs multiple stories into fixed-length sequences.

    Stories are concatenated with EOS tokens between them, then chunked
    into sequences of max_seq_len. This is more efficient than padding
    each story individually since TinyStories has short stories.
    """

    def __init__(
        self,
        stories: list[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunks = self._prepare_chunks(stories)

    def _prepare_chunks(self, stories: list[str]) -> list[list[int]]:
        """Tokenize all stories and pack into fixed-length chunks."""
        # Tokenize all stories and concatenate with EOS
        all_tokens = []
        eos_id = self.tokenizer.eos_token_id

        for story in stories:
            tokens = self.tokenizer.encode(story, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)  # Separator between stories

        # Chunk into sequences of max_seq_len + 1 (for input/target split)
        chunk_size = self.max_seq_len + 1
        chunks = []

        for i in range(0, len(all_tokens) - chunk_size + 1, self.max_seq_len):
            chunk = all_tokens[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return input_ids and labels (shifted by 1 for next-token prediction)."""
        tokens = self.chunks[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
        }


class SimpleSequenceDataset(Dataset):
    """
    Dataset that treats each story as a separate sequence.

    Stories are padded/truncated to max_seq_len. Less efficient than
    packing but simpler and preserves story boundaries.
    """

    def __init__(
        self,
        stories: list[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stories = stories

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize story and return input/labels."""
        tokens = self.tokenizer.encode(
            self.stories[idx],
            max_length=self.max_seq_len + 1,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        ).squeeze(0)

        # Create labels with -100 for padding (ignored in loss)
        labels = tokens.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokens[:-1],
            'labels': labels[1:],
        }


def create_dataloaders(
    config: DataConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for TinyStories.

    Args:
        config: Data configuration
        tokenizer: Optional pre-loaded tokenizer (loads if not provided)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(config.tokenizer_name)

    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="train")

    # Extract story texts (with optional limit for quick testing)
    if config.max_stories:
        stories = [example['text'] for example in dataset.select(range(min(config.max_stories, len(dataset))))]
        print(f"Loaded {len(stories)} stories (limited from {len(dataset)})")
    else:
        stories = [example['text'] for example in dataset]
        print(f"Loaded {len(stories)} stories")

    # Split into train and validation
    val_stories = stories[:config.val_split_size]
    train_stories = stories[config.val_split_size:]

    print(f"Train: {len(train_stories)} stories, Val: {len(val_stories)} stories")

    # Create datasets
    DatasetClass = PackedSequenceDataset if config.pack_sequences else SimpleSequenceDataset

    train_dataset = DatasetClass(
        stories=train_stories,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )

    val_dataset = DatasetClass(
        stories=val_stories,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )

    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # Create dataloaders (pin_memory only useful with GPU)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of data loading
    config = DataConfig(
        max_seq_len=128,
        batch_size=4,
        num_workers=0,
        val_split_size=100,
    )

    tokenizer = load_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    train_loader, val_loader = create_dataloaders(config, tokenizer)

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Decode a sample
    sample_ids = batch['input_ids'][0]
    decoded = tokenizer.decode(sample_ids)
    print(f"\nSample decoded (first 200 chars):\n{decoded[:200]}...")
