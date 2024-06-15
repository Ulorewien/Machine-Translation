import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]

def get_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset("opus_books", f"{config["language_source"]}-{config["language_target"]}", split="train")

    tokenizer_source = get_tokenizer(config, dataset_raw, config["language_source"])
    tokenizer_target = get_tokenizer(config, dataset_raw, config["language_target"])

    train_size = int(0.9 * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_size, val_size])