import torch
import torch.nn as nn
from torch.utils.data import Dataset

class OPUSBooksDataset(Dataset):
    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_language, target_language, seq_len):
        super().__init__()
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.seq_len = seq_len

        self.start_token = torch.Tensor([source_tokenizer.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.end_token = torch.Tensor([source_tokenizer.token_to_id(["[EOS]"])], dtype=torch.int64)
        self.pad_token = torch.Tensor([source_tokenizer.token_to_id(["[PAD]"])], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        source_target_pair = self.dataset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]

        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        encoder_num_padding = self.seq_len - len(encoder_input_tokens)
        decoder_num_padding = self.seq_len - len(decoder_input_tokens)

        if encoder_num_padding < 0 or decoder_num_padding < 0:
            raise ValueError("Sentence is too long.")
        
        encoder_input = torch.cat([
            self.start_token,
            torch.tensor(encoder_input_tokens, dtype=torch.int64),
            self.end_token,
            torch.tensor([self.pad_token] * encoder_num_padding, dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.start_token,
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * decoder_num_padding, dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            self.end_token,
            torch.tensor([self.pad_token] * decoder_num_padding, dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input, 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
