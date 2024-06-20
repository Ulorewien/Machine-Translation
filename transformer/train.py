import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

from dataset import OPUSBooksDataset, causal_mask
from model import build_transformer
from config import get_config
from util import get_weights_file_path, plot_loss, plot_metric

import warnings
warnings.filterwarnings("ignore")

def greedy_decode(model, encoder, encoder_mask, tokenizer_source, tokenizer_target, max_len, device):
    start_id = tokenizer_target.token_to_id("[SOS]")
    end_id = tokenizer_target.token_to_id("[EOS]")

    encoder_output = model.encode(encoder, encoder_mask)
    decoder_input = torch.empty(1, 1).fill_(start_id).type_as(encoder).to(device)

    while(True):
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        probs = model.project(out[:,-1])
        _, next_word = torch.max(probs, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder).fill_(next_word.item()).to(device)], dim=1)

        if next_word == end_id:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, val_dataset, tokenizer_source, tokenizer_target, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    ct = 0
    console_width = 80

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in val_dataset:
            ct += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-"*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if ct == num_examples:
                break

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()

    return cer, wer, bleu

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
    # print("a")
    dataset_raw = load_dataset("Helsinki-NLP/opus_books", f"{config["language_source"]}-{config["language_target"]}", split="train")
    # print("b")

    tokenizer_source = get_tokenizer(config, dataset_raw, config["language_source"])
    tokenizer_target = get_tokenizer(config, dataset_raw, config["language_target"])
    # print("c")

    train_size = int(0.9 * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_size, val_size])
    # print("d")

    train_dataset = OPUSBooksDataset(train_dataset_raw, tokenizer_source, tokenizer_target, config["language_source"], config["language_target"], config["seq_len"])
    val_dataset = OPUSBooksDataset(val_dataset_raw, tokenizer_source, tokenizer_target, config["language_source"], config["language_target"], config["seq_len"])
    # print("e")
    
    max_len_source = 0
    max_len_target = 0

    # print(len(dataset_raw))

    for i, item in enumerate(dataset_raw):
        # print(i)
        source_ids = tokenizer_source.encode(item["translation"][config["language_source"]]).ids
        target_ids = tokenizer_target.encode(item["translation"][config["language_target"]]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f"Maximum length of source sentence: {max_len_source}")
    print(f"Maximum length of target sentence: {max_len_target}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_source, tokenizer_target

def get_model(config, source_vocab_size, target_vocab_size):
    model = build_transformer(source_vocab_size, target_vocab_size, config["seq_len"], config["seq_len"], config["d_model"], config["n_layer"], config["n_heads"], config["dropout"], config["d_ff"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer_source, tokenizer_target = get_dataset(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    losses = []
    char_error_rates = []
    word_error_rates = []
    bleu_scores = []

    for epoch in range(initial_epoch, config["n_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")
        epoch_loss = []
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_function(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            
            epoch_loss.append(loss.item())
        
        losses.append(sum(epoch_loss)/len(epoch_loss))
        
        cer, wer, bleu = run_validation(model, val_loader, tokenizer_source, tokenizer_target, config["seq_len"], device, lambda x: batch_iterator.write(x), global_step, writer)
        
        char_error_rates.append(cer)
        word_error_rates.append(wer)
        bleu_scores.append(bleu)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epcoh": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

    plot_loss(losses, "Transformer Training Loss", "transformer_loss.png")
    plot_metric(char_error_rates, "Character Error Rate (CER)", "transformer_cer.png")
    plot_metric(word_error_rates, "Word Error Rate (WER)", "transformer_wer.png")
    plot_metric(bleu_scores, "BLEU Score", "transformer_bleu.png")

if __name__ == "__main__":
    config = get_config()
    train_model(config)
