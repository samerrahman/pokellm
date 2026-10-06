import argparse
import json
import os
import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class DoublesSFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, torch.Tensor]] = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        raw_entries = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_entries.append(json.loads(line))

        print(f"Loaded {len(raw_entries)} entries from {data_path}. Tokenizing with action-loss masking...")

        for entry in tqdm(raw_entries, desc="Tokenizing"):
            prompt = entry["prompt"]
            output = entry["output"]

            if not prompt.endswith(" ") and not prompt.endswith("\n"):
                prompt = prompt + " "

            full_text = prompt + output + self.tokenizer.eos_token

            encoded_full = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_prompt = tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded_full["input_ids"][0]
            attention_mask = encoded_full["attention_mask"][0]
            prompt_len = encoded_prompt["input_ids"].shape[1]

            labels = input_ids.clone()
            # Mask out the prompt tokens so the model is only trained to predict the action output
            labels[:prompt_len] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(ex["input_ids"].size(0) for ex in batch)
    input_ids_list = []
    attn_mask_list = []
    labels_list = []

    for ex in batch:
        pad_len = max_len - ex["input_ids"].size(0)
        if pad_len > 0:
            input_ids = torch.cat([ex["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            attn_mask = torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            input_ids = ex["input_ids"]
            attn_mask = ex["attention_mask"]
            labels = ex["labels"]

        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "labels": torch.stack(labels_list),
    }


def train(args):
    # Select compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer & model
    print(f"Loading base model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)

    # Load dataset
    dataset = DoublesSFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Run data generation first.")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(10, total_steps // 10), num_training_steps=total_steps)

    print(f"Starting training for {args.epochs} epochs ({total_steps} total steps)...")
    model.train()

    step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving fine-tuned Doubles bot to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train a PokéLLMon bot for VGC Doubles via Supervised Fine-Tuning.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model identifier or path")
    parser.add_argument("--data_path", type=str, default="battle_data/doubles/doubles_sft.jsonl", help="Path to SFT data")
    parser.add_argument("--output_dir", type=str, default="saved_models/doubles_bot", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
