# # figure 7 in paper

# # 3. Each unique token location (in AR and parallel) should get the same positional embeddings.
# positional_embeddings = get_positional_embeddings(seq_len=seq_len)
# positional_embeddings = positional_embeddings.repeat(2, 1)

# loss = model(input_ids, mask=attention_mask, targets=label_ids, positional_embeddings=positional_embeddings)
    
# loss.backward()

import argparse
import torch
from transformers import GPT2LMHeadModel
from torch.optim import AdamW

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_tokenizer, get_dataloader, apply_sbd_masking
from training.loss import compute_sbd_loss
from models.flex_masks import build_sbd_train_mask_dense

import wandb

def train_step_baseline(model, batch, device):
    """
    One NTP training step. Returns loss.
    input_ids: (batch_size, seq_len)
    """
    input_ids = batch[0].to(device)
    outputs = model(input_ids, labels=input_ids)
    return outputs.loss


def train_step_sbd(model, batch, mask_token_id, block_len, device):
    """
    One SBD training step (equation 14, figure 7). Returns total_loss, ntp_loss, matp_loss.
    input_ids: (B, T)
    Steps:
    1. masking
        (B, T), True = masked
    2. doubled sequence
        concat, B, T -> B, 2T
    3. forward  (B, 2T, V)
    4. loss 
    return total_loss, ntp_loss, matp_loss
    """
    input_ids = batch[0].to(device)  # (B, T)
    T = input_ids.shape[1]
 
    # Mask random tokens for MATP
    masked_input, input_ids_mask = apply_sbd_masking(input_ids, mask_token_id)
 
    # Concat: [original seq | masked seq]
    # input_ids:     (B, T)  [x1, x2, x3, x4, x5, x6, x7, x8]
    # masked_input:  (B, T)  [x1, <m>, x3, <m>, <m>, x6, <m>, x8]
    # doubled_input: (B, 2T) [x1, x2, x3, x4, x5, x6, x7, x8 | x1, <m>, x3, <m>, <m>, x6, <m>, x8]
    doubled_input = torch.cat([input_ids, masked_input], dim=1)  # (B, 2T)

    # position_ids
    pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    position_ids = pos.repeat(1, 2)  # (1, 2T): [0..T-1 | 0..T-1]
 
    # Dense 4D mask for HF GPT-2
    attn_mask = build_sbd_train_mask_dense(T, block_len, device)
    
    logits = model(doubled_input, attention_mask=attn_mask, position_ids=position_ids).logits

    total_loss, ntp_loss, matp_loss = compute_sbd_loss(
        logits, input_ids, input_ids_mask, block_len
    )
    return total_loss, ntp_loss, matp_loss


def main(args):
    # 1. Setup: tokenizer, model, dataloader, optimizer
    # 2. Training loop
    # Tokenizer with [MASK] and [PAD] added
    tokenizer = get_tokenizer()
    mask_token_id = tokenizer.mask_token_id
 
    # Model: resize embeddings to cover new special tokens
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
 
    dl = get_dataloader("train", tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    wandb.init(project="sbd", config=vars(args), name=f"{args.mode}_{args.steps}steps")
 
    for step, batch in enumerate(dl):
        if step >= args.steps:
            break
 
        optimizer.zero_grad()
 
        if args.mode == "baseline":
            loss = train_step_baseline(model, batch, device)
            loss.backward()
            print(f"Step {step:4d} | loss: {loss.item():.4f}")
            wandb.log({"ntp_loss": loss.item()}, step=step)
 
        elif args.mode == "sbd":
            total_loss, ntp_loss, matp_loss = train_step_sbd(
                model, batch, mask_token_id, args.block_len, device
            )
            total_loss.backward()
            print(
                f"Step {step:4d} | total: {total_loss.item():.4f}  "
                f"ntp: {ntp_loss.item():.4f}  matp: {matp_loss.item():.4f}"
            )
            wandb.log({
                "total_loss": total_loss.item(),
                "ntp_loss": ntp_loss.item(),
                "matp_loss": matp_loss.item(),
            }, step=step)
 
        optimizer.step()
 
    print("Training finished.")

if __name__ == "__main__":
    # usage: python training/train.py --mode sbd --steps 50
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "sbd"], default="baseline")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--block_len", type=int, default=4)
    args = parser.parse_args()
    main(args)
