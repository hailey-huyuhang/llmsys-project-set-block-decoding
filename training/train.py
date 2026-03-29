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


def train_step_baseline(model, batch, device):
    """
    One NTP training step. Returns loss.
    input_ids: (batch_size, seq_len)
    """
    pass


def train_step_sbd(model, batch, mask_token_id, block_len, device):
    """
    One SBD training step. Returns total_loss, ntp_loss, matp_loss.
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
    pass


def main(args):
    # 1. Setup: tokenizer, model, dataloader, optimizer
    # 2. Training loop
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "sbd"], default="baseline")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--block_len", type=int, default=4)
    args = parser.parse_args()
    main(args)
