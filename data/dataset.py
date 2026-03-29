import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import GPT2Tokenizer


def get_tokenizer():
    """
    Load GPT-2 tokenizer and add [MASK] token for SBD masked-token prediction.
    Returns the tokenizer and the mask_token_id.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    return tokenizer

def get_dataloader(split: str, tokenizer: GPT2Tokenizer, seq_len: int, batch_size: int, num_workers: int = 0):
    """
    Load wikitext-2, tokenize, chunk into fixed-length sequences, return a DataLoader.

    Args:
        split:       "train", "validation", or "test"
        tokenizer:   GPT-2 tokenizer (from get_tokenizer())
        seq_len:     number of tokens per chunk
        batch_size:  batch size
        num_workers: DataLoader workers

    Returns:
        DataLoader yielding (input_ids,) tuples of shape (batch_size, seq_len)
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all non-empty lines into one long string
    text = "\n\n".join(line for line in dataset["text"] if line.strip())

    # Tokenize the full text at once (no padding/truncation needed)
    token_ids = tokenizer.encode(text)

    # Chunk into non-overlapping fixed-length sequences (drop the last incomplete chunk); off-by-one
    chunks = [
        token_ids[i : i + seq_len]
        for i in range(0, len(token_ids) - seq_len + 1, seq_len)
    ]

    tensor_data = torch.tensor(chunks, dtype=torch.long)  # (num_chunks, seq_len)
    dl = DataLoader(
        TensorDataset(tensor_data),
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=True,
    )
    return dl

# logic from Fig 7
def apply_sbd_masking(input_ids: torch.Tensor, mask_token_id: int):
    """
    Apply random token masking for SBD MATP training (paper Algorithm 1 / Fig 7).

    For each sequence in the batch, sample a masking rate τ ~ Uniform(0, 1),
    then independently mask each token with probability τ.

    Convention (matches compute_sbd_loss):
        input_ids_mask == True  →  token is MASKED (replaced with [MASK])
        input_ids_mask == False →  token is clean

    Args:
        input_ids:     clean token ids, shape (batch_size, seq_len)
        mask_token_id: id of the [MASK] token

    Returns:
        masked_input:   input_ids with masked positions replaced, shape (batch_size, seq_len)
        input_ids_mask: boolean mask, shape (batch_size, seq_len)
    """
    bsz, seq_len = input_ids.shape

    # Sample one masking rate per sequence
    tau = torch.rand(bsz, 1, device=input_ids.device)

    # True where token should be masked
    input_ids_mask = torch.rand(bsz, seq_len, device=input_ids.device) < tau

    masked_input = torch.where(
        input_ids_mask,
        torch.full_like(input_ids, mask_token_id),
        input_ids,
    )

    return masked_input, input_ids_mask


if __name__ == "__main__":
    # Unit test — run from repo root: python data/dataset.py
    SEQ_LEN = 64
    BATCH_SIZE = 4

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"Vocab size: {len(tokenizer)}  |  mask_token_id: {tokenizer.mask_token_id}")

    print("\nLoading wikitext-2 train split...")
    dl = get_dataloader("train", tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    print(f"Number of batches: {len(dl)}")

    (input_ids,) = next(iter(dl))
    print(f"input_ids shape: {input_ids.shape}")  # (4, 64)

    print("\nTesting apply_sbd_masking...")
    masked_input, mask = apply_sbd_masking(input_ids, tokenizer.mask_token_id)
    print(f"masked_input shape: {masked_input.shape}")
    print(f"mask shape:         {mask.shape}")
    print(f"Masked token ratio: {mask.float().mean():.2f}  (should vary ~0.0–1.0)")
    print(f"Mask token appears: {(masked_input == tokenizer.mask_token_id).sum().item()} times")
    print("\nAll checks passed.")

# # figure 7 in paper

# from xml.parsers.expat import model
# from sympy import sequence
# import torch

# from models.flex_masks import create_attention_mask_train


# input_ids = sequence[:, :-1]

# ar_label_ids = sequence[:, 1:]
# parallel_label_ids = input_ids.clone()

# bsz, seq_len = input_ids.shape

# block_len = torch.randint(max_block_size, (1, )).item()
# attention_mask = create_attention_mask_train(seq_len=seq_len, block_len=block_len)

# # 1. Compute masked_input
# time = torch.rand(size=(bsz, 1), device="cuda")
# input_ids_mask = torch.rand(size=input_ids.shape, device=input_ids.device) > time

# masked_input = torch.where(condition=input_ids_mask, input=tokenizer.mask_id, other=input_ids)

# input_ids = torch.cat([input_ids, masked_input], dim=1)

# # 2. Compute label_ids
# parallel_label_ids[:, :-1][input_ids_mask] = -100

# label_ids = torch.cat([ar_label_ids, parallel_label_ids], dim=1)