# figure 7 in paper

from xml.parsers.expat import model
from sympy import sequence
import torch

from models.flex_masks import create_attention_mask_train


input_ids = sequence[:, :-1]

ar_label_ids = sequence[:, 1:]
parallel_label_ids = input_ids.clone()

bsz, seq_len = input_ids.shape

block_len = torch.randint(max_block_size, (1, )).item()
attention_mask = create_attention_mask_train(seq_len=seq_len, block_len=block_len)

# 1. Compute masked_input
time = torch.rand(size=(bsz, 1), device="cuda")
input_ids_mask = torch.rand(size=input_ids.shape, device=input_ids.device) > time

masked_input = torch.where(condition=input_ids_mask, input=tokenizer.mask_id, other=input_ids)

input_ids = torch.cat([input_ids, masked_input], dim=1)

# 2. Compute label_ids
parallel_label_ids[:, :-1][input_ids_mask] = -100

label_ids = torch.cat([ar_label_ids, parallel_label_ids], dim=1)

# 3. Each unique token location (in AR and parallel) should get the same positional embeddings.
positional_embeddings = get_positional_embeddings(seq_len=seq_len)
positional_embeddings = positional_embeddings.repeat(2, 1)

loss = model(input_ids, mask=attention_mask, targets=label_ids, positional_embeddings=positional_embeddings)
    
loss.backward()