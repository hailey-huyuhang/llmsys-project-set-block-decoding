# figure 9 in the paper
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
def create_attention_mask_train(seq_len: int, block_len: int) -> BlockMask:
    half_seq_len: int = seq_len // 2

    def mask_mod(b, h, q_idx, kv_idx):    
        # Top-left quadrant (x1 -> x1, standard causal)
        # True if query and key are in the first half and key is before or at query.
        is_in_top_left_causal = (q_idx < half_seq_len) & (kv_idx < half_seq_len) & (kv_idx <= q_idx)
        # Bottom-right quadrant (xt -> xt, block attention)
        # True if query and key are in the second half and belong to the same block.
        q_block_xt = (q_idx - half_seq_len) // block_len
        kv_block_xt = (kv_idx - half_seq_len) // block_len
        is_in_bottom_right_block = (q_idx >= half_seq_len) & (kv_idx >= half_seq_len) & (q_block_xt == kv_block_xt)
        
        # Bottom-left quadrant (xt -> x1, block causal past)
        # True if query is in the second half, key is in the first half, and the queries block index is strictly greater than the keys block index.
        q_block_idx_bl = (q_idx - half_seq_len) // block_len
        kv_block_idx_bl = kv_idx // block_len
        is_in_bottom_left_block_causal_past = (q_idx >= half_seq_len) & (kv_idx < half_seq_len) & (q_block_idx_bl > kv_block_idx_bl)
        return is_in_top_left_causal | is_in_bottom_right_block | is_in_bottom_left_block_causal_past
    
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

from torch.nn.attention.flex_attention import BlockMask, create_block_mask
def create_attention_mask_inference(causal_point: int) -> BlockMask:
    # causal_point is the index of the first token in the prediction block
    def mask_mod(b, h, q_idx, kv_idx):
        is_causal = (kv_idx <= q_idx)
        is_past_causal_point = (causal_point <= q_idx)
        return is_causal | is_past_causal_point
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)