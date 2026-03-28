import torch
from torch.nn.attention.flex_attention import create_block_mask

def create_attention_mask_train(seq_len: int, block_len: int):
    half_seq_len = seq_len // 2

    def mask_mod(b, h, q_idx, kv_idx):
        # Top left quadrant for standard causal attention
        is_top_left = (q_idx < half_seq_len) & (kv_idx < half_seq_len) & (kv_idx <= q_idx)

        # Bottom right quadrant for future block bidirectional attention
        q_block_br = (q_idx - half_seq_len) // block_len
        kv_block_br = (kv_idx - half_seq_len) // block_len
        is_bottom_right = (q_idx >= half_seq_len) & (kv_idx >= half_seq_len) & (q_block_br == kv_block_br)

        # Bottom left quadrant for future block attending past blocks
        q_block_bl = (q_idx - half_seq_len) // block_len
        kv_block_bl = kv_idx // block_len
        is_bottom_left = (q_idx >= half_seq_len) & (kv_idx < half_seq_len) & (q_block_bl > kv_block_bl)

        return is_top_left | is_bottom_right | is_bottom_left

    # Keep B and H as None so they broadcast correctly during actual training
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

def create_attention_mask_inference(seq_len: int, causal_point: int):
    def mask_mod(b, h, q_idx, kv_idx):
        # Standard causal masking
        is_causal = kv_idx <= q_idx
        
        # Allow full attention for tokens past the causal point
        is_past_causal = causal_point <= q_idx
        
        return is_causal | is_past_causal

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


if __name__ == "__main__":
    SEQ_LEN = 16
    BLOCK_LEN = 4
    CAUSAL_POINT = 12 # from which query index to start enabling full bidirectional attention
    
    print("Compiling FlexAttention Masks...")
    # Verify that the FlexAttention API accepts our logic without crashing
    train_mask = create_attention_mask_train(seq_len=SEQ_LEN, block_len=BLOCK_LEN)
    inference_mask = create_attention_mask_inference(seq_len=SEQ_LEN, causal_point=CAUSAL_POINT)
    
    print("\n--- Train Mask (Figure 8) ---")
    # Simulate the q_idx and kv_idx grids to explicitly visualize the layout
    q_grid = torch.arange(SEQ_LEN).unsqueeze(1)  # Shape: (16, 1)
    kv_grid = torch.arange(SEQ_LEN).unsqueeze(0) # Shape: (1, 16)
    
    half_seq = SEQ_LEN // 2
    top_left = (q_grid < half_seq) & (kv_grid < half_seq) & (kv_grid <= q_grid)
    q_br = (q_grid - half_seq) // BLOCK_LEN
    kv_br = (kv_grid - half_seq) // BLOCK_LEN
    bottom_right = (q_grid >= half_seq) & (kv_grid >= half_seq) & (q_br == kv_br)
    q_bl = (q_grid - half_seq) // BLOCK_LEN
    kv_bl = kv_grid // BLOCK_LEN
    bottom_left = (q_grid >= half_seq) & (kv_grid < half_seq) & (q_bl > kv_bl)
    
    dense_train = top_left | bottom_right | bottom_left
    
    for row in dense_train:
        print(" ".join(["1" if val.item() else "." for val in row]))
        
    print("\n--- Inference Mask ---")
    is_causal = kv_grid <= q_grid
    is_past_causal = CAUSAL_POINT <= q_grid
    dense_infer = is_causal | is_past_causal
    
    for row in dense_infer:
        print(" ".join(["1" if val.item() else "." for val in row]))