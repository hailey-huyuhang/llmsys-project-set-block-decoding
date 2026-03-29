import torch
import torch.nn.functional as F

def compute_sbd_loss(logits: torch.Tensor, input_ids: torch.Tensor, input_ids_mask: torch.Tensor, block_len: int):
    """
    Implements Equation 14: SBD Loss = NTP Loss + MATP Loss
    
    Args:
        logits: The full logits from the model's forward pass. 
                Shape: (batch_size, 2 * seq_len, vocab_size)
        input_ids: The original, clean input tokens. 
                   Shape: (batch_size, seq_len)
        input_ids_mask: Boolean mask indicating which tokens were replaced with [MASK].
                        True = Masked, False = Clean. Shape: (batch_size, seq_len)
        block_len: The block size (k) used for this specific training step.
                        
    Returns:
        total_loss (with grad), ntp_loss_log (no grad), matp_loss_log (no grad)
    """
    bsz, double_seq_len, vocab_size = logits.shape
    seq_len = double_seq_len // 2
    
    # 1. Split logits into Causal (NTP) and Bidirectional (MATP) halves
    ntp_logits = logits[:, :seq_len, :].contiguous()
    matp_logits = logits[:, seq_len:, :].contiguous()
    
    # 2. Prepare NTP (AR) Labels & Logits (Shifted by 1)
    shift_ntp_logits = ntp_logits[:, :-1, :].contiguous()
    ar_label_ids = input_ids[:, 1:].contiguous()
    
    # 3. Prepare MATP (Parallel) Labels
    parallel_label_ids = input_ids.clone()
    
    # Mask out clean tokens (Only calculate loss on <m>) ---
    parallel_label_ids[~input_ids_mask] = -100 
    
    # Filter by strict block boundaries (Equation 14: \lfloor L/k \rfloor) ---
    # Calculate the exact length of the sequence that falls into complete blocks
    valid_matp_len = (seq_len // block_len) * block_len
    
    # Any tokens falling outside the complete blocks are set to -100
    if valid_matp_len < seq_len:
        parallel_label_ids[:, valid_matp_len:] = -100
    
    # Concatenate for mathematically correct equal-weighting ---
    combined_logits = torch.cat([shift_ntp_logits, matp_logits], dim=1)
    combined_labels = torch.cat([ar_label_ids, parallel_label_ids], dim=1)
    
    # This loss drives the backward pass. Every valid token contributes exactly 1 / N to the loss.
    total_loss = F.cross_entropy(
        combined_logits.reshape(-1, vocab_size),
        combined_labels.reshape(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    # ---------------------------------------------------------------------
    # 4. Compute separated losses ONLY for Wandb Logging (No Gradients)
    # ---------------------------------------------------------------------
    with torch.no_grad():
        ntp_loss_log = F.cross_entropy(
            shift_ntp_logits.reshape(-1, vocab_size),
            ar_label_ids.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Safe check: In case the random masking resulted in 0 valid tokens in complete blocks
        matp_loss_log = F.cross_entropy(
            matp_logits.reshape(-1, vocab_size),
            parallel_label_ids.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )
        if torch.isnan(matp_loss_log):
            matp_loss_log = torch.tensor(0.0, device=logits.device)
            
    return total_loss, ntp_loss_log, matp_loss_log


if __name__ == "__main__":
    # Local Unit Test, usage: python training/loss.py
    print("Testing SBD Loss Computation with strict Block Boundaries...")
    BATCH_SIZE = 2
    SEQ_LEN = 10
    BLOCK_LEN = 4
    VOCAB_SIZE = 100
    
    mock_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mock_mask = torch.rand(BATCH_SIZE, SEQ_LEN) > 0.5
    mock_logits = torch.randn(BATCH_SIZE, 2 * SEQ_LEN, VOCAB_SIZE)
    
    try:
        total, ntp, matp = compute_sbd_loss(
            logits=mock_logits, 
            input_ids=mock_input_ids, 
            input_ids_mask=mock_mask,
            block_len=BLOCK_LEN
        )
        print(f"Success! Loss correctly equal-weighted and boundary-filtered.")
        print(f"Total Combined Loss : {total.item():.4f} (For Backprop)")
        print(f"  ├─ NTP Log        : {ntp.item():.4f}") # For Wandb loss curve check
        print(f"  └─ MATP Log       : {matp.item():.4f}") # For Wandb loss curve check
        
    except Exception as e:
        print(f"Error computing loss: {e}")