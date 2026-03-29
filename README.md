# Set Block Decoding (SBD) Reproduction

Welcome to our reproduction repository for the paper: **[Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/abs/2509.04185)**.

This project aims to implement a miniTorch version of the SBD methodology. SBD is a decoding paradigm that enables parallel sampling of non-contiguous future token blocks without altering the underlying Transformer architecture. By using a hybrid attention mechanism—causal for historical tokens and bidirectional for the future block—the model can predict multiple future tokens simultaneously.

## Project Structure & Function Mapping

To keep our implementation modular and easy to debug, the repository is split into four core components: Data, Models, Training, and Inference.

### 1. Data & Masking (`data/`) - *[PyTorch]*
* **`dataset.py`**
  * **Core Functions:** `get_dataloader()`, `apply_sbd_masking()`
  * Handles the data pipeline. Implement the random noising logic to create the masked sequence `x_hat` where tokens are replaced with a `[MASK]` token based on a probability $\tau$. It is responsible for outputting the `input_ids` and the concatenated `label_ids` as shown in the paper's Code Block 7.

### 2. Core Architecture (`models/`) - *[Hybrid]*
* **`flex_masks.py`**
  * **Core Functions:** `create_attention_mask_train()` (Fig 9), `create_attention_mask_inference()` (Fig 10).
  * Uses PyTorch's `FlexAttention` to create the mixed causal (for past tokens) and bidirectional (for future block tokens) attention masks.
* **`modeling_sbd.py`**
  * **Core Functions:** `class SBDModelWrapper`
  * Wraps a lightweight base LLM (e.g., Qwen2.5-0.5B or Llama-3.2-1B) to inject the custom FlexAttention masks during the forward pass.

### 3. Training & Loss (`training/`) - *[PyTorch]*
* **`loss.py`**
  * **Core Functions:** `compute_sbd_loss()`
  * Implements Equation 14 from the paper. This function calculates and sums the standard autoregressive (NTP) loss and the masked prediction (MATP) loss.
* **`train.py`**
  * **Core Functions:** `train_step()`, `main()`
  * The main training loop. It handles the positional embeddings duplication (`positional_embeddings.repeat(2, 1)`), passes the data through the model, and executes the backpropagation step as outlined in Algorithm 1 (Set Block Decoding training) and Code Block 7 (Fig 7).

### 4. Inference & Decoding (`inference/`) - *[miniTorch]*
* **`eb_sampler.py`**
  * **Core Functions:** `entropy_bounded_sample()`
  * Implements the Entropy Bounded (EB) Sampler. It calculates the marginal entropy of the predicted tokens and unmasks them based on the $\gamma$ threshold (Equation 8).
* **`generate.py`**
  * **Core Functions:** `generate()` (Algorithm 2 Set Block Decoding inference), `sample_block()` (Algorithm 3 Sample block).
  * The decoding loop. Manages the KV-cache updates for both causal and bidirectional tokens while generating text block by block.


## MIDTERM REPORT TODO

> Goal: baseline training + SBD training integrated + real loss curves + report
> Total estimated: ~18-20h (buffer included)

### Step 1 — Autoregressive Baseline (~4h)
- [x] `dataset.py`: load wikitext-2, tokenize with GPT-2 tokenizer (+ add `[MASK]` token), chunk into fixed-length sequences, return dataloader (~1.5h)
- [ ] `train.py`: NTP training loop using `GPT2LMHeadModel` directly (no wrapper), `--mode baseline` flag, prints step/loss (~1.5h)
- [ ] **Signal**: run 50-100 steps, confirm loss is not NaN and is decreasing (~1h, includes debug)

### Step 2A — SBD Loss Path (~4h, must complete)
- [ ] `dataset.py`: add `apply_sbd_masking()` — random mask tokens by probability τ, output `input_ids_mask` (~1h)
- [ ] `train.py`: doubled sequence input, switch to `compute_sbd_loss()` in `--mode sbd`, log total/NTP/MATP loss separately (~2h)
- [ ] **Signal**: run 50 steps in SBD mode, confirm all three losses are valid and not NaN (~1h, includes debug)

### Step 2B — Hybrid Attention Mask (~3h, optional)
- [ ] `modeling_sbd.py`: wrap GPT-2, inject `create_attention_mask_train()` into forward pass
- [ ] **Stop if shape mismatch or forward crash** — do not over-invest here

### Step 3 — Midterm Report (~5-6h)
- [ ] Pipeline diagram: baseline vs SBD training flow (~1h)
- [ ] Results: baseline vs SBD loss curves (~0.5h)
- [ ] Method: NTP vs MATP, doubled sequence, hybrid attention (~1h)
- [ ] Limitations: hybrid mask status, no inference yet, no MiniTorch yet (~0.5h)
- [ ] **MiniTorch explanation**: explicitly state that PyTorch-first approach validates algorithmic correctness; MiniTorch port targets inference components (EB Sampler, KV-cache) in Week 6–7, consistent with proposal timeline (~0.5h)
- [ ] Next steps: EB sampler, KV-cache, MiniTorch port in Week 6–7 (~0.5h)
- [ ] Polish + submit (~1h)

### Known Risks
- **Doubled sequence length mismatch**: `input_ids` fed to model must be `2 * seq_len` in SBD mode — easy to get shape wrong
- **`input_ids_mask` semantics**: be explicit — `True` = token is masked (replaced with `[MASK]`), `False` = token is clean
- **Loss label alignment**: `compute_sbd_loss()` handles its own label construction — do not also construct labels in `dataset.py` or `train.py`

### After Midterm
- [ ] `eb_sampler.py`: Entropy Bounded Sampler
- [ ] `generate.py`: SBD decoding loop with KV-cache
- [ ] MiniTorch port (inference components)