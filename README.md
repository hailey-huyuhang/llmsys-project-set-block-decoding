# Set Block Decoding (SBD) Reproduction

Welcome to our reproduction repository for the paper: **[Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/abs/2509.04185)**.

This project aims to implement a miniTorch version of the SBD methodology. SBD is a decoding paradigm that enables parallel sampling of non-contiguous future token blocks without altering the underlying Transformer architecture. By using a hybrid attention mechanism—causal for historical tokens and bidirectional for the future block—the model can predict multiple future tokens simultaneously.

The project is split into two stages. Training runs in PyTorch on top of GPT-2, and inference runs in MiniTorch, the custom framework built throughout the course. The final benchmark compares standard NTP decoding against SBD decoding, both running inside MiniTorch, measuring NFE reduction and wall-clock speedup.

## Project Structure
To keep our implementation modular and easy to debug, the repository is split into four core components: Data, Models, Training, and Inference.
 
```
.
├── data/
│   └── dataset.py
├── models/
│   └── flex_masks.py
├── training/
│   ├── loss.py
│   └── train.py
├── inference/
│   ├── sbd_mask.py
│   ├── eb_sampler.py
│   └── generate.py
├── minitorch/
└── benchmark_sbd.py
```

## Component Details

### 1. Data & Masking (`data/`) - *[PyTorch]*
* **`dataset.py`**
  * **Core Functions:** `get_dataloader()`, `apply_sbd_masking()`
  * Handles the data pipeline. Implement the random noising logic to create the masked sequence `x_hat` where tokens are replaced with a `[MASK]` token based on a probability $\tau$. It is responsible for outputting the `input_ids` and the concatenated `label_ids` as shown in the paper's Code Block 7.

### 2. Core Architecture (`models/`) - *[PyTorch]*
* **`flex_masks.py`**
  * **Core Functions:** `build_sbd_train_mask_dense()`, `create_attention_mask_train()` (Fig 9), `create_attention_mask_inference()` (Fig 10).
  * Uses PyTorch's `FlexAttention` to create the mixed causal (for past tokens) and bidirectional (for future block tokens) attention masks.
  * Builds attention masks for SBD training and inference. Dense 4D mask is passed directly to HF GPT-2 via attention_mask parameter.
<!-- * **`modeling_sbd.py`**
  * **Core Functions:** `class SBDModelWrapper`
  * Wraps a lightweight base LLM (e.g., Qwen2.5-0.5B or Llama-3.2-1B) to inject the custom FlexAttention masks during the forward pass. -->

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
* **`sbd_mask.py`**
  * **Core Functions:** `build_sbd_inference_mask()`
  * Builds the 4D attention mask used during block decoding at inference time. Tokens up to `causal_point` attend causally. Tokens from `causal_point` onward form the current prediction block and attend to all positions, following Figure 10 of the paper. Output is a MiniTorch tensor compatible with the modified `DecoderLM.forward()`.

### 5. MiniTorch Model (`minitorch/`) - *[miniTorch]*
* **`modules_transfomer.py`**
  * **Modified Functions:** `MultiHeadAttention.self_attention()`, `MultiHeadAttention.forward()`, `TransformerLayer.forward()`, `DecoderLM.forward()`
  * Adds an optional `mask` parameter to the four functions above so that an externally constructed SBD attention mask can be passed in at inference time. When `mask` is `None` the behaviour is identical to the original implementation.

### 6. Benchmark (`benchmark_sbd.py`) - *[miniTorch]*
* **Core Functions:** `run_benchmark()`
* Runs `generate_ntp()` and `generate_sbd()` on the same MiniTorch model and the same prompt set, then reports NFE per generated token, NFE speedup, wall-clock time per token, and wall-clock speedup. Results are collected across $\gamma$ values of 0.1, 0.35, and 0.6 to reproduce the speed-accuracy tradeoff shown in Figure 1 of the paper.


## TODO

> Goal: baseline training + SBD training integrated + real loss curves + report
> Total estimated: ~18-20h (buffer included)

### Step 1 — Autoregressive Baseline (~4h)
- [x] `dataset.py`: load wikitext-2, tokenize with GPT-2 tokenizer (+ add `[MASK]` token), chunk into fixed-length sequences, return dataloader (~1.5h)
- [x] `train.py`: NTP training loop using `GPT2LMHeadModel` directly (no wrapper), `--mode baseline` flag, prints step/loss (~1.5h)
- [x] **Signal**: run 50-100 steps, confirm loss is not NaN and is decreasing (~1h, includes debug)

### Step 2A — SBD Loss Path (~4h, must complete)
- [x] `dataset.py`: add `apply_sbd_masking()` — random mask tokens by probability τ, output `input_ids_mask` (~1h)
- [x] `train.py`: doubled sequence input, switch to `compute_sbd_loss()` in `--mode sbd`, log total/NTP/MATP loss separately (~2h)
- [x] **Signal**: run 500 steps in SBD mode, confirm all three losses are valid and not NaN (~1h, includes debug)

### Step 2B — Hybrid Attention Mask (~3h, optional)
<!-- - [ ] `modeling_sbd.py`: wrap GPT-2, inject `create_attention_mask_train()` into forward pass -->
- [x] `flex_masks.py`: added `build_sbd_train_mask_dense()`, dense 4D mask for HF GPT-2, passed via attention_mask in train_step_sbd()
- [x] **Signal**: run 500 steps in SBD mode, confirm all three losses are valid and not NaN (~1h)
<!-- - [ ] **Stop if shape mismatch or forward crash** — do not over-invest here -->

<!-- ### Step 3 — Midterm Report (~5-6h)
- [ ] Pipeline diagram: baseline vs SBD training flow (~1h)
- [ ] Results: baseline vs SBD loss curves (~0.5h)
- [ ] Method: NTP vs MATP, doubled sequence, hybrid attention (~1h)
- [ ] Limitations: hybrid mask status, no inference yet, no MiniTorch yet (~0.5h)
- [ ] **MiniTorch explanation**: explicitly state that PyTorch-first approach validates algorithmic correctness; MiniTorch port targets inference components (EB Sampler, KV-cache) in Week 6–7, consistent with proposal timeline (~0.5h)
- [ ] Next steps: EB sampler, KV-cache, MiniTorch port in Week 6–7 (~0.5h)
- [ ] Polish + submit (~1h) -->

### Risks
- **Doubled sequence length mismatch**: `input_ids` fed to model must be `2 * seq_len` in SBD mode — easy to get shape wrong
- **`input_ids_mask` semantics**: be explicit — `True` = token is masked (replaced with `[MASK]`), `False` = token is clean
- **Loss label alignment**: `compute_sbd_loss()` handles its own label construction — do not also construct labels in `dataset.py` or `train.py`

<!-- ### After Midterm
- [ ] `eb_sampler.py`: Entropy Bounded Sampler
- [ ] `generate.py`: SBD decoding loop with KV-cache
- [ ] MiniTorch port (inference components) -->

### Step 3 — Training Fixes and Longer Run (~3h)
- [x] Fix position embeddings by setting `position_ids = torch.arange(T).repeat(1, 2)` in `train_step_sbd()` so both halves of the doubled input share the same indices 0 to T-1 (~15min)
- [ ] Re-run SBD training for 2000+ steps with the position fix applied and compare MATP convergence against the old run (~2h including GPU wait)
- [ ] **Signal**: MATP loss drops below 5.0 (currently 5.61 at 500 steps)

### Step 4 — MiniTorch Model Modification (~2h)
- [ ] Add an optional `mask` parameter to `MultiHeadAttention.self_attention`, `MultiHeadAttention.forward`, `TransformerLayer.forward`, and `DecoderLM.forward` in `modules_transfomer.py` so an external SBD attention mask can be injected at inference time (~1h)
- [ ] Verify that calling these functions without a mask argument produces bit-identical outputs to the original implementation (~1h)
- [ ] **Signal**: the existing machine translation training run completes with no change in loss

### Step 5 — MiniTorch Inference Components (~5h)
- [ ] `inference/sbd_mask.py`: implement `build_sbd_inference_mask` — positions before `causal_point` attend causally, positions in the prediction block attend to all positions, output as a 4D MiniTorch tensor (~1h)
- [ ] `inference/eb_sampler.py`: implement `entropy_bounded_sample` — compute per-position entropy for all masked positions, sort ascending, unmask the largest group whose cumulative entropy stays within γ, guarantee at least one token revealed per call (~2h)
- [ ] `inference/generate.py`: implement `generate_ntp` for one-token-at-a-time decoding, `generate_sbd` for the outer block-level loop, and `sample_block` for the inner unmasking loop, tracking forward pass count per block for NFE reporting (~2h)
- [ ] **Signal**: `generate_ntp` produces readable text and average NFE per block in `generate_sbd` is less than `block_size`

### Step 6 — Benchmark (~3h)
- [ ] `benchmark_sbd.py`: implement `run_benchmark` — run `generate_ntp` and `generate_sbd` on the same model and prompt set, collect NFE speedup and wall-clock speedup at γ = 0.1, 0.35, and 0.6 (~2h)
- [ ] Compare collected results against the Roofline analysis in Table 3 and Table 4 of the paper (~1h)
- [ ] **Signal**: NFE reduction of 2x or more at γ = 0.35 with no significant drop in output quality

### Risks
- **Regression in `modules_transfomer.py`**: confirm the modified file is numerically identical to the original when no mask is passed before moving on to inference work
- **Missing MiniTorch ops**: `entropy_bounded_sample` may rely on operations not available in MiniTorch; use numpy for entropy computation as a fallback if needed
- **NFE vs wall-clock gap**: without KV-cache, wall-clock speedup will be smaller than NFE speedup; report both metrics and compare against the paper's Roofline bounds