# Set Block Decoding (SBD) Reproduction

Welcome to our reproduction repository for the paper: **[Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/abs/2509.04185)**.

This project aims to implement a miniTorch version of the SBD methodology. SBD is a decoding paradigm that enables parallel sampling of non-contiguous future token blocks without altering the underlying Transformer architecture. By using a hybrid attention mechanism—causal for historical tokens and bidirectional for the future block—the model can predict multiple future tokens simultaneously.

## Project Structure & Function Mapping

To keep our implementation modular and easy to debug, the repository is split into four core components: Data, Models, Training, and Inference.

### 1. Data & Masking (`data/`) - *[PyTorch]*
* **`dataset.py`**
  * **Core Functions:** `get_dataloader()`, `apply_sbd_masking()`
  * **Role:** Handles the data pipeline. It must implement the random noising logic to create the masked sequence `x_hat` where tokens are replaced with a `[MASK]` token based on a probability $\tau$. It is responsible for outputting the `input_ids` and the concatenated `label_ids` as shown in the paper's Code Block 7.

### 2. Core Architecture (`models/`) - *[Hybrid]*
* **`flex_masks.py`**
  * **Core Functions:** `create_attention_mask_train()` (Fig 9), `create_attention_mask_inference()` (Fig 10).
  * **Role:** The heart of the system. Uses PyTorch's `FlexAttention` to create the mixed causal (for past tokens) and bidirectional (for future block tokens) attention masks.
* **`modeling_sbd.py`**
  * **Core Functions:** `class SBDModelWrapper`
  * **Role:** Wraps a lightweight base LLM (e.g., Qwen2.5-0.5B or Llama-3.2-1B) to inject the custom FlexAttention masks during the forward pass.

### 3. Training & Loss (`training/`) - *[PyTorch]*
* **`loss.py`**
  * **Core Functions:** `compute_sbd_loss()`
  * **Role:** Implements Equation 14 from the paper. This function calculates and sums the standard autoregressive (NTP) loss and the masked prediction (MATP) loss.
* **`train.py`**
  * **Core Functions:** `train_step()`, `main()`
  * **Role:** The main training loop. It handles the positional embeddings duplication (`positional_embeddings.repeat(2, 1)`), passes the data through the model, and executes the backpropagation step as outlined in Algorithm 1 (Set Block Decoding training) and Code Block 7 (Fig 7).

### 4. Inference & Decoding (`inference/`) - *[Custom Systems Logic]*
* **`eb_sampler.py`**
  * **Core Functions:** `entropy_bounded_sample()`
  * **Role:** Implements the Entropy Bounded (EB) Sampler. It calculates the marginal entropy of the predicted tokens and unmasks them based on the $\gamma$ threshold (Equation 8).
* **`generate.py`**
  * **Core Functions:** `generate()` (Algorithm 2 Set Block Decoding inference), `sample_block()` (Algorithm 3 Sample block).
  * **Role:** The decoding loop. Manages the KV-cache updates for both causal and bidirectional tokens while generating text block by block.