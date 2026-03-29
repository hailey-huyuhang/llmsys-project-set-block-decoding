# # figure 7 in paper

# # 3. Each unique token location (in AR and parallel) should get the same positional embeddings.
# positional_embeddings = get_positional_embeddings(seq_len=seq_len)
# positional_embeddings = positional_embeddings.repeat(2, 1)

# loss = model(input_ids, mask=attention_mask, targets=label_ids, positional_embeddings=positional_embeddings)
    
# loss.backward()