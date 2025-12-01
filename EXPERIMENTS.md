# ARC Transformer Architecture Experiments

## Core Hypothesis

The Value (V) network in standard transformer attention may obstruct information flow, creating a bottleneck for learning ARC patterns. By analyzing/enhancing attention matrices directly (before applying to values), we might extract more meaningful patterns.

## Experiments Conducted

### 1. Baseline Transformer
- Standard multi-head attention with Q, K, V projections
- Standard transformer encoder architecture
- **Status**: Baseline established

### 2. U-Net Filtered Attention
- **Approach**: Compute full attention matrix `[batch, seq_len, seq_len]` from Q @ K^T, then process with U-Net to detect hierarchical patterns
- **Memory Issue**: Quadratic complexity (5400×5400 = 29M elements) + U-Net overhead causes GPU memory explosion
- **Optimizations Tried**: FP16, removed skip connections, reduced U-Net size (base_channels=2, num_downsample=2), minimal model config
- **Result**: Still too memory-intensive for production use

## Next Experiment: Upper Triangular Attention + Bottleneck

### Approach
1. **Symmetric Attention**: Compute only upper triangular attention matrix (since attention should be symmetric: `attn[i,j] = attn[j,i]`)
   - Reduces memory from `seq_len × seq_len` to `seq_len × (seq_len + 1) / 2`
   - For seq_len=5400: ~14.6M elements (50% reduction)

2. **Enhanced Vectors**: For each attention score `attn[i,j]`, create enhanced vector:
   - `[attention_score, vec_i, vec_j]` where `vec_i` and `vec_j` are the original input vectors at positions i and j
   - Shape: `[num_attention_pairs, d_model*2 + 1]`

3. **Bottleneck Layer**: Reduce enhanced vectors back to sequence length
   - Input: `[num_attention_pairs, d_model*2 + 1]`
   - Output: `[seq_len, d_model]` (one vector per sequence position)
   - Single layer transformation

### Expected Benefits
- Memory efficient: Only compute/store upper triangular matrix
- Preserves context: Original vectors appended to attention scores
- Information flow: Bottleneck forces compression of attention patterns back to sequence representation
- No V network bottleneck: Direct transformation from attention patterns to output

### Implementation Notes
- Use `torch.triu()` or similar to extract upper triangular attention
- Concatenate original input embeddings to attention scores
- Apply bottleneck MLP: `[d_model*2 + 1] → d_model` per attention pair, then aggregate to `[seq_len, d_model]`
