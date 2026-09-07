"""Lab 2: inspect attention, multi-head attention, and causal masking."""

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from bayan.attention import attention, MultiHeadAttention


def attention_weights(q, k, mask=None):
    """Compute attention weights for inspection."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    return torch.softmax(scores, dim=-1)


def main():
    torch.manual_seed(42)

    print("=== 1. Scaled Dot-Product Attention ===")

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    # Our implementation
    ours = attention(q, k, v)

    # PyTorch reference
    reference = F.scaled_dot_product_attention(q, k, v)

    max_difference = (ours - reference).abs().max().item()

    print(f"Output shape: {ours.shape}")
    print(f"Maximum difference vs PyTorch: {max_difference:.10f}")
    print(f"Equivalent within 1e-6: {max_difference < 1e-6}")

    assert max_difference < 1e-6


    print("\n=== 2. Attention Weight Matrix ===")

    weights = attention_weights(q, k)

    print("Head 0 attention weights:")
    print(weights[0, 0])

    print("\nRow sums:")
    print(weights[0, 0].sum(dim=-1))


    print("\n=== 3. Multi-Head Attention ===")

    x = torch.randn(2, 5, 16)

    mha = MultiHeadAttention(
        d_model=16,
        num_heads=4,
    )

    mha_output = mha(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {mha_output.shape}")

    assert mha_output.shape == x.shape


    print("\n=== 4. Causal Mask ===")

    seq_len = q.size(-2)

    # Lower-triangular mask:
    # position i can only attend to positions <= i
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    masked_output = attention(
        q,
        k,
        v,
        mask=causal_mask,
    )

    masked_weights = attention_weights(
        q,
        k,
        mask=causal_mask,
    )

    print("Causal mask:")
    print(causal_mask[0, 0].int())

    print("\nMasked attention weights — Head 0:")
    print(masked_weights[0, 0])

    # Values above the diagonal must be zero
    future_attention = torch.triu(
        masked_weights[0, 0],
        diagonal=1,
    )

    max_future_attention = future_attention.abs().max().item()

    print(
        f"\nMaximum attention paid to future tokens: "
        f"{max_future_attention:.10f}"
    )

    print(
        "Lower-triangular causal behaviour:",
        max_future_attention == 0.0,
    )

    assert masked_output.shape == q.shape
    assert max_future_attention == 0.0

    

    print("\nDecoder-style causal attention verified ✅")

    print("\n=== 5. Real Attention Diagnostics + PAD Leak ===")

    checkpoint = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.eval()

    examples = [
        "There is a water leak in Jeddah. Please fix it urgently.",
        "يوجد تسرب مياه في الحي ونحتاج إلى إصلاحه بشكل عاجل.",
    ]

    encoded = tokenizer(
        examples,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Run once WITHOUT a padding mask
    with torch.no_grad():
        no_mask_output = model(
            input_ids=input_ids,
            output_attentions=True,
        )

    # Run again WITH the correct padding mask
    with torch.no_grad():
        masked_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    no_mask_att = no_mask_output.attentions[-1]
    masked_att = masked_output.attentions[-1]

    # [batch, head, query, key]
    pad_keys = (attention_mask == 0)[:, None, None, :]

    def mean_pad_mass(attentions):
        pad_mass_per_query = (
            attentions * pad_keys
        ).sum(dim=-1)

        real_queries = (
            attention_mask[:, None, :]
            .expand_as(pad_mass_per_query)
            .bool()
        )

        return (
            pad_mass_per_query[real_queries]
            .mean()
            .item()
        )

    pad_mass_without_mask = mean_pad_mass(no_mask_att)
    pad_mass_with_mask = mean_pad_mass(masked_att)

    print(
        f"Mean PAD attention without mask: "
        f"{pad_mass_without_mask:.6f}"
    )

    print(
        f"Mean PAD attention with mask:    "
        f"{pad_mass_with_mask:.6f}"
    )

    # Inspect the first example
    valid_length = int(attention_mask[0].sum().item())

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0, :valid_length]
    )

    inspected = masked_att[
        0, :, :valid_length, :valid_length
    ]

    # Find a head that pays relatively high attention
    # to neighbouring tokens.
    adjacency_scores = []

    for head in range(inspected.size(0)):
        matrix = inspected[head]

        neighbour_values = []

        for i in range(valid_length):
            if i > 0:
                neighbour_values.append(matrix[i, i - 1])

            if i + 1 < valid_length:
                neighbour_values.append(matrix[i, i + 1])

        adjacency_scores.append(
            torch.stack(neighbour_values).mean().item()
        )

    adjacency_head = max(
        range(len(adjacency_scores)),
        key=adjacency_scores.__getitem__,
    )

    # Check whether one head treats [SEP] as an attention sink.
    sep_position = tokens.index("[SEP]")

    sep_scores = (
        inspected[:, :, sep_position]
        .mean(dim=-1)
    )

    sep_head = int(torch.argmax(sep_scores).item())

    print("\nTokens:")
    print(tokens)

    print(
        f"\nMost adjacency-looking head: {adjacency_head} "
        f"(score={adjacency_scores[adjacency_head]:.6f})"
    )

    print(
        f"Strongest [SEP]-sink head: {sep_head} "
        f"(mean SEP attention={sep_scores[sep_head].item():.6f})"
    )

    print(
        "\nAttention matrix for adjacency-looking head:"
    )

    print(inspected[adjacency_head])

    print("\nPAD leakage removed by correct attention mask:",
          pad_mass_with_mask < 1e-8)


if __name__ == "__main__":
    main()