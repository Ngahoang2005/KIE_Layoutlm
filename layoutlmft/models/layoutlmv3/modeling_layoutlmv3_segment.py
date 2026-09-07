#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification (relative segment-position variant)

Same core idea as the base version (segment pooling + inter-segment context
Transformer + broadcast), but the segment ordering signal is now injected
as a RELATIVE position bias directly inside the self-attention score of the
segment-context Transformer, following the same relative_position_bucket
mechanism LayoutLMv3 already uses for token-level attention -- instead of
an absolute positional embedding added to the input.

This requires a small custom self-attention layer (PyTorch's
nn.TransformerEncoderLayer has no hook for an extra per-head attention
bias), mirroring how LayoutLMv3SelfAttention itself is hand-written rather
than using nn.MultiheadAttention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)


def relative_position_bucket(relative_position, num_buckets=32, max_distance=32):
    """
    Same log-bucket scheme as LayoutLMv3Encoder.relative_position_bucket,
    specialized for 1D segment order (bidirectional).
    relative_position: LongTensor, any shape.
    """
    num_buckets = num_buckets // 2
    ret = (relative_position > 0).long() * num_buckets
    n = torch.abs(relative_position)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret = ret + torch.where(is_small, n, val_if_large)
    return ret


class RelPosSegmentSelfAttention(nn.Module):
    """
    Minimal multi-head self-attention over the sequence of segment vectors,
    with a learned relative-position bias added to the raw attention score
    (before softmax), exactly like LayoutLMv3SelfAttention does for tokens.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x, n_seg):
        # x: (n_seg, H) -> (num_heads, n_seg, head_dim)
        return x.view(n_seg, self.num_heads, self.head_dim).permute(1, 0, 2)

    def forward(self, x, rel_bias):
        """
        x:        (n_seg, H) -- single "batch" of segment vectors (we run
                   one document at a time, same as the original loop).
        rel_bias: (num_heads, n_seg, n_seg) -- additive bias per head,
                   already computed from the relative position buckets.
        """
        n_seg, H = x.shape

        q = self._shape(self.query(x), n_seg)   # (heads, n_seg, head_dim)
        k = self._shape(self.key(x), n_seg)
        v = self._shape(self.value(x), n_seg)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (heads, n_seg, n_seg)
        attn_scores = attn_scores + rel_bias

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)                     # (heads, n_seg, head_dim)
        context = context.permute(1, 0, 2).contiguous().view(n_seg, H)
        return self.out_proj(context)


class RelPosSegmentContextLayer(nn.Module):
    """
    One Transformer block: self-attn (with relative bias) + FFN, pre/post-LN.
    Mirrors nn.TransformerEncoderLayer's structure (post-LN, like the
    default PyTorch implementation) so behavior is comparable to the
    absolute-position baseline.
    """

    def __init__(self, hidden_size, num_heads, ffn_size, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosSegmentSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, rel_bias):
        attn_out = self.self_attn(x, rel_bias)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.norm2(x + ffn_out)
        return x


class RelPosSegmentContextEncoder(nn.Module):
    """Stack of RelPosSegmentContextLayer, plus the relative-bias generator."""

    def __init__(self, hidden_size, num_heads, ffn_size, num_layers, dropout, rel_bins, max_distance):
        super().__init__()
        self.layers = nn.ModuleList([
            RelPosSegmentContextLayer(hidden_size, num_heads, ffn_size, dropout)
            for _ in range(num_layers)
        ])
        self.rel_bins = rel_bins
        self.max_distance = max_distance
        self.num_heads = num_heads
        # one bias value per head per bucket, learned
        self.rel_pos_bias = nn.Linear(rel_bins, num_heads, bias=False)

    def _compute_rel_bias(self, n_seg, device):
        positions = torch.arange(n_seg, device=device)
        rel_mat = positions.unsqueeze(0) - positions.unsqueeze(1)     # (n_seg, n_seg)
        rel_bucket = relative_position_bucket(rel_mat, num_buckets=self.rel_bins, max_distance=self.max_distance)
        rel_onehot = F.one_hot(rel_bucket, num_classes=self.rel_bins).float()  # (n_seg, n_seg, rel_bins)
        rel_bias = self.rel_pos_bias(rel_onehot)                       # (n_seg, n_seg, num_heads)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()              # (num_heads, n_seg, n_seg)
        return rel_bias

    def forward(self, x):
        # x: (n_seg, H)
        n_seg = x.shape[0]
        rel_bias = self._compute_rel_bias(n_seg, x.device)
        for layer in self.layers:
            x = layer(x, rel_bias)
        return x


class LayoutLMv3ForSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", None)
        if seg_ctx_dropout is None:
            seg_ctx_dropout = config.hidden_dropout_prob
        # NEW: relative-position config knobs
        rel_seg_bins = getattr(config, "segment_relative_position_bins", 32)
        rel_seg_max_distance = getattr(config, "segment_relative_max_distance", 32)

        if seg_ctx_layers > 0:
            self.segment_context = RelPosSegmentContextEncoder(
                hidden_size=config.hidden_size,
                num_heads=seg_ctx_heads,
                ffn_size=config.hidden_size * 2,
                num_layers=seg_ctx_layers,
                dropout=seg_ctx_dropout,
                rel_bins=rel_seg_bins,
                max_distance=rel_seg_max_distance,
            )
            self.segment_context_gate = nn.Parameter(torch.zeros(1))
            # NOTE: no more segment_position_embedding -- position info now
            # lives entirely inside the attention bias, not the input.
        else:
            self.segment_context = None
            self.segment_context_gate = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        self.init_weights()

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)

            if self.segment_context is not None:
                # NEW: no positional embedding added to input anymore --
                # relative order is injected inside attention via rel_bias.
                ctx_out = self.segment_context(seg_vecs)   # (n_seg, H)
                seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)
            else:
                seg_vecs_ctx = seg_vecs

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        return broadcast_hidden

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_id=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
        )

        sequence_output = outputs[0]
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        if seg_id is not None:
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0
            if seg_id.shape[1] > 1:
                prev = seg_id[:, :-1]
                cur = seg_id[:, 1:]
                changed = (cur != prev) & (cur >= 0)
                is_first[:, 1:] = changed.long()
            is_first = is_first * (seg_id >= 0).long()

            text_hidden = text_hidden + self.is_first_token_embedding(is_first)

        if image_hidden.shape[1] > 0:
            pooled_sequence = torch.cat([text_hidden, image_hidden], dim=1)
        else:
            pooled_sequence = text_hidden

        pooled_sequence = self.dropout(pooled_sequence)
        logits = self.classifier(pooled_sequence)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
