#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment_residual_v2.py
# coding=utf-8
"""
LayoutLMv3ForSegmentResidualTokenClassification (residual-v2, corrected)

Design (grounded in error analysis + v1 diagnosis):
  - Segment pooling + inter-segment Transformer context: unchanged from
    the original segment model (mean-pool tokens per segment, add a
    segment positional embedding for reading order, run a small
    TransformerEncoder over the sequence of segment vectors).
  - NO broadcast replacement. The token keeps its own hidden state h_i;
    segment context is added as a residual correction:

        s'_k = TransformerContext(seg_vecs + pos_emb)      # full context, no gate
        r_k  = W_s(s'_k)                                    # W_s zero-initialized
        z_i  = h_i + r_k                                    # broadcast r_k to every
                                                              # token in segment k

    At step 0, W_s = 0 => r_k = 0 => z_i = h_i identically (exact vanilla
    baseline behavior at init).

  - Why this differs from residual-v1 (and why v1 likely stalled):
    v1 used TWO zero-initialized scalars in series
    (segment_context_gate, segment_fusion_alpha). Chain rule shows that
    with alpha=0, d(loss)/d(W_s) and d(loss)/d(gate) are BOTH exactly
    zero at init (they're multiplied by alpha=0 in the backward pass).
    Only alpha itself gets a non-zero gradient directly. So every
    upstream parameter (the segment_context Transformer, the segment
    positional embedding, W_s) is gradient-starved until alpha has
    already drifted away from zero on its own -- an unnecessary,
    slow, serial warm-up that likely explains why v1 underperformed
    within a short 1000-step budget.

    v2 uses exactly ONE zero-initialized point: the Linear projection
    W_s. d(loss)/d(W_s) = d(loss)/d(z_i) outer s'_k, which is NON-ZERO
    from step 0 (s'_k is not itself gated to zero). So W_s starts
    learning immediately, and gradient also reaches segment_context and
    segment_position_embedding through s'_k from step 0 -- no serial
    bottleneck.

  - is_first_token_embedding is REMOVED. It encoded "first token OF
    SEGMENT" (not "first token of WORD"), which under
    label_all_tokens=True actively taught the classifier a spurious
    B/I signal whenever a segment's first tokenizer-token was not the
    first token of a multi-token word (see the DATE:/FAX: regressions
    in error analysis). Since we no longer overwrite h_i, the token's
    own hidden state (with its own 1D position + content) already
    carries the information the classifier needs for B/I boundaries,
    so no extra learned embedding is required here.

Interface is unchanged from the previous segment model: same forward()
signature, same seg_id convention (-1 = not in any segment, else local
segment index in reading order), same fallback to vanilla per-token
behavior when seg_id is None.
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)


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
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)

        if seg_ctx_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=seg_ctx_heads,
                dim_feedforward=config.hidden_size * 2,
                dropout=seg_ctx_dropout,
                batch_first=True,
            )
            self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=seg_ctx_layers)

            max_pos = getattr(config, "segment_context_max_positions", 128)
            self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)

            # The ONLY zero-init point in the whole branch.
            self.segment_residual_proj = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.zeros_(self.segment_residual_proj.weight)
            nn.init.zeros_(self.segment_residual_proj.bias)
        else:
            self.segment_context = None
            self.segment_position_embedding = None
            self.segment_residual_proj = None

        self.init_weights()

    def _segment_residual(self, text_hidden, seg_id):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only.
        seg_id:      (B, L) long tensor, -1 for tokens outside any segment,
                     else local segment index (0, 1, 2, ...) in reading order.

        Returns z = h + r, where r is the broadcast residual correction
        (identically zero at initialization).
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        residual = torch.zeros_like(text_hidden)

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)  # reading order
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)

            max_pos = self.segment_position_embedding.num_embeddings
            positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
            seg_vecs_with_pos = seg_vecs + self.segment_position_embedding(positions)
            ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)  # s'_k, full context, no gate

            r_k = self.segment_residual_proj(ctx_out)  # zero at init, learns from step 0

            for i, mask in enumerate(seg_masks):
                residual[b, mask] = r_k[i]

        return text_hidden + residual

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

        sequence_output = outputs[0]  # (B, text_len + image_len, H)
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        if seg_id is not None and self.segment_context is not None:
            text_hidden = self._segment_residual(text_hidden, seg_id)
        # if seg_id is None (or segment_context disabled): exact vanilla fallback.

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
