#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

Core idea (grounded in error analysis on FUNSD + CORD):
  - Segment self-consistency is already ~98-99% solved by the base model
    (confirmed empirically) -> a consistency REGULARIZER has little to gain.
  - The real errors are (a) whole segments classified wrong as a unit
    (esp. long free-text spans dropped entirely via BIO "drift"), and
    (b) confusions that depend on the NEIGHBORING segment's role
    (HEADER vs QUESTION on FUNSD; parent vs sub-item on CORD).
  - Fix: pool each segment's token hidden states into one vector, run a
    tiny Transformer encoder over the SEQUENCE of segment vectors (reading
    order) so adjacent segments exchange information, then inject the
    contextualized segment representation back into each token through a
    residual adapter before the token classifier.
  - The segment representation is fused RESIDUALLY instead of replacing
    token representations: z_i = h_i + alpha * s'_k. This keeps
    token-specific evidence while allowing segment context to refine it.
  - alpha is a learned scalar initialized at 0, so the model starts from
    the original token representation and learns how much segment context
    to use.
  - We intentionally remove the previous "is-first-token-of-segment"
    embedding because the current training setup uses label_all_tokens=True,
    and segment-first is not the same thing as word-first/subword behavior.

This class does NOT touch the pretrained LayoutLMv3 self-attention, does
NOT build any graph/hypergraph, and does NOT modify the pretrained
backbone architecture. The only new operation after the backbone is a
segment-context residual adapter before token classification.
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

        # ---- NEW: lightweight inter-segment context module ----
        # Config knobs (optional; safe defaults if not set on the config object).
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
            self.segment_context_gate = nn.Parameter(torch.zeros(1))
            
            # NEW: positional embedding cho THỨ TỰ segment trong document (reading order)
            max_pos = getattr(config, "segment_context_max_positions", 128)
            self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.segment_context = None
            self.segment_context_gate = None
            self.segment_position_embedding = None

        # ------------------------------------------------------------------
        # Residual fusion scale.
        #
        # New fusion:
        #     z_i = h_i + alpha * s'_k
        #
        # alpha starts at 0 so the model does not initially overwrite the
        # token-specific representation produced by the LayoutLMv3 backbone.
        # ------------------------------------------------------------------
        self.segment_fusion_alpha = nn.Parameter(torch.zeros(1))

        # Learned projection of the segment context before residual fusion.
        # This lets the classifier learn which directions of the segment
        # representation are useful at token level instead of adding the raw
        # pooled vector directly to every token.
        self.segment_context_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
        )

        self.init_weights()
        # for param in self.layoutlmv3.parameters():
        #     param.requires_grad = False

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only
                     (image-patch positions, if any, are handled separately
                     by the caller and never enter this function).
        seg_id:      (B, L) long tensor. -1 marks tokens that do not belong
                     to any segment (special tokens / padding). Non-negative
                     values are LOCAL segment indices per example, assigned
                     in reading order (0, 1, 2, ...), exactly matching the
                     bbox-equality grouping used in run_funsd_cord.py's
                     tokenize_and_align_labels (see patch).

        Returns:
            fused_hidden: (B, L, H) -- original token representations plus
                a residual contribution from their segment context.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        # Preserve the original token-level representation.
        fused_hidden = text_hidden.clone()

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

            if self.segment_context is not None:
                max_pos = self.segment_position_embedding.num_embeddings
                positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                seg_vecs_with_pos = seg_vecs + self.segment_position_embedding(positions)
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)
                seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)
            else:
                seg_vecs_ctx = seg_vecs

            for i, mask in enumerate(seg_masks):
                # ------------------------------------------------------
                # Residual segment fusion.
                #
                # OLD: token hidden state was REPLACED by seg_vecs_ctx[i].
                # NEW: keep h_i and add learned segment context.
                # ------------------------------------------------------
                segment_residual = self.segment_context_projection(
                    seg_vecs_ctx[i]
                )

                fused_hidden[b, mask] = (
                    text_hidden[b, mask]
                    + self.segment_fusion_alpha * segment_residual
                )

        return fused_hidden

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
        seg_id=None,  # NEW input: (batch, text_seq_len), see docstring above
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

        if seg_id is not None:
            # Residual segment fusion only. The token-level hidden state is
            # never replaced by a pooled segment vector.
            text_hidden = self._segment_pool_and_contextualize(
                text_hidden,
                seg_id,
            )
        # If seg_id is None (e.g. vanilla token-classification inference),
        # fall back to plain per-token behavior -- text_hidden is untouched.

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
