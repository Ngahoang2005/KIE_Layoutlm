# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

Conservative segment + token late-fusion model for FUNSD/CORD.

Motivation:
  The strongest observed segment model in this project uses:
      h_i -> mean pool by segment -> inter-segment context -> broadcast
      -> first-token embedding -> classifier

  Several later variants became worse after changing the hidden representation
  itself or adding a large extra classifier branch. This version therefore
  keeps the original segment representation as the MAIN path and introduces
  only a small late token shortcut.

Architecture:

  Backbone token states:
      h_i

  Segment path:
      s_k = mean(h_i in segment k)
      c_k = Transformer(s_k + segment-position)
      s'_k = s_k + g * (c_k - s_k)
      z_seg_i = s'_k + first_token_embedding(i)
      logits_seg_i = Classifier(z_seg_i)

  Token path:
      z_tok_i = h_i + first_token_embedding(i)
      logits_tok_i = Classifier(z_tok_i)

  Final logits:
      logits_i = logits_seg_i
                 + alpha * (logits_tok_i - logits_seg_i)

  The token shortcut is activated only for NON-FIRST tokens inside
  multi-token segments. This preserves the strong segment decision for the
  first B-token while allowing the following I-tokens to recover their own
  token-level information.

  alpha starts very close to zero, so the model starts essentially as the
  previously working hard-segment model. There is only ONE new trainable
  scalar and no extra classifier/MLP.

Why this is deliberately conservative:
  - No adaptive pooling.
  - No within-segment positional embedding.
  - No token residual hidden-state projection.
  - No auxiliary loss.
  - No second classifier with independent parameters.
  - The original segment path remains the dominant path.

This file keeps the same interface expected by run_funsd_cord.py and is meant
for use through --use_segment_head only.
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

        # ------------------------------------------------------------------
        # LayoutLMv3 backbone + original classifier
        # ------------------------------------------------------------------
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_labels < 10:
            self.classifier = nn.Linear(
                config.hidden_size,
                config.num_labels,
            )
        else:
            self.classifier = LayoutLMv3ClassificationHead(
                config,
                pool_feature=False,
            )

        # ------------------------------------------------------------------
        # Inter-segment context: same core mechanism as the strong segment
        # model used earlier in this project.
        # ------------------------------------------------------------------
        seg_ctx_layers = int(
            getattr(config, "segment_context_layers", 1)
        )
        seg_ctx_heads = int(
            getattr(config, "segment_context_heads", 4)
        )
        seg_ctx_dropout = float(
            getattr(
                config,
                "segment_context_dropout",
                config.hidden_dropout_prob,
            )
        )
        max_seg_pos = int(
            getattr(config, "segment_context_max_positions", 128)
        )

        if seg_ctx_layers <= 0:
            raise ValueError("segment_context_layers must be > 0")
        if seg_ctx_heads <= 0:
            raise ValueError("segment_context_heads must be > 0")
        if config.hidden_size % seg_ctx_heads != 0:
            raise ValueError(
                "segment_context_heads must divide hidden_size: "
                f"hidden_size={config.hidden_size}, "
                f"segment_context_heads={seg_ctx_heads}"
            )
        if max_seg_pos <= 0:
            raise ValueError("segment_context_max_positions must be > 0")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=seg_ctx_heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=seg_ctx_dropout,
            batch_first=True,
        )
        self.segment_context = nn.TransformerEncoder(
            encoder_layer,
            num_layers=seg_ctx_layers,
        )
        self.segment_position_embedding = nn.Embedding(
            max_seg_pos,
            config.hidden_size,
        )

        # ReZero gate. At initialization the context Transformer contributes
        # nothing, preserving the mean-pooled segment representation.
        self.segment_context_gate = nn.Parameter(torch.zeros(1))

        # ------------------------------------------------------------------
        # Existing BIO-preserving first-token cue.
        # ------------------------------------------------------------------
        self.is_first_token_embedding = nn.Embedding(
            2,
            config.hidden_size,
        )

        # ------------------------------------------------------------------
        # NEW: one scalar controlling late token shortcut.
        # ------------------------------------------------------------------
        # We use a bounded positive coefficient. The initialization is small,
        # but not exactly zero, so the scalar has a healthy gradient from step
        # 1. With max=0.15 and raw=-2, alpha ~= 0.018.
        self.token_blend_max = float(
            getattr(config, "token_blend_max", 0.15)
        )
        if not (0.0 < self.token_blend_max <= 0.5):
            raise ValueError(
                "token_blend_max must be in (0, 0.5], got "
                f"{self.token_blend_max}"
            )
        self.token_blend_raw = nn.Parameter(
            torch.tensor(-1.3863, dtype=torch.float32)
        )

        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        self.init_weights()

        with torch.no_grad():
            self.segment_context_gate.zero_()
            self.segment_position_embedding.weight.normal_(
                mean=0.0,
                std=0.02,
            )
            self.is_first_token_embedding.weight.normal_(
                mean=0.0,
                std=0.02,
            )

    @staticmethod
    def _normalize_seg_id(seg_id, text_len):
        """Return seg_id with shape (B, text_len), padding with -1."""
        if seg_id is None:
            return None

        if seg_id.ndim != 2:
            raise ValueError(
                f"seg_id must be 2-D, got shape={tuple(seg_id.shape)}"
            )

        if seg_id.shape[1] == text_len:
            return seg_id

        if seg_id.shape[1] > text_len:
            return seg_id[:, :text_len]

        pad_len = text_len - seg_id.shape[1]
        pad = torch.full(
            (seg_id.shape[0], pad_len),
            -1,
            device=seg_id.device,
            dtype=seg_id.dtype,
        )
        return torch.cat([seg_id, pad], dim=1)

    @staticmethod
    def _build_first_token_flags(seg_id):
        """1 for the first valid token of each segment, else 0."""
        flags = torch.zeros_like(seg_id, dtype=torch.long)
        if seg_id.shape[1] == 0:
            return flags

        valid = seg_id >= 0
        flags[:, 0] = valid[:, 0].long()

        if seg_id.shape[1] > 1:
            prev = seg_id[:, :-1]
            cur = seg_id[:, 1:]
            flags[:, 1:] = (
                (cur >= 0)
                & (cur != prev)
            ).long()

        return flags

    @staticmethod
    def _build_nonfirst_multitoken_mask(seg_id):
        """
        True only for tokens that:
          1) belong to a valid segment,
          2) are NOT the first token of that segment,
          3) belong to a segment containing at least two tokens.

        This intentionally excludes the first B-token of every segment.
        """
        B, L = seg_id.shape
        mask_out = torch.zeros(
            B,
            L,
            device=seg_id.device,
            dtype=torch.bool,
        )

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if not bool(valid.any()):
                continue

            unique_segments = torch.unique(ids[valid], sorted=True)
            for seg_value in unique_segments:
                mask = ids == seg_value
                indices = torch.nonzero(
                    mask,
                    as_tuple=False,
                ).squeeze(-1)

                # Only the I/middle/last tokens get the token shortcut.
                if int(indices.numel()) > 1:
                    mask_out[b, indices[1:]] = True

        return mask_out

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        Hard segment mean-pooling + reading-order context.

        Every token in a valid segment receives the same segment-context
        vector. The first-token embedding is added later in forward().
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device

        if seg_id.ndim != 2 or tuple(seg_id.shape) != (B, L):
            raise ValueError(
                "seg_id must have shape (batch, text_len) matching text_hidden; "
                f"got seg_id={tuple(seg_id.shape)}, "
                f"text_hidden={tuple(text_hidden.shape)}"
            )

        broadcast_hidden = text_hidden.clone()
        gate = self.segment_context_gate
        max_pos = self.segment_position_embedding.num_embeddings

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if not bool(valid.any()):
                continue

            unique_segments = torch.unique(
                ids[valid],
                sorted=True,
            )
            n_seg = int(unique_segments.numel())

            segment_vectors = torch.zeros(
                n_seg,
                H,
                device=device,
                dtype=text_hidden.dtype,
            )
            segment_masks = []

            for i, seg_value in enumerate(unique_segments):
                mask = ids == seg_value
                segment_masks.append(mask)
                segment_vectors[i] = text_hidden[b, mask].mean(dim=0)

            positions = torch.arange(
                n_seg,
                device=device,
            ).clamp(max=max_pos - 1)

            segment_input = (
                segment_vectors
                + self.segment_position_embedding(positions)
            )
            context_out = self.segment_context(
                segment_input.unsqueeze(0)
            ).squeeze(0)

            segment_vectors_context = (
                segment_vectors
                + gate * (context_out - segment_vectors)
            )

            for i, mask in enumerate(segment_masks):
                broadcast_hidden[b, mask] = segment_vectors_context[i]

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
        line_ids=None,
        block_ids=None,
        column_ids=None,
        entity_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

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
            line_ids=line_ids,
            block_ids=block_ids,
            column_ids=column_ids,
        )

        sequence_output = outputs[0]

        if input_ids is not None:
            text_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            text_len = inputs_embeds.shape[1]
        else:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided"
            )

        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        if seg_id is not None:
            seg_id = self._normalize_seg_id(seg_id, text_len)

            # --------------------------------------------------------------
            # Main segment path: this is the model's dominant representation.
            # --------------------------------------------------------------
            segment_hidden = self._segment_pool_and_contextualize(
                text_hidden,
                seg_id,
            )

            first_flags = self._build_first_token_flags(seg_id)
            first_embedding = self.is_first_token_embedding(first_flags)

            segment_hidden = segment_hidden + first_embedding

            # --------------------------------------------------------------
            # Token shortcut.
            # --------------------------------------------------------------
            # Use the SAME classifier weights as the segment path. This keeps
            # the new branch parameter-light and avoids another high-LR head.
            token_hidden = text_hidden + first_embedding

            # Both paths use the same dropout layer implementation. The
            # shortcut only matters on non-first tokens of multi-token
            # segments, where collapsing to one pooled vector loses information.
            segment_for_classifier = self.dropout(segment_hidden)
            token_for_classifier = self.dropout(token_hidden)

            segment_logits_text = self.classifier(segment_for_classifier)
            token_logits_text = self.classifier(token_for_classifier)

            # Positive bounded mixing coefficient, initially small.
            alpha = self.token_blend_max * torch.sigmoid(
                self.token_blend_raw
            )

            shortcut_mask = self._build_nonfirst_multitoken_mask(seg_id)
            shortcut_mask = shortcut_mask.unsqueeze(-1)

            logits_text = segment_logits_text + (
                alpha
                * shortcut_mask.to(segment_logits_text.dtype)
                * (token_logits_text - segment_logits_text)
            )
        else:
            # Vanilla fallback when no seg_id is supplied.
            classifier_sequence = self.dropout(text_hidden)
            logits_text = self.classifier(classifier_sequence)

        if image_hidden.shape[1] > 0:
            if seg_id is not None:
                # Text branch has already passed through its classifier.
                # Image tokens still use the original classifier path.
                image_for_classifier = self.dropout(image_hidden)
                logits_image = self.classifier(image_for_classifier)
                logits = torch.cat(
                    [logits_text, logits_image],
                    dim=1,
                )
            else:
                image_for_classifier = self.dropout(image_hidden)
                logits_image = self.classifier(image_for_classifier)
                logits = torch.cat(
                    [logits_text, logits_image],
                    dim=1,
                )
        else:
            logits = logits_text

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                ignore_labels = torch.full_like(
                    labels.view(-1),
                    loss_fct.ignore_index,
                )
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    ignore_labels,
                )
                loss = loss_fct(
                    active_logits,
                    active_labels,
                )
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (
                (loss,) + output
                if loss is not None
                else output
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
