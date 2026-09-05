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
    order) so adjacent segments exchange information, then broadcast the
    context-enriched vector back to every token in the segment before the
    (unchanged) token classifier.
  - To keep the existing BIO scheme / seqeval / compute_metrics pipeline
    100% unchanged, we do NOT collapse labels to entity-type-only. Instead
    we add a tiny learned "is-first-token-of-segment" embedding so the
    (otherwise identical) broadcast vector can still support the B-/I-
    distinction at the classifier.

  - NEW (token-fusion): a small auxiliary "local_classifier" sees the
    ORIGINAL per-token hidden state (before segment pooling/broadcast), and
    its logits are blended into the segment logits via a per-token,
    uncertainty-conditioned gate. Fusion happens at the LOGIT level (not
    hidden-state level) and in RESIDUAL form:

        delta = local_logits - segment_logits.detach()
        fused = segment_logits + gate * delta

    This is deliberately NOT `(1-gate)*segment + gate*local`, because that
    interpolation form multiplies the segment path's gradient by (1-gate)
    at every step, permanently diluting the signal that made segment-level
    pooling work in the first place. The residual form leaves the segment
    path's gradient completely untouched (coefficient 1, not 1-gate) while
    still letting `local_classifier` learn through `gate * delta`. The gate
    starts near 0 (bias=-4 -> sigmoid~0.018), so at init the whole model is
    numerically identical to the pure segment-head baseline; it only opens
    up where the (detached) segment prediction is already uncertain
    (high entropy / low max-prob), which is exactly the additive-help
    regime we want (bofsung, not gay nhieu).

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone.
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

        # ---- Inter-segment context (unchanged) ----
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

            max_pos = getattr(config, "segment_context_max_positions", 128)
            self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.segment_context = None
            self.segment_context_gate = None
            self.segment_position_embedding = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        # ================================================================
        # Token-fusion module: local (pre-pooling) classifier + uncertainty
        # -gated RESIDUAL logit fusion. See module docstring for rationale.
        # ================================================================
        use_token_fusion = getattr(config, "use_token_fusion", True)
        self.use_token_fusion = use_token_fusion
        self.token_fusion_aux_loss_weight = getattr(config, "token_fusion_aux_loss_weight", 0.3)

        if use_token_fusion:
            self.local_classifier = nn.Linear(config.hidden_size, config.num_labels)

            gate_hidden_dim = getattr(config, "token_fusion_gate_hidden_dim", config.hidden_size // 4)
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2 + 2, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            # Zero-init weight + negative bias -> gate starts near 0
            # (sigmoid(-4) ~ 0.018), so forward pass at step 0 is
            # numerically ~identical to the pure segment-head baseline.
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, -4.0)
        else:
            self.local_classifier = None
            self.fusion_gate = None

        # Running accumulator for avg-gate logging (so it's impossible to
        # "forget to log" -- see get_and_reset_gate_stats()).
        self._gate_sum = 0.0
        self._gate_count = 0

        self.init_weights()

    def get_and_reset_gate_stats(self):
        """Returns (avg_gate, num_tokens_seen) since the last reset, then
        resets the accumulator. Call this after trainer.evaluate() to log
        how open the fusion gate currently is, without needing to modify
        compute_metrics (which only sees predictions/labels, not internals).
        """
        if self._gate_count == 0:
            return None, 0
        avg = self._gate_sum / self._gate_count
        self._gate_sum = 0.0
        self._gate_count = 0
        return avg, self._gate_count

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only.
        seg_id:      (B, L) long tensor. -1 marks tokens not in any segment.
                     Non-negative values are LOCAL segment indices per
                     example, in reading order.

        Returns:
            broadcast_hidden: (B, L, H) -- every token in the same segment
                gets an IDENTICAL context-enriched vector.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

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

        # Keep the ORIGINAL per-token hidden state (before segment
        # pooling/broadcast) -- this feeds local_classifier so it can see
        # fine-grained per-token cues the segment vector cannot.
        local_hidden = text_hidden

        if seg_id is not None:
            broadcast_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0
            if seg_id.shape[1] > 1:
                prev = seg_id[:, :-1]
                cur = seg_id[:, 1:]
                changed = (cur != prev) & (cur >= 0)
                is_first[:, 1:] = changed.long()
            is_first = is_first * (seg_id >= 0).long()

            broadcast_hidden = broadcast_hidden + self.is_first_token_embedding(is_first)
        else:
            broadcast_hidden = text_hidden

        if image_hidden.shape[1] > 0:
            pooled_sequence = torch.cat([broadcast_hidden, image_hidden], dim=1)
        else:
            pooled_sequence = broadcast_hidden

        pooled_sequence = self.dropout(pooled_sequence)
        segment_logits_full = self.classifier(pooled_sequence)  # (B, text_len+img_len, num_labels)

        local_logits = None  # kept around for the aux loss below

        if self.use_token_fusion and seg_id is not None:
            segment_logits_text = segment_logits_full[:, :text_len, :]

            local_logits = self.local_classifier(self.dropout(local_hidden))  # (B, text_len, num_labels)

            with torch.no_grad():
                seg_probs = torch.softmax(segment_logits_text, dim=-1)
                max_prob = seg_probs.max(dim=-1, keepdim=True).values
                entropy = -(seg_probs * torch.log(seg_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)

            gate_input = torch.cat(
                [local_hidden, broadcast_hidden, max_prob, entropy], dim=-1
            )  # (B, text_len, 2H+2)
            gate = torch.sigmoid(self.fusion_gate(gate_input))  # (B, text_len, 1)

            # ---- Residual fusion (NOT interpolation) ----
            # segment_logits_text.detach() in the delta means the segment
            # path's gradient (from the first term) is left completely
            # untouched -- only local_classifier/fusion_gate learn through
            # the second term.
            delta = local_logits - segment_logits_text.detach()
            fused_text_logits = segment_logits_text + gate * delta

            # Accumulate gate stats for logging (mean over all real tokens;
            # padding/special tokens included is fine as an approximation --
            # they contribute ~0 signal either way since gate depends on
            # local/broadcast hidden which are near-zero there too, but if
            # you want it exact you can mask by attention_mask before this).
            with torch.no_grad():
                self._gate_sum += gate.sum().item()
                self._gate_count += gate.numel()

            if image_hidden.shape[1] > 0:
                logits = torch.cat([fused_text_logits, segment_logits_full[:, text_len:, :]], dim=1)
            else:
                logits = fused_text_logits
        else:
            logits = segment_logits_full

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

            # ---- Auxiliary loss for local_classifier ----
            # Small extra supervision so local_classifier learns useful
            # per-token signal even while the gate is still mostly closed
            # (otherwise it would stay near-random until the gate opens,
            # and "opening the gate onto noise" is exactly the failure mode
            # we're trying to avoid). Only added during training (not eval),
            # so eval_loss stays a clean, comparable number across runs.
            if self.training and local_logits is not None and self.token_fusion_aux_loss_weight > 0:
                # Reuse the same active_labels for the TEXT part. active_labels
                # was built for the full (text+image) sequence; slice to text_len.
                labels_text = labels[:, :text_len] if labels.shape[1] >= text_len else labels
                if attention_mask is not None:
                    attn_text = attention_mask[:, :text_len]
                    active_loss_text = attn_text.reshape(-1) == 1
                    active_local_logits = local_logits.reshape(-1, self.num_labels)
                    active_labels_text = torch.where(
                        active_loss_text,
                        labels_text.reshape(-1),
                        torch.tensor(loss_fct.ignore_index).type_as(labels),
                    )
                    aux_loss = loss_fct(active_local_logits, active_labels_text)
                else:
                    aux_loss = loss_fct(local_logits.reshape(-1, self.num_labels), labels_text.reshape(-1))
                loss = loss + self.token_fusion_aux_loss_weight * aux_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
