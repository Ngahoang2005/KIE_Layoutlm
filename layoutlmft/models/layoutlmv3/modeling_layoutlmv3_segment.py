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

  - Token-fusion (local_classifier + fusion_gate): a small auxiliary
    classifier sees the ORIGINAL per-token hidden state (before segment
    pooling/broadcast), and its logits are blended into the segment logits
    via a per-token, uncertainty-conditioned gate:

        delta = local_logits - segment_logits.detach()
        fused = segment_logits + gate * delta

    RESIDUAL form (not `(1-gate)*segment + gate*local`), so the segment
    path's gradient is never diluted by (1-gate) -- it stays exactly as
    strong as in the pure segment-head baseline. Only local_classifier /
    fusion_gate learn through the second term.

    Two extra safeguards keep the gate honest (see forward() and the loss
    computation below):
      1. entropy fed to the gate is NORMALIZED to [0,1] (divided by
         ln(num_labels)) so it's on the same scale as max_prob -- otherwise
         the raw ln(7)~1.95 entropy dominates the gate's decision in an
         uncontrolled way.
      2. an L1-style SPARSITY penalty on the gate's mean activation is
         added to the training loss. Without it, the optimizer can cheat:
         local_classifier (a single Linear layer seeing raw per-token
         hidden states) overfits the tiny FUNSD train set very fast, and
         the "cheapest" way to lower train loss becomes "open the gate wide
         so fused_logits just tracks the overfit local_logits" -- which is
         exactly what caused avg_gate to jump to ~0.6 and eval accuracy to
         drop despite train loss looking fine. The sparsity penalty gives
         opening the gate a real cost, so it only opens where doing so
         actually reduces the MAIN (non-aux) loss by more than that cost.

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
        # -gated RESIDUAL logit fusion, with sparsity regularization.
        # ================================================================
        use_token_fusion = getattr(config, "use_token_fusion", True)
        self.use_token_fusion = use_token_fusion

        # Aux loss weight: kept small on purpose -- just enough to "prime"
        # local_classifier so it isn't pure noise once the gate opens a bit,
        # but not so large that it overfits the tiny FUNSD train set fast
        # and drags the gate open with it. (Was 0.3, caused avg_gate~0.6.)
        self.token_fusion_aux_loss_weight = getattr(config, "token_fusion_aux_loss_weight", 0.05)

        # Sparsity penalty on gate.mean() -- gives "opening the gate" a
        # real cost in the loss, preventing the overfitting shortcut above.
        self.token_fusion_gate_reg_weight = getattr(config, "token_fusion_gate_reg_weight", 0.05)

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
        how open the fusion gate currently is.
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

        local_logits = None      # used by the aux loss below
        gate_for_reg = None      # used by the sparsity penalty below

        if self.use_token_fusion and seg_id is not None:
            segment_logits_text = segment_logits_full[:, :text_len, :]

            local_logits = self.local_classifier(self.dropout(local_hidden))  # (B, text_len, num_labels)

            with torch.no_grad():
                seg_probs = torch.softmax(segment_logits_text, dim=-1)
                max_prob = seg_probs.max(dim=-1, keepdim=True).values
                entropy = -(seg_probs * torch.log(seg_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
                # Normalize entropy to [0, 1] (max possible entropy is
                # ln(num_labels)) so it's on the same scale as max_prob --
                # otherwise the raw ~ln(7)=1.95 value dominates the gate
                # input in an uncontrolled, unintended way.
                max_entropy = torch.log(torch.tensor(float(self.num_labels), device=entropy.device))
                entropy = entropy / max_entropy.clamp_min(1e-8)

            gate_input = torch.cat(
                [local_hidden, broadcast_hidden, max_prob, entropy], dim=-1
            )  # (B, text_len, 2H+2)
            gate = torch.sigmoid(self.fusion_gate(gate_input))  # (B, text_len, 1)

            # ---- Residual fusion (NOT interpolation) ----
            # segment_logits_text.detach() in delta means the segment path's
            # gradient (first term) is completely untouched -- only
            # local_classifier/fusion_gate learn through the second term.
            delta = local_logits - segment_logits_text.detach()
            fused_text_logits = segment_logits_text + gate * delta

            gate_for_reg = gate if self.training else None

            # Accumulate gate stats for logging (mean over all tokens incl.
            # padding is an acceptable approximation here).
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

            # ---- Auxiliary loss for local_classifier (small weight) ----
            if self.training and local_logits is not None and self.token_fusion_aux_loss_weight > 0:
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

            # ---- Gate sparsity regularization ----
            # Gives opening the gate a real cost, so the optimizer can't
            # cheaply lower train loss just by tracking an overfit
            # local_classifier -- it must actually help the MAIN loss more
            # than the penalty costs.
            if self.training and gate_for_reg is not None and self.token_fusion_gate_reg_weight > 0:
                if attention_mask is not None:
                    attn_text = attention_mask[:, :text_len].unsqueeze(-1).float()
                    gate_reg = (gate_for_reg * attn_text).sum() / attn_text.sum().clamp_min(1.0)
                else:
                    gate_reg = gate_for_reg.mean()
                loss = loss + self.token_fusion_gate_reg_weight * gate_reg

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
