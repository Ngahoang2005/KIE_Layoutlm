#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

Core idea (grounded in error analysis on FUNSD + CORD):
  - Segment self-consistency is already ~98-99% solved by the base model
    -> pool each segment's token hidden states into one vector, run a
    tiny Transformer encoder over the SEQUENCE of segment vectors (reading
    order) so adjacent segments exchange information, then broadcast the
    context-enriched vector back to every token in the segment.
  - is_first_token_embedding lets the (shared, broadcast) classifier input
    still distinguish B- from I- despite every token in a segment sharing
    one pooled vector.

  - Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) on the
    segment vectors (seg_vecs_ctx -- AFTER inter-segment context, BEFORE
    broadcasting to tokens). Gold TYPE groups are parsed automatically
    from config.id2label (stripping B-/I- prefixes; "O"/"OTHER" excluded),
    so this works unchanged on FUNSD, CORD, or any other BIO-tagged
    dataset -- no hand-picked confusable pairs needed.

    IMPORTANT: whether this helps or hurts is DATASET-DEPENDENT. It was
    motivated and validated via debug.py's nearest-centroid error
    attribution on CORD (77.2% +/- 1.9% of misclassified segments were
    attributable to embedding overlap there). On FUNSD, the dominant
    error mode described in the accompanying report is CONTEXTUAL
    confusion between neighboring segments (HEADER vs QUESTION, disambiguated
    by whether an ANSWER follows) -- a positional/contextual signal that
    segment_context (order-aware) already targets, not a static
    same-type-clustering problem that SupCon (order-agnostic) addresses.
    Always run debug.py's error-attribution analysis on the TARGET
    dataset before assuming SupCon will help there too.

  Instrumentation: get_segment_gate_value() and get_and_reset_supcon_stats()
  expose diagnostics consumed by CustomTrainer in run_funsd_cord.py (no
  wandb dependency -- plain console/logger output, Kaggle-friendly).

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone.
"""
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

        # ---- ablation knob: is-first-token embedding ----
        self.use_first_token_embedding = getattr(config, "use_first_token_embedding", True)
        if self.use_first_token_embedding:
            self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
            nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)
        else:
            self.is_first_token_embedding = None

        # ---- inter-segment context module ----
        segment_pooling_only = getattr(config, "segment_pooling_only", False)
        seg_ctx_layers = 0 if segment_pooling_only else getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)
        self.segment_context_layers = seg_ctx_layers

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

        # ================================================================
        # Supervised Contrastive Loss on segment vectors.
        # ================================================================
        self.use_supcon_loss = getattr(config, "use_supcon_loss", True)
        self.supcon_weight = getattr(config, "supcon_weight", 0.05)
        self.supcon_temperature = getattr(config, "supcon_temperature", 0.07)

        if self.use_supcon_loss:
            type_of_label, type_vocab = self._build_type_of_label(config)
            self.register_buffer("type_of_label", type_of_label, persistent=True)
            self.type_vocab = type_vocab  # kept for debugging/inspection only
            n_excluded = int((type_of_label < 0).sum().item())
            print(
                f"[SupCon init] parsed {len(type_vocab)} entity type(s) from config.id2label: "
                f"{type_vocab} | {n_excluded}/{len(type_of_label)} label ids excluded (O/OTHER/unknown). "
                f"If type_vocab looks empty or wrong, config.id2label was likely not set correctly "
                f"before from_pretrained() -- SupCon will silently no-op in that case."
            )
        else:
            self.type_of_label = None
            self.type_vocab = []

        # ---- diagnostics accumulators (training-time only; reset via
        # get_and_reset_supcon_stats(), typically called every logging_steps) ----
        self._supcon_loss_sum = 0.0
        self._supcon_loss_count = 0
        self._supcon_anchor_total = 0
        self._supcon_anchor_with_pos = 0

        self.init_weights()

    @staticmethod
    def _build_type_of_label(config):
        """
        Parses config.id2label into a LongTensor (num_labels,) mapping
        label_id -> type_id, where "O"/"OTHER" maps to -1 (excluded from
        SupCon), and B-/I- variants of the same type share the same
        nonnegative type id.
        """
        id2label = getattr(config, "id2label", None)
        num_labels = config.num_labels

        if not id2label or len(id2label) != num_labels:
            return torch.full((num_labels,), -1, dtype=torch.long), []

        items = sorted(((int(k), v) for k, v in id2label.items()), key=lambda kv: kv[0])

        type_vocab = []
        type_index = {}
        type_of_label = [-1] * num_labels

        for label_id, label_str in items:
            s = str(label_str)
            if s == "O" or s.upper() == "OTHER":
                type_of_label[label_id] = -1
                continue
            if s.startswith("B-") or s.startswith("I-"):
                type_name = s[2:]
            else:
                type_name = s
            if type_name in ("", "O"):
                type_of_label[label_id] = -1
                continue
            if type_name not in type_index:
                type_index[type_name] = len(type_vocab)
                type_vocab.append(type_name)
            type_of_label[label_id] = type_index[type_name]

        return torch.tensor(type_of_label, dtype=torch.long), type_vocab

    def get_segment_gate_value(self):
        """Optional introspection hook. Returns None if
        segment_context_layers == 0 (no gate exists)."""
        if self.segment_context_gate is None:
            return None
        return self.segment_context_gate.detach().float().item()

    def get_and_reset_supcon_stats(self):
        """Returns a dict of SupCon diagnostics accumulated since the last
        reset (TRAINING-mode forward passes only), then resets the
        accumulators.

        Keys:
          avg_supcon_loss: mean SupCon loss value over accumulated steps,
              or None if SupCon never fired.
          frac_anchors_with_pos: fraction of type-labeled (non-"O")
              segments that had at least one same-type "positive" partner
              elsewhere in their micro-batch, averaged over accumulated
              steps. Low values -> batch too small / too few segments per
              type for the loss to have anything to work with.
        """
        stats = {}
        if self._supcon_loss_count > 0:
            stats["avg_supcon_loss"] = self._supcon_loss_sum / self._supcon_loss_count
        else:
            stats["avg_supcon_loss"] = None
        if self._supcon_anchor_total > 0:
            stats["frac_anchors_with_pos"] = self._supcon_anchor_with_pos / self._supcon_anchor_total
        else:
            stats["frac_anchors_with_pos"] = None

        self._supcon_loss_sum = 0.0
        self._supcon_loss_count = 0
        self._supcon_anchor_total = 0
        self._supcon_anchor_with_pos = 0
        return stats

    def _compute_supcon_loss(self, vecs, type_ids):
        """
        vecs:     (N, H) float tensor, WITH gradient.
        type_ids: (N,) long tensor -- entity-type group id per vector.

        Returns (loss, frac_anchors_with_positive). loss is None if N < 2
        or no anchor has a same-type partner in this batch.
        """
        N = vecs.shape[0]
        if N < 2:
            return None, 0.0

        z = F.normalize(vecs, p=2, dim=-1)
        sim = torch.matmul(z, z.t()) / self.supcon_temperature  # (N, N)

        same_type = type_ids.unsqueeze(0) == type_ids.unsqueeze(1)  # (N, N)
        self_mask = torch.eye(N, dtype=torch.bool, device=vecs.device)
        positive_mask = same_type & (~self_mask)

        sim_for_denom = sim.masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_for_denom, dim=1, keepdim=True)
        log_prob = sim - log_denom

        pos_count = positive_mask.sum(dim=1)
        has_positive = pos_count > 0
        if not bool(has_positive.any()):
            return None, 0.0

        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1).float()
        loss = -mean_log_prob_pos[has_positive].mean()

        frac = has_positive.float().mean().item()
        return loss, frac

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, labels_text=None):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only.
        seg_id:      (B, L) long tensor. -1 marks tokens not in any segment.
        labels_text: optional (B, L) long tensor of gold BIO label ids,
                     ALREADY SLICED to the text-only length. Only used
                     (and only when self.training) to collect
                     (segment_vector, gold_type) pairs for SupCon.

        Returns:
            fused_hidden: (B, L, H)
            supcon_vecs: (N, H) tensor WITH gradient, or None.
            supcon_types: (N,) long tensor, or None.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        fused_hidden = text_hidden.clone()

        collect_supcon = (
            self.use_supcon_loss and self.training and labels_text is not None
            and self.type_of_label is not None
        )
        supcon_vec_list = []
        supcon_type_list = []

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
                max_pos = self.segment_position_embedding.num_embeddings
                positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                seg_vecs_with_pos = seg_vecs + self.segment_position_embedding(positions)
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)
                seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)
            else:
                seg_vecs_ctx = seg_vecs

            if collect_supcon:
                for i, mask in enumerate(seg_masks):
                    seg_label_ids = labels_text[b][mask]
                    valid_lbl = seg_label_ids != -100
                    if not bool(valid_lbl.any()):
                        continue
                    rep_label = int(seg_label_ids[valid_lbl][0].item())
                    if rep_label < 0 or rep_label >= self.type_of_label.shape[0]:
                        continue
                    type_id = int(self.type_of_label[rep_label].item())
                    if type_id < 0:
                        continue
                    supcon_vec_list.append(seg_vecs_ctx[i])
                    supcon_type_list.append(type_id)

            for i, mask in enumerate(seg_masks):
                fused_hidden[b, mask] = seg_vecs_ctx[i]

        if supcon_vec_list:
            supcon_vecs = torch.stack(supcon_vec_list, dim=0)
            supcon_types = torch.tensor(supcon_type_list, dtype=torch.long, device=device)
        else:
            supcon_vecs, supcon_types = None, None

        return fused_hidden, supcon_vecs, supcon_types

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

        supcon_vecs = None
        supcon_types = None

        if seg_id is not None:
            labels_text = labels[:, :text_len] if labels is not None else None
            text_hidden, supcon_vecs, supcon_types = self._segment_pool_and_contextualize(
                text_hidden, seg_id, labels_text=labels_text
            )

            if self.use_first_token_embedding:
                is_first = torch.zeros_like(seg_id, dtype=torch.long)
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

            if self.training and self.use_supcon_loss and supcon_vecs is not None:
                supcon_loss, frac_pos = self._compute_supcon_loss(supcon_vecs, supcon_types)
                if supcon_loss is not None:
                    loss = loss + self.supcon_weight * supcon_loss
                    with torch.no_grad():
                        self._supcon_loss_sum += supcon_loss.item()
                        self._supcon_loss_count += 1
                        self._supcon_anchor_total += 1
                        self._supcon_anchor_with_pos += frac_pos

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
