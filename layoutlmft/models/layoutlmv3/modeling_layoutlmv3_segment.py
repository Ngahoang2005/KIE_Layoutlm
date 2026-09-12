#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

Core idea (grounded in error analysis on FUNSD + CORD):
  - Segment self-consistency is already ~98-99% solved by the base model
    -> pool each segment's token hidden states into one vector, run a
    tiny Transformer encoder over the SEQUENCE of segment vectors (reading
    order) so adjacent segments exchange information, then broadcast the
    context-enriched vector back to every token in the segment before the
    (unchanged) token classifier.
  - is_first_token_embedding lets the (shared, broadcast) classifier input
    still distinguish B- from I- despite every token in a segment sharing
    one pooled vector.

  - NEW: Segment-level linear-chain CRF, added as an AUXILIARY loss (does
    NOT replace or touch the existing classifier/logits/CE-loss pipeline).

    Motivation: XY-cut-style reading-order fixes help FUNSD (multi-column
    forms, where segment ORDER itself can be wrong) but do nothing for
    CORD (single-column receipts, already in near-correct order). CORD's
    dominant error mode (confirmed via debug.py: 77%+ of misclassified
    segments are embedding-overlap cases like MENU.ITEMSUBTOTAL vs
    MENU.PRICE) is not an ordering problem -- it's that the model doesn't
    exploit the fact that certain entity TYPES have strong SEQUENTIAL
    dependencies on their neighbors (e.g. ITEMSUBTOTAL almost always
    follows a block of MENU.* segments; on FUNSD, ANSWER almost always
    follows QUESTION, which is exactly what distinguishes QUESTION from
    HEADER). A linear-chain CRF over the SEQUENCE of segment types is the
    standard, dataset-agnostic tool for exactly this: it learns a
    transition matrix P(type_t | type_{t-1}) purely from label co-occurrence
    statistics in the training set, with no hand-crafted rule and no
    knowledge of what the type names even mean -- so it works unchanged on
    FUNSD (3 types), CORD (~23 types), or any other BIO-tagged dataset.

    Design choices (each picked to avoid disturbing the existing working
    pipeline):
      * CRF operates on a NEW, separate segment_type_classifier head (not
        the main token classifier). Its loss is added to the total loss
        with a single small weight (crf_weight) -- exactly one
        hyperparameter to tune, unlike SupCon's two entangled knobs
        (weight/temperature), because a CRF's "temperature" is implicitly
        absorbed into the learned transition/emission scale itself.
      * The CRF sequence per document EXCLUDES "O"/"OTHER" segments (a
        semantically heterogeneous catch-all whose position carries little
        transition signal), so it models only the entity-type subsequence
        in reading order.
      * Main classifier, CE loss, and returned `logits` are COMPLETELY
        UNCHANGED -- CRF's contribution is purely an auxiliary gradient
        shaping segment_context/segment vectors, plus a diagnostic
        (Viterbi-decoded segment-type accuracy) exposed via
        get_and_reset_crf_stats(). This keeps compute_metrics/seqeval
        working exactly as before with zero changes to run_funsd_cord.py's
        evaluation logic.
      * transitions/start/end scores are zero-initialized (neutral prior:
        no transition is favored/disfavored at step 0), while
        segment_type_classifier uses normal (non-zero) initialization,
        since -- unlike token_reads_segment's residual read -- this is a
        genuinely NEW auxiliary head whose whole point is to learn useful
        features from scratch; there is no "must match baseline exactly at
        step 0" requirement here (identical in spirit to how the SupCon
        auxiliary loss was added).

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


class _LinearChainCRF(nn.Module):
    """
    Minimal, self-contained linear-chain CRF (no external dependency on
    torchcrf/pytorch-crf, to avoid install issues on Kaggle/offline runs).
    Standard forward-algorithm NLL + Viterbi decode, batched with a
    boolean mask for variable-length sequences.

    emissions: (B, L, num_tags) float, WITH gradient.
    tags:      (B, L) long, gold tag ids (ignored where mask is False).
    mask:      (B, L) bool, True for valid (non-padding) positions. Every
               sequence must have mask[:, 0] == True (at least one valid
               position), which the caller guarantees by construction.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))  # [from, to]

    def _score(self, emissions, tags, mask):
        B, L = tags.shape
        batch_idx = torch.arange(B, device=tags.device)
        score = self.start_transitions[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]
        for i in range(1, L):
            m = mask[:, i].float()
            trans_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emit_score = emissions[batch_idx, i, tags[:, i]]
            score = score + m * (trans_score + emit_score)
        seq_ends = mask.long().sum(dim=1) - 1  # index of last valid position per sequence
        last_tags = tags[batch_idx, seq_ends]
        score = score + self.end_transitions[last_tags]
        return score

    def _normalizer(self, emissions, mask):
        B, L, _ = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, num_tags)
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)                    # (B, num_tags, 1)
            broadcast_emit = emissions[:, i].unsqueeze(1)            # (B, 1, num_tags)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)          # (B, num_tags)
            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, next_score, score)
        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)  # (B,)

    def neg_log_likelihood(self, emissions, tags, mask):
        gold_score = self._score(emissions, tags, mask)
        log_partition = self._normalizer(emissions, mask)
        return (log_partition - gold_score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi decode. Returns a list of python lists (one per batch
        element), each the predicted tag sequence for its valid length."""
        B, L, num_tags = emissions.shape
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, num_tags)
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emit  # (B, from, to)
            best_score, best_idx = next_score.max(dim=1)  # (B, num_tags) each
            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_idx)  # (B, num_tags): best previous tag for each current tag

        score = score + self.end_transitions.unsqueeze(0)
        seq_ends = mask.long().sum(dim=1) - 1

        best_paths = []
        for b in range(B):
            length = int(seq_ends[b].item()) + 1
            best_last_tag = int(score[b].argmax().item())
            path = [best_last_tag]
            for i in reversed(range(length - 1)):
                best_last_tag = int(history[i][b, path[-1]].item())
                path.append(best_last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths


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

        # ---- inter-segment context module (unchanged) ----
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

        # is-first-token embedding (unchanged) -- keeps token-level B-/I-
        # distinction working exactly as before.
        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        # ================================================================
        # NEW: segment-level CRF (auxiliary loss only -- see docstring).
        # ================================================================
        self.use_crf_loss = getattr(config, "use_crf_loss", True)
        self.crf_weight = getattr(config, "crf_weight", 0.1)

        if self.use_crf_loss:
            type_of_label, type_vocab = self._build_type_of_label(config)
            self.register_buffer("type_of_label", type_of_label, persistent=True)
            self.type_vocab = type_vocab
            n_types = max(len(type_vocab), 1)
            self.segment_type_classifier = nn.Linear(config.hidden_size, n_types)
            self.crf = _LinearChainCRF(n_types)
            print(
                f"[CRF init] parsed {len(type_vocab)} entity type(s) for segment-level CRF: {type_vocab}. "
                f"If this list looks empty or wrong, config.id2label was not set correctly before "
                f"from_pretrained() -- CRF will silently no-op in that case."
            )
        else:
            self.type_of_label = None
            self.type_vocab = []
            self.segment_type_classifier = None
            self.crf = None

        # ---- diagnostics accumulators ----
        self._crf_loss_sum = 0.0
        self._crf_loss_count = 0
        self._crf_correct = 0
        self._crf_total = 0

        self.init_weights()

    @staticmethod
    def _build_type_of_label(config):
        """Same parsing logic used for the earlier SupCon experiment:
        label_id -> type_id (B-/I- stripped, "O"/"OTHER" -> -1, excluded)."""
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
        if self.segment_context_gate is None:
            return None
        return self.segment_context_gate.detach().float().item()

    def get_and_reset_crf_stats(self):
        """Returns dict with avg_crf_loss and crf_type_accuracy (Viterbi
        decode vs gold, at SEGMENT granularity) since the last reset, then
        resets accumulators. crf_type_accuracy is computed both at train
        and eval time (unlike the loss, which only affects training),
        because it's the clearest signal of whether the CRF has actually
        learned useful transition structure -- watch this number, not just
        the loss value, when deciding on crf_weight.
        """
        stats = {}
        stats["avg_crf_loss"] = (self._crf_loss_sum / self._crf_loss_count) if self._crf_loss_count > 0 else None
        stats["crf_type_accuracy"] = (self._crf_correct / self._crf_total) if self._crf_total > 0 else None

        self._crf_loss_sum = 0.0
        self._crf_loss_count = 0
        self._crf_correct = 0
        self._crf_total = 0
        return stats

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, labels_text=None):
        """
        Returns:
            broadcast_hidden: (B, L, H) -- unchanged behavior.
            crf_batch: optional dict with keys "emissions" (B', maxN, n_types),
                "tags" (B', maxN), "mask" (B', maxN) ready to feed the CRF,
                built ONLY from documents that have >=1 non-"O" segment.
                None if CRF disabled / no labels / nothing to collect.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        collect_crf = (
            self.use_crf_loss and labels_text is not None and self.type_of_label is not None
        )
        per_doc_emissions = []  # list of (n_i, n_types) tensors, WITH gradient
        per_doc_tags = []       # list of (n_i,) long tensors

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

            if collect_crf:
                doc_types = []
                doc_emission_idxs = []
                for i, mask in enumerate(seg_masks):
                    seg_label_ids = labels_text[b][mask]
                    valid_lbl = seg_label_ids != -100
                    if not bool(valid_lbl.any()):
                        continue
                    rep_label = int(seg_label_ids[valid_lbl][0].item())
                    if rep_label < 0 or rep_label >= self.type_of_label.shape[0]:
                        continue
                    type_id = int(self.type_of_label[rep_label].item())
                    if type_id < 0:  # "O"/"OTHER"/unknown -- excluded from the CRF sequence
                        continue
                    doc_types.append(type_id)
                    doc_emission_idxs.append(i)

                if len(doc_types) >= 1:
                    doc_seg_vecs = seg_vecs_ctx[doc_emission_idxs]  # (n_i, H), WITH gradient
                    doc_emissions = self.segment_type_classifier(doc_seg_vecs)  # (n_i, n_types)
                    per_doc_emissions.append(doc_emissions)
                    per_doc_tags.append(torch.tensor(doc_types, dtype=torch.long, device=device))

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        crf_batch = None
        if per_doc_emissions:
            max_n = max(e.shape[0] for e in per_doc_emissions)
            n_types = per_doc_emissions[0].shape[1]
            n_docs = len(per_doc_emissions)
            padded_emissions = torch.zeros(n_docs, max_n, n_types, device=device, dtype=per_doc_emissions[0].dtype)
            padded_tags = torch.zeros(n_docs, max_n, dtype=torch.long, device=device)
            mask = torch.zeros(n_docs, max_n, dtype=torch.bool, device=device)
            for i, (emis, tags) in enumerate(zip(per_doc_emissions, per_doc_tags)):
                n_i = emis.shape[0]
                padded_emissions[i, :n_i] = emis
                padded_tags[i, :n_i] = tags
                mask[i, :n_i] = True
            crf_batch = {"emissions": padded_emissions, "tags": padded_tags, "mask": mask}

        return broadcast_hidden, crf_batch

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

        crf_batch = None
        if seg_id is not None:
            labels_text = labels[:, :text_len] if labels is not None else None
            text_hidden, crf_batch = self._segment_pool_and_contextualize(
                text_hidden, seg_id, labels_text=labels_text
            )

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

            # ---- CRF: NLL loss (training only) + Viterbi-accuracy diagnostic (train & eval) ----
            if self.use_crf_loss and crf_batch is not None:
                emissions = crf_batch["emissions"]
                tags = crf_batch["tags"]
                mask = crf_batch["mask"]

                if self.training:
                    crf_nll = self.crf.neg_log_likelihood(emissions, tags, mask)
                    loss = loss + self.crf_weight * crf_nll
                    with torch.no_grad():
                        self._crf_loss_sum += crf_nll.item()
                        self._crf_loss_count += 1

                with torch.no_grad():
                    decoded = self.crf.decode(emissions.detach(), mask)
                    for i, path in enumerate(decoded):
                        n_i = int(mask[i].sum().item())
                        gold = tags[i, :n_i].tolist()
                        correct = sum(1 for a, b in zip(path, gold) if a == b)
                        self._crf_correct += correct
                        self._crf_total += n_i

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
