#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

... (phần docstring gốc giữ nguyên) ...

FIXED (this revision): the earlier CRF integration only used the CRF's
NLL as an AUXILIARY loss term -- gradient reshaped segment_context, but
the final `logits` (what compute_metrics/argmax actually see) never used
the CRF's beliefs at all. Empirically this measured crf_type_accuracy as
high as 97.6% while eval_f1 stayed BELOW the no-CRF baseline: the CRF was
learning transitions perfectly well, but that knowledge never reached the
decision the classifier makes.

Fix: compute the CRF's per-position log-MARGINAL probability P(type_t = k
| whole segment-type sequence) via a differentiable forward-backward
pass, then ADD this (scaled by a zero-initialized scalar gate, so at step
0 logits are byte-for-byte unchanged from baseline) directly into the
corresponding B-/I- label columns of `logits`, broadcast to every token in
that segment. Now the CRF's structural knowledge directly participates in
what argmax picks, both at eval/predict time and at train time (where the
resulting gradient flows back through the marginals into the CRF's own
transition/emission parameters -- a more direct training signal than the
old NLL-only auxiliary loss, which is kept as an additional term).
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
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags[batch_idx, seq_ends]
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_alpha(self, emissions, mask):
        B, L, K = emissions.shape
        alpha = torch.zeros(B, L, K, device=emissions.device, dtype=emissions.dtype)
        alpha[:, 0] = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, L):
            broadcast_alpha = alpha[:, t - 1].unsqueeze(2)             # (B,K,1)
            scores = broadcast_alpha + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)  # (B,K,K)
            new_alpha = torch.logsumexp(scores, dim=1)                 # (B,K)
            m = mask[:, t].unsqueeze(1)
            alpha[:, t] = torch.where(m, new_alpha, alpha[:, t - 1])
        return alpha

    def _normalizer(self, emissions, mask):
        alpha = self._forward_alpha(emissions, mask)
        B = emissions.shape[0]
        seq_ends = mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(B, device=emissions.device)
        final = alpha[batch_idx, seq_ends] + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(final, dim=1)  # (B,)

    def neg_log_likelihood(self, emissions, tags, mask):
        gold_score = self._score(emissions, tags, mask)
        log_partition = self._normalizer(emissions, mask)
        return (log_partition - gold_score).mean()

    def marginals(self, emissions, mask):
        """
        Differentiable log P(type_t = k | full sequence) via forward-backward.
        Returns (B, L, K). Only valid at positions where mask is True.
        """
        B, L, K = emissions.shape
        device = emissions.device
        alpha = self._forward_alpha(emissions, mask)  # (B, L, K)

        seq_ends = mask.long().sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(B, device=device)
        normalizer = torch.logsumexp(
            alpha[batch_idx, seq_ends] + self.end_transitions.unsqueeze(0), dim=1
        )  # (B,)

        beta = torch.zeros(B, L, K, device=device, dtype=emissions.dtype)
        beta[batch_idx, seq_ends] = self.end_transitions.unsqueeze(0).expand(B, K)
        for t in range(L - 2, -1, -1):
            broadcast_beta_next = beta[:, t + 1].unsqueeze(1)          # (B,1,K) -- "to" tag at t+1
            emissions_next = emissions[:, t + 1].unsqueeze(1)          # (B,1,K)
            scores = self.transitions.unsqueeze(0) + emissions_next + broadcast_beta_next  # (B,K,K) [from,to]
            new_beta = torch.logsumexp(scores, dim=2)                  # (B,K)
            valid_t = (t < seq_ends).unsqueeze(1)
            m = mask[:, t].unsqueeze(1) & valid_t
            beta[:, t] = torch.where(m, new_beta, beta[:, t])

        log_marginal = alpha + beta - normalizer.view(B, 1, 1)
        return log_marginal  # (B, L, K)

    @torch.no_grad()
    def decode(self, emissions, mask):
        B, L, num_tags = emissions.shape
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emit
            best_score, best_idx = next_score.max(dim=1)
            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_idx)
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

        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)

        if seg_ctx_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=seg_ctx_heads,
                dim_feedforward=config.hidden_size * 2, dropout=seg_ctx_dropout, batch_first=True,
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
        # CRF: NLL auxiliary loss (unchanged) + NEW logit-fusion via
        # zero-init gate (this is the actual fix).
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

            # NEW: gate controlling how much of the CRF's log-marginal is
            # fused into the final logits. Zero-init -> at step 0 logits
            # are IDENTICAL to the no-CRF baseline.
            self.crf_logit_fusion_gate = nn.Parameter(torch.zeros(1))

            # NEW: precompute, for each type_id, the list of label_ids
            # (both B- and I- variants) that belong to it, so the marginal
            # bonus for type k can be scattered into all matching label
            # columns of `logits` at once.
            label_ids_for_type = [[] for _ in range(n_types)]
            for label_id, t_id in enumerate(type_of_label.tolist()):
                if t_id >= 0:
                    label_ids_for_type[t_id].append(label_id)
            self._label_ids_for_type = label_ids_for_type  # python list, not a tensor (ragged)

            print(
                f"[CRF init] parsed {len(type_vocab)} entity type(s): {type_vocab}. "
                f"If empty/wrong, config.id2label was not set before from_pretrained()."
            )
        else:
            self.type_of_label = None
            self.type_vocab = []
            self.segment_type_classifier = None
            self.crf = None
            self.crf_logit_fusion_gate = None
            self._label_ids_for_type = []

        self._crf_loss_sum = 0.0
        self._crf_loss_count = 0
        self._crf_correct = 0
        self._crf_total = 0

        self.init_weights()

    @staticmethod
    def _build_type_of_label(config):
        id2label = getattr(config, "id2label", None)
        num_labels = config.num_labels
        if not id2label or len(id2label) != num_labels:
            return torch.full((num_labels,), -1, dtype=torch.long), []
        items = sorted(((int(k), v) for k, v in id2label.items()), key=lambda kv: kv[0])
        type_vocab, type_index = [], {}
        type_of_label = [-1] * num_labels
        for label_id, label_str in items:
            s = str(label_str)
            if s == "O" or s.upper() == "OTHER":
                continue
            type_name = s[2:] if (s.startswith("B-") or s.startswith("I-")) else s
            if type_name in ("", "O"):
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

    def get_crf_fusion_gate_value(self):
        if self.crf_logit_fusion_gate is None:
            return None
        return self.crf_logit_fusion_gate.detach().float().item()

    def get_and_reset_crf_stats(self):
        stats = {
            "avg_crf_loss": (self._crf_loss_sum / self._crf_loss_count) if self._crf_loss_count > 0 else None,
            "crf_type_accuracy": (self._crf_correct / self._crf_total) if self._crf_total > 0 else None,
        }
        self._crf_loss_sum = 0.0
        self._crf_loss_count = 0
        self._crf_correct = 0
        self._crf_total = 0
        return stats

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, labels_text=None):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        collect_crf = self.use_crf_loss and labels_text is not None and self.type_of_label is not None
        per_doc_emissions, per_doc_tags = [], []
        # NEW: keep, per doc, the token-mask of each segment that entered
        # the CRF sequence, so we can later scatter the marginal bonus
        # back onto those tokens' `logits`.
        per_doc_seg_masks = []

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

            if collect_crf:
                doc_types, doc_emission_idxs, doc_masks_for_crf = [], [], []
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
                    doc_types.append(type_id)
                    doc_emission_idxs.append(i)
                    doc_masks_for_crf.append((b, mask))  # remember which (doc, token-mask) this segment maps to

                if len(doc_types) >= 1:
                    doc_seg_vecs = seg_vecs_ctx[doc_emission_idxs]
                    doc_emissions = self.segment_type_classifier(doc_seg_vecs)
                    per_doc_emissions.append(doc_emissions)
                    per_doc_tags.append(torch.tensor(doc_types, dtype=torch.long, device=device))
                    per_doc_seg_masks.append(doc_masks_for_crf)

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        crf_batch = None
        if per_doc_emissions:
            max_n = max(e.shape[0] for e in per_doc_emissions)
            n_types = per_doc_emissions[0].shape[1]
            n_docs = len(per_doc_emissions)
            padded_emissions = torch.zeros(n_docs, max_n, n_types, device=device, dtype=per_doc_emissions[0].dtype)
            padded_tags = torch.zeros(n_docs, max_n, dtype=torch.long, device=device)
            crf_mask = torch.zeros(n_docs, max_n, dtype=torch.bool, device=device)
            for i, (emis, tags) in enumerate(zip(per_doc_emissions, per_doc_tags)):
                n_i = emis.shape[0]
                padded_emissions[i, :n_i] = emis
                padded_tags[i, :n_i] = tags
                crf_mask[i, :n_i] = True
            crf_batch = {
                "emissions": padded_emissions, "tags": padded_tags, "mask": crf_mask,
                "seg_masks": per_doc_seg_masks,  # list (per crf-doc-index) of [(b, token_mask), ...]
            }

        return broadcast_hidden, crf_batch

    def _fuse_crf_marginals_into_logits(self, logits, crf_batch):
        """
        Adds crf_logit_fusion_gate * log_marginal[type] into every label
        column belonging to that type, at every token of the corresponding
        segment. logits is modified via non-in-place ops to keep autograd
        clean (returns a new tensor).
        """
        log_marginal = self.crf.marginals(crf_batch["emissions"], crf_batch["mask"])  # (n_crf_docs, maxN, n_types)
        bonus = self.crf_logit_fusion_gate * log_marginal  # (n_crf_docs, maxN, n_types)

        fused = logits
        for crf_doc_idx, seg_list in enumerate(crf_batch["seg_masks"]):
            for seg_pos, (b, token_mask) in enumerate(seg_list):
                for type_id, label_ids in enumerate(self._label_ids_for_type):
                    if not label_ids:
                        continue
                    add_val = bonus[crf_doc_idx, seg_pos, type_id]
                    idx = torch.tensor(label_ids, device=logits.device)
                    fused = fused.clone()
                    fused[b, token_mask][:, idx] = fused[b, token_mask][:, idx] + add_val
        return fused

    def forward(
        self, input_ids=None, bbox=None, attention_mask=None, token_type_ids=None,
        position_ids=None, valid_span=None, head_mask=None, inputs_embeds=None,
        labels=None, seg_id=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, images=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, images=images, valid_span=valid_span,
        )

        sequence_output = outputs[0]
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        crf_batch = None
        if seg_id is not None:
            labels_text = labels[:, :text_len] if labels is not None else None
            text_hidden, crf_batch = self._segment_pool_and_contextualize(text_hidden, seg_id, labels_text=labels_text)

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

        # ---- NEW: fuse CRF marginals into logits (train AND eval) ----
        if self.use_crf_loss and crf_batch is not None:
            logits = self._fuse_crf_marginals_into_logits(logits, crf_batch)

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

            if self.use_crf_loss and crf_batch is not None:
                emissions, tags, mask = crf_batch["emissions"], crf_batch["tags"], crf_batch["mask"]
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
                        self._crf_correct += sum(1 for a, b in zip(path, gold) if a == b)
                        self._crf_total += n_i

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
