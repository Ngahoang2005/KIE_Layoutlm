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
    order) so adjacent segments exchange information.

  - Token-reads-segment (DSpERT + DEPTH inspired), FIXED single-key bug:
    key/value set is [other tokens in the same segment] PLUS [the
    segment's context vector as one extra "virtual" key], so softmax is
    no longer forced to 1 (see earlier single-key version for why that
    degenerated into a fixed, token-independent linear transform).

  - FIXED (this revision): the earlier attempt at magnitude regularization
    accumulated the read-vector norm via `.item()` inside `torch.no_grad()`
    before adding it to the loss -- this DETACHES it from the computation
    graph, so it contributed literally zero gradient and did not constrain
    anything (a "regularization" term that does nothing). Fixed by keeping
    a SEPARATE, gradient-carrying accumulation (a list of tensors, not
    floats) built during `_segment_pool_and_contextualize` and only
    reduced (`.mean()`) at the very end -- this stays attached to the
    graph so `loss.backward()` actually penalizes large injected
    magnitudes. Regularization is applied to the POST-gate quantity
    (token_read_gate * read), i.e. what's actually added to the token's
    hidden state -- not the raw out_proj output -- because that's the only
    quantity that actually affects the model's behavior; if the gate stays
    near 0, out_proj is free to have any internal scale without penalty
    (harmless, since it's gated down to ~0 contribution anyway).

  - FIXED (this revision): the diagnostic accumulators (_read_entropy_sum,
    _read_norm_sum, ...) previously accumulated on EVERY forward() call,
    including ordinary training steps -- so by the time
    get_and_reset_token_read_stats() was called after trainer.evaluate(),
    the returned "eval_*" numbers were actually a mix of ~1000 training
    batches' worth of stats plus a handful of eval batches, NOT a clean
    eval-set-only measurement. Fixed by only accumulating when
    `self.training` is False (i.e. only during trainer.evaluate() /
    trainer.predict() forward passes), so "eval_*" metrics genuinely
    reflect eval-set behavior.

  out_proj is still zero-initialized AND token_read_gate starts at 0
  (belt-and-suspenders), so at step 0 this module contributes exactly 0.

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

        # ---- ablation knob: is-first-token embedding ----
        self.use_first_token_embedding = getattr(config, "use_first_token_embedding", True)
        if self.use_first_token_embedding:
            self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
            nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)
        else:
            self.is_first_token_embedding = None

        # ---- inter-segment context module (unchanged from before) ----
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
        # Token-reads-segment cross-attention (DSpERT + DEPTH inspired),
        # real multi-key attention + separate scale gate + gradient-carrying
        # magnitude regularization. See module docstring.
        # ================================================================
        self.use_token_segment_read = getattr(config, "use_token_segment_read", True)
        if self.use_token_segment_read:
            read_heads = getattr(config, "token_segment_read_heads", 4)
            self.token_reads_segment = nn.MultiheadAttention(
                embed_dim=config.hidden_size, num_heads=read_heads, batch_first=True,
                dropout=getattr(config, "token_segment_read_dropout", 0.0),
            )
            # Separate scalar gate: controls HOW MUCH of the read to use,
            # decoupled from out_proj (which controls the DIRECTION/content
            # of what's read). Starts at 0 (ReZero-style).
            self.token_read_gate = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.token_reads_segment.out_proj.weight)
            nn.init.zeros_(self.token_reads_segment.out_proj.bias)

            # Weight for the (now gradient-carrying) magnitude regularizer
            # applied to the POST-gate injected vector during training.
            self.token_read_norm_reg_weight = getattr(config, "token_read_norm_reg_weight", 0.01)
        else:
            self.token_reads_segment = None
            self.token_read_gate = None
            self.token_read_norm_reg_weight = 0.0

        # ---- Instrumentation accumulators (LOGGING ONLY -- detached,
        # eval-mode-only; see get_and_reset_token_read_stats) ----
        self._read_entropy_sum = 0.0
        self._read_entropy_count = 0
        self._read_norm_sum = 0.0   # post-gate norm, for logging
        self._read_norm_count = 0
        self._read_raw_norm_sum = 0.0  # pre-gate (out_proj) norm, for logging
        self._read_raw_norm_count = 0
        self._single_key_seg_count = 0
        self._total_seg_count = 0

        self.init_weights()

    def get_segment_gate_value(self):
        """Optional introspection hook. Returns None if
        segment_context_layers == 0 (no gate exists)."""
        if self.segment_context_gate is None:
            return None
        return self.segment_context_gate.detach().float().item()

    def get_token_read_gate_value(self):
        """Optional introspection hook. Returns None if
        use_token_segment_read == False (no gate exists)."""
        if self.token_read_gate is None:
            return None
        return self.token_read_gate.detach().float().item()

    def get_and_reset_token_read_stats(self):
        """Returns a dict of token-read diagnostics accumulated since the
        last reset (EVAL-MODE FORWARD PASSES ONLY -- see module docstring
        for why training-time forward passes are excluded), then resets
        the accumulators. Call this after trainer.evaluate().
        """
        stats = {}
        if self._read_entropy_count > 0:
            stats["avg_read_attn_entropy"] = self._read_entropy_sum / self._read_entropy_count
        else:
            stats["avg_read_attn_entropy"] = None
        if self._read_norm_count > 0:
            stats["avg_read_vector_norm"] = self._read_norm_sum / self._read_norm_count
        else:
            stats["avg_read_vector_norm"] = None
        if self._read_raw_norm_count > 0:
            stats["avg_read_raw_norm"] = self._read_raw_norm_sum / self._read_raw_norm_count
        else:
            stats["avg_read_raw_norm"] = None
        if self._total_seg_count > 0:
            stats["frac_single_key_segments"] = self._single_key_seg_count / self._total_seg_count
        else:
            stats["frac_single_key_segments"] = None

        self._read_entropy_sum = 0.0
        self._read_entropy_count = 0
        self._read_norm_sum = 0.0
        self._read_norm_count = 0
        self._read_raw_norm_sum = 0.0
        self._read_raw_norm_count = 0
        self._single_key_seg_count = 0
        self._total_seg_count = 0
        return stats

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only.
        seg_id:      (B, L) long tensor. -1 marks tokens that do not belong
                     to any segment (special tokens / padding). Non-negative
                     values are LOCAL segment indices per example, assigned
                     in reading order (0, 1, 2, ...), matching the
                     bbox-equality grouping in run_funsd_cord.py's
                     tokenize_and_align_labels.

        Returns:
            fused_hidden: (B, L, H) -- token_hidden PLUS an optional residual
                read from its own segment (other tokens + context vector).
            read_norm_reg_term: scalar tensor (WITH gradient) equal to the
                mean L2 norm of the POST-gate injected read vectors across
                every token that went through the multi-key branch, or
                None if token-read is disabled / no segment had >=2 tokens
                in this batch. Add this (scaled) to the loss during
                training to discourage the module from exploiting large,
                non-selective magnitude as a shortcut.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        fused_hidden = text_hidden.clone()
        read_norms_for_reg = []  # gradient-carrying tensors, reduced at the end

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

            if self.token_reads_segment is not None:
                for i, mask in enumerate(seg_masks):
                    tokens = text_hidden[b, mask].unsqueeze(0)  # (1, n_tok, H) -- query
                    n_tok = tokens.shape[1]
                    seg_ctx_kv = seg_vecs_ctx[i].view(1, 1, -1)  # (1, 1, H)

                    if not self.training:
                        self._total_seg_count += 1

                    if n_tok == 1:
                        kv = seg_ctx_kv
                        if not self.training:
                            self._single_key_seg_count += 1
                        read, _ = self.token_reads_segment(
                            tokens, kv, kv, need_weights=False
                        )
                    else:
                        kv = torch.cat([tokens, seg_ctx_kv], dim=1)  # (1, n_tok+1, H)
                        read, attn_weights = self.token_reads_segment(
                            tokens, kv, kv, need_weights=True
                        )
                        # attn_weights: (1, n_tok, n_tok+1), already averaged
                        # over heads by MultiheadAttention when need_weights=True.
                        if not self.training:
                            with torch.no_grad():
                                w = attn_weights.squeeze(0)  # (n_tok, n_tok+1)
                                ent = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)
                                max_ent = torch.log(torch.tensor(float(w.shape[-1]), device=device))
                                norm_ent = ent / max_ent.clamp_min(1e-8)
                                self._read_entropy_sum += norm_ent.sum().item()
                                self._read_entropy_count += norm_ent.numel()

                    read_sq = read.squeeze(0)  # (n_tok, H) -- pre-gate, WITH gradient
                    gated_read = self.token_read_gate * read_sq  # WITH gradient

                    # ---- gradient-carrying accumulation for regularization ----
                    if self.training:
                        read_norms_for_reg.append(gated_read.norm(dim=-1))

                    # ---- detached, eval-only accumulation for logging ----
                    if not self.training:
                        with torch.no_grad():
                            self._read_raw_norm_sum += read_sq.norm(dim=-1).sum().item()
                            self._read_raw_norm_count += read_sq.shape[0]
                            self._read_norm_sum += gated_read.norm(dim=-1).sum().item()
                            self._read_norm_count += gated_read.shape[0]

                    fused_hidden[b, mask] = text_hidden[b, mask] + gated_read
            else:
                for i, mask in enumerate(seg_masks):
                    fused_hidden[b, mask] = seg_vecs_ctx[i]

        if read_norms_for_reg:
            read_norm_reg_term = torch.cat(read_norms_for_reg).mean()
        else:
            read_norm_reg_term = None

        return fused_hidden, read_norm_reg_term

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

        read_norm_reg_term = None
        if seg_id is not None:
            text_hidden, read_norm_reg_term = self._segment_pool_and_contextualize(text_hidden, seg_id)

            if self.use_first_token_embedding:
                is_first = torch.zeros_like(seg_id, dtype=torch.long)
                if seg_id.shape[1] > 1:
                    prev = seg_id[:, :-1]
                    cur = seg_id[:, 1:]
                    changed = (cur != prev) & (cur >= 0)
                    is_first[:, 1:] = changed.long()
                is_first = is_first * (seg_id >= 0).long()
                text_hidden = text_hidden + self.is_first_token_embedding(is_first)
        # if seg_id is None (e.g. an old checkpoint / different dataloader),
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

            # ---- Magnitude regularization (gradient-carrying, training only) ----
            if self.training and read_norm_reg_term is not None and self.token_read_norm_reg_weight > 0:
                loss = loss + self.token_read_norm_reg_weight * read_norm_reg_term

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
