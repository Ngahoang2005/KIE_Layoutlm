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
    An earlier version used the segment's (single) context vector as the
    ONLY key/value for each token's cross-attention query. With exactly
    one key, softmax(x) = 1 regardless of the query -- the attention
    degenerates into a fixed linear transform of the segment vector,
    IDENTICAL for every token in the segment. This defeated the entire
    purpose (each token should be able to decide, from its own content,
    how much and what to read from its segment) while still adding
    trainable Q/K/V/out_proj parameters that only add optimization noise
    -- which is exactly why the earlier version scored slightly WORSE than
    plain segment+position (no token-read).

    Fix: give each token's query a REAL multi-key attention target -- the
    key/value set is now [other tokens in the same segment] PLUS [the
    segment's context vector as one extra "virtual" key]. With >=2 keys,
    softmax is no longer forced to 1, so different tokens can genuinely
    attend differently based on their own content. Segments with only one
    token (no "other tokens" to attend to) fall back to the single-key
    case, which is an unavoidable degenerate case, not a design flaw.

    out_proj is still zero-initialized, so at step 0 this module
    contributes exactly 0 -- forward pass is byte-for-byte identical to
    "no token-read at all" (the A configuration). Training then gradually
    learns how much of the (now genuinely token-dependent) read to use.

  - Instrumentation: this class exposes two introspection hooks used by
    CustomTrainer in run_funsd_cord.py to log, after every eval:
      * segment_context_gate value (how open the inter-segment context
        blend is)
      * token-read attention entropy (normalized [0,1]; ~1 means the
        module hasn't learned to discriminate between keys yet, lower
        means it has) and the average L2 norm of the residual "read"
        vector actually added to token hidden states (near-zero means the
        module is contributing almost nothing regardless of entropy).
    Both are essential to tell apart "hasn't learned anything useful yet"
    from "learned something but it's not helping" during tuning.

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone. It only ADDS an optional,
zero-initialized residual read to what the token classifier "sees" -- an
orthogonal mechanism to HGA / GraphLayoutLM.
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
        # FIXED to use real multi-key attention. See module docstring.
        # ================================================================
        self.use_token_segment_read = getattr(config, "use_token_segment_read", True)
        if self.use_token_segment_read:
            read_heads = getattr(config, "token_segment_read_heads", 4)
            self.token_reads_segment = nn.MultiheadAttention(
                embed_dim=config.hidden_size, num_heads=read_heads, batch_first=True,
                dropout=getattr(config, "token_segment_read_dropout", 0.0),
            )
            # Zero-init: at step 0 the read contributes exactly 0, so
            # behavior is identical to the model with this module absent.
            nn.init.zeros_(self.token_reads_segment.out_proj.weight)
            nn.init.zeros_(self.token_reads_segment.out_proj.bias)
        else:
            self.token_reads_segment = None

        # ---- Instrumentation accumulators (reset after each eval) ----
        # Attention entropy: mean, over all (token, segment) pairs with
        # >=2 keys (single-key segments are skipped -- their entropy is
        # trivially 0 / undefined and would just dilute the signal).
        self._read_entropy_sum = 0.0
        self._read_entropy_count = 0
        # Average L2 norm of the residual "read" vector actually added to
        # each token's hidden state -- tells you whether the module is
        # contributing anything in absolute magnitude, independent of how
        # "sharp" or "flat" its attention pattern is.
        self._read_norm_sum = 0.0
        self._read_norm_count = 0
        # Fraction of segments that hit the degenerate single-key case
        # (n_tok == 1) -- if this is very high, most of the dataset's
        # segments simply can't benefit from this module at all.
        self._single_key_seg_count = 0
        self._total_seg_count = 0

        self.init_weights()

    def get_segment_gate_value(self):
        """Optional introspection hook. Returns None if
        segment_context_layers == 0 (no gate exists)."""
        if self.segment_context_gate is None:
            return None
        return self.segment_context_gate.detach().float().item()

    def get_and_reset_token_read_stats(self):
        """Returns a dict of token-read diagnostics accumulated since the
        last reset, then resets the accumulators. Call this after
        trainer.evaluate(). Values are None where nothing was accumulated
        (e.g. use_token_segment_read=False, or seg_id was never passed).
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
        if self._total_seg_count > 0:
            stats["frac_single_key_segments"] = self._single_key_seg_count / self._total_seg_count
        else:
            stats["frac_single_key_segments"] = None

        self._read_entropy_sum = 0.0
        self._read_entropy_count = 0
        self._read_norm_sum = 0.0
        self._read_norm_count = 0
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
                Tokens in the same segment are NOT forced identical: each
                keeps its own base hidden state.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
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
                # Mean-pool to build the segment's SUMMARY vector (unchanged).
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

                    self._total_seg_count += 1

                    if n_tok == 1:
                        # Degenerate case: no "other tokens" to attend to.
                        # Fall back to the single key (segment context
                        # vector). Entropy is trivially undefined here, so
                        # it's excluded from the entropy stat, but counted
                        # separately as frac_single_key_segments.
                        kv = seg_ctx_kv
                        self._single_key_seg_count += 1
                        read, _ = self.token_reads_segment(
                            tokens, kv, kv, need_weights=False
                        )
                    else:
                        # Real multi-key attention: other tokens in the
                        # segment + the context vector as one extra key.
                        kv = torch.cat([tokens, seg_ctx_kv], dim=1)  # (1, n_tok+1, H)
                        read, attn_weights = self.token_reads_segment(
    tokens, kv, kv, need_weights=True
)
                        # attn_weights: (1, n_tok, n_tok+1) after averaging
                        # over heads. Log normalized entropy per query row.
                        with torch.no_grad():
                            w = attn_weights.squeeze(0)  # (n_tok, n_tok+1)
                            ent = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)  # (n_tok,)
                            max_ent = torch.log(torch.tensor(float(w.shape[-1]), device=device))
                            norm_ent = (ent / max_ent.clamp_min(1e-8))
                            self._read_entropy_sum += norm_ent.sum().item()
                            self._read_entropy_count += norm_ent.numel()

                    with torch.no_grad():
                        read_norm = read.squeeze(0).norm(dim=-1)  # (n_tok,)
                        self._read_norm_sum += read_norm.sum().item()
                        self._read_norm_count += read_norm.numel()

                    fused_hidden[b, mask] = text_hidden[b, mask] + read.squeeze(0)
            else:
                # Fallback: old hard-broadcast behavior (ablation / back-compat).
                for i, mask in enumerate(seg_masks):
                    fused_hidden[b, mask] = seg_vecs_ctx[i]

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

        if seg_id is not None:
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
