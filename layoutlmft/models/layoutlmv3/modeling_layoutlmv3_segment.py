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

  - NEW (token-reads-segment, replaces the old hard broadcast):
    Earlier versions OVERWROTE every token's hidden state with the (shared)
    segment vector -- this destroyed per-token information and required a
    separate "is-first-token" embedding just to let the classifier tell
    B- from I- again. Grounded in two published designs:
      * DSpERT (Zhu et al., ACL Findings 2023, arXiv:2210.04182) shows that
        SHALLOW one-shot pooling of tokens into a span representation is
        "significantly ineffective for long-span entities" -- exactly the
        failure mode observed here (the 114-word "NOTE..." block on FUNSD
        doc 82092117, completely missed by the vanilla model). Their fix:
        treat the span as a QUERY and tokens as KEY/VALUE via cross-attention,
        rather than a single mean/weighted pool.
      * DEPTH (arXiv:2405.07788) keeps ordinary tokens completely unmodified
        in the shared self-attention; only a separate SEGMENT-SUMMARY token
        is constrained to attend within its segment. Regular tokens are
        never overwritten -- they only gain the OPTION to look at the
        segment summary.
    Combining both: each token QUERIES its own segment's vector via a
    single-key cross-attention and ADDS the result as a residual --
    the token's own hidden state is the base, never replaced. The
    attention out_proj is zero-initialized, so at step 0 this read
    contributes exactly 0 -> behavior is byte-for-byte identical to
    "no fusion at all" (the current well-tested backbone). Training then
    gradually learns how much of the segment summary is worth reading
    per token -- there is no global broadcast, no risk of drowning a
    long segment's fine-grained tokens in one shared vector.

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
        # With the new token-reads-segment design, tokens are no longer
        # forced identical within a segment, so the classifier can in
        # principle tell B- from I- from the (untouched) token hidden state
        # alone. Kept ON by default for safety/back-compat; try turning it
        # OFF as an ablation once the new fusion is validated.
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
        # NEW: token-reads-segment cross-attention (DSpERT + DEPTH inspired).
        # Each token QUERIES the (single) vector of its own segment and ADDS
        # the result as a residual -- the token's own hidden state is never
        # replaced. num_heads=1 is deliberate: there is exactly ONE key
        # (the token's own segment vector) per query, so extra heads would
        # just be redundant copies of the same single-key attention.
        # ================================================================
        self.use_token_segment_read = getattr(config, "use_token_segment_read", True)
        if self.use_token_segment_read:
            read_heads = getattr(config, "token_segment_read_heads", 1)
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

        self.init_weights()

    def get_segment_gate_value(self):
        """Optional introspection hook (used by a logging callback, if any).
        Returns None if segment_context_layers == 0 (no gate exists)."""
        if self.segment_context_gate is None:
            return None
        return self.segment_context_gate.detach().float().item()

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
                read from its own segment's context-enriched vector. Unlike
                the old design, tokens in the same segment are NOT forced
                identical: each keeps its own base hidden state.
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
                # Mean-pool to build the segment's SUMMARY vector (this part
                # is unchanged -- only what happens to it afterward differs).
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
                # NEW: per-token residual read, NO overwrite/broadcast.
                for i, mask in enumerate(seg_masks):
                    tokens = text_hidden[b, mask].unsqueeze(0)              # (1, n_tok, H) query
                    seg_kv = seg_vecs_ctx[i].view(1, 1, -1)                  # (1, 1, H) key=value
                    read, _ = self.token_reads_segment(tokens, seg_kv, seg_kv)
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
