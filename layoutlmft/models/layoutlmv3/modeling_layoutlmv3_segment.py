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

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone. It only replaces what the
token classifier head "sees" for tokens inside multi-token segments -- an
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

        # ====== NEW: dual-stream classifiers ======
        self.use_dual_stream = getattr(config, "use_dual_stream", True)

        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)          # token-level stream (dùng nếu dual_stream=False, hoặc là "logits_token" nếu True)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        if self.use_dual_stream:
            if config.num_labels < 10:
                self.classifier_seg = nn.Linear(config.hidden_size, config.num_labels)   # segment-context stream, TÁCH RIÊNG trọng số
            else:
                self.classifier_seg = LayoutLMv3ClassificationHead(config, pool_feature=False)
            # scalar học được để trộn 2 luồng logits, init = 0 -> sigmoid=0.5 (cân bằng đều lúc đầu)
            # có thể đổi thành per-label hoặc per-token nếu muốn linh hoạt hơn
            self.fusion_alpha = nn.Parameter(torch.zeros(1))

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

        self.init_weights()
    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        Trả về:
            seg_context_hidden: (B, L, H) - vector segment-context được broadcast (dùng cho nhánh classifier_seg)
            Token gốc (text_hidden) KHÔNG bị đổi, giữ nguyên để dùng cho nhánh classifier token-level.
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        seg_context_hidden = text_hidden.clone()   # sẽ bị ghi đè theo segment, nhưng KHÔNG ảnh hưởng tới text_hidden gốc

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

            for i, mask in enumerate(seg_masks):
                seg_context_hidden[b, mask] = seg_vecs_ctx[i]

        return seg_context_hidden    
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
        line_ids=line_ids,
        block_ids=block_ids,
    )

    sequence_output = outputs[0]
    text_len = input_ids.shape[1]
    text_hidden = sequence_output[:, :text_len, :]     # GIỮ NGUYÊN, không bị đổi bởi segment module nữa
    image_hidden = sequence_output[:, text_len:, :]

    if seg_id is not None:
        if seg_id.shape[1] != text_len:
            if seg_id.shape[1] > text_len:
                seg_id = seg_id[:, :text_len]
            else:
                pad_len = text_len - seg_id.shape[1]
                pad_tensor = torch.ones(seg_id.shape[0], pad_len, device=seg_id.device, dtype=seg_id.dtype) * -1
                seg_id = torch.cat([seg_id, pad_tensor], dim=1)

        # is-first-token vẫn tính như cũ
        is_first = torch.zeros_like(seg_id, dtype=torch.long)
        if seg_id.shape[1] > 1:
            prev = seg_id[:, :-1]
            cur = seg_id[:, 1:]
            changed = (cur != prev) & (cur >= 0)
            is_first[:, 1:] = changed.long()
        is_first = is_first * (seg_id >= 0).long()

        if self.use_dual_stream:
            # ====== NHÁNH 1: token-level thuần, KHÔNG đụng vào seg context ======
            token_stream_hidden = text_hidden  # giữ nguyên, không cộng gì cả

            # ====== NHÁNH 2: segment-context, cộng thêm is-first để phân biệt B-/I- ======
            seg_context_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)
            seg_stream_hidden = seg_context_hidden + self.is_first_token_embedding(is_first)

            logits_token = self.classifier(self.dropout(
                torch.cat([token_stream_hidden, image_hidden], dim=1) if image_hidden.shape[1] > 0 else token_stream_hidden
            ))
            logits_seg = self.classifier_seg(self.dropout(
                torch.cat([seg_stream_hidden, image_hidden], dim=1) if image_hidden.shape[1] > 0 else seg_stream_hidden
            ))

            alpha = torch.sigmoid(self.fusion_alpha)   # scalar 0..1, học được
            logits = alpha * logits_token + (1 - alpha) * logits_seg

        else:
            # baseline cũ: chỉ 1 luồng, hard broadcast + is-first
            text_hidden_ctx = self._segment_pool_and_contextualize(text_hidden, seg_id)
            text_hidden_ctx = text_hidden_ctx + self.is_first_token_embedding(is_first)
            pooled_sequence = torch.cat([text_hidden_ctx, image_hidden], dim=1) if image_hidden.shape[1] > 0 else text_hidden_ctx
            logits = self.classifier(self.dropout(pooled_sequence))
    else:
        pooled_sequence = torch.cat([text_hidden, image_hidden], dim=1) if image_hidden.shape[1] > 0 else text_hidden
        logits = self.classifier(self.dropout(pooled_sequence))

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