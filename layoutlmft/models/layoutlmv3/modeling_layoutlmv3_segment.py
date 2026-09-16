#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

[PATCH v3 -- ablation suite de chung minh Transformer segment_context hoc
duoc gi that, khong chi la nhieu tham so]

Them:
  1. segment_use_position_embedding TACH RIENG khoi segment_context_layers
     -- truoc day tat layers=0 keo tat luon position embedding, gay confound
     khi so sanh ctx=0 vs ctx=1.
  2. self.eval_shuffle_mode (dat TU NGOAI, khong qua forward() kwargs) --
     3 gia tri:
       None          : hanh vi binh thuong
       "order"       : xao THU TU segment truoc khi dua vao Transformer
       "membership"  : xao NGAU NHIEN token nao thuoc segment nao
     CHI dung 2 mode nay o EVAL, khong dung luc train.
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

        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)
        use_pos_embed = getattr(config, "segment_use_position_embedding", True)

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
        else:
            self.segment_context = None
            self.segment_context_gate = None

        if use_pos_embed:
            max_pos = getattr(config, "segment_context_max_positions", 128)
            self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.segment_position_embedding = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        self.token_gate = nn.Parameter(torch.zeros(1))

        # None | "order" | "membership" -- dat tu ben ngoai truoc khi eval
        self.eval_shuffle_mode = None

        self.init_weights()

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        for b in range(B):
            ids = seg_id[b].clone()

            if self.eval_shuffle_mode == "membership" and not self.training:
                valid_mask = ids >= 0
                valid_ids = ids[valid_mask]
                perm = torch.randperm(valid_ids.shape[0], device=device)
                ids[valid_mask] = valid_ids[perm]

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

            order_perm = None
            if self.eval_shuffle_mode == "order" and not self.training and n_seg > 1:
                order_perm = torch.randperm(n_seg, device=device)
                seg_vecs_input = seg_vecs[order_perm]
            else:
                seg_vecs_input = seg_vecs

            if self.segment_context is not None:
                if self.segment_position_embedding is not None:
                    max_pos = self.segment_position_embedding.num_embeddings
                    positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                    seg_vecs_with_pos = seg_vecs_input + self.segment_position_embedding(positions)
                else:
                    seg_vecs_with_pos = seg_vecs_input
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)

                if order_perm is not None:
                    inv_perm = torch.empty_like(order_perm)
                    inv_perm[order_perm] = torch.arange(n_seg, device=device)
                    ctx_out = ctx_out[inv_perm]

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

        if seg_id is not None:
            per_token_hidden = text_hidden
            pooled_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0
            if seg_id.shape[1] > 1:
                prev = seg_id[:, :-1]
                cur = seg_id[:, 1:]
                changed = (cur != prev) & (cur >= 0)
                is_first[:, 1:] = changed.long()
            is_first = is_first * (seg_id >= 0).long()

            valid_mask = (seg_id >= 0).unsqueeze(-1).to(pooled_hidden.dtype)
            pooled_hidden = pooled_hidden + self.is_first_token_embedding(is_first) * valid_mask

            text_hidden = per_token_hidden + self.token_gate * (pooled_hidden - per_token_hidden)

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
