#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

[PATCH v4 -- dua tren ket qua ablation T1-T6 + Run D/E, chuyen han sang
GOLD annotation, khong con lo phong thu truoc segmentation nhieu]

Hai cai tien rut ra truc tiep tu du lieu ablation:

  1. THAY position embedding theo THU TU DOC (segment_position_embedding,
     index bang arange(n_seg)) bang 2D SPATIAL EMBEDDING theo TOA DO TAM
     THAT cua tung segment.
     Ly do: T4 (xao thu tu segment luc test) -> F1 khong doi (91.41+-0.09).
     Run E (bo han position embedding luc train) -> F1 giam 0.99 diem.
     => model dang dung no nhu THE DINH DANH pha doi xung, KHONG dung nhu
     thu tu doc thuc su. Doi sang toa do that tan dung dung tin hieu hinh
     hoc (segment nay o tren/duoi/trai/phai segment kia), thay vi lang phi
     slot tham so cho 1 tin hieu ma model khong dung dung muc dich.

  2. LENGTH-CONDITIONAL GATE: nhan segment_context_gate voi 1 he so phu
     thuoc SO TU trong segment (sigmoid theo threshold/slope hoc duoc).
     Ly do: case study cho thay Transformer sua dung 50 token nhung lam
     sai them 52 -- over-smoothing khong chon loc, ap dung deu cho moi
     segment ke ca segment ngan von da du tin hieu per-token. Length-
     conditional gate cho phep model tu hoc "segment ngan -> tin it vao
     context, segment dai -> tin nhieu" thay vi 1 gate scalar toan cuc.
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

            n_buckets = getattr(config, "segment_spatial_buckets", 1024)
            self.segment_x_embedding = nn.Embedding(n_buckets, config.hidden_size)
            self.segment_y_embedding = nn.Embedding(n_buckets, config.hidden_size)
            nn.init.normal_(self.segment_x_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.segment_y_embedding.weight, mean=0.0, std=0.02)
            self._spatial_n_buckets = n_buckets
        else:
            self.segment_context = None
            self.segment_context_gate = None
            self.segment_x_embedding = None
            self.segment_y_embedding = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        if seg_ctx_layers > 0:
            # PATCH: hieu chinh lai gia tri khoi tao theo so lieu thuc te tu
            # probe_segment_context.py -- median segment SUA DUNG = 9.0 tu,
            # median segment LAM SAI = 5.0 tu. Threshold=3.0 (gia tri doan
            # truoc, CHUA co so lieu) dat qua thap: tai seg_len=5 (dung
            # nhom can bi chan), sigmoid((5-3)*1)=0.88 -- gan nhu KHONG chan
            # gi ca, di nguoc lai muc dich thiet ke. Doi threshold ve diem
            # giua 2 median (7.0) de co chan dung nhom can chan tu dau,
            # thay vi bat model tu hoc lai tu 1 diem khoi tao sai lech xa.
            self.seg_len_gate_threshold = nn.Parameter(torch.tensor(7.0))
            self.seg_len_gate_slope = nn.Parameter(torch.tensor(1.0))
        else:
            self.seg_len_gate_threshold = None
            self.seg_len_gate_slope = None

        self.init_weights()

    def _bbox_to_bucket(self, coord_0_1000):
        idx = (coord_0_1000.clamp(0, 1000) / 1000.0 * (self._spatial_n_buckets - 1)).long()
        return idx

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, bbox):
        """bbox: (B, L, 4) [x0,y0,x1,y1] scale 0-1000, phan TEXT (khop voi
        text_hidden), dung de tinh tam moi segment."""
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_lens = torch.zeros(n_seg, device=device, dtype=text_hidden.dtype)
            seg_cx = torch.zeros(n_seg, device=device, dtype=torch.float32)
            seg_cy = torch.zeros(n_seg, device=device, dtype=torch.float32)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)
                seg_lens[i] = mask.sum().float()
                member_boxes = bbox[b, mask].float()
                cx = (member_boxes[:, 0] + member_boxes[:, 2]) / 2
                cy = (member_boxes[:, 1] + member_boxes[:, 3]) / 2
                seg_cx[i] = cx.mean()
                seg_cy[i] = cy.mean()

            if self.segment_context is not None:
                x_bucket = self._bbox_to_bucket(seg_cx)
                y_bucket = self._bbox_to_bucket(seg_cy)
                spatial_embed = self.segment_x_embedding(x_bucket) + self.segment_y_embedding(y_bucket)
                seg_vecs_with_pos = seg_vecs + spatial_embed

                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)

                length_factor = torch.sigmoid(
                    (seg_lens - self.seg_len_gate_threshold) * self.seg_len_gate_slope
                )
                effective_gate = self.segment_context_gate * length_factor.unsqueeze(-1)
                seg_vecs_ctx = seg_vecs + effective_gate * (ctx_out - seg_vecs)
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
            bbox_text = bbox[:, :text_len, :] if bbox is not None else None
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id, bbox_text)

            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0
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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
