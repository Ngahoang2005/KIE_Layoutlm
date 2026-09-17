#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

[PATCH v6 -- PHUONG AN CUOI, sau khi 2 huong truoc that bai co kiem chung]

Lich su ngan gon (de nguoi doc sau nay hieu vi sao thiet ke nhu the nay):
  - v3 (nen goc): order-based position embedding + segment_context_gate
    (scalar). F1 = 91.41, TOT NHAT trong moi lan thu.
  - v4 (2D spatial embedding 1024 bucket + length-conditional gate):
    F1 GIAM con 90.08. Nghi ngo qua nhieu tham so moi (~1.6M) tren 149
    van ban train.
  - v5 (co lap 2 thay doi, giam bucket con 32): CA HAI VAN GIAM rieng le
    (spatial-only: 91.20, length-gate-only: 90.99), cong lai giam manh
    hon (90.56). Debug print xac nhan: voi threshold=7 (hieu chinh tu case
    study), sigmoid(length) BAO HOA GAN 0 cho ~90% segment (FUNSD da so
    dai 1-6 tu) -> gan nhu khong co gradient huu ich, threshold/slope gan
    nhu dung yen suot 1000 step (6.98->6.87). Ket luan: length-conditional
    gate SAI VE BAN CHAT co che (khong phai sai so, ma sigmoid tren 1 dac
    trung vo huong khong phu hop voi phan phoi lech cua FUNSD).
  - v6 (BAN NAY): quay ve nen v3 (da kiem chung tot nhat), BO HAN length-
    conditional gate. Thay vao do, THEM 1 co che MOI o vi tri KHAC: GRUGate
    -- residual connection giua per-token hidden GOC va pooled_hidden (sau
    segment pooling + Transformer context), dua tren GTrXL (Parisotto et
    al., ICML 2020, "Stabilizing Transformers for Reinforcement Learning").
    Khac biet cot loi: gate la VECTOR H-chieu hoc TU NOI DUNG bieu dien
    (ca x va y), khong phai 1 scalar/1 dac trung be mat (do dai) nhu truoc
    -- tranh dung loi bao hoa da gap. Thu gon bang bottleneck (kieu
    Adapter, Houlsby et al. 2019) de kiem soat so tham so moi.
    Dong thoi GIAI QUYET dung van de goc: doc file document 17 (ban goc)
    cho thay KHONG HE co residual ve per_token_hidden -- pooling luon bi
    AP DAT CUNG NHUC bat ke co loi hay khong, ke ca voi GOLD segmentation
    (case study 42 sua/38 sai xac nhan dieu nay). GRUGate cho phep model
    TU QUYET DINH moi token co nen tin pooled_hidden hay khong, dua tren
    noi dung thuc te, thay vi ep buoc.
"""
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)


def _is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


class GRUGate(nn.Module):
    """GRU-style gated residual connection (Parisotto et al., ICML 2020,
    "Stabilizing Transformers for Reinforcement Learning" -- GTrXL). Thu
    gon bang bottleneck (kieu Adapter, Houlsby et al. 2019) de kiem soat
    so tham so moi tren tap du lieu nho (149 van ban).

    QUAN TRONG: gate `z` la VECTOR H-chieu, tinh tu NOI DUNG thuc su cua
    x (per-token hidden goc) va y (pooled_hidden sau segment context) --
    khong phai 1 scalar, khong dua vao 1 dac trung be mat nhu do dai.
    Day la diem khac biet cot loi so voi length-conditional gate da that
    bai (xem lich su o dau file).
    """

    def __init__(self, hidden_size, bottleneck=64, bias_init=2.0):
        super().__init__()
        self.down_r = nn.Linear(hidden_size * 2, bottleneck, bias=False)
        self.up_r = nn.Linear(bottleneck, hidden_size, bias=False)
        self.down_z = nn.Linear(hidden_size * 2, bottleneck, bias=False)
        self.up_z = nn.Linear(bottleneck, hidden_size, bias=False)
        self.down_g = nn.Linear(hidden_size * 2, bottleneck, bias=False)
        self.up_g = nn.Linear(bottleneck, hidden_size, bias=False)
        # bias_init lon (~2) -> sigmoid(... - bg) ban dau GAN 0 -> z gan 0
        # -> output gan bang x (per_token_hidden) -- AN TOAN luc khoi tao
        # (giong tinh than token_gate=0 truoc day, nhung o day la per-dim,
        # hoc duoc tu noi dung thay vi 1 scalar co dinh).
        self.bg = nn.Parameter(torch.full((hidden_size,), bias_init))

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        r = torch.sigmoid(self.up_r(self.down_r(xy)))
        z = torch.sigmoid(self.up_z(self.down_z(xy)) - self.bg)
        gated_input = torch.cat([r * x, y], dim=-1)
        h_hat = torch.tanh(self.up_g(self.down_g(gated_input)))
        return (1 - z) * x + z * h_hat


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

        self.debug_mode = getattr(config, "segment_debug", False)
        self._debug_printed = False

        # ---- Nen v3 (da kiem chung tot nhat): order-based position embedding ----
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

        # ---- PATCH v6: GRUGate thay cho viec ap dat pooling vo dieu kien ----
        self.use_token_gru_gate = getattr(config, "segment_use_token_gru_gate", True)
        if seg_ctx_layers > 0 and self.use_token_gru_gate:
            bottleneck = getattr(config, "segment_gate_bottleneck", 64)
            self.token_gru_gate = GRUGate(config.hidden_size, bottleneck=bottleneck)
        else:
            self.token_gru_gate = None

        # None | "order" | "membership" -- giu lai hook cho T4/T5-style
        # ablation neu can dung lai, khong anh huong hanh vi mac dinh.
        self.eval_shuffle_mode = None

        self.init_weights()

        if _is_main_process():
            n_gru_params = sum(p.numel() for p in self.token_gru_gate.parameters()) if self.token_gru_gate else 0
            print(
                f"[LayoutLMv3ForSegmentTokenClassification v6] segment_context_layers={seg_ctx_layers} "
                f"use_token_gru_gate={self.use_token_gru_gate} (tham so GRUGate: {n_gru_params:,})"
            )

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
                max_pos = self.segment_position_embedding.num_embeddings
                positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                seg_vecs_with_pos = seg_vecs_input + self.segment_position_embedding(positions)
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

            # ---- PATCH v6: GRUGate thay vi ghi de vo dieu kien ----
            if self.token_gru_gate is not None:
                text_hidden = self.token_gru_gate(per_token_hidden, pooled_hidden)

                if self.debug_mode and not self._debug_printed and _is_main_process():
                    with torch.no_grad():
                        diff_norm = (pooled_hidden - per_token_hidden).norm(dim=-1)
                        out_diff_norm = (text_hidden - per_token_hidden).norm(dim=-1)
                        ratio = (out_diff_norm / (diff_norm + 1e-9)).mean().item()
                    print("=" * 60)
                    print("[DEBUG GRUGate] batch0:")
                    print(f"  ||pooled - per_token|| mean = {diff_norm.mean().item():.4f}")
                    print(f"  ||output - per_token|| mean = {out_diff_norm.mean().item():.4f}")
                    print(f"  ty le output di chuyen ve phia pooled = {ratio:.4f} "
                          f"(0=giu nguyen per_token, 1=dung het pooled)")
                    print("=" * 60)
                    self._debug_printed = True
            else:
                text_hidden = pooled_hidden

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
