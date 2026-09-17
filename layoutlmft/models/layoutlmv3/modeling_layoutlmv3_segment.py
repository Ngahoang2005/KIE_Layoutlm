#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

[PATCH v5 -- CO LAP 2 thay doi cua v4 (2D spatial embedding, length-
conditional gate) thanh 2 co RIENG BIET, vi v4-full lam F1 GIAM (91.41 ->
90.08, recall giam manh nhat -1.85) va khong biet chinh xac thanh phan nao
gay hai. Gio co the bat/tat tung phan doc lap de co lap nguyen nhan.

Config flags moi (default = GIONG v4 full, doi rieng tung cai khi ablation):
  segment_use_spatial_embed (bool, default True)
      True  -> dung 2D spatial embedding (toa do tam segment, bucket-hoa)
      False -> dung LAI order-based position embedding (ban v3 cu, dua
               vao arange(n_seg))
  segment_use_length_gate (bool, default True)
      True  -> gate = segment_context_gate * sigmoid((seg_len-threshold)*slope)
      False -> gate = segment_context_gate (scalar thuong, ban v3 cu)
  segment_spatial_buckets (int, default 32)
      Da giam tu 1024 (v4) xuong 32 -- 1024 bucket ~1.6M tham so moi qua
      lon so voi 149 van ban train, nghi ngo la nguyen nhan chinh gay F1
      giam (chua hoc noi bieu dien co nghia, phat nhieu thay vi tin hieu).
  segment_debug (bool, default False)
      Bat debug print (shape/gia tri mau) o step dau tien cua forward,
      chi 1 lan, de kiem tra nhanh khong can cho het training.

Giu nguyen eval_shuffle_mode (order/membership) tu ban truoc -- van dung
duoc cho T4/T5-style ablation neu can, khong anh huong hanh vi mac dinh.
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
    # Tranh spam print khi chay DDP nhieu process (--nproc_per_node=2)
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


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

        # ---- Co doc lap (PATCH v5) ----
        self.use_spatial_embed = getattr(config, "segment_use_spatial_embed", True)
        self.use_length_gate = getattr(config, "segment_use_length_gate", True)
        self.debug_mode = getattr(config, "segment_debug", False)
        self._debug_printed = False

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

            # ---- Nhanh A: order-based position embedding (v3 cu) ----
            if not self.use_spatial_embed:
                max_pos = getattr(config, "segment_context_max_positions", 128)
                self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
                nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
                self.segment_x_embedding = None
                self.segment_y_embedding = None
            # ---- Nhanh B: 2D spatial embedding (v4 moi, bucket giam con 32) ----
            else:
                n_buckets = getattr(config, "segment_spatial_buckets", 32)
                self.segment_x_embedding = nn.Embedding(n_buckets, config.hidden_size)
                self.segment_y_embedding = nn.Embedding(n_buckets, config.hidden_size)
                nn.init.normal_(self.segment_x_embedding.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.segment_y_embedding.weight, mean=0.0, std=0.02)
                self._spatial_n_buckets = n_buckets
                self.segment_position_embedding = None
        else:
            self.segment_context = None
            self.segment_context_gate = None
            self.segment_position_embedding = None
            self.segment_x_embedding = None
            self.segment_y_embedding = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        # ---- Length-conditional gate (chi tao tham so khi bat) ----
        if seg_ctx_layers > 0 and self.use_length_gate:
            # threshold=7.0: hieu chinh theo so lieu that (median SUA DUNG=9,
            # median LAM SAI=5, diem giua=7) -- xem thao luan truoc.
            self.seg_len_gate_threshold = nn.Parameter(torch.tensor(7.0))
            self.seg_len_gate_slope = nn.Parameter(torch.tensor(1.0))
        else:
            self.seg_len_gate_threshold = None
            self.seg_len_gate_slope = None

        # None | "order" | "membership" -- dat tu ben ngoai truoc khi eval,
        # dung cho T4/T5-style ablation, khong anh huong hanh vi mac dinh.
        self.eval_shuffle_mode = None

        self.init_weights()

        if _is_main_process():
            print(
                f"[LayoutLMv3ForSegmentTokenClassification] segment_context_layers={seg_ctx_layers} "
                f"use_spatial_embed={self.use_spatial_embed} use_length_gate={self.use_length_gate} "
                f"spatial_buckets={getattr(config, 'segment_spatial_buckets', 32) if self.use_spatial_embed else None}"
            )

    def _bbox_to_bucket(self, coord_0_1000):
        idx = (coord_0_1000.clamp(0, 1000) / 1000.0 * (self._spatial_n_buckets - 1)).long()
        return idx

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, bbox):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        debug_this_call = self.debug_mode and not self._debug_printed and _is_main_process()

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
            seg_lens = torch.zeros(n_seg, device=device, dtype=text_hidden.dtype)
            seg_cx = torch.zeros(n_seg, device=device, dtype=torch.float32)
            seg_cy = torch.zeros(n_seg, device=device, dtype=torch.float32)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)
                seg_lens[i] = mask.sum().float()
                if self.use_spatial_embed and bbox is not None:
                    member_boxes = bbox[b, mask].float()
                    cx = (member_boxes[:, 0] + member_boxes[:, 2]) / 2
                    cy = (member_boxes[:, 1] + member_boxes[:, 3]) / 2
                    seg_cx[i] = cx.mean()
                    seg_cy[i] = cy.mean()

            order_perm = None
            if self.eval_shuffle_mode == "order" and not self.training and n_seg > 1:
                order_perm = torch.randperm(n_seg, device=device)
                seg_vecs_input = seg_vecs[order_perm]
            else:
                seg_vecs_input = seg_vecs

            if self.segment_context is not None:
                # ---- Nhanh A/B: chon nguon "vi tri" theo co use_spatial_embed ----
                if self.use_spatial_embed:
                    x_bucket = self._bbox_to_bucket(seg_cx)
                    y_bucket = self._bbox_to_bucket(seg_cy)
                    pos_embed = self.segment_x_embedding(x_bucket) + self.segment_y_embedding(y_bucket)
                    if order_perm is not None:
                        pos_embed = pos_embed[order_perm]
                else:
                    max_pos = self.segment_position_embedding.num_embeddings
                    positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                    pos_embed = self.segment_position_embedding(positions)

                seg_vecs_with_pos = seg_vecs_input + pos_embed
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)

                if order_perm is not None:
                    inv_perm = torch.empty_like(order_perm)
                    inv_perm[order_perm] = torch.arange(n_seg, device=device)
                    ctx_out = ctx_out[inv_perm]

                # ---- Nhanh A/B: gate thuong hay length-conditional ----
                if self.use_length_gate:
                    length_factor = torch.sigmoid(
                        (seg_lens - self.seg_len_gate_threshold)
                        * torch.clamp(self.seg_len_gate_slope, min=0.05, max=2.0)
                    )
                    effective_gate = self.segment_context_gate * length_factor.unsqueeze(-1)
                else:
                    effective_gate = self.segment_context_gate

                seg_vecs_ctx = seg_vecs + effective_gate * (ctx_out - seg_vecs)

                if debug_this_call and b == 0:
                    print("=" * 60)
                    print("[DEBUG segment_pool] batch0, n_seg =", n_seg)
                    print("  seg_lens        :", seg_lens.tolist()[:10], "...")
                    if self.use_spatial_embed:
                        print("  seg_cx (0-1000) :", seg_cx.tolist()[:10], "...")
                        print("  seg_cy (0-1000) :", seg_cy.tolist()[:10], "...")
                        print("  x_bucket        :", x_bucket.tolist()[:10], "...")
                        print("  y_bucket        :", y_bucket.tolist()[:10], "...")
                    if self.use_length_gate:
                        print("  threshold =", self.seg_len_gate_threshold.item(),
                              " slope =", self.seg_len_gate_slope.item())
                        print("  length_factor   :", length_factor.tolist()[:10], "...")
                    print("  segment_context_gate =", self.segment_context_gate.item())
                    print("  ||ctx_out - seg_vecs|| / ||seg_vecs|| =",
                          ((ctx_out - seg_vecs).norm() / (seg_vecs.norm() + 1e-9)).item())
                    print("=" * 60)
                    self._debug_printed = True
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
