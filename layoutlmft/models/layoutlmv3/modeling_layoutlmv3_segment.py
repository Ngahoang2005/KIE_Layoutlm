# layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
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
        self.use_xy_cut = getattr(config, "segment_use_xy_cut", False)

        if seg_ctx_layers > 0:
            # Ép cứng dropout = 0.3 cho module Context để chống Overfitting
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=seg_ctx_heads,
                dim_feedforward=config.hidden_size * 2,
                dropout=0.3, 
                batch_first=True,
            )
            self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=seg_ctx_layers)
            self.segment_context_gate = nn.Parameter(torch.zeros(1))

            max_pos = getattr(config, "segment_context_max_positions", 256)
            self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.segment_context = None
            self.segment_context_gate = None
            self.segment_position_embedding = None

        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        self.init_weights()

        if _is_main_process():
            print(f"[LayoutLMv3ForSegmentTokenClassification] Layers={seg_ctx_layers} | Overlap-XY-Cut={self.use_xy_cut}")

    def _get_xy_cut_order(self, boxes):
        """
        Heuristic Overlap-based Sort: 
        1. Sắp xếp thô từ trên xuống dưới (theo Y_min).
        2. Nhóm các cụm vào chung một dòng nếu độ giao nhau dọc (Vertical Overlap) > 50%.
        3. Trong mỗi dòng, sắp xếp lại từ trái qua phải (theo X_min).
        """
        N = boxes.shape[0]
        if N <= 1:
            return torch.arange(N, device=boxes.device)
        
        y_min = boxes[:, 1]
        y_max = boxes[:, 3]
        x_min = boxes[:, 0]
        
        initial_order = torch.argsort(y_min)
        sorted_ymin = y_min[initial_order]
        sorted_ymax = y_max[initial_order]
        sorted_xmin = x_min[initial_order]
        
        lines = []
        current_line = [0]
        current_line_ymin = sorted_ymin[0].item()
        current_line_ymax = sorted_ymax[0].item()
        
        for i in range(1, N):
            box_ymin = sorted_ymin[i].item()
            box_ymax = sorted_ymax[i].item()
            
            overlap = max(0.0, min(current_line_ymax, box_ymax) - max(current_line_ymin, box_ymin))
            box_height = box_ymax - box_ymin
            
            if box_height > 0 and (overlap / box_height) > 0.5:
                current_line.append(i)
                current_line_ymin = min(current_line_ymin, box_ymin)
                current_line_ymax = max(current_line_ymax, box_ymax)
            else:
                lines.append(current_line)
                current_line = [i]
                current_line_ymin = box_ymin
                current_line_ymax = box_ymax
        
        if current_line:
            lines.append(current_line)
            
        final_order = []
        for line in lines:
            if len(line) > 1:
                line_tensor = torch.tensor(line, device=boxes.device)
                line_xmin = sorted_xmin[line_tensor]
                x_order = torch.argsort(line_xmin)
                final_order.extend(initial_order[line_tensor[x_order]].tolist())
            else:
                final_order.append(initial_order[line[0]].item())
                
        return torch.tensor(final_order, device=boxes.device)

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, bbox):
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
            seg_boxes = torch.zeros(n_seg, 4, device=device, dtype=torch.float32)
            seg_masks = []
            
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)
                if bbox is not None:
                    member_boxes = bbox[b, mask].float()
                    seg_boxes[i, 0] = member_boxes[:, 0].min()
                    seg_boxes[i, 1] = member_boxes[:, 1].min()
                    seg_boxes[i, 2] = member_boxes[:, 2].max()
                    seg_boxes[i, 3] = member_boxes[:, 3].max()

            if self.segment_context is not None:
                order_perm = None
                if self.use_xy_cut and bbox is not None and n_seg > 1:
                    order_perm = self._get_xy_cut_order(seg_boxes)
                    seg_vecs_input = seg_vecs[order_perm]
                else:
                    seg_vecs_input = seg_vecs

                max_pos = self.segment_position_embedding.num_embeddings
                positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                pos_embed = self.segment_position_embedding(positions)

                seg_vecs_with_pos = seg_vecs_input + pos_embed
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
