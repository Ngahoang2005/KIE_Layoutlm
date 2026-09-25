import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy

from typing import NewType
InputDataClass = NewType("InputDataClass", Any)

def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span

@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    # ... giữ nguyên docstring ...

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        has_line_ids_input = "line_ids" in features[0]
        has_block_ids_input = "block_ids" in features[0]
        has_seg_id_input = "seg_id" in features[0]
        has_column_ids_input = "column_ids" in features[0]   # ✅ Đã có
        
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        
        # ====== PADDING LABELS ======
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]

        # ====== PADDING BBOX ======
        if has_bbox_input:
            if padding_side == "right":
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            else:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            batch["bbox"] = torch.tensor(batch["bbox"], dtype=torch.long)

        # ====== PADDING POSITION_IDS ======
        if has_position_input:
            if padding_side == "right":
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]
            else:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]

        # ====== PADDING LINE_IDS ======
        if has_line_ids_input:
            if padding_side == "right":
                batch["line_ids"] = [line_id + [-1] * (sequence_length - len(line_id)) for line_id in batch["line_ids"]]
            else:
                batch["line_ids"] = [[-1] * (sequence_length - len(line_id)) + line_id for line_id in batch["line_ids"]]

        # ====== PADDING BLOCK_IDS ======
        if has_block_ids_input:
            if padding_side == "right":
                batch["block_ids"] = [block_id + [-1] * (sequence_length - len(block_id)) for block_id in batch["block_ids"]]
            else:
                batch["block_ids"] = [[-1] * (sequence_length - len(block_id)) + block_id for block_id in batch["block_ids"]]

        # ====== ✅ THÊM MỚI: PADDING COLUMN_IDS ======
        if has_column_ids_input:
            if padding_side == "right":
                batch["column_ids"] = [col_id + [-1] * (sequence_length - len(col_id)) for col_id in batch["column_ids"]]
            else:
                batch["column_ids"] = [[-1] * (sequence_length - len(col_id)) + col_id for col_id in batch["column_ids"]]

        # ====== PADDING SEG_ID ======
        if has_seg_id_input:
            if padding_side == "right":
                batch["seg_id"] = [seg_id + [-1] * (sequence_length - len(seg_id)) for seg_id in batch["seg_id"]]
            else:
                batch["seg_id"] = [[-1] * (sequence_length - len(seg_id)) + seg_id for seg_id in batch["seg_id"]]

        # ====== XỬ LÝ SEGMENT_IDS ======
        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (sequence_length - len(batch['segment_ids'][i])) + [
                    batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        # ====== CHUYỂN TẤT CẢ LIST THÀNH TENSOR ======
        for k, v in batch.items():
            if isinstance(v, list) and len(v) > 0:
                if k == 'images':
                    continue
                if isinstance(v[0], list):
                    # ✅ THÊM 'column_ids' vào danh sách này
                    if k in ['line_ids', 'block_ids', 'seg_id', 'column_ids']:
                        continue
                    try:
                        max_len = max([len(item) for item in v])
                        for i in range(len(v)):
                            if len(v[i]) < max_len:
                                if k == 'bbox':
                                    v[i] = v[i] + [[0, 0, 0, 0]] * (max_len - len(v[i]))
                                else:
                                    v[i] = v[i] + [0] * (max_len - len(v[i]))
                        batch[k] = torch.tensor(v, dtype=torch.int64)
                    except Exception as e:
                        print(f"Warning: Could not convert {k} to tensor: {e}")
                        continue
                else:
                    try:
                        batch[k] = torch.tensor(v, dtype=torch.int64)
                    except Exception:
                        continue

        # ====== XỬ LÝ SEGMENT IDS ======
        if 'segment_ids' in batch:
            batch['segment_ids'] = torch.tensor(batch['segment_ids'], dtype=torch.int64)
            valid_span = pre_calc_rel_mat(segment_ids=batch['segment_ids'])
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        # ====== CHUYỂN LINE_IDS, BLOCK_IDS, SEG_ID, COLUMN_IDS THÀNH TENSOR ======
        if "line_ids" in batch:
            max_len = max([len(x) for x in batch["line_ids"]])
            for i in range(len(batch["line_ids"])):
                if len(batch["line_ids"][i]) < max_len:
                    batch["line_ids"][i] = batch["line_ids"][i] + [-1] * (max_len - len(batch["line_ids"][i]))
            batch["line_ids"] = torch.tensor(batch["line_ids"], dtype=torch.long)

        if "block_ids" in batch:
            max_len = max([len(x) for x in batch["block_ids"]])
            for i in range(len(batch["block_ids"])):
                if len(batch["block_ids"][i]) < max_len:
                    batch["block_ids"][i] = batch["block_ids"][i] + [-1] * (max_len - len(batch["block_ids"][i]))
            batch["block_ids"] = torch.tensor(batch["block_ids"], dtype=torch.long)

        if "seg_id" in batch:
            max_len = max([len(x) for x in batch["seg_id"]])
            for i in range(len(batch["seg_id"])):
                if len(batch["seg_id"][i]) < max_len:
                    batch["seg_id"][i] = batch["seg_id"][i] + [-1] * (max_len - len(batch["seg_id"][i]))
            batch["seg_id"] = torch.tensor(batch["seg_id"], dtype=torch.long)

        # ====== ✅ THÊM MỚI: CHUYỂN COLUMN_IDS THÀNH TENSOR ======
        if "column_ids" in batch:
            max_len = max([len(x) for x in batch["column_ids"]])
            for i in range(len(batch["column_ids"])):
                if len(batch["column_ids"][i]) < max_len:
                    batch["column_ids"][i] = batch["column_ids"][i] + [-1] * (max_len - len(batch["column_ids"][i]))
            batch["column_ids"] = torch.tensor(batch["column_ids"], dtype=torch.long)

        # ====== XỬ LÝ IMAGE PATCHES ======
        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)
            
            if "line_ids" in batch:
                image_line_ids = torch.ones((batch["line_ids"].shape[0], IMAGE_LEN), dtype=torch.long) * -1
                batch["line_ids"] = torch.cat([batch["line_ids"], image_line_ids], dim=1)
            
            if "block_ids" in batch:
                image_block_ids = torch.ones((batch["block_ids"].shape[0], IMAGE_LEN), dtype=torch.long) * -1
                batch["block_ids"] = torch.cat([batch["block_ids"], image_block_ids], dim=1)
            
            if "seg_id" in batch:
                image_seg_ids = torch.ones((batch["seg_id"].shape[0], IMAGE_LEN), dtype=torch.long) * -1
                batch["seg_id"] = torch.cat([batch["seg_id"], image_seg_ids], dim=1)
            
            # ====== ✅ THÊM MỚI: Mở rộng column_ids cho image patches ======
            if "column_ids" in batch:
                image_col_ids = torch.ones((batch["column_ids"].shape[0], IMAGE_LEN), dtype=torch.long) * -1
                batch["column_ids"] = torch.cat([batch["column_ids"], image_col_ids], dim=1)

        return batch
