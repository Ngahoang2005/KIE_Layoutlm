#!/usr/bin/env python
# coding=utf-8
"""
run_ablation_eval.py -- Ablation E/F: xao thu tu segment / xao thanh vien
segment luc EVAL tren 1 checkpoint ctx=1 DA TRAIN XONG, khong train lai.

3 lan eval tren CUNG 1 checkpoint, CUNG 1 test set:
  1. normal      -- eval_shuffle_mode = None
  2. order       -- xao THU TU cac segment nhin thay nhau (giu nguyen
                     token nao thuoc segment nao)
  3. membership  -- xao NGAU NHIEN token nao thuoc segment nao (giu
                     nguyen phan phoi kich thuoc segment)

Neu Transformer thuc su hoc duoc gi do co y nghia tu cau truc segment:
  F1(order)      << F1(normal)  neu no dung THU TU doc
  F1(membership) << F1(normal)  neu no dung DUNG cach gom nhom

Neu ca 2 deu ~F1(normal) -> Transformer khong dua vao cau truc that,
chi dang lam 1 phep bien doi vo nghia (hoac gan nhu identity vi gate thap).

Chay:
    python run_ablation_eval.py \
        --checkpoint ./output/funsd-layoutlmv3-base-ctx1 \
        --dataset_name funsd \
        --n_repeats 5
"""
import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
    LayoutLMv3ForSegmentTokenClassification,
)
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def build_eval_dataset(dataset_name, tokenizer, label_to_id, label_column_name):
    if dataset_name == "funsd":
        import layoutlmft.data.funsd as mod
    elif dataset_name == "cord":
        import layoutlmft.data.cord as mod
    else:
        raise NotImplementedError
    raw = load_dataset(os.path.abspath(mod.__file__))["test"]
    text_column = "words" if "words" in raw.column_names else "tokens"

    common_transform = Compose([RandomResizedCropAndInterpolationWithTwoPic(size=224, interpolation="bicubic")])
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(IMAGENET_INCEPTION_MEAN), std=torch.tensor(IMAGENET_INCEPTION_STD)),
    ])

    def _prep(examples):
        tok = tokenizer(examples[text_column], padding=False, truncation=True,
                         return_overflowing_tokens=True, is_split_into_words=True)
        labels, bboxes, images, seg_ids = [], [], [], []
        for bi in range(len(tok["input_ids"])):
            word_ids = tok.word_ids(batch_index=bi)
            oi = tok["overflow_to_sample_mapping"][bi]
            label = examples[label_column_name][oi]
            bbox = examples["bboxes"][oi]

            word_seg_id, counter, prev = [], -1, None
            for wb in bbox:
                t = tuple(wb)
                if t != prev:
                    counter += 1
                    prev = t
                word_seg_id.append(counter)

            prev_widx = None
            lab_ids, bbox_in, seg_in = [], [], []
            for widx in word_ids:
                if widx is None:
                    lab_ids.append(-100); bbox_in.append([0, 0, 0, 0]); seg_in.append(-1)
                elif widx != prev_widx:
                    lab_ids.append(label_to_id[label[widx]]); bbox_in.append(bbox[widx]); seg_in.append(word_seg_id[widx])
                else:
                    lab_ids.append(-100); bbox_in.append(bbox[widx]); seg_in.append(word_seg_id[widx])
                prev_widx = widx
            labels.append(lab_ids); bboxes.append(bbox_in); seg_ids.append(seg_in)

            img = pil_loader(examples["image_path"][oi])
            for_patches, _ = common_transform(img, augmentation=False)
            images.append(patch_transform(for_patches))

        tok["labels"] = labels
        tok["bbox"] = bboxes
        tok["seg_id"] = seg_ids
        tok["images"] = images
        return tok

    return raw.map(_prep, batched=True, remove_columns=raw.column_names, load_from_cache_file=False)


def seqeval_f1(preds, gold, label_names):
    metric = load_metric("seqeval")
    pred_seq, gold_seq = [], []
    for pr, gr in zip(preds, gold):
        p_row, g_row = [], []
        for p, g in zip(pr, gr):
            if g == -100:
                continue
            p_row.append(label_names[p]); g_row.append(label_names[g])
        pred_seq.append(p_row); gold_seq.append(g_row)
    r = metric.compute(predictions=pred_seq, references=gold_seq)
    return r["overall_f1"] * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint ctx=1 đã train xong")
    parser.add_argument("--dataset_name", default="funsd", choices=["funsd", "cord"])
    parser.add_argument("--n_repeats", type=int, default=5,
                         help="Số lần lặp lại mỗi mode shuffle (vì random) để lấy mean±std")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.dataset_name == "funsd":
        import layoutlmft.data.funsd as mod
    else:
        import layoutlmft.data.cord as mod
    raw_features = load_dataset(os.path.abspath(mod.__file__))["test"].features
    label_column_name = "ner_tags" if "ner_tags" in raw_features else list(raw_features.keys())[1]
    label_names = raw_features[label_column_name].feature.names
    label_to_id = {l: i for i, l in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, add_prefix_space=True, tokenizer_file=None)
    config = AutoConfig.from_pretrained(args.checkpoint, num_labels=len(label_names), finetuning_task="ner")
    model = LayoutLMv3ForSegmentTokenClassification.from_pretrained(args.checkpoint, config=config)

    if model.segment_context is None:
        raise SystemExit(
            "Checkpoint này có segment_context_layers=0 -- ablation E/F chỉ có ý nghĩa "
            "trên checkpoint ctx=1 (có Transformer). Dùng đúng checkpoint đã train với "
            "--segment_context_layers 1."
        )

    print(f"[Check] segment_context_gate (trước eval) = {model.segment_context_gate.item():.4f}")
    print(f"[Check] token_gate (trước eval) = {model.token_gate.item():.4f}")

    eval_dataset = build_eval_dataset(args.dataset_name, tokenizer, label_to_id, label_column_name)
    data_collator = DataCollatorForKeyValueExtraction(tokenizer, padding=True, max_length=512)

    def run_eval(shuffle_mode, tag):
        model.eval_shuffle_mode = shuffle_mode
        training_args = TrainingArguments(
            output_dir=f"/tmp/ablation_ef_{tag}",
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to=[], remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=training_args, data_collator=data_collator)
        out = trainer.predict(eval_dataset)
        preds = np.argmax(out.predictions, axis=-1)
        return seqeval_f1(preds, out.label_ids, label_names)

    print("\n[1/3] normal (không xáo)...")
    f1_normal = run_eval(None, "normal")
    print(f"  F1 = {f1_normal:.2f}")

    print(f"\n[2/3] order-shuffle (lặp {args.n_repeats} lần, vì random)...")
    f1_order_list = [run_eval("order", f"order_{i}") for i in range(args.n_repeats)]
    print(f"  F1 = {np.mean(f1_order_list):.2f} ± {np.std(f1_order_list):.2f}  (các lần: {[f'{x:.2f}' for x in f1_order_list]})")

    print(f"\n[3/3] membership-shuffle (lặp {args.n_repeats} lần)...")
    f1_member_list = [run_eval("membership", f"member_{i}") for i in range(args.n_repeats)]
    print(f"  F1 = {np.mean(f1_member_list):.2f} ± {np.std(f1_member_list):.2f}  (các lần: {[f'{x:.2f}' for x in f1_member_list]})")

    print("\n" + "=" * 60)
    print("KẾT LUẬN")
    print("=" * 60)
    print(f"  F1 normal              = {f1_normal:.2f}")
    print(f"  F1 order-shuffle       = {np.mean(f1_order_list):.2f} (Δ = {f1_normal - np.mean(f1_order_list):+.2f})")
    print(f"  F1 membership-shuffle  = {np.mean(f1_member_list):.2f} (Δ = {f1_normal - np.mean(f1_member_list):+.2f})")
    print("  Δ lớn (>1-2 điểm, vượt std) -> Transformer THỰC SỰ dựa vào đúng cấu trúc đó.")
    print("  Δ nhỏ (trong khoảng std)    -> Transformer KHÔNG dựa vào cấu trúc đó một cách có ý nghĩa.")


if __name__ == "__main__":
    main()
