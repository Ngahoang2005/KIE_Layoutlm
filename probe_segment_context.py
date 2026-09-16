#!/usr/bin/env python
# coding=utf-8
"""
probe_segment_context.py

Hỏi đúng một câu: Transformer `segment_context` CÓ THỰC SỰ học được quan hệ
giữa các segment không, hay chỉ là tham số thừa?

Script này KHÔNG yêu cầu sửa modeling_layoutlmv3_segment.py -- nó
monkey-patch `_segment_pool_and_contextualize` lúc runtime, nên chạy được
thẳng trên checkpoint đã train.

SÁU PHÉP THỬ, xếp theo thứ tự "rẻ -> đắt", và quan trọng là theo thứ tự
"phải đúng thì phép sau mới có nghĩa":

  [T1] Giá trị gate.
       `segment_context_gate` là hệ số nhân TRỰC TIẾP lên toàn bộ đóng góp
       của Transformer:  seg_vecs_ctx = seg_vecs + gate * (ctx_out - seg_vecs)
       Nếu |gate| ~ 0 thì Transformer KHÔNG đóng góp gì, và MỌI phép thử
       shuffle phía sau sẽ cho Δ=0 một cách tầm thường -- không phải vì
       "Transformer không dùng cấu trúc", mà vì nó đã bị tắt. Đây chính là
       cái bẫy diễn giải cần loại trừ TRƯỚC.

  [T2] Độ lớn thực tế của phần Transformer thêm vào.
       Đo ||gate * (ctx_out - seg_vecs)|| / ||seg_vecs|| trên dữ liệu thật.
       Cho biết Transformer làm thay đổi biểu diễn bao nhiêu PHẦN TRĂM.
       gate nhỏ nhưng (ctx_out - seg_vecs) lớn vẫn có thể có tác dụng, nên
       phải đo tích thực tế chứ không chỉ nhìn gate.

  [T3] Zero-gate ablation (ép gate=0 lúc eval).
       So F1 thật vs F1 khi TẮT HẲN Transformer. Đây là thước đo trực tiếp
       nhất cho "Transformer đóng góp bao nhiêu điểm F1", và không phụ
       thuộc vào bất kỳ giả định nào về cách nó học.

  [T4] Order shuffle: xáo THỨ TỰ segment trước khi đưa vào Transformer,
       rồi hoán vị ngược output về đúng chỗ.
       Δ lớn -> Transformer dùng thứ tự đọc (position embedding + vị trí
       tương đối trong chuỗi). Δ ~ 0 -> nó bất biến với thứ tự.

  [T5] Membership shuffle: xáo ngẫu nhiên token nào thuộc segment nào
       (giữ nguyên số segment và kích thước phân bố).
       Δ lớn -> việc gom đúng token vào đúng segment là thiết yếu.

  [T6] Entropy của attention trong segment_context.
       Attention gần như ĐỀU (entropy ~ log(n_seg)) nghĩa là mỗi segment
       "nhìn" mọi segment khác như nhau -> thực chất chỉ đang tính trung
       bình toàn cục, KHÔNG học quan hệ có chọn lọc. Entropy thấp hơn
       đáng kể so với uniform -> có chọn lọc thật.

CÁCH ĐỌC KẾT QUẢ (quan trọng):
  - T1 cho |gate| ~ 0 VÀ T3 cho ΔF1 ~ 0  -> Transformer vô dụng, T4/T5/T6
    không cần diễn giải thêm.
  - T3 cho ΔF1 rõ rệt nhưng T4/T5 đều ~0 -> nó có đóng góp, nhưng KHÔNG
    qua thứ tự cũng không qua cấu trúc nhóm; nhiều khả năng chỉ hoạt động
    như một phép chuẩn hoá/trung bình toàn cục (kiểm chứng bằng T6).
  - T4 hoặc T5 lớn -> có bằng chứng thật rằng nó dùng đúng cấu trúc đó.

LƯU Ý KỸ THUẬT: chạy inference thủ công trên MỘT GPU (không dùng Trainer /
DataParallel). Lý do: DataParallel nhân bản module sang từng GPU, nên
monkey-patch hay thuộc tính đặt trên module gốc có thể KHÔNG tới được
replica -> shuffle im lặng không chạy và mọi Δ ra 0.00 (một cách giả tạo).

Usage:
    CUDA_VISIBLE_DEVICES=0 python probe_segment_context.py \\
        --checkpoint ./output/funsd-B-ctx1-seed42 \\
        --dataset_name funsd \\
        --n_repeats 5
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer

from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
    LayoutLMv3ForSegmentTokenClassification,
)
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


# ===========================================================================
# Dataset
# ===========================================================================
def build_eval_dataset(dataset_name, tokenizer, label_column_name):
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
                    lab_ids.append(label[widx]); bbox_in.append(bbox[widx]); seg_in.append(word_seg_id[widx])
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
        tok.pop("overflow_to_sample_mapping", None)
        return tok

    return raw.map(_prep, batched=True, remove_columns=raw.column_names, load_from_cache_file=False)


# ===========================================================================
# Monkey-patched pooling: mode-aware + thu thập thống kê
# ===========================================================================
class PatchedPooling:
    """Thay thế model._segment_pool_and_contextualize. Giữ nguyên logic gốc,
    chỉ thêm: (a) chế độ shuffle, (b) ép gate=0, (c) thu thập thống kê về
    độ lớn đóng góp của Transformer và entropy attention."""

    def __init__(self, model):
        self.model = model
        self.mode = None           # None | "order" | "membership"
        self.force_gate_zero = False
        self.collect_stats = False
        self.rng_seed = None

        # thống kê thu thập được
        self.delta_ratios = []     # ||gate*(ctx-seg)|| / ||seg||
        self.attn_entropies = []   # entropy attention (nat)
        self.attn_uniform_entropies = []  # log(n_seg) để so sánh

    def reset_stats(self):
        self.delta_ratios = []
        self.attn_entropies = []
        self.attn_uniform_entropies = []

    def __call__(self, text_hidden, seg_id):
        model = self.model
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        gen = None
        if self.rng_seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.rng_seed)

        for b in range(B):
            ids = seg_id[b].clone()

            # --- T5: membership shuffle ---
            if self.mode == "membership":
                valid_mask = ids >= 0
                valid_ids = ids[valid_mask]
                n_valid = valid_ids.shape[0]
                if n_valid > 1:
                    perm = torch.randperm(n_valid, generator=gen).to(device)
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

            # --- T4: order shuffle ---
            order_perm = None
            if self.mode == "order" and n_seg > 1:
                order_perm = torch.randperm(n_seg, generator=gen).to(device)
                seg_vecs_input = seg_vecs[order_perm]
            else:
                seg_vecs_input = seg_vecs

            if model.segment_context is not None:
                max_pos = model.segment_position_embedding.num_embeddings
                positions = torch.arange(n_seg, device=device).clamp(max=max_pos - 1)
                seg_vecs_with_pos = seg_vecs_input + model.segment_position_embedding(positions)

                ctx_out = model.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)

                # --- T6: entropy attention của layer đầu ---
                if self.collect_stats and n_seg > 1:
                    try:
                        attn_layer = model.segment_context.layers[0].self_attn
                        x = seg_vecs_with_pos.unsqueeze(0)
                        _, attn_w = attn_layer(x, x, x, need_weights=True, average_attn_weights=True)
                        # attn_w: (1, n_seg, n_seg)
                        p = attn_w.squeeze(0).clamp(min=1e-12)
                        ent = -(p * p.log()).sum(dim=-1).mean().item()
                        self.attn_entropies.append(ent)
                        self.attn_uniform_entropies.append(float(np.log(n_seg)))
                    except TypeError:
                        # bản torch cũ không có average_attn_weights
                        attn_layer = model.segment_context.layers[0].self_attn
                        x = seg_vecs_with_pos.unsqueeze(0)
                        _, attn_w = attn_layer(x, x, x, need_weights=True)
                        p = attn_w.squeeze(0).clamp(min=1e-12)
                        ent = -(p * p.log()).sum(dim=-1).mean().item()
                        self.attn_entropies.append(ent)
                        self.attn_uniform_entropies.append(float(np.log(n_seg)))
                    except Exception:
                        pass

                if order_perm is not None:
                    inv_perm = torch.empty_like(order_perm)
                    inv_perm[order_perm] = torch.arange(n_seg, device=device)
                    ctx_out = ctx_out[inv_perm]

                gate = torch.zeros_like(model.segment_context_gate) if self.force_gate_zero \
                    else model.segment_context_gate

                delta = gate * (ctx_out - seg_vecs)

                # --- T2: độ lớn đóng góp thực tế ---
                if self.collect_stats:
                    num = delta.norm(dim=-1)
                    den = seg_vecs.norm(dim=-1).clamp(min=1e-9)
                    self.delta_ratios.extend((num / den).detach().cpu().tolist())

                seg_vecs_ctx = seg_vecs + delta
            else:
                seg_vecs_ctx = seg_vecs

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        return broadcast_hidden


# ===========================================================================
# Manual inference (KHÔNG dùng Trainer/DataParallel)
# ===========================================================================
@torch.no_grad()
def run_inference(model, dataset, data_collator, device, batch_size=4):
    model.eval()
    all_preds, all_golds = [], []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = data_collator(rows)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = batch.pop("labels")
        out = model(**batch)
        logits = out.logits if hasattr(out, "logits") else out[0]
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_golds.append(labels.cpu().numpy())
    return all_preds, all_golds


def seqeval_f1_from_batches(all_preds, all_golds, label_names):
    pred_seq, gold_seq = [], []
    for preds_b, golds_b in zip(all_preds, all_golds):
        for pr, gr in zip(preds_b, golds_b):
            p_row, g_row = [], []
            for p, g in zip(pr, gr):
                if g == -100:
                    continue
                p_row.append(label_names[p]); g_row.append(label_names[g])
            pred_seq.append(p_row); gold_seq.append(g_row)
    return f1_score(gold_seq, pred_seq) * 100


# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default="funsd", choices=["funsd", "cord"])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        print(f"[CẢNH BÁO] Thấy {torch.cuda.device_count()} GPU. Script này chạy inference thủ công "
              f"trên 1 GPU để tránh DataParallel làm shuffle không có tác dụng. "
              f"Nên chạy với CUDA_VISIBLE_DEVICES=0 cho chắc chắn.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "funsd":
        import layoutlmft.data.funsd as mod
    else:
        import layoutlmft.data.cord as mod
    raw_features = load_dataset(os.path.abspath(mod.__file__))["test"].features
    label_column_name = "ner_tags" if "ner_tags" in raw_features else list(raw_features.keys())[1]
    label_names = raw_features[label_column_name].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True,
                                               add_prefix_space=True, tokenizer_file=None)
    config = AutoConfig.from_pretrained(args.checkpoint, num_labels=len(label_names), finetuning_task="ner")
    model = LayoutLMv3ForSegmentTokenClassification.from_pretrained(args.checkpoint, config=config)
    model.to(device)

    if model.segment_context is None:
        raise SystemExit("Checkpoint này có segment_context_layers=0 -- không có Transformer để probe. "
                          "Dùng checkpoint train với --segment_context_layers 1.")

    eval_dataset = build_eval_dataset(args.dataset_name, tokenizer, label_column_name)
    data_collator = DataCollatorForKeyValueExtraction(tokenizer, padding=True, max_length=512)

    # cài monkey-patch
    patched = PatchedPooling(model)
    model._segment_pool_and_contextualize = patched

    print("\n" + "=" * 70)
    print("[T1] GIÁ TRỊ GATE")
    print("=" * 70)
    gate_val = model.segment_context_gate.item()
    print(f"  segment_context_gate = {gate_val:+.6f}")
    if abs(gate_val) < 0.01:
        print("  !!! |gate| < 0.01 -- đóng góp của Transformer gần như BẰNG KHÔNG.")
        print("      Mọi Δ ở T4/T5 sẽ ra ~0 một cách TẦM THƯỜNG (không có gì để xáo),")
        print("      KHÔNG được diễn giải là 'Transformer không dùng cấu trúc'.")

    # ---- chạy normal + thu thống kê ----
    print("\n" + "=" * 70)
    print("[T2] ĐỘ LỚN ĐÓNG GÓP THỰC TẾ CỦA TRANSFORMER  +  [T6] ENTROPY ATTENTION")
    print("=" * 70)
    patched.mode = None
    patched.force_gate_zero = False
    patched.collect_stats = True
    patched.reset_stats()
    preds_normal, golds = run_inference(model, eval_dataset, data_collator, device, args.batch_size)
    f1_normal = seqeval_f1_from_batches(preds_normal, golds, label_names)
    patched.collect_stats = False

    if patched.delta_ratios:
        dr = np.array(patched.delta_ratios)
        print(f"  ||gate*(ctx-seg)|| / ||seg||  :  mean={dr.mean():.4f}  median={np.median(dr):.4f}  "
              f"p95={np.percentile(dr, 95):.4f}  max={dr.max():.4f}")
        print(f"  -> Transformer làm đổi biểu diễn segment trung bình {100*dr.mean():.2f}%.")
        if dr.mean() < 0.01:
            print("     (< 1% -- thay đổi không đáng kể về mặt biểu diễn)")

    if patched.attn_entropies:
        ae = np.array(patched.attn_entropies)
        ue = np.array(patched.attn_uniform_entropies)
        ratio = (ae / np.clip(ue, 1e-9, None))
        print(f"  Entropy attention: mean={ae.mean():.4f} nat   |   uniform baseline log(n_seg)={ue.mean():.4f} nat")
        print(f"  Tỉ lệ entropy/uniform: mean={ratio.mean():.4f}  (1.0 = attention ĐỀU hoàn toàn)")
        if ratio.mean() > 0.95:
            print("     -> attention gần như ĐỀU: mỗi segment nhìn mọi segment như nhau,")
            print("        tức thực chất chỉ là trung bình toàn cục, KHÔNG chọn lọc quan hệ.")
        else:
            print("     -> attention CÓ chọn lọc (tập trung hơn đáng kể so với đều).")

    print(f"\n  F1 normal = {f1_normal:.2f}")

    # ---- T3: zero-gate ablation ----
    print("\n" + "=" * 70)
    print("[T3] ZERO-GATE ABLATION (tắt hẳn Transformer lúc eval)")
    print("=" * 70)
    patched.mode = None
    patched.force_gate_zero = True
    preds_zero, golds_z = run_inference(model, eval_dataset, data_collator, device, args.batch_size)
    f1_zero = seqeval_f1_from_batches(preds_zero, golds_z, label_names)
    patched.force_gate_zero = False
    print(f"  F1 với gate thật = {f1_normal:.2f}")
    print(f"  F1 với gate=0    = {f1_zero:.2f}")
    print(f"  ĐÓNG GÓP THỰC CỦA TRANSFORMER = {f1_normal - f1_zero:+.2f} điểm F1")

    # đếm số token dự đoán khác nhau giữa 2 chế độ
    n_diff = sum(int((a != b).sum()) for a, b in zip(preds_normal, preds_zero))
    n_tot = sum(int(a.size) for a in preds_normal)
    print(f"  Số vị trí dự đoán KHÁC nhau: {n_diff}/{n_tot} ({100*n_diff/max(n_tot,1):.3f}%)")
    if n_diff == 0:
        print("  !!! KHÔNG MỘT dự đoán nào thay đổi -> Transformer hoàn toàn không ảnh hưởng đầu ra.")

    # ---- T4: order shuffle ----
    print("\n" + "=" * 70)
    print(f"[T4] ORDER SHUFFLE (lặp {args.n_repeats} lần)")
    print("=" * 70)
    f1_order = []
    for i in range(args.n_repeats):
        patched.mode = "order"
        patched.rng_seed = 1000 + i
        p, g = run_inference(model, eval_dataset, data_collator, device, args.batch_size)
        f1_order.append(seqeval_f1_from_batches(p, g, label_names))
    patched.mode = None
    patched.rng_seed = None
    print(f"  F1 = {np.mean(f1_order):.2f} ± {np.std(f1_order):.2f}   {[f'{x:.2f}' for x in f1_order]}")
    print(f"  Δ so với normal = {f1_normal - np.mean(f1_order):+.2f}")

    # ---- T5: membership shuffle ----
    print("\n" + "=" * 70)
    print(f"[T5] MEMBERSHIP SHUFFLE (lặp {args.n_repeats} lần)")
    print("=" * 70)
    f1_member = []
    for i in range(args.n_repeats):
        patched.mode = "membership"
        patched.rng_seed = 2000 + i
        p, g = run_inference(model, eval_dataset, data_collator, device, args.batch_size)
        f1_member.append(seqeval_f1_from_batches(p, g, label_names))
    patched.mode = None
    patched.rng_seed = None
    print(f"  F1 = {np.mean(f1_member):.2f} ± {np.std(f1_member):.2f}   {[f'{x:.2f}' for x in f1_member]}")
    print(f"  Δ so với normal = {f1_normal - np.mean(f1_member):+.2f}")

    # ---- tổng kết ----
    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(f"  gate                              = {gate_val:+.6f}")
    if patched.delta_ratios:
        print(f"  thay đổi biểu diễn trung bình      = {100*np.mean(patched.delta_ratios):.2f}%")
    print(f"  F1 normal                          = {f1_normal:.2f}")
    print(f"  F1 gate=0 (tắt Transformer)        = {f1_zero:.2f}   (Δ = {f1_normal - f1_zero:+.2f})")
    print(f"  F1 order-shuffle                   = {np.mean(f1_order):.2f}   (Δ = {f1_normal - np.mean(f1_order):+.2f})")
    print(f"  F1 membership-shuffle              = {np.mean(f1_member):.2f}   (Δ = {f1_normal - np.mean(f1_member):+.2f})")
    print()
    if abs(f1_normal - f1_zero) < 0.1 and n_diff == 0:
        print("  KẾT LUẬN: Transformer segment_context KHÔNG đóng góp gì đo được.")
        print("  (T4/T5 ra 0 là hệ quả tất yếu, không phải bằng chứng độc lập.)")
    elif abs(f1_normal - f1_zero) >= 0.1:
        print("  Transformer CÓ đóng góp đo được. Giờ Δ ở T4/T5 mới có ý nghĩa:")
        print("   - Δ(order) lớn     -> dùng thứ tự đọc.")
        print("   - Δ(membership) lớn -> cần gom đúng token vào đúng segment.")
        print("   - Cả hai ~0 nhưng T3 lớn -> chỉ hoạt động như trung bình/chuẩn hoá toàn cục")
        print("     (đối chiếu với entropy attention ở T6 để xác nhận).")


if __name__ == "__main__":
    main()
