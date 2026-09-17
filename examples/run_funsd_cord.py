#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset
import evaluate
import transformers
import torch
from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default="ner")
    dataset_name: Optional[str] = field(default='funsd')
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)
    label_all_tokens: bool = field(default=False)
    return_entity_level_metrics: bool = field(default=False)
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    use_segment_head: bool = field(default=False)

    segment_context_layers: int = field(
        default=1,
        metadata={"help": "Ablation test: 0 = Tat Transformer context, 1 = Bat Transformer context."}
    )
    # ---- PATCH v6: GRUGate (GTrXL-style), thay the length-conditional
    # gate va spatial-embedding-lon da that bai (xem lich su trong
    # modeling_layoutlmv3_segment.py docstring). ----
    segment_use_token_gru_gate: bool = field(
        default=True,
        metadata={"help": "True = dung GRUGate (residual gate hoc tu noi "
                  "dung, per-token per-chieu) giua per-token hidden goc va "
                  "pooled_hidden. False = ghi de vo dieu kien (ban v3 cu)."}
    )
    segment_gate_bottleneck: int = field(
        default=64,
        metadata={"help": "Chieu bottleneck cho GRUGate -- kiem soat so "
                  "tham so moi tren tap du lieu nho (149 van ban FUNSD)."}
    )
    segment_debug: bool = field(
        default=False,
        metadata={"help": "Bat debug print o forward dau tien, 1 lan."}
    )

    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224)
    second_input_size: int = field(default=112)
    train_interpolation: str = field(default='bicubic')
    second_interpolation: str = field(default='lanczos')
    imagenet_default_mean_and_std: bool = field(default=False)


class GateLoggingCallback(TrainerCallback):
    """Log gia tri gate moi lan eval -- khong phu thuoc seed/nhieu train,
    xem TRUOC KHI tin vao chenh lech F1 giua cac run."""
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        msgs = []
        if getattr(model, "segment_context_gate", None) is not None:
            msgs.append(f"segment_context_gate={model.segment_context_gate.item():.4f}")
        if getattr(model, "token_gru_gate", None) is not None:
            with torch.no_grad():
                bg_mean = model.token_gru_gate.bg.mean().item()
                bg_std = model.token_gru_gate.bg.std().item()
            msgs.append(f"gru_gate_bias_mean={bg_mean:.4f}(std={bg_std:.4f}) "
                        f"[thap hon init(2.0) -> gate dang mo ra, tin pooled_hidden nhieu hon]")
        if msgs:
            logger.info(f"[GateLoggingCallback] step={state.global_step} " + " ".join(msgs))


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    if data_args.dataset_name == 'funsd':
        import layoutlmft.data.funsd
        datasets = load_dataset(os.path.abspath(layoutlmft.data.funsd.__file__), cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord
        datasets = load_dataset(os.path.abspath(layoutlmft.data.cord.__file__), cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"
    label_column_name = (f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1])
    remove_columns = column_names

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.segment_context_layers = getattr(data_args, "segment_context_layers", 1)
    config.segment_use_token_gru_gate = getattr(data_args, "segment_use_token_gru_gate", True)
    config.segment_gate_bottleneck = getattr(data_args, "segment_gate_bottleneck", 64)
    config.segment_debug = getattr(data_args, "segment_debug", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if getattr(data_args, "use_segment_head", False):
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import LayoutLMv3ForSegmentTokenClassification
        model = LayoutLMv3ForSegmentTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])

    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        seg_ids = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]

            word_seg_id = None
            if getattr(data_args, "use_segment_head", False):
                word_seg_id = []
                seg_counter = -1
                prev_bbox_tuple = None
                for wb in bbox:
                    wb_tuple = tuple(wb)
                    if wb_tuple != prev_bbox_tuple:
                        seg_counter += 1
                        prev_bbox_tuple = wb_tuple
                    word_seg_id.append(seg_counter)

            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            seg_id_inputs = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    if word_seg_id is not None:
                        seg_id_inputs.append(-1)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            if word_seg_id is not None:
                seg_ids.append(seg_id_inputs)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        if getattr(data_args, "use_segment_head", False):
            tokenized_inputs["seg_id"] = seg_ids
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        eval_dataset = datasets["test"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    class CustomTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                backbone_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" in n and p.requires_grad]
                new_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" not in n and p.requires_grad]

                optimizer_grouped_parameters = [
                    {"params": backbone_params, "lr": self.args.learning_rate},
                    {"params": new_params, "lr": 1e-4}
                ]

                self.optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[GateLoggingCallback()] if getattr(data_args, "use_segment_head", False) else None,
    )

    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

if __name__ == "__main__":
    main()
