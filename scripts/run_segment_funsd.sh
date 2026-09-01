#!/bin/bash
set -e

cd /home/s24gbn1/Documents/httn/unilm/layoutlmv3
export PYTHONPATH="/home/s24gbn1/Documents/httn/unilm/layoutlmv3:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="funsd-LISC-Experiment"

SEEDS=(42 123 1993)

for SEED in "${SEEDS[@]}"
do
    OUT_DIR="./layoutlmv3-large-finetuned-funsd-segctx-seed${SEED}"

    echo ""
    echo "============================================================"
    echo "RUNNING SEED = ${SEED}"
    echo "============================================================"
    # Dọn dẹp checkpoint cũ của seed hiện tại
    rm -rf "$OUT_DIR"

    python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port 4398 examples/run_funsd_cord.py \
      --dataset_name funsd \
      --do_train \
      --do_eval \
      --do_predict \
      --use_segment_head \
      --model_name_or_path models/layoutlmv3-base \
      --output_dir "$OUT_DIR" \
      --segment_level_layout 1 \
      --visual_embed 1 \
      --input_size 224 \
      --max_steps 1000 \
      --save_steps -1 \
      --evaluation_strategy steps \
      --eval_steps 100 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --dataloader_num_workers 4 \
      --report_to wandb \
      --run_name "FUnSD-LR-Split-seed${SEED}" \
      --seed "$SEED" \
      --overwrite_output_dir \
      --overwrite_cache

done

echo ""
echo "============================================================"
echo "CALCULATING 3-SEED MEAN ± STD (LISC)"
echo "============================================================"

python - <<'PY'
import os
import json
import numpy as np

seeds = [42, 123, 1993]
metrics = [
    "eval_accuracy",
    "eval_f1",
    "eval_precision",
    "eval_recall",
    "eval_loss",
]

results = {m: [] for m in metrics}

for seed in seeds:
    path = f"./layoutlmv3-large-finetuned-cord-segctx-seed{seed}/eval_results.json"
    print(f"\nSeed {seed}:")

    if not os.path.exists(path):
        print(f"  [WARNING] Missing: {path}")
        continue

    with open(path, "r") as f:
        data = json.load(f)

    for metric in metrics:
        if metric in data:
            value = float(data[metric])
            results[metric].append(value)
            print("  {:18s} = {:.6f}".format(metric, value))

print("\n" + "=" * 70)
print("FINAL RESULT: MEAN ± STD")
print("=" * 70)

summary = {}
for metric in metrics:
    values = results[metric]
    if not values:
        continue
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    print("{:18s}: {:.4f} ± {:.4f}".format(metric, mean, std))
    summary[metric] = {"values": values, "mean": float(mean), "std": float(std)}

with open("funsd_lisc_3seed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

PY