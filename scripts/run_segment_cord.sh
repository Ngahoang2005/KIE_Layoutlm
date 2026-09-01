#!/bin/bash

set -e

cd /home/s24gbn1/Documents/httn/unilm/layoutlmv3

export PYTHONPATH="/home/s24gbn1/Documents/httn/unilm/layoutlmv3:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="CORD-LISC-Experiment"

SEEDS=(42 123 1993)

for SEED in "${SEEDS[@]}"
do

    OUT="./cord-lisc-base-seed${SEED}"

    echo ""
    echo "============================================================"
    echo "CORD SEGMENT CONTEXT (LISC) - SEED = ${SEED}"
    echo "============================================================"

    # Nếu seed đã có kết quả thì bỏ qua
    if [ -f "$OUT/eval_results.json" ]; then
        echo "[SKIP] Seed ${SEED} đã có kết quả."
        continue
    fi

    rm -rf "$OUT"

    # Giữ nguyên torch.distributed.launch như bản gốc để đảm bảo công bằng 100%
    python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port 4398 examples/run_funsd_cord.py \
    --dataset_name cord \
    --do_train --do_eval \
    --use_segment_head \
    --model_name_or_path models/layoutlmv3-base \
    --output_dir "$OUT" \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 \
    --save_steps -1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 8 \
    --report_to none \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed "$SEED" \
    
done

echo ""
echo "============================================================"
echo "CALCULATING SEGMENT CONTEXT 3-SEED MEAN ± STD"
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
    path = "./cord-lisc-base-seed{}/eval_results.json".format(seed)

    print("\nSeed {}:".format(seed))

    if not os.path.exists(path):
        print("  [WARNING] Missing:", path)
        continue

    with open(path, "r") as f:
        data = json.load(f)

    for metric in metrics:
        if metric in data:
            value = float(data[metric])
            results[metric].append(value)
            print("  {:18s} = {:.6f}".format(metric, value))

print("\n" + "=" * 70)
print("FINAL LISC RESULT: MEAN ± STD")
print("=" * 70)

summary = {}

for metric in metrics:
    values = results[metric]
    if not values:
        print("{:18s}: NO DATA".format(metric))
        continue

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0

    print(
        "{:18s}: {:.4f} ± {:.4f}".format(
            metric,
            mean,
            std
        )
    )

    summary[metric] = {
        "values": values,
        "mean": float(mean),
        "std": float(std),
    }

output_summary_file = "cord_lisc_3seed_summary.json"
with open(output_summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("Saved:", output_summary_file)
PY