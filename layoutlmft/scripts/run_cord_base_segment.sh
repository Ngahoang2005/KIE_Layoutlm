#!/bin/bash
set -e

cd /home/s24gbn1/Documents/httn/unilm/layoutlmv3
export PYTHONPATH="/home/s24gbn1/Documents/httn/unilm/layoutlmv3:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true 

SEEDS=(42 123 1993)
BASE_OUT_DIR="./logs/cord-base-segment-token-0.03"
BROADCAST_MODE=${BROADCAST_MODE:-"additive"}  # default = additive (fix: cộng dồn context-delta vào token gốc thay vì ghi đè hoàn toàn)
for SEED in "${SEEDS[@]}"
do
    OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
    
    echo ""
    echo "============================================================"
    echo "RUNNING CORD BASE (SEGMENT HEAD) - SEED = ${SEED}"
    echo "============================================================"
    
    rm -rf "$OUT_DIR"

    python -m torch.distributed.run \
      --nproc_per_node=1 \
      --master_port=4400 \
      examples/run_funsd_cord.py \
      --dataset_name cord \
      --do_train --do_eval --do_predict \
      --use_segment_head \
      --model_name_or_path models/layoutlmv3-base \
      --output_dir "$OUT_DIR" \
      --segment_level_layout 1 --visual_embed 1 --input_size 224 \
      --max_steps 1000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
      --learning_rate 5e-5 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 32 \
      --dataloader_num_workers 4 \
      --report_to none \
      --seed "$SEED" \
      --overwrite_output_dir --overwrite_cache \
      --use_hierarchical_position_encoding \
      --use_hierarchical_position_encoding \
      --max_line_position 80 \
      --max_block_position 15 \
      --use_column_encoding True \
      --max_column_position 8 \
      --use_intra_line_boundary True \
      --lambda_bound_init 0.1 \
     

done

echo ""
echo "============================================================"
echo "CALCULATING 3-SEED MEAN ± STD (CORD BASE)"
echo "============================================================"

export BASE_OUT_DIR
python - <<'PY'
import os, json, numpy as np
seeds = [42, 123, 1993]
metrics = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_loss"]
results = {m: [] for m in metrics}
base_dir = os.environ['BASE_OUT_DIR']

for seed in seeds:
    path = f"{base_dir}-seed{seed}/eval_results.json"
    print(f"\nSeed {seed}:")
    if not os.path.exists(path):
        print("  [WARNING] Missing:", path)
        continue
    with open(path, "r") as f:
        data = json.load(f)
    for metric in metrics:
        if metric in data:
            results[metric].append(float(data[metric]))
            print(f"  {metric:18s} = {data[metric]:.6f}")

print("\n" + "=" * 70)
print("FINAL RESULT: MEAN ± STD")
print("=" * 70)
summary = {}
for metric in metrics:
    values = results[metric]
    if not values:
        continue
    mean, std = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0
    print(f"{metric:18s}: {mean:.4f} ± {std:.4f}")
    summary[metric] = {"values": values, "mean": float(mean), "std": float(std)}

out_file = f"{base_dir}_3seed_summary.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)
print("=" * 70)
print("Saved:", out_file)
PY
