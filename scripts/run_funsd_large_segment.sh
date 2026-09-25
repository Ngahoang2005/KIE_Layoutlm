#!/bin/bash
set -e

cd /home/s24gbn1/Documents/httn/unilm/layoutlmv3
export PYTHONPATH="/home/s24gbn1/Documents/httn/unilm/layoutlmv3:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true 

PYTHON_BIN="${PYTHON:-$(command -v python || true)}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: No Python interpreter found. Activate the project env first: conda activate layoutlmv3"
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

SEEDS=(42 123 1993)
BASE_OUT_DIR="./logs/funsd-large-segment-token-0.03"

for SEED in "${SEEDS[@]}"
do
    OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
    
    echo ""
    echo "============================================================"
    echo "RUNNING FUNSD LARGE (SEGMENT HEAD) - SEED = ${SEED}"
    echo "============================================================"

    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node=1 \
      --master_port=4398 \
      examples/run_funsd_cord.py \
      --dataset_name funsd \
      --do_train --do_eval \
      --use_segment_head \
      --model_name_or_path models/layoutlmv3-large \
      --output_dir "$OUT_DIR" \
      --segment_level_layout 1 --visual_embed 1 --input_size 224 \
      --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --dataloader_num_workers 4 \
      --report_to none \
      --seed "$SEED" \
      --overwrite_output_dir --overwrite_cache
done

echo ""
echo "============================================================"
echo "CALCULATING 3-SEED MEAN ± STD (FUNSD LARGE)"
echo "============================================================"

export BASE_OUT_DIR
"${PYTHON_BIN}" - <<'PY'
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
