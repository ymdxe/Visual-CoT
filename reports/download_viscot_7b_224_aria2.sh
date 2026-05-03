#!/usr/bin/env bash
set -euo pipefail
cd /root/code/Visual-CoT
mkdir -p checkpoints/VisCoT-7b-224 reports
LOG="reports/viscot_7b_224_aria2.log"
exec >> "$LOG" 2>&1
printf '\n=== VisCoT-7b-224 aria2 download start: %s ===\n' "$(date '+%F %T')"
BASE="https://huggingface.co/deepcs233/VisCoT-7b-224/resolve/main"
FILES=(
  "pytorch_model-00001-of-00002.bin"
  "pytorch_model-00002-of-00002.bin"
)
for f in "${FILES[@]}"; do
  printf '[%s] downloading %s\n' "$(date '+%F %T')" "$f"
  aria2c \
    -c -x 8 -s 8 -j 1 --min-split-size=4M --max-connection-per-server=8 \
    --connect-timeout=30 --timeout=60 --retry-wait=10 --max-tries=0 \
    --summary-interval=60 --file-allocation=none --allow-overwrite=true --auto-file-renaming=false \
    -d checkpoints/VisCoT-7b-224 -o "$f" \
    "$BASE/$f"
  ls -lh "checkpoints/VisCoT-7b-224/$f"
done
printf '[%s] final checkpoint files\n' "$(date '+%F %T')"
find checkpoints/VisCoT-7b-224 -maxdepth 1 -type f -printf '%f %s\n' | sort
du -sh checkpoints/VisCoT-7b-224
touch reports/viscot_7b_224_aria2.done
printf '=== VisCoT-7b-224 aria2 download done: %s ===\n' "$(date '+%F %T')"
