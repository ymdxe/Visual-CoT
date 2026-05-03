#!/usr/bin/env bash
set -euo pipefail
cd /root/code/Visual-CoT
LOG="reports/cub_finalize.log"
exec >> "$LOG" 2>&1
printf '\n=== finalize start: %s ===\n' "$(date '+%F %T')"
CUB_TGZ="downloads/cub/CUB_200_2011.tgz"
PID=""
if [ -f reports/cub_aria2.pid ]; then
  PID="$(cat reports/cub_aria2.pid || true)"
fi
if [ -n "$PID" ]; then
  while ps -p "$PID" >/dev/null 2>&1; do
    printf '[%s] waiting for aria2 pid=%s\n' "$(date '+%F %T')" "$PID"
    tail -n 5 reports/cub_aria2_download.log || true
    sleep 60
  done
fi
printf '[%s] download process ended\n' "$(date '+%F %T')"
ls -lh "$CUB_TGZ" "$CUB_TGZ.aria2" 2>/dev/null || true
if [ -f "$CUB_TGZ.aria2" ]; then
  echo '[ERROR] aria2 control file still exists, archive is incomplete.'
  exit 1
fi
if [ ! -s "$CUB_TGZ" ]; then
  echo '[ERROR] CUB archive missing or empty.'
  exit 1
fi
printf '[%s] testing archive\n' "$(date '+%F %T')"
tar -tzf "$CUB_TGZ" >/dev/null
if [ ! -d downloads/cub/CUB_200_2011/images ] || [ "$(find downloads/cub/CUB_200_2011/images -type f 2>/dev/null | wc -l)" -lt 10000 ]; then
  printf '[%s] extracting CUB archive\n' "$(date '+%F %T')"
  tar -xzf "$CUB_TGZ" -C downloads/cub
else
  printf '[%s] CUB images already extracted\n' "$(date '+%F %T')"
fi
mkdir -p playground/data/cot
if [ -e playground/data/cot/cub ] && [ ! -L playground/data/cot/cub ]; then
  echo '[ERROR] playground/data/cot/cub exists and is not a symlink; leaving it unchanged.'
  exit 2
fi
ln -sfn ../../../downloads/cub/CUB_200_2011/images playground/data/cot/cub
printf '[%s] CUB image count: ' "$(date '+%F %T')"
find downloads/cub/CUB_200_2011/images -type f | wc -l
ls -ld playground/data/cot/cub
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate viscot
printf '[%s] building small CUB benchmark\n' "$(date '+%F %T')"
python tools/build_small_benchmark.py \
  --benchmark-dir ./viscot_benchmark/benchmark \
  --image-folder ./playground/data \
  --datasets cub \
  --num-per-dataset 3 \
  --output ./small_benchmark.json \
  --det-output ./results/region_compress/detection.jsonl
printf '[%s] testing region compression on one CUB image\n' "$(date '+%F %T')"
SAMPLE_IMAGE=$(find playground/data/cot/cub -type f | head -n 1)
mkdir -p results/region_compress/compressed_images
python tools/region_compress.py \
  --image "$SAMPLE_IMAGE" \
  --bbox '[0.2,0.2,0.6,0.6]' \
  --mode blur \
  --output results/region_compress/compressed_images/cub_blur_smoke.jpg
ls -lh small_benchmark.json results/region_compress/detection.jsonl results/region_compress/compressed_images/cub_blur_smoke.jpg
printf '=== finalize done: %s ===\n' "$(date '+%F %T')"
