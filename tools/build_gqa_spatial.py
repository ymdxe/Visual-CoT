#!/usr/bin/env python3
"""Filter GQA samples by spatial-relation keywords in the question expression.

Outputs:
- viscot_benchmark/benchmark/gqa_spatial.json (conversation-style, subset of gqa.json)
- viscot_benchmark/benchmark_det/gqa_spatial.jsonl (detection-style, subset of gqa.jsonl)

Whitelist regex: prepositional / locational relations only. Avoid pure attribute questions.
"""
import json
import re
from pathlib import Path

SRC_BENCH = Path("viscot_benchmark/benchmark/gqa.json")
SRC_DET = Path("viscot_benchmark/benchmark_det/gqa.jsonl")
OUT_BENCH = Path("viscot_benchmark/benchmark/gqa_spatial.json")
OUT_DET = Path("viscot_benchmark/benchmark_det/gqa_spatial.jsonl")

PATTERN = re.compile(
    r"\b("
    r"in front of|behind|to the (left|right) of|above|below|under|on top of|"
    r"next to|near|beside|between|"
    r"on the (left|right) (of|side)|behind the|in the middle of|"
    r"in the (back|front)|to the (left|right)"
    r")\b",
    re.IGNORECASE,
)


def is_spatial_text(text: str) -> bool:
    if not text:
        return False
    return bool(PATTERN.search(text))


def conv_question(sample):
    convs = sample.get("conversations") or []
    if not convs:
        return ""
    t = convs[0].get("value", "")
    t = t.replace("<image>\n", "").replace("<image>", "")
    for marker in (
        "Please provide the bounding box coordinate of the region",
        "Please provide the bounding box",
    ):
        if marker in t:
            t = t.split(marker)[0]
            break
    return t.strip()


def main():
    bench = json.load(open(SRC_BENCH))
    kept_qids = set()
    kept_bench = []
    for s in bench:
        q = conv_question(s)
        if is_spatial_text(q):
            kept_bench.append(s)
            kept_qids.add(s.get("question_id"))
    OUT_BENCH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(kept_bench, open(OUT_BENCH, "w"), ensure_ascii=False, indent=2)
    print(f"[bench] {len(kept_bench)}/{len(bench)} samples written to {OUT_BENCH}")

    det_kept = 0
    det_total = 0
    with open(OUT_DET, "w") as out:
        for line in open(SRC_DET):
            if not line.strip():
                continue
            det_total += 1
            r = json.loads(line)
            qid = r.get("question_id")
            expr = r.get("expression", "")
            if qid in kept_qids or is_spatial_text(expr):
                out.write(line)
                det_kept += 1
    print(f"[det]   {det_kept}/{det_total} rows written to {OUT_DET}")


if __name__ == "__main__":
    main()
