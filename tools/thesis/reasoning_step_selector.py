import argparse
import json
import re
from dataclasses import dataclass


KEYWORDS = {
    "color", "wing", "head", "tail", "beak", "body", "breast", "stripe",
    "spot", "shape", "pattern", "texture", "left", "right", "bird",
}


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+|\n+", text or "") if p.strip()]


def score_sentence(sentence):
    lower = sentence.lower()
    keyword_score = sum(1 for word in KEYWORDS if word in lower)
    length_penalty = len(sentence.split()) / 80.0
    return keyword_score - length_penalty


def heuristic_select(text, topk=2):
    scored = sorted(((score_sentence(s), s) for s in split_sentences(text)), reverse=True)
    return [sentence for _score, sentence in scored[:topk]]


@dataclass
class RLDesign:
    state: str = "candidate evidence text, bbox quality features, answer confidence, token length, latency"
    action: str = "select or drop each candidate reasoning/evidence step"
    reward: str = "answer_correct_reward - length_penalty - latency_penalty"
    policy: str = "placeholder lightweight policy network; not trained in this run"


def main():
    parser = argparse.ArgumentParser(description="Heuristic key-step selector and RL design stub.")
    parser.add_argument("--input-text", default="")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--print-rl-design", action="store_true")
    args = parser.parse_args()
    payload = {"selected_steps": heuristic_select(args.input_text, args.topk)}
    if args.print_rl_design:
        payload["rl_design"] = RLDesign().__dict__
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
