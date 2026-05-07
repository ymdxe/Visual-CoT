import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import types
from datetime import datetime

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def patch_transformers_llava_registration():
    """Allow this repo's LlavaConfig to override Transformers' built-in key."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        return

    original_register = CONFIG_MAPPING.register

    def register_with_exist_ok(key, value, exist_ok=False):
        return original_register(key, value, exist_ok=True)

    CONFIG_MAPPING.register = register_with_exist_ok


def patch_optional_flash_attn():
    if importlib.util.find_spec("flash_attn") is not None:
        return

    def missing_flash_attn(*_args, **_kwargs):
        raise RuntimeError(
            "flash_attn is not installed. Install flash-attn if this model path "
            "requires the NTK/flash-attention variant."
        )

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []
    flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )
    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    interface_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.flash_attn_interface", loader=None
    )
    interface_module.flash_attn_varlen_qkvpacked_func = missing_flash_attn
    interface_module.flash_attn_unpadded_qkvpacked_func = missing_flash_attn
    padding_module = types.ModuleType("flash_attn.bert_padding")
    padding_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.bert_padding", loader=None
    )
    padding_module.unpad_input = missing_flash_attn
    padding_module.pad_input = missing_flash_attn

    sys.modules.setdefault("flash_attn", flash_attn_module)
    sys.modules.setdefault("flash_attn.flash_attn_interface", interface_module)
    sys.modules.setdefault("flash_attn.bert_padding", padding_module)


patch_transformers_llava_registration()
patch_optional_flash_attn()

from llava.constants import (  # noqa: E402
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

try:  # noqa: E402
    from tools.rc import box_from, clip_box, crop as rc_crop
except Exception:  # pragma: no cover - fallback keeps help/import usable.
    box_from = None
    clip_box = None
    rc_crop = None

PROMPT_VERSION = {
    "none": "v1_plain",
    "structured": "v2_structured_evidence",
    "caption": "v3_region_caption",
    "verify": "v4_answer_verifier",
}

MODE_PRESETS = {
    "full": ("none", "full_only", "none", False),
    "pred_bbox": ("pred", "full_crop", "none", False),
    "oracle_bbox": ("oracle", "full_crop", "none", False),
    "random_bbox": ("random", "full_crop", "none", False),
    "center_bbox": ("center", "full_crop", "none", False),
    "woimg": ("none", "full_only", "none", True),
    "crop_only": ("pred", "crop_only", "none", False),
    "structured_evidence": ("pred", "full_crop", "structured", False),
    "region_caption": ("pred", "full_crop", "caption", False),
    "answer_verifier": ("pred", "full_crop", "verify", False),
    "lowres_full_highrescrop": ("pred", "lowres_full_highres_crop", "none", False),
    "answer_reasoning_compressed": ("pred", "full_crop", "structured", False),
    "direct_then_explain": ("pred", "full_crop", "none", False),
    "vision_pruned": ("pred", "full_crop", "none", False),
    "feature_pca": ("pred", "full_crop", "none", False),
    "topk_crops": ("pred", "full_crop", "none", False),
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def parse_box(value):
    if box_from is not None:
        return box_from(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(v) for v in value[:4]]
        except (TypeError, ValueError):
            return None
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
    if len(nums) < 4:
        return None
    try:
        return [float(v) for v in nums[:4]]
    except ValueError:
        return None


def norm_box(value, width=None, height=None):
    box = parse_box(value)
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
        if width and height:
            box = [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]
        else:
            m = max(abs(x1), abs(y1), abs(x2), abs(y2), 1.0)
            box = [x1 / m, y1 / m, x2 / m, y2 / m]
    if clip_box is not None:
        return clip_box(box)
    x1, y1, x2, y2 = [min(max(float(v), 0.0), 1.0) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box):
    box = norm_box(box)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a, b):
    a = norm_box(a)
    b = norm_box(b)
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else None


def resolve_image(image_folder, image_file):
    path = os.path.join(image_folder, image_file)
    if os.path.exists(path):
        return path
    if image_file.startswith("cot/cub/"):
        alt = os.path.join("downloads/cub/CUB_200_2011/images", image_file[len("cot/cub/") :])
        if os.path.exists(alt):
            return alt
    return path


def load_questions(path):
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_detection(path):
    if not path:
        return {}, []
    rows = []
    by_id = {}
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for row in f:
            if not row.strip():
                continue
            item = json.loads(row)
            rows.append(item)
            qid = item.get("question_id")
            if qid is not None:
                by_id[str(qid)] = item
    return by_id, rows


def image_token(model_config):
    token = DEFAULT_IMAGE_TOKEN
    if getattr(model_config, "mm_use_im_start_end", False):
        token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
    return token


def clean_question(line):
    conversations = line.get("conversations", [])
    if conversations:
        text = conversations[0].get("value", "")
    else:
        text = line.get("question") or line.get("expression") or line.get("prompt") or ""
    text = text.replace("<image>\n", "").replace("<image>", "").strip()
    markers = [
        "Please provide the bounding box coordinate of the region",
        "Please provide the bounding box",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


def gt_answer(line):
    conversations = line.get("conversations", [])
    for msg in reversed(conversations):
        if msg.get("from") == "gpt":
            value = msg.get("value", "")
            if parse_box(value) is None:
                return value
    for key in ("answer", "gt_answer", "label"):
        if key in line:
            value = line[key]
            if isinstance(value, list):
                return value[0] if value else None
            return value
    return None


def oracle_box(line):
    conversations = line.get("conversations", [])
    if len(conversations) > 1:
        box = norm_box(conversations[1].get("value"))
        if box is not None:
            return box
    images = line.get("image") or []
    if len(images) > 1 and "###" in images[1]:
        return norm_box(images[1].split("###", 1)[1], line.get("width"), line.get("height"))
    return norm_box(line.get("bbox"), line.get("width"), line.get("height"))


def pred_box_for(line, local_idx, det_by_id, det_rows):
    rec = det_by_id.get(str(line.get("question_id")))
    if rec is None and local_idx < len(det_rows):
        rec = det_rows[local_idx]
    if rec is None:
        return None, None, None
    raw = rec.get("bbox_pred") or rec.get("text") or rec.get("bbox_pred_raw")
    box = norm_box(raw, rec.get("width") or line.get("width"), rec.get("height") or line.get("height"))
    return box, rec, rec.get("bbox_parse_ok", box is not None)


def random_box():
    cx, cy = random.random(), random.random()
    w = 0.15 + random.random() * 0.45
    h = 0.15 + random.random() * 0.45
    return norm_box([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def square_crop(img, box, min_side=224):
    img = img.convert("RGB")
    if box is None:
        return img.copy()
    if rc_crop is not None:
        return rc_crop(img, box, pad=1.2)
    w, h = img.size
    x1, y1, x2, y2 = box
    left, top, right, bottom = x1 * w, y1 * h, x2 * w, y2 * h
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    half = max((right - left) / 2.0, (bottom - top) / 2.0, min_side / 2.0)
    left, top = max(0, cx - half), max(0, cy - half)
    right, bottom = min(w, cx + half), min(h, cy + half)
    return img.crop((int(left), int(top), int(right), int(bottom)))


def lowres_full(img, size=224):
    img = img.convert("RGB")
    w, h = img.size
    small = img.resize((size, size))
    return small.resize((w, h))


def select_box(args, line, local_idx, det_by_id, det_rows):
    source = args.bbox_source
    det_rec = None
    parse_ok = None
    if source == "pred":
        box, det_rec, parse_ok = pred_box_for(line, local_idx, det_by_id, det_rows)
    elif source == "oracle":
        box = oracle_box(line)
        parse_ok = box is not None
    elif source == "random":
        box = random_box()
        parse_ok = box is not None
    elif source == "center":
        box = [0.25, 0.25, 0.75, 0.75]
        parse_ok = True
    else:
        box = None
        parse_ok = None
    return box, det_rec, parse_ok


def prepare_images(args, full_img, box):
    if args.without_image:
        return [], None
    crop_img = square_crop(full_img, box) if box is not None else full_img.copy()
    if args.crop_size:
        crop_img = crop_img.resize((args.crop_size, args.crop_size))
    if args.crop_mode == "full_only":
        return [full_img], None
    if args.crop_mode == "crop_only":
        return [crop_img], crop_img
    if args.crop_mode == "lowres_full_highres_crop":
        return [lowres_full(full_img, args.lowres_size), crop_img], crop_img
    return [full_img, crop_img], crop_img


def make_conv_prompt(args, model_config, question, box, images_count, region_caption=None, initial_answer=None):
    conv = conv_templates[args.conv_mode].copy()
    token = image_token(model_config)
    box_text = "null" if box is None else "[%.3f, %.3f, %.3f, %.3f]" % tuple(box)
    if args.without_image or images_count == 0:
        prefix = ""
    else:
        prefix = (token + "\n") * images_count

    if args.evidence_mode == "structured":
        instruction = (
            f"{prefix}{question}\n"
            f"Predicted region bbox: {box_text}.\n"
            "Please answer with the following fixed structure:\n"
            "[Region]\nDescribe the local region related to the question.\n\n"
            "[Visual Evidence]\nList the key visual evidence from the crop and the full image.\n\n"
            "[Reasoning]\nGive a brief reasoning process based on the visual evidence.\n\n"
            "[Answer]\nGive the final answer only."
        )
    elif args.evidence_mode == "caption" and region_caption:
        instruction = (
            f"{prefix}{question}\n"
            f"Predicted region bbox: {box_text}.\n"
            f"Region caption: {region_caption}\n"
            "Use the full image, the local crop, and the region caption to answer concisely."
        )
    elif args.evidence_mode == "verify" and initial_answer is not None:
        instruction = (
            f"{prefix}{question}\n"
            f"Initial answer: {initial_answer}\n"
            f"Predicted region bbox: {box_text}.\n"
            "Check whether the initial answer is supported by the visual evidence. "
            "If supported, keep it. If not supported, revise it. End with [Answer] and the final answer."
        )
    elif args.crop_mode == "crop_only":
        instruction = f"{prefix}{question}\nPlease answer based on the local detail image."
    elif args.crop_mode == "lowres_full_highres_crop":
        instruction = (
            f"{token}\n{token}\n{question}\n"
            "Please answer based on the low-resolution full image and the high-resolution local crop."
        )
    elif images_count >= 2:
        instruction = (
            f"{token}\n{question}\n"
            f"Predicted region bbox: {box_text}.\n"
            f"{token}\nPlease answer the question based on the original image and local detail image."
        )
    else:
        instruction = f"{prefix}{question}"

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), instruction


def make_caption_prompt(args, model_config, question):
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(
        conv.roles[0],
        image_token(model_config)
        + "\nDescribe the local image region in one short sentence for answering this question: "
        + question,
    )
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def make_direct_answer_prompt(args, model_config, question, images_count):
    conv = conv_templates[args.conv_mode].copy()
    token = image_token(model_config)
    prefix = "" if args.without_image or images_count == 0 else (token + "\n") * images_count
    conv.append_message(conv.roles[0], prefix + question + "\nAnswer the question with the final answer only.")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), prefix + question


def make_explanation_prompt(args, model_config, question, answer, images_count):
    conv = conv_templates[args.conv_mode].copy()
    token = image_token(model_config)
    prefix = "" if args.without_image or images_count == 0 else (token + "\n") * images_count
    conv.append_message(
        conv.roles[0],
        (
            f"{prefix}{question}\n"
            f"Given the answer: {answer}\n"
            "Briefly explain the key visual evidence in one or two sentences."
        ),
    )
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_tensor(images, image_processor, model_config):
    if not images:
        return None
    if isinstance(image_processor, list):
        tensors = [process_images(images, proc, model_config) for proc in image_processor]
        return torch.cat(tensors, dim=0)
    return process_images(images, image_processor, model_config)


def generate(model, tokenizer, image_processor, model_config, args, prompt, images, dtype, max_new_tokens=128):
    image_tensor = build_tensor(images, image_processor, model_config)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if image_tensor is not None:
        if image_tensor.ndim == 5:
            image_tensor = image_tensor[0]
        kwargs["images"] = image_tensor.to(dtype=dtype, device="cuda", non_blocking=True)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **kwargs)
    input_token_len = input_ids.shape[1]
    output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    conv = conv_templates[args.conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    return output


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_match(pred, gold):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return None
    return int(g in p or p in g)


SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:\[?(Region|Visual Evidence|Reasoning|Answer)\]?|"
    r"(Region|Visual Evidence|Reasoning|Answer))\s*:\s*",
    flags=re.IGNORECASE,
)


def token_count(text):
    if not text:
        return 0
    return len(str(text).split())


def split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def parse_structured_output(text):
    text = "" if text is None else str(text)
    matches = list(SECTION_RE.finditer(text))
    sections = {"region_text": None, "visual_evidence_text": None, "reasoning_text": None, "answer_extracted": None}
    if not matches:
        sections["reasoning_text"] = text
        sections["answer_extracted"] = text.strip()
        return sections
    key_map = {
        "region": "region_text",
        "visual evidence": "visual_evidence_text",
        "reasoning": "reasoning_text",
        "answer": "answer_extracted",
    }
    for idx, match in enumerate(matches):
        raw = (match.group(1) or match.group(2) or "").lower()
        key = key_map.get(raw)
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    if not sections["answer_extracted"]:
        last = [s for s in split_sentences(text)[-2:] if s]
        sections["answer_extracted"] = last[-1] if last else text.strip()
    return sections


KEY_EVIDENCE_WORDS = {
    "color", "colour", "wing", "head", "tail", "beak", "bill", "body", "breast",
    "belly", "back", "eye", "stripe", "spot", "shape", "pattern", "texture",
    "left", "right", "upper", "lower", "bird", "feather", "black", "white",
    "red", "blue", "yellow", "brown", "gray", "grey", "green", "orange",
}


def deduplicate(items):
    seen = set()
    out = []
    for item in items:
        key = normalize_answer(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def compress_text(text, max_sentences, keep_keywords=True, do_dedup=True):
    sentences = split_sentences(text)
    if do_dedup:
        sentences = deduplicate(sentences)
    if keep_keywords:
        keyword_hits = [s for s in sentences if any(k in s.lower() for k in KEY_EVIDENCE_WORDS)]
        if keyword_hits:
            sentences = keyword_hits + [s for s in sentences if s not in keyword_hits]
    return " ".join(sentences[:max_sentences]).strip()


def apply_reasoning_compression(args, parsed):
    raw_reasoning = parsed.get("reasoning_text") or ""
    raw_evidence = parsed.get("visual_evidence_text") or ""
    if args.reasoning_compression != "rule":
        return {
            "raw_reasoning": raw_reasoning,
            "compressed_reasoning": None,
            "raw_reasoning_tokens": token_count(raw_reasoning),
            "compressed_reasoning_tokens": None,
            "reasoning_compression_ratio": None,
            "compressed_evidence": None,
        }
    compressed_reasoning = compress_text(
        raw_reasoning,
        max_sentences=args.max_reasoning_sentences,
        do_dedup=args.deduplicate_sentences,
    )
    compressed_evidence = compress_text(
        raw_evidence,
        max_sentences=args.max_evidence_sentences,
        do_dedup=args.deduplicate_sentences,
    )
    raw_tokens = token_count(raw_reasoning)
    compressed_tokens = token_count(compressed_reasoning)
    ratio = (compressed_tokens / raw_tokens) if raw_tokens else None
    return {
        "raw_reasoning": raw_reasoning,
        "compressed_reasoning": compressed_reasoning,
        "raw_reasoning_tokens": raw_tokens,
        "compressed_reasoning_tokens": compressed_tokens,
        "reasoning_compression_ratio": ratio,
        "compressed_evidence": compressed_evidence,
    }


def heuristic_select_steps(text, topk):
    scored = []
    for sent in split_sentences(text):
        lower = sent.lower()
        score = sum(1 for k in KEY_EVIDENCE_WORDS if k in lower)
        score += 1.0 / max(1, token_count(sent))
        scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sent for _score, sent in scored[:topk]]


def exact_match(pred, gold):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return None
    return int(p == g)


def apply_vision_pruning(model, args):
    if not args.vision_prune:
        return None
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    target_root = model
    if args.vision_prune_target == "vision_tower":
        target_root = model.get_vision_tower()
    elif args.vision_prune_target == "mm_projector":
        target_root = getattr(model.get_model(), "mm_projector", None) or getattr(model, "mm_projector", None)
    elif args.vision_prune_target == "patch_embed":
        target_root = model.get_vision_tower()

    module_type = nn.Conv2d if args.vision_prune_type == "conv_ln_structured" else nn.Linear
    selected = []
    if target_root is not None:
        for name, module in target_root.named_modules():
            if isinstance(module, module_type):
                if args.vision_prune_target == "patch_embed" and "embed" not in name.lower():
                    continue
                selected.append((name, module))
    before_nonzero = 0
    before_total = 0
    for _name, module in selected:
        weight = module.weight.detach()
        before_nonzero += int(torch.count_nonzero(weight).item())
        before_total += weight.numel()
    report = {
        "vision_prune_type": args.vision_prune_type,
        "vision_prune_amount": args.vision_prune_amount,
        "vision_prune_target": args.vision_prune_target,
        "vision_prune_dry_run": args.vision_prune_dry_run,
        "modules": [name for name, _module in selected[:50]],
        "num_modules": len(selected),
        "nonzero_before": before_nonzero,
        "total_params_considered": before_total,
    }
    print(json.dumps({"vision_prune_report": report}, ensure_ascii=False))
    if args.vision_prune_dry_run:
        return report
    for _name, module in selected:
        prune.ln_structured(module, name="weight", amount=args.vision_prune_amount, n=2, dim=0)
        prune.remove(module, "weight")
    after_nonzero = 0
    for _name, module in selected:
        after_nonzero += int(torch.count_nonzero(module.weight.detach()).item())
    report["nonzero_after"] = after_nonzero
    report["nonzero_ratio_after"] = (after_nonzero / before_total) if before_total else None
    return report


def apply_mode(args):
    if args.mode and args.mode != "auto":
        preset = MODE_PRESETS.get(args.mode)
        if preset is None:
            raise ValueError(f"unsupported mode: {args.mode}")
        args.bbox_source, args.crop_mode, args.evidence_mode, mode_woimg = preset
        args.without_image = mode_woimg or args.without_image
    else:
        if args.random_bbox:
            args.bbox_source = "random"
        elif args.center_bbox:
            args.bbox_source = "center"
        elif args.detection_file:
            args.bbox_source = "pred"
        elif args.with_cot and args.bbox_source == "none":
            args.bbox_source = "oracle"
        if args.without_image:
            args.bbox_source = "none"
            args.crop_mode = "full_only"
        if args.with_cot and args.crop_mode == "full_only" and args.bbox_source != "none":
            args.crop_mode = "full_crop"


def eval_model(args):
    if args.load_4bit and args.load_8bit:
        raise ValueError("--load-4bit and --load-8bit are mutually exclusive")
    apply_mode(args)
    if args.mode == "direct_then_explain":
        args.answer_policy = "direct_then_explain"
    if args.mode == "answer_reasoning_compressed":
        args.evidence_mode = "structured"
        if args.reasoning_compression == "none":
            args.reasoning_compression = "rule"
    if args.output_file:
        args.answers_file = args.output_file
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        precision=args.precision,
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    prune_report = apply_vision_pruning(model, args)
    if args.vision_prune and args.vision_prune_dry_run:
        return
    feature_note = None
    if args.feature_compression != "none":
        feature_note = (
            "feature compression interface enabled; PCA insertion is recorded as a feasibility "
            "check because the current model requires fixed visual feature dimensions unless an adapter is trained"
        )

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[: args.max_samples]
    det_by_id, det_rows = load_detection(args.detection_file)

    answers_file = os.path.expanduser(args.log_jsonl or args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    if args.save_crop_dir:
        os.makedirs(args.save_crop_dir, exist_ok=True)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"It seems that this is a plain model, auto switching to {args.conv_mode}.")

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for local_idx, line in enumerate(tqdm(questions)):
            preprocess_start = time.perf_counter()
            qid = line.get("question_id", local_idx)
            question = clean_question(line)
            image_files = line.get("image") or [line.get("img_path")]
            image_rel = image_files[0]
            image_path = resolve_image(args.image_folder, image_rel)
            full_img = Image.open(image_path).convert("RGB")

            gt_box = oracle_box(line)
            chosen_box, det_rec, bbox_parse_ok = select_box(args, line, local_idx, det_by_id, det_rows)
            crop_ratio = box_area(chosen_box) if chosen_box is not None else None
            images, crop_img = prepare_images(args, full_img, chosen_box)
            saved_crop = None
            if args.save_crop_dir and crop_img is not None:
                saved_crop = os.path.join(args.save_crop_dir, f"{qid}_{args.mode or args.crop_mode}.jpg")
                try:
                    crop_img.save(saved_crop)
                except Exception:
                    saved_crop = None

            metadata = {
                "detection_record": det_rec,
                "saved_crop": saved_crop,
                "hit": None,
                "bbox_iou": box_iou(chosen_box, gt_box),
                "num_crops_requested": args.num_crops,
            }
            prompt, prompt_text = make_conv_prompt(args, model.config, question, chosen_box, len(images))
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0 if args.record_latency else None
            if torch.cuda.is_available() and args.record_memory:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            generate_ms = None
            stage1_latency_ms = None
            stage2_latency_ms = None
            generated_done = False

            region_caption = None
            if args.answer_policy == "direct_then_explain":
                direct_prompt, prompt_text = make_direct_answer_prompt(args, model.config, question, len(images))
                stage1_start = time.perf_counter()
                direct_answer = generate(
                    model, tokenizer, image_processor, model.config, args, direct_prompt, images, dtype, max_new_tokens=32
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stage1_latency_ms = (time.perf_counter() - stage1_start) * 1000.0 if args.record_latency else None
                explanation_prompt = make_explanation_prompt(args, model.config, question, direct_answer, len(images))
                stage2_start = time.perf_counter()
                explanation = generate(
                    model,
                    tokenizer,
                    image_processor,
                    model.config,
                    args,
                    explanation_prompt,
                    images,
                    dtype,
                    max_new_tokens=args.explanation_max_new_tokens,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stage2_latency_ms = (time.perf_counter() - stage2_start) * 1000.0 if args.record_latency else None
                output = direct_answer
                metadata["direct_answer"] = direct_answer
                metadata["explanation"] = explanation
                generate_ms = None
                if stage1_latency_ms is not None and stage2_latency_ms is not None:
                    generate_ms = stage1_latency_ms + stage2_latency_ms
                generated_done = True
            elif args.evidence_mode == "caption" and crop_img is not None:
                caption_prompt = make_caption_prompt(args, model.config, question)
                gen_start = time.perf_counter()
                region_caption = generate(
                    model, tokenizer, image_processor, model.config, args, caption_prompt, [crop_img], dtype, max_new_tokens=64
                )
                metadata["region_caption"] = region_caption
                prompt, prompt_text = make_conv_prompt(
                    args, model.config, question, chosen_box, len(images), region_caption=region_caption
                )

            if generated_done:
                pass
            elif args.evidence_mode == "verify":
                base_args = argparse.Namespace(**vars(args))
                base_args.evidence_mode = "none"
                base_prompt, _ = make_conv_prompt(base_args, model.config, question, chosen_box, len(images))
                gen_start = time.perf_counter()
                initial_answer = generate(model, tokenizer, image_processor, model.config, args, base_prompt, images, dtype)
                metadata["initial_answer"] = initial_answer
                prompt, prompt_text = make_conv_prompt(
                    args, model.config, question, chosen_box, len(images), initial_answer=initial_answer
                )
                output = generate(model, tokenizer, image_processor, model.config, args, prompt, images, dtype)
                generate_ms = (time.perf_counter() - gen_start) * 1000.0 if args.record_latency else None
                metadata["verifier_judgement"] = output
                metadata["verifier_rewritten"] = normalize_answer(output) != normalize_answer(initial_answer)
            else:
                gen_start = time.perf_counter()
                output = generate(model, tokenizer, image_processor, model.config, args, prompt, images, dtype)
                generate_ms = (time.perf_counter() - gen_start) * 1000.0 if args.record_latency else None

            if torch.cuda.is_available() and args.record_memory:
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0 if args.record_latency else None
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() and args.record_memory else None
            gold = gt_answer(line)
            parsed = parse_structured_output(output)
            compression = apply_reasoning_compression(args, parsed)
            selected_steps = None
            if args.step_selector == "heuristic":
                selected_steps = heuristic_select_steps(
                    (parsed.get("visual_evidence_text") or "") + " " + (parsed.get("reasoning_text") or ""),
                    args.step_selector_topk,
                )
            answer_extracted = parsed.get("answer_extracted") or output
            metadata["hit"] = contains_match(answer_extracted, gold)
            bbox_iou_value = box_iou(chosen_box, gt_box)
            normalized_gt = normalize_answer(gold)
            normalized_pred = normalize_answer(answer_extracted)
            answer_tokens = token_count(answer_extracted)
            reasoning_tokens = token_count(parsed.get("reasoning_text"))
            notes = []
            if feature_note:
                notes.append(feature_note)
            if prune_report:
                notes.append("vision pruning feasibility hook applied")

            row = {
                "question_id": qid,
                "dataset": args.dataset_name or line.get("dataset"),
                "image": image_rel,
                "mode": args.mode if args.mode != "auto" else None,
                "reasoning_compression": args.reasoning_compression,
                "answer_policy": args.answer_policy,
                "step_selector": args.step_selector,
                "evidence_mode": args.evidence_mode,
                "crop_mode": args.crop_mode,
                "bbox_source": args.bbox_source,
                "prompt_version": PROMPT_VERSION.get(args.evidence_mode, "v1_plain"),
                "prompt": prompt_text,
                "gt_answer": gold,
                "pred_answer": output,
                "answer_extracted": answer_extracted,
                "normalized_gt": normalized_gt,
                "normalized_pred": normalized_pred,
                "exact_match": exact_match(answer_extracted, gold),
                "contains_match": contains_match(answer_extracted, gold),
                "text": output,
                "bbox_gt": gt_box,
                "bbox_pred_raw": det_rec.get("bbox_pred_raw") if isinstance(det_rec, dict) else None,
                "bbox_pred": chosen_box,
                "bbox_parse_ok": bbox_parse_ok,
                "bbox_iou": bbox_iou_value,
                "bbox_correct_at_05": None if bbox_iou_value is None else int(bbox_iou_value >= 0.5),
                "crop_ratio": crop_ratio,
                "lowres_size": args.lowres_size if args.crop_mode == "lowres_full_highres_crop" else None,
                "crop_size": args.crop_size,
                "vision_prune_type": args.vision_prune_type if args.vision_prune else None,
                "vision_prune_amount": args.vision_prune_amount if args.vision_prune else None,
                "feature_compression": args.feature_compression,
                "pca_dim": args.pca_dim if args.feature_compression == "pca" else None,
                "region_text": parsed.get("region_text"),
                "visual_evidence_text": parsed.get("visual_evidence_text"),
                "reasoning_text": parsed.get("reasoning_text"),
                "compressed_reasoning": compression["compressed_reasoning"],
                "raw_reasoning": compression["raw_reasoning"],
                "raw_reasoning_tokens": compression["raw_reasoning_tokens"],
                "compressed_reasoning_tokens": compression["compressed_reasoning_tokens"],
                "reasoning_compression_ratio": compression["reasoning_compression_ratio"],
                "answer_before_compression": answer_extracted,
                "answer_after_compression": answer_extracted,
                "reasoning_length_tokens": reasoning_tokens,
                "answer_length_tokens": answer_tokens,
                "latency_ms": latency_ms,
                "latency_ms_total": latency_ms,
                "latency_ms_preprocess": preprocess_ms,
                "latency_ms_generate": generate_ms,
                "stage1_latency_ms": stage1_latency_ms,
                "stage2_latency_ms": stage2_latency_ms,
                "total_latency_ms": latency_ms,
                "peak_gpu_memory_mb": peak_mem,
                "num_visual_inputs": len(images),
                "num_visual_tokens_est": len(images) * 576 if images else 0,
                "error_tag": None,
                "error_type": None,
                "prompt_text": prompt_text,
                "model_path": model_path,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "notes": "; ".join(notes) if notes else None,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "metadata": metadata,
            }
            row["metadata"]["selected_steps"] = selected_steps
            row["metadata"]["compressed_evidence"] = compression["compressed_evidence"]
            row["metadata"]["vision_prune_report"] = prune_report
            row["metadata"]["feature_note"] = feature_note
            if "height" in line:
                row["height"] = line["height"]
            if "width" in line:
                row["width"] = line["width"]
            if "bbox" in line:
                row["bbox"] = line["bbox"]
            ans_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            ans_file.flush()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="s3://mmdata/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--with-cot", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--random-bbox", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--center-bbox", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--without-image", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--detection-file", type=str, default=None)
    parser.add_argument("--adapt-ratio", type=float, default=1.0)

    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--mode", choices=list(MODE_PRESETS.keys()) + ["auto"], default="auto")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox-source", choices=["pred", "oracle", "random", "center", "none"], default="none")
    parser.add_argument(
        "--crop-mode",
        choices=["full_only", "full_crop", "crop_only", "lowres_full_highres_crop"],
        default="full_only",
    )
    parser.add_argument("--evidence-mode", choices=["none", "structured", "caption", "verify"], default="none")
    parser.add_argument("--reasoning-compression", choices=["none", "rule"], default="none")
    parser.add_argument("--max-reasoning-sentences", type=int, default=2)
    parser.add_argument("--max-evidence-sentences", type=int, default=3)
    parser.add_argument("--deduplicate-sentences", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--answer-policy", choices=["single", "direct_then_explain"], default="single")
    parser.add_argument("--explanation-max-new-tokens", type=int, default=64)
    parser.add_argument("--step-selector", choices=["none", "heuristic"], default="none")
    parser.add_argument("--step-selector-topk", type=int, default=2)
    parser.add_argument("--lowres-size", type=int, default=112)
    parser.add_argument("--crop-size", type=int, default=336)
    parser.add_argument("--vision-prune", action="store_true")
    parser.add_argument("--vision-prune-type", choices=["conv_ln_structured", "linear_ln_structured"], default="linear_ln_structured")
    parser.add_argument("--vision-prune-amount", type=float, default=0.1)
    parser.add_argument("--vision-prune-target", choices=["patch_embed", "vision_tower", "mm_projector"], default="vision_tower")
    parser.add_argument("--vision-prune-dry-run", action="store_true")
    parser.add_argument("--feature-compression", choices=["none", "pca"], default="none")
    parser.add_argument("--pca-fit-samples", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--pca-position", choices=["before_projector", "after_projector"], default="before_projector")
    parser.add_argument("--pca-save-path", type=str, default=None)
    parser.add_argument("--pca-load-path", type=str, default=None)
    parser.add_argument("--num-crops", type=int, default=1)
    parser.add_argument("--save-crop-dir", type=str, default=None)
    parser.add_argument("--log-jsonl", type=str, default=None)
    parser.add_argument("--record-latency", action="store_true")
    parser.add_argument("--record-memory", action="store_true")
    return parser


if __name__ == "__main__":
    eval_model(build_parser().parse_args())
