import argparse
import json
import math
import os
import time
import sys
import types
import importlib.util
import importlib.machinery

import shortuuid
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def patch_transformers_llava_registration():
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
        raise RuntimeError("flash_attn is not installed for this model path.")

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []
    flash_attn_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    interface_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    interface_module.flash_attn_varlen_qkvpacked_func = missing_flash_attn
    interface_module.flash_attn_unpadded_qkvpacked_func = missing_flash_attn
    padding_module = types.ModuleType("flash_attn.bert_padding")
    padding_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    padding_module.unpad_input = missing_flash_attn
    padding_module.pad_input = missing_flash_attn
    sys.modules.setdefault("flash_attn", flash_attn_module)
    sys.modules.setdefault("flash_attn.flash_attn_interface", interface_module)
    sys.modules.setdefault("flash_attn.bert_padding", padding_module)


patch_transformers_llava_registration()
patch_optional_flash_attn()

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

try:
    from tools.rc import box_from, clip_box
except Exception:
    box_from = None
    clip_box = None


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def parse_bbox(text):
    if box_from is not None:
        return box_from(text)
    if text is None:
        return None
    import re
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(text))
    if len(nums) < 4:
        return None
    try:
        return [float(v) for v in nums[:4]]
    except ValueError:
        return None


def norm_bbox(box):
    if clip_box is not None:
        return clip_box(box)
    if box is None:
        return None
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pixel_to_norm(box, width, height):
    if box is None or not width or not height:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return norm_bbox([x1, y1, x2, y2])
    return norm_bbox([x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)])


def resolve_image(image_folder, image_file):
    path = os.path.join(image_folder, image_file)
    if os.path.exists(path):
        return path
    if image_file.startswith("cot/cub/"):
        alt = os.path.join("downloads/cub/CUB_200_2011/images", image_file[len("cot/cub/"):])
        if os.path.exists(alt):
            return alt
    return path


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["img_path"]
        expr = line.get("expression", "")
        qs = f"{expr.lower()}. Please provide the bounding box coordinate of the region that can help you answer the question better."
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_path = resolve_image(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor, prompt

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def maybe_save_vis(args, line, bbox_pred, local_idx):
    if not args.save_vis or bbox_pred is None:
        return None
    out_dir = args.save_vis or os.path.join(os.path.dirname(args.answers_file), "vis")
    os.makedirs(out_dir, exist_ok=True)
    image_path = resolve_image(args.image_folder, line["img_path"])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    w, h = image.size
    x1, y1, x2, y2 = bbox_pred
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline="red", width=max(2, w // 300))
    out = os.path.join(out_dir, f"{local_idx:06d}_{line.get('question_id', local_idx)}.jpg")
    image.save(out)
    return out


def eval_model(args):
    if args.load_4bit and args.load_8bit:
        raise ValueError("--load-4bit and --load-8bit are mutually exclusive")
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

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding="utf-8")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        questions = questions[: args.max_samples]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"It seems that this is a plain model, auto switching to {args.conv_mode}.")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for local_idx, ((input_ids, image_tensor, prompt), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions))):
            idx = line.get("question_id", local_idx)
            height = line.get("height")
            width = line.get("width")
            bbox_gt = line.get("bbox")
            bbox_gt_norm = pixel_to_norm(bbox_gt, width, height)
            cur_prompt = f"{line.get('expression', '').lower()}. Please provide the bounding box coordinate of the region that can help you answer the question better."
            stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            if torch.cuda.is_available() and args.record_memory:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=dtype, device="cuda", non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )
            if torch.cuda.is_available() and args.record_memory:
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0 if args.record_latency else None
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() and args.record_memory else None
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()
            raw_box = outputs
            pred_box = parse_bbox(raw_box)
            pred_box = norm_bbox(pred_box)
            parse_ok = pred_box is not None
            vis_path = maybe_save_vis(args, line, pred_box, local_idx)
            ans_file.write(json.dumps({
                "question_id": idx,
                "dataset": args.dataset_name or line.get("dataset"),
                "image": line.get("img_path"),
                "prompt": cur_prompt,
                "bbox_gt": bbox_gt_norm or bbox_gt,
                "bbox_pred_raw": raw_box,
                "bbox_pred": pred_box,
                "bbox_parse_ok": parse_ok,
                "height": height,
                "width": width,
                "latency_ms": latency_ms,
                "peak_gpu_memory_mb": peak_mem,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "text": outputs,
                "bbox": bbox_gt,
                "metadata": {"split": line.get("split"), "num_boxes": args.num_boxes, "vis_path": vis_path},
            }, ensure_ascii=False) + "\n")
            if args.save_jsonl:
                ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-vis", nargs="?", const="", default=None)
    parser.add_argument("--save-jsonl", action="store_true")
    parser.add_argument("--num-boxes", type=int, default=1)
    parser.add_argument("--record-latency", action="store_true")
    parser.add_argument("--record-memory", action="store_true")
    eval_model(parser.parse_args())
