import argparse
import random
import re
import copy
import torch
import os
import json
import time
import sys
import types
import importlib.util
import importlib.machinery
from tqdm import tqdm
import shortuuid


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
            "flash_attn is not installed. Install flash-attn if the selected "
            "model path requires the NTK/flash-attention variant."
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

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from tools.region_compress import (
    apply_region_compression,
    bbox_area,
    parse_bbox,
    validate_bbox,
)



SUBIMAGE_PATTERN = r".*\#\#\#\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def strip_image_token(text):
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def question_without_bbox_request(text):
    if "Please provide the bounding box coordinate of the region" in text:
        text = text.split("Please provide the bounding box coordinate of the region")[0]
    elif "Please provide the bounding box" in text:
        text = text.split("Please provide the bounding box")[0]
    return text.strip()


def get_ground_truth(line):
    for turn in reversed(line.get("conversations", [])):
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            value = turn.get("value")
            if value:
                return value
    return None


def normalize_bbox_coords(bbox, width=None, height=None):
    coords = parse_bbox(bbox)
    if coords is None:
        return None
    if width and height and max(abs(v) for v in coords) > 1.0:
        x1, y1, x2, y2 = coords
        coords = [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]
    return coords


def bbox_to_text(bbox):
    if bbox is None:
        return None
    return "[%.6f, %.6f, %.6f, %.6f]" % tuple(bbox[:4])


def detection_record_bbox(record):
    if not isinstance(record, dict):
        return None, None
    width = record.get("width")
    height = record.get("height")
    for key in ("text", "bbox", "bbox_normalized", "pred_bbox", "box"):
        if key not in record or record.get(key) in (None, ""):
            continue
        if key == "bbox_normalized":
            coords = normalize_bbox_coords(record.get(key))
        else:
            coords = normalize_bbox_coords(record.get(key), width=width, height=height)
        if coords is not None:
            return coords, record.get(key) if key == "text" else bbox_to_text(coords)
    return None, None


def safe_filename(value):
    value = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "sample"


def cropwithbbox(pil_img, sub_image_info):
    width, height = pil_img.size
    x_min, y_min, x_max, y_max = sub_image_info
    if sum([x_min, y_min, x_max, y_max]) < 5:
        x_min = x_min * max(width, height)
        y_min = y_min * max(width, height)
        x_max = x_max * max(width, height)
        y_max = y_max * max(width, height)
    if width > height:
        overlay = (width - height) // 2
        y_min = max(0, y_min - overlay)
        y_max = max(0, y_max - overlay)
    else:
        overlay = (height - width) // 2
        x_min = max(0, x_min - overlay)
        x_max = max(0, x_max - overlay)
    center_point = [(x_min + x_max)//2, (y_min + y_max)//2]
    half_sizes = [(x_max - x_min)//2, (y_max - y_min)//2]
    cropped_half_size = max(max(half_sizes), 112)
    upper_left_point = [center_point[0]-cropped_half_size, center_point[1]-cropped_half_size]
    if upper_left_point[0] < 0:
        center_point[0] += (-upper_left_point[0])
    if upper_left_point[1] < 0:
        center_point[1] += (-upper_left_point[1])
    lower_right_point = [center_point[0]+cropped_half_size, center_point[1]+cropped_half_size]
    if lower_right_point[0] > width:
        center_point[0] -= (lower_right_point[0] - width)
    if lower_right_point[1] > height:
        center_point[1] -= (lower_right_point[1] - height)
    cropped_region = [
        max(0, center_point[0]-cropped_half_size),
        max(0, center_point[1]-cropped_half_size),
        min(width, center_point[0]+cropped_half_size),
        min(height, center_point[1]+cropped_half_size),
    ]
    cropped_image = pil_img.crop(cropped_region)
    return cropped_image

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_name,
        with_cot,
        detection_results,
        random_bbox,
        center_bbox,
        without_image,
        adapt_ratio,
        compress_mode,
        visual_input_policy,
        bbox_expand_ratio,
        save_compressed_images,
        compressed_image_dir,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model_name = model_name
        self.with_cot = with_cot
        self.detection_results = detection_results
        self.random_bbox = random_bbox
        self.center_bbox = center_bbox
        self.without_image = without_image
        self.adapt_ratio = adapt_ratio
        self.compress_mode = compress_mode
        self.visual_input_policy = visual_input_policy
        self.bbox_expand_ratio = bbox_expand_ratio
        self.save_compressed_images = save_compressed_images
        self.compressed_image_dir = compressed_image_dir

    def __getitem__(self, index):
        preprocess_start = time.perf_counter()
        line = self.questions[index]
        image_files = list(line["image"])
        raw_conversations = line["conversations"]

        conv = conv_templates[args.conv_mode].copy()
        bbox_text = None
        coords = None
        if self.random_bbox:
            center = [random.random(), random.random()]
            height = random.random() * 0.5
            width = random.random() * 0.5
            coords = [max(0, center[0]-width), max(0, center[1]-height), min(1, center[0]+width), min(1, center[1]+height)]
            bbox_text = '[%.3f, %.3f, %.3f, %.3f]' % (coords[0], coords[1], coords[2], coords[3])

        elif self.center_bbox:
            coords = [0.25, 0.25, 0.75, 0.75]
            bbox_text = '[%.3f, %.3f, %.3f, %.3f]' % (coords[0], coords[1], coords[2], coords[3])

        elif self.detection_results is not None:
            coords, bbox_text = detection_record_bbox(self.detection_results[index])
        else:
            bbox_text = raw_conversations[1].get('value') if len(raw_conversations) > 1 else None
            coords = normalize_bbox_coords(
                bbox_text,
                width=line.get('width'),
                height=line.get('height'),
            )

        bbox_valid = validate_bbox(coords)
        bbox_area_value = bbox_area(coords) if bbox_valid else None
        question_with_image = question_without_bbox_request(raw_conversations[0]['value'])
        question_text = strip_image_token(question_with_image)
        active_compression = self.compress_mode != "none"
        single_visual_input = (
            self.visual_input_policy == "single"
            or (self.visual_input_policy == "auto" and active_compression)
        )
        dual_visual_input = (
            self.visual_input_policy == "dual"
            or (
                self.visual_input_policy == "auto"
                and not active_compression
                and self.with_cot
                and self.without_image is False
            )
        )

        if (active_compression or single_visual_input) and self.with_cot:
            conv.append_message(conv.roles[0], question_text)
            conv.append_message(conv.roles[1], bbox_text or "")
            if dual_visual_input:
                answer_prompt = (
                    DEFAULT_IMAGE_TOKEN
                    + "\nPlease answer the question based on the image and local detail image. "
                    + question_text
                )
            elif self.compress_mode == "none":
                answer_prompt = (
                    DEFAULT_IMAGE_TOKEN
                    + "\nPlease answer the question based on the original image. "
                    + question_text
                )
            elif bbox_valid:
                if self.compress_mode == "crop_only":
                    answer_prompt = (
                        DEFAULT_IMAGE_TOKEN
                        + "\nPlease answer the question based on the key region image. "
                        + question_text
                    )
                else:
                    answer_prompt = (
                        DEFAULT_IMAGE_TOKEN
                        + "\nPlease answer the question based on the region-aware compressed image. "
                        + question_text
                    )
            else:
                answer_prompt = (
                    DEFAULT_IMAGE_TOKEN
                    + "\nPlease answer the question based on the image. "
                    + question_text
                )
            conv.append_message(conv.roles[0], answer_prompt)
            conv.append_message(conv.roles[1], None)
        elif self.with_cot and self.without_image is False:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'].split(' Please provide the bounding box coordinate of the region')[0])
            conv.append_message(conv.roles[1], bbox_text or raw_conversations[1]['value'])

            # conv.append_message(conv.roles[0], raw_conversations[2]['value'])
            conv.append_message(conv.roles[0], raw_conversations[2]['value'] + '\nPlease answer the question based on the original image and local detail image.'+  raw_conversations[0]['value'].split('Please provide the bounding box coordinate of the region')[0].replace('<image>\n', ''))
            conv.append_message(conv.roles[1], None)
        elif self.with_cot and self.without_image is True:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            conv.append_message(conv.roles[1], bbox_text or raw_conversations[1]['value'])
            conv.append_message(conv.roles[0], '')
            conv.append_message(conv.roles[1], None)
        else:
            if 'Please provide the bounding box' in raw_conversations[0]['value']:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'].split('Please provide the bounding box')[0])
            else:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = []
        image_path = os.path.join(self.image_folder, image_files[0])
        original_image = Image.open(image_path).convert("RGB")
        image = original_image.copy()
        compressed_image_path = None
        use_region_compressed_primary = active_compression and not (
            dual_visual_input and self.compress_mode == "crop_only"
        )
        if use_region_compressed_primary:
            if bbox_valid:
                image = apply_region_compression(
                    original_image,
                    coords,
                    self.compress_mode,
                    expand_ratio=self.bbox_expand_ratio,
                )
            else:
                image = original_image.copy()

            if self.save_compressed_images:
                os.makedirs(self.compressed_image_dir, exist_ok=True)
                sample_id = safe_filename(line.get("question_id", index))
                compressed_image_path = os.path.join(
                    self.compressed_image_dir,
                    f"{index:06d}_{sample_id}_{self.compress_mode}.png",
                )
                image.save(compressed_image_path)
            images.append(image)
        else:
            images.append(image)


        if dual_visual_input and self.with_cot and self.without_image is False and len(image_files) > 1:
            if coords is None:
                if self.detection_results is None:
                    if '###' not in image_files[1]:
                        raise ValueError("%s is not a valid cot path" % image_path)
                    coords = parse_bbox(raw_conversations[1]['value'])
            if coords is None:
                print("Can not parse the coords: %s" % (bbox_text or image_files[1]))
                coords = [0.0, 0.0, 1.0, 1.0]
            image_files[1] = image_files[1].split('###')[0]
            image_path2 = os.path.join(self.image_folder, image_files[1])
            if image_path2 == image_path:
                image = copy.copy(original_image)
            else:
                image = Image.open(image_path2).convert("RGB")
            image = cropwithbbox(image, coords)
            images.append(image)


        if isinstance(self.image_processor, list):
            image_tensor_0 = process_images(
                images, self.image_processor[0], self.model_config
            )
            image_tensor_1 = process_images(
                images, self.image_processor[1], self.model_config
            )
            image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
        else:
            image_tensor = process_images(
                images, self.image_processor, self.model_config
            )

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        metadata = {
            "compress_mode": self.compress_mode,
            "visual_input_policy": self.visual_input_policy,
            "num_visual_images": len(images),
            "bbox": coords,
            "bbox_valid": bbox_valid,
            "bbox_area": bbox_area_value,
            "compressed_image_path": compressed_image_path,
            "preprocess_latency_sec": time.perf_counter() - preprocess_start,
        }

        return input_ids, image_tensor, prompt, json.dumps(metadata)

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    model_name,
    with_cot,
    detection_results,
    random_bbox,
    center_bbox,
    without_image,
    adapt_ratio,
    compress_mode,
    visual_input_policy,
    bbox_expand_ratio,
    save_compressed_images,
    compressed_image_dir,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_name,
        with_cot,
        detection_results,
        random_bbox,
        center_bbox,
        without_image,
        adapt_ratio,
        compress_mode,
        visual_input_policy,
        bbox_expand_ratio,
        save_compressed_images,
        compressed_image_dir,
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    if args.random_bbox is True and args.center_bbox is True:
        raise ValueError("random-bbox and center-bbox cannot all be true!")

    if args.question_file.endswith('.jsonl'):
        questions = [
            json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
        ]
    else:
        questions = json.load(open(args.question_file))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.detection_file is not None:
        detection_results = [
            json.loads(r) for r in open(args.detection_file, 'r')
        ]
    else:
        detection_results = None

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        model_name,
        args.with_cot,
        detection_results,
        args.random_bbox,
        args.center_bbox,
        args.without_image,
        args.adapt_ratio,
        args.compress_mode,
        args.visual_input_policy,
        args.bbox_expand_ratio,
        args.save_compressed_images,
        args.compressed_image_dir,
    )

    for (input_ids, image_tensor, prompt, metadata_json), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]
        sample_metadata = json.loads(metadata_json[0] if isinstance(metadata_json, (list, tuple)) else metadata_json)

        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        if image_tensor.ndim == 5:
            image_tensor = image_tensor[0]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        generation_start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.bfloat16, device="cuda", non_blocking=True
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            max_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            max_gpu_memory_mb = None
        generation_latency_sec = time.perf_counter() - generation_start_time

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        prompt_q = line['conversations'][0]['value']
        if prompt_q.startswith('<image>\n'):
            prompt_q = prompt_q.replace('<image>\n', '')
        if 'Please provide the bounding box coordinate of the region' in prompt_q:
            prompt_q = prompt_q.split('Please provide the bounding box coordinate of the region')[0]
        #print(outputs, line['conversations'][1]['value'])
        dumped_dict = {
                    "question_id": idx,
                    "conversations": prompt[0],
                    "text": outputs,
                    "gt_answer": get_ground_truth(line),
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "prompt": prompt_q,
                    "metadata": sample_metadata,
                }
        dumped_dict["metadata"]["latency_sec"] = generation_latency_sec
        dumped_dict["metadata"]["max_gpu_memory_mb"] = max_gpu_memory_mb
        if 'height' in line:
            dumped_dict['height'] = line['height']
        if 'width' in line:
            dumped_dict['width'] = line['width']
        if 'bbox' in line:
            dumped_dict['bbox'] = line['bbox']
        preprocess_latency_sec = sample_metadata.get("preprocess_latency_sec")
        if preprocess_latency_sec is None:
            end_to_end_latency_sec = generation_latency_sec
        else:
            end_to_end_latency_sec = preprocess_latency_sec + (
                time.perf_counter() - generation_start_time
            )
        dumped_dict["metadata"]["generation_latency_sec"] = generation_latency_sec
        dumped_dict["metadata"]["end_to_end_latency_sec"] = end_to_end_latency_sec
        ans_file.write(
            json.dumps(dumped_dict)
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="s3://mmdata/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--with-cot', type=str_to_bool, default=False)
    parser.add_argument('--random-bbox', type=str_to_bool, default=False)
    parser.add_argument('--center-bbox', type=str_to_bool, default=False)
    parser.add_argument('--without-image', type=str_to_bool, default=False)
    parser.add_argument('--detection-file', type=str, default=None)
    parser.add_argument('--adapt-ratio', type=float, default=1.0)
    parser.add_argument(
        '--compress-mode',
        type=str,
        default='none',
        choices=['none', 'crop_only', 'blur', 'downsample', 'mask'],
    )
    parser.add_argument(
        '--visual-input-policy',
        type=str,
        default='auto',
        choices=['auto', 'single', 'dual'],
        help='auto: original dual-image flow for none and single image for compression; '
             'single: force one visual input; dual: force main image plus local crop.',
    )
    parser.add_argument('--bbox-expand-ratio', type=float, default=1.2)
    parser.add_argument('--save-compressed-images', action='store_true', default=False)
    parser.add_argument(
        '--compressed-image-dir',
        type=str,
        default='results/region_compress/compressed_images',
    )
    args = parser.parse_args()
    eval_model(args)
