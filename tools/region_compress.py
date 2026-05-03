import argparse
import os
import re
from typing import Iterable, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_bbox(text) -> Optional[list]:
    """Parse a normalized bbox string into [x1, y1, x2, y2]."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)) and len(text) >= 4:
        try:
            return [float(v) for v in text[:4]]
        except (TypeError, ValueError):
            return None

    text = str(text).strip()
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    target = bracket_match.group(1) if bracket_match else text
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", target)
    if len(nums) < 4:
        return None
    try:
        return [float(v) for v in nums[:4]]
    except ValueError:
        return None


def clamp_bbox(bbox) -> Optional[list]:
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None

    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_bbox(bbox, ratio=1.2) -> Optional[list]:
    bbox = clamp_bbox(bbox)
    if bbox is None:
        return None
    ratio = max(float(ratio), 1.0)
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) * ratio / 2.0
    half_h = (y2 - y1) * ratio / 2.0
    return clamp_bbox([cx - half_w, cy - half_h, cx + half_w, cy + half_h])


def bbox_area(bbox) -> Optional[float]:
    bbox = clamp_bbox(bbox)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def validate_bbox(bbox, min_area=0.01, max_area=0.95) -> bool:
    area = bbox_area(bbox)
    if area is None:
        return False
    return min_area <= area <= max_area


def _bbox_to_pixels(image: Image.Image, bbox, expand_ratio=1.2) -> Optional[tuple]:
    bbox = expand_bbox(bbox, expand_ratio)
    if bbox is None:
        return None
    width, height = image.size
    x1, y1, x2, y2 = bbox
    left = int(round(x1 * width))
    upper = int(round(y1 * height))
    right = int(round(x2 * width))
    lower = int(round(y2 * height))
    left = max(0, min(left, width - 1))
    upper = max(0, min(upper, height - 1))
    right = max(left + 1, min(right, width))
    lower = max(upper + 1, min(lower, height))
    return left, upper, right, lower


def crop_only(image: Image.Image, bbox, expand_ratio=1.2) -> Image.Image:
    image = image.convert("RGB")
    box = _bbox_to_pixels(image, bbox, expand_ratio)
    if box is None:
        return image.copy()
    return image.crop(box)


def blur_background(
    image: Image.Image, bbox, expand_ratio=1.2, blur_radius=8
) -> Image.Image:
    image = image.convert("RGB")
    box = _bbox_to_pixels(image, bbox, expand_ratio)
    if box is None:
        return image.copy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred.paste(image.crop(box), box)
    return blurred


def downsample_background(
    image: Image.Image, bbox, expand_ratio=1.2, scale=0.25
) -> Image.Image:
    image = image.convert("RGB")
    box = _bbox_to_pixels(image, bbox, expand_ratio)
    if box is None:
        return image.copy()
    width, height = image.size
    scale = min(max(float(scale), 0.05), 1.0)
    small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    downsampled = image.resize(small_size, RESAMPLE_BILINEAR).resize(
        (width, height), RESAMPLE_BICUBIC
    )
    downsampled.paste(image.crop(box), box)
    return downsampled


def mask_background(image: Image.Image, bbox, expand_ratio=1.2) -> Image.Image:
    image = image.convert("RGB")
    box = _bbox_to_pixels(image, bbox, expand_ratio)
    if box is None:
        return image.copy()
    gray = ImageOps.grayscale(image).convert("RGB")
    muted = ImageEnhance.Brightness(gray).enhance(0.55)
    muted.paste(image.crop(box), box)
    return muted


def apply_region_compression(
    image: Image.Image,
    bbox,
    mode,
    expand_ratio=1.2,
    blur_radius=8,
    downsample_scale=0.25,
) -> Image.Image:
    mode = (mode or "none").lower()
    if mode == "none":
        return image.convert("RGB").copy()
    if mode == "crop_only":
        return crop_only(image, bbox, expand_ratio=expand_ratio)
    if mode == "blur":
        return blur_background(
            image, bbox, expand_ratio=expand_ratio, blur_radius=blur_radius
        )
    if mode == "downsample":
        return downsample_background(
            image, bbox, expand_ratio=expand_ratio, scale=downsample_scale
        )
    if mode == "mask":
        return mask_background(image, bbox, expand_ratio=expand_ratio)
    raise ValueError(f"Unsupported compression mode: {mode}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument(
        "--mode",
        default="blur",
        choices=["none", "crop_only", "blur", "downsample", "mask"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expand-ratio", type=float, default=1.2)
    parser.add_argument("--blur-radius", type=float, default=8)
    parser.add_argument("--downsample-scale", type=float, default=0.25)
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    if not validate_bbox(bbox):
        raise ValueError(f"Invalid bbox: {args.bbox}")

    image = Image.open(args.image).convert("RGB")
    compressed = apply_region_compression(
        image,
        bbox,
        args.mode,
        expand_ratio=args.expand_ratio,
        blur_radius=args.blur_radius,
        downsample_scale=args.downsample_scale,
    )
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    compressed.save(args.output)
    print(args.output)


if __name__ == "__main__":
    _main()
