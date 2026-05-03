import argparse
import os
import re
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


try:
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR


def box_from(x) -> Optional[list]:
    """Parse bbox-like input into [x1, y1, x2, y2]."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) >= 4:
        try:
            return [float(v) for v in x[:4]]
        except (TypeError, ValueError):
            return None

    text = str(x).strip()
    hit = re.search(r"\[([^\]]+)\]", text)
    target = hit.group(1) if hit else text
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", target)
    if len(nums) < 4:
        return None
    try:
        return [float(v) for v in nums[:4]]
    except ValueError:
        return None


def clip_box(box) -> Optional[list]:
    box = box_from(box)
    if box is None:
        return None
    x1, y1, x2, y2 = [min(max(float(v), 0.0), 1.0) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pad_box(box, pad=1.2) -> Optional[list]:
    box = clip_box(box)
    if box is None:
        return None
    pad = max(float(pad), 1.0)
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) * pad / 2.0
    hh = (y2 - y1) * pad / 2.0
    return clip_box([cx - hw, cy - hh, cx + hw, cy + hh])


def area(box) -> Optional[float]:
    box = clip_box(box)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def ok_box(box, min_area=0.01, max_area=0.95) -> bool:
    a = area(box)
    return a is not None and min_area <= a <= max_area


def pix_box(img: Image.Image, box, pad=1.2) -> Optional[tuple]:
    box = pad_box(box, pad)
    if box is None:
        return None
    w, h = img.size
    x1, y1, x2, y2 = box
    left = int(round(x1 * w))
    top = int(round(y1 * h))
    right = int(round(x2 * w))
    bottom = int(round(y2 * h))
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return left, top, right, bottom


def crop(img: Image.Image, box, pad=1.2) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    return img.crop(pbox)


def blur_bg(img: Image.Image, box, pad=1.2, r=8) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    out = img.filter(ImageFilter.GaussianBlur(radius=r))
    out.paste(img.crop(pbox), pbox)
    return out


def low_bg(img: Image.Image, box, pad=1.2, scale=0.25) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    w, h = img.size
    scale = min(max(float(scale), 0.05), 1.0)
    small = (max(1, int(w * scale)), max(1, int(h * scale)))
    out = img.resize(small, BILINEAR).resize((w, h), BICUBIC)
    out.paste(img.crop(pbox), pbox)
    return out


def mask_bg(img: Image.Image, box, pad=1.2, dim=0.55) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    out = ImageOps.grayscale(img).convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(dim)
    out.paste(img.crop(pbox), pbox)
    return out


def mark(img: Image.Image, box, pad=1.2) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    out = img.copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(pbox, outline=(255, 36, 36, 255), width=max(3, out.size[0] // 160))
    draw.rectangle(pbox, fill=(255, 36, 36, 36))
    return Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")


def zoom(img: Image.Image, box, pad=1.35) -> Image.Image:
    img = img.convert("RGB")
    pbox = pix_box(img, box, pad)
    if pbox is None:
        return img.copy()
    return img.crop(pbox).resize(img.size, BICUBIC)


def pick(box, question=""):
    a = area(box)
    if a is None:
        return "base"
    q = (question or "").lower()
    local_words = (
        "color",
        "leg",
        "wing",
        "beak",
        "eye",
        "tail",
        "text",
        "word",
        "number",
        "detail",
    )
    global_words = ("scene", "background", "left", "right", "where", "around")
    if a < 0.03:
        return "zoom" if any(w in q for w in local_words) else "mark"
    if a > 0.65:
        return "base" if any(w in q for w in global_words) else "mask"
    if any(w in q for w in local_words):
        return "zoom" if a < 0.12 else "blur"
    if any(w in q for w in global_words):
        return "mark"
    return "blur"


def apply(img: Image.Image, box, mode="base", pad=1.2, q="", blur=8, scale=0.25):
    mode = (mode or "base").lower()
    if mode == "mix":
        mode = pick(box, q)
    if mode == "base":
        return img.convert("RGB").copy(), mode
    if mode == "crop":
        return crop(img, box, pad), mode
    if mode == "blur":
        return blur_bg(img, box, pad, r=blur), mode
    if mode == "low":
        return low_bg(img, box, pad, scale=scale), mode
    if mode == "mask":
        return mask_bg(img, box, pad), mode
    if mode == "mark":
        return mark(img, box, pad), mode
    if mode == "zoom":
        return zoom(img, box, pad=max(pad, 1.35)), mode
    raise ValueError(f"unsupported mode: {mode}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", required=True)
    parser.add_argument(
        "--mode",
        default="blur",
        choices=["base", "crop", "blur", "low", "mask", "mark", "zoom", "mix"],
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--pad", type=float, default=1.2)
    parser.add_argument("--blur", type=float, default=8)
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()

    box = box_from(args.box)
    if not ok_box(box):
        raise ValueError(f"invalid box: {args.box}")
    img = Image.open(args.image).convert("RGB")
    out, _ = apply(img, box, args.mode, pad=args.pad, blur=args.blur, scale=args.scale)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.save(args.out)
    print(args.out)


if __name__ == "__main__":
    _main()
