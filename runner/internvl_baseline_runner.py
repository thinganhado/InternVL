#!/usr/bin/env python3
import argparse
import csv
import json
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# Prefer repo-level prompt files: <repo>/baseline_prompts/*.txt
DEFAULT_SYSTEM_FILE = REPO_ROOT / "baseline_prompts" / "baseline_system.txt"
DEFAULT_USER_TEMPLATE_FILE = REPO_ROOT / "baseline_prompts" / "baseline_user.txt"
# Backward-compatible fallback: <repo>/runner/baseline_prompts/*.txt
LEGACY_SYSTEM_FILE = THIS_DIR / "baseline_prompts" / "baseline_system.txt"
LEGACY_USER_TEMPLATE_FILE = THIS_DIR / "baseline_prompts" / "baseline_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv"
DEFAULT_META_JSON = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_val.json"
DEFAULT_IMAGE_FOLDER = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/"
DEFAULT_OUTPUT_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_file:
        return _load_text_file(Path(args.system_file), "--system-file")
    if DEFAULT_SYSTEM_FILE.exists():
        return _load_text_file(DEFAULT_SYSTEM_FILE, "--system-file")
    if LEGACY_SYSTEM_FILE.exists():
        return _load_text_file(LEGACY_SYSTEM_FILE, "--system-file")
    return "You are a helpful language and vision assistant."


def _resolve_user_template(args: argparse.Namespace) -> str:
    if args.user_template_file:
        return _load_text_file(Path(args.user_template_file), "--user-template-file")
    if DEFAULT_USER_TEMPLATE_FILE.exists():
        return _load_text_file(DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")
    if LEGACY_USER_TEMPLATE_FILE.exists():
        return _load_text_file(LEGACY_USER_TEMPLATE_FILE, "--user-template-file")
    return "Please analyze this image."


def _resolve_image_path(image_path_raw: str, image_folder: str) -> Optional[Path]:
    p = Path(str(image_path_raw)).expanduser()

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if image_folder:
            candidates.append(Path(image_folder).expanduser() / p)
        candidates.append(Path.cwd() / p)

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.exists():
            return resolved
    return None


def _extract_first_image_field(example: dict) -> str:
    image = example.get("image")
    if isinstance(image, str):
        return image
    if isinstance(image, list) and image:
        return str(image[0])
    return ""


def _extract_gt_regions_from_example(example: dict) -> str:
    if "regions" in example:
        return str(example.get("regions", "")).strip()

    conversations = example.get("conversations", [])
    if not isinstance(conversations, list):
        return ""

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).strip().lower() == "gpt":
            return str(turn.get("value", "")).strip()
    return ""


def _discover_items_from_json(args: argparse.Namespace):
    meta_json = Path(args.meta_json).expanduser().resolve()
    if not meta_json.exists():
        raise FileNotFoundError(f"--meta-json does not exist: {meta_json}")

    try:
        data = json.loads(meta_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse --meta-json: {meta_json}\n{e}") from e

    if not isinstance(data, list):
        raise ValueError("--meta-json should be a JSON list of examples.")

    items = []
    for ex in data:
        if not isinstance(ex, dict):
            continue

        img_path_raw = _extract_first_image_field(ex)
        if not img_path_raw:
            continue

        img_path = _resolve_image_path(img_path_raw, args.image_folder)
        if img_path is None:
            continue

        sample_id = str(ex.get("id", "")).strip() or img_path.stem
        if args.sample_id_glob and not fnmatch(sample_id, args.sample_id_glob):
            continue

        items.append(
            {
                "sample_id": sample_id,
                "img_path": str(img_path),
                "gt_regions": _extract_gt_regions_from_example(ex),
                "source": f"json:{meta_json.name}",
            }
        )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from --meta-json.")
    return sorted(items, key=lambda x: x["sample_id"])


def _discover_items_from_csv(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")

    items = []
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path_raw = str(row.get("img_path", "")).strip()
            if not img_path_raw:
                continue

            img_path = _resolve_image_path(img_path_raw, args.image_folder)
            if img_path is None:
                continue

            sample_id = img_path.stem
            if args.sample_id_glob and not fnmatch(sample_id, args.sample_id_glob):
                continue

            items.append(
                {
                    "sample_id": sample_id,
                    "img_path": str(img_path),
                    "gt_regions": str(row.get("regions", "")).strip(),
                    "source": f"csv:{meta_csv.name}",
                }
            )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from --meta-csv.")
    return sorted(items, key=lambda x: x["sample_id"])


def _discover_items(args: argparse.Namespace):
    if args.meta_json:
        meta_json = Path(args.meta_json).expanduser().resolve()
        if meta_json.exists():
            return _discover_items_from_json(args)
        if args.require_meta_json:
            raise FileNotFoundError(f"--meta-json does not exist: {meta_json}")
        print(f"[warn] --meta-json not found, falling back to --meta-csv: {meta_json}")
    return _discover_items_from_csv(args)


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image(image_file: str, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def _resolve_torch_dtype(dtype_str: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: {list(mapping.keys())}")
    return mapping[dtype_str]


def _generate_one(model, tokenizer, pixel_values, question, max_new_tokens, do_sample, temperature, top_p):
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample and temperature > 0.0),
    }
    if generation_config["do_sample"]:
        generation_config["temperature"] = temperature
        generation_config["top_p"] = top_p

    return model.chat(tokenizer, pixel_values, question, generation_config)


def _load_existing_sample_ids(output_jsonl: Path) -> set:
    done = set()
    if not output_jsonl.exists():
        return done

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sample_id = str(rec.get("sample_id", "")).strip()
            if sample_id:
                done.add(sample_id)
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot InternVL prompts on image data from JSON/CSV metadata.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")
    parser.add_argument("--meta-json", default=DEFAULT_META_JSON, help="JSON path for split metadata (preferred).")
    parser.add_argument("--require-meta-json", action="store_true", help="Fail if --meta-json is missing.")
    parser.add_argument("--meta-csv", default=DEFAULT_META_CSV, help="Fallback CSV path with img_path and regions columns.")
    parser.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER, help="Base folder for relative image paths.")
    parser.add_argument("--sample-id-glob", default="", help="Only include rows whose sample_id/stem matches this glob.")

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument("--user-template-file", default=None, help=f"Path to user prompt txt. Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}")

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use-flash-attn", action="store_true", help="Enable use_flash_attn=True on model load.")
    parser.add_argument("--input-size", type=int, default=448, help="InternVL image tile input size.")
    parser.add_argument("--max-num", type=int, default=12, help="Max dynamic image tiles.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat output jsonl path.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if already present.")
    parser.add_argument("--print-prompts", action="store_true", help="Print system/user prompts before generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    items = _discover_items(args)
    system_prompt = _resolve_system_prompt(args)
    user_prompt = _resolve_user_template(args)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "internvl_outputs.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        done = _load_existing_sample_ids(output_jsonl)
        before = len(items)
        items = [it for it in items if it["sample_id"] not in done]
        skipped = before - len(items)
        if skipped > 0:
            print(f"[resume] skipped_existing_samples={skipped}")
        if not items:
            print("[resume] no pending samples; nothing to generate.")
            return

    print(f"[model] {args.model_id}")
    print(f"[items] {len(items)}")
    print(f"[image_folder] {args.image_folder}")

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    if args.use_flash_attn:
        model_kwargs["use_flash_attn"] = True

    model = AutoModel.from_pretrained(args.model_id, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)

    # InternVL chat API uses the model-level system message.
    model.system_message = system_prompt

    mode = "w" if args.overwrite else "a"
    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as jsonl_fp:
        for idx, item in enumerate(items, start=1):
            pixel_values = _load_image(item["img_path"], input_size=args.input_size, max_num=args.max_num)
            pixel_values = pixel_values.to(torch_dtype if torch_dtype != "auto" else torch.bfloat16)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            question = f"<image>\nSpectrogram (GRID with axes):\n{user_prompt}"
            if args.print_prompts:
                print(f"[system]\n{system_prompt}\n[user]\n{question}")

            output_text = _generate_one(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "sample_id": item["sample_id"],
                "img_path": item["img_path"],
                "gt_regions": item["gt_regions"],
                "source": item.get("source", ""),
                "model_id": args.model_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": output_text,
            }

            sample_dir = output_dir / item["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fp.flush()
            try:
                os.fsync(jsonl_fp.fileno())
            except OSError:
                pass

            print(f"[{idx}/{len(items)}] {item['sample_id']}")
            print(output_text)


if __name__ == "__main__":
    main()
