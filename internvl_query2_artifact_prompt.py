#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "query2_prompts" / "query2_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "query2_prompts" / "query2_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_gt_with_transcript.csv"
DEFAULT_IMAGE_FOLDER = ""
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/"
DEFAULT_OUTPUT_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/ZS_Q2"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.system_file) if args.system_file else DEFAULT_SYSTEM_FILE, "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.user_template_file) if args.user_template_file else DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")


def _resolve_image_path(img_path_raw: str, image_folder: str, csv_dir: Path):
    p = Path(str(img_path_raw)).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if image_folder:
            candidates.append(Path(image_folder).expanduser() / p)
        candidates.append(csv_dir / p)
        candidates.append(Path.cwd() / p)

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.exists():
            return resolved
    return None


def _discover_items(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    csv_dir = meta_csv.parent

    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")

    items = []
    used_sample_ids = set()
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            img_path_raw = str(row.get("img_path", "")).strip()
            prompt1_output = str(row.get("regions", "")).strip()
            transcript = str(row.get("transcript", "")).strip()
            prompt2_target = str(row.get("prompt2_target", "")).strip()
            if not img_path_raw:
                continue

            img_path = _resolve_image_path(img_path_raw, args.image_folder, csv_dir)
            if img_path is None:
                continue
            if args.img_stem_contains and args.img_stem_contains not in img_path.stem:
                continue

            sample_id = str(row.get("id", "")).strip() or img_path.stem or f"row_{row_idx}"
            unique_sample_id = sample_id
            if unique_sample_id in used_sample_ids:
                unique_sample_id = f"{sample_id}__row{row_idx}"
            used_sample_ids.add(unique_sample_id)

            items.append(
                {
                    "sample_id": unique_sample_id,
                    "crop_method": "GRID",
                    "img_path": str(img_path),
                    "prompt1_output": prompt1_output,
                    "transcript": transcript,
                    "prompt2_target": prompt2_target,
                }
            )

    items = sorted(items, key=lambda x: x["sample_id"])
    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from --meta-csv with current filters.")
    return items


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


def _model_tag_from_model_id(model_id: str) -> str:
    return Path(str(model_id).rstrip("/\\")).name or "model"


def _default_output_dir_from_model(model_id: str) -> str:
    return str(Path(DEFAULT_OUTPUT_ROOT) / f"{_model_tag_from_model_id(model_id)}_test")


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
    parser = argparse.ArgumentParser(description="Run InternVL query2 prompt on stage1_gt_with_transcript CSV.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")

    parser.add_argument(
        "--meta-csv",
        default=DEFAULT_META_CSV,
        help="CSV with columns: img_path, regions, transcript, prompt2_target.",
    )
    parser.add_argument(
        "--image-folder",
        default=DEFAULT_IMAGE_FOLDER,
        help="Optional base folder for resolving relative img_path entries.",
    )
    parser.add_argument(
        "--img-stem-contains",
        default="_LA_D_",
        help="Only run rows where resolved img_path stem contains this substring. Set empty string to disable.",
    )

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=(
            "Path to user prompt template txt. Supports placeholders: "
            "{prompt1_output}, {transcript}, {sample_id}. "
            f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}"
        ),
    )

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use-flash-attn", action="store_true", help="Enable use_flash_attn=True on model load.")
    parser.add_argument("--input-size", type=int, default=448, help="InternVL image tile input size.")
    parser.add_argument("--max-num", type=int, default=12, help="Max dynamic image tiles.")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}/<model_name>_test",
    )
    parser.add_argument("--output-file", default=None, help="Single-item output file.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat output jsonl path.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if already present.")
    parser.add_argument("--print-prompts", action="store_true", help="Print built system/user prompts before generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    items = _discover_items(args)
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    if len(items) > 1 and args.output_file:
        raise ValueError("--output-file is only for single item. Use --output-dir for grouped outputs.")

    output_dir_str = args.output_dir if args.output_dir else _default_output_dir_from_model(args.model_id)
    output_dir = Path(output_dir_str).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "internvl_query2_outputs.jsonl"
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
    print(f"[img_stem_contains] {args.img_stem_contains}")

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
    model.system_message = system_prompt

    mode = "w" if args.overwrite else "a"
    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as jsonl_fp:
        for idx, item in enumerate(items, start=1):
            transcript_text = str(item.get("transcript", "")).strip()
            prompt1_output = str(item.get("prompt1_output", "")).strip()
            user_prompt = user_template.format_map(
                defaultdict(
                    str,
                    {
                        "prompt1_output": prompt1_output,
                        "transcript": transcript_text,
                        "sample_id": item["sample_id"],
                    },
                )
            )

            pixel_values = _load_image(item["img_path"], input_size=args.input_size, max_num=args.max_num)
            pixel_values = pixel_values.to(torch_dtype if torch_dtype != "auto" else torch.bfloat16)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            question = f"<image>\nSpectrogram ({item['crop_method']}):\n{user_prompt}"
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
                "crop_method": item["crop_method"],
                "img_path": item["img_path"],
                "prompt1_output": prompt1_output,
                "transcript": transcript_text,
                "prompt2_target": item.get("prompt2_target", ""),
                "model_id": args.model_id,
                "response": output_text,
            }

            sample_dir = output_dir / item["crop_method"].lower() / item["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fp.flush()
            try:
                os.fsync(jsonl_fp.fileno())
            except OSError:
                pass

            if len(items) == 1 and args.output_file:
                out_file = Path(args.output_file).expanduser().resolve()
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(output_text, encoding="utf-8")

            print(f"[{idx}/{len(items)}] {item['sample_id']}")
            print(output_text)


if __name__ == "__main__":
    main()
