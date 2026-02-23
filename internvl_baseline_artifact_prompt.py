#!/usr/bin/env python3
import argparse
import json
import os
from fnmatch import fnmatch
from pathlib import Path

from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "baseline_prompts" / "baseline_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "baseline_prompts" / "baseline_user.txt"

DEFAULT_META_JSON = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json"
DEFAULT_IMAGE_FOLDER = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/"
DEFAULT_OUTPUT_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final___baseline__strong__InternVL/"


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.system_file) if args.system_file else DEFAULT_SYSTEM_FILE, "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.user_template_file) if args.user_template_file else DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")


def _resolve_image_path(image_path_raw: str, image_folder: str) -> Path | None:
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

    messages = example.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                img = item.get("image")
                if isinstance(img, str) and img.strip():
                    return img.strip()
                image_url = item.get("image_url")
                if isinstance(image_url, str) and image_url.strip():
                    return image_url.strip()
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()

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

        sample_id = str(ex.get("sample_id", "")).strip() or str(ex.get("id", "")).strip() or img_path.stem
        if args.sample_id_glob and not fnmatch(sample_id, args.sample_id_glob):
            continue

        gt_regions = _extract_gt_regions_from_example(ex)
        items.append(
            {
                "sample_id": sample_id,
                "img_path": str(img_path),
                "gt_regions": gt_regions,
                "source": f"json:{meta_json.name}",
            }
        )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from --meta-json.")

    return sorted(items, key=lambda x: x["sample_id"])


def _build_messages(args: argparse.Namespace, item: dict):
    system_prompt = _resolve_system_prompt(args)
    user_prompt = _resolve_user_template(args)

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Spectrogram (GRID with axes):"},
                {"type": "image", "image": item["img_path"]},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _normalize_dtype_name(dtype_str: str) -> str:
    key = str(dtype_str).strip().lower()
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    normalized = aliases.get(key, key)
    if normalized not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: auto, fp16/float16, bf16/bfloat16, float32.")
    return normalized


def _import_vllm_deps():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError("vllm is required for this script.") from e
    return LLM, SamplingParams


def _infer_tp_size_from_env() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if parts:
            return len(parts)
    return 1


def _load_chat_renderer(model_id: str):
    renderer = None
    renderer_name = "none"

    try:
        from transformers import AutoProcessor

        renderer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        renderer_name = "AutoProcessor"
        return renderer, renderer_name
    except Exception:
        pass

    from transformers import AutoTokenizer

    renderer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    renderer_name = "AutoTokenizer"
    return renderer, renderer_name


def _render_prompt(renderer, messages, fallback_system: str, fallback_user: str):
    if hasattr(renderer, "apply_chat_template"):
        try:
            return renderer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"{fallback_system}\n\n<image>\n{fallback_user}\n"


def _prepare_input_for_vllm(messages, image_path: str, renderer):
    system_text = ""
    user_text = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        text_parts = [str(part.get("text", "")).strip() for part in content if isinstance(part, dict) and part.get("type") == "text"]
        joined = "\n".join([t for t in text_parts if t])
        if role == "system" and joined:
            system_text = joined
        if role == "user" and joined:
            user_text = joined

    prompt = _render_prompt(renderer, messages, fallback_system=system_text, fallback_user=user_text)
    pil_image = Image.open(image_path).convert("RGB")
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": pil_image},
    }


def _build_vllm_sampling_params(args, SamplingParams):
    sample_flag = bool(args.do_sample and args.temperature > 0.0)
    temperature = args.temperature if sample_flag else 0.0
    top_p = args.top_p if sample_flag else 1.0
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )


def _generate_one_vllm(llm, renderer, messages, image_path: str, sampling_params):
    model_input = _prepare_input_for_vllm(messages=messages, image_path=image_path, renderer=renderer)
    outputs = llm.generate([model_input], sampling_params=sampling_params)
    if outputs and getattr(outputs[0], "outputs", None):
        return outputs[0].outputs[0].text
    return ""


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
    parser = argparse.ArgumentParser(description="Run baseline InternVL prompt on JSON split with vLLM.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")

    parser.add_argument("--meta-json", default=DEFAULT_META_JSON, help="JSON path for val split.")
    parser.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER, help="Base folder for resolving relative image paths.")
    parser.add_argument("--sample-id-glob", default="", help="Only include sample ids matching this glob.")

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument("--user-template-file", default=None, help=f"Path to user prompt txt. Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}")

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--backend", default="vllm", choices=["vllm"], help="Only vLLM is supported.")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM tensor parallel size. Default: GPU count.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="Enable vLLM eager mode.")
    parser.add_argument("--vllm-max-model-len", type=int, default=768, help="vLLM max model length.")
    parser.add_argument("--vllm-max-images-per-prompt", type=int, default=1, help="vLLM multimodal image limit per prompt.")

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--output-jsonl", default=None, help="Optional output jsonl path.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if sample already exists.")
    parser.add_argument("--print-messages", action="store_true", help="Print built messages before generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    items = _discover_items_from_json(args)

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "internvl_baseline_outputs.jsonl"
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

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    LLM, SamplingParams = _import_vllm_deps()

    renderer, renderer_name = _load_chat_renderer(args.model_id)
    print(f"[renderer] {renderer_name}")

    tp_size = args.tensor_parallel_size
    if tp_size is None:
        tp_size = _infer_tp_size_from_env()

    llm_kwargs = {
        "model": args.model_id,
        "trust_remote_code": True,
        "tensor_parallel_size": tp_size,
        "dtype": _normalize_dtype_name(args.dtype),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enforce_eager": args.vllm_enforce_eager,
        "limit_mm_per_prompt": {"image": args.vllm_max_images_per_prompt},
        "seed": 0,
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = _build_vllm_sampling_params(args, SamplingParams)

    mode = "w" if args.overwrite else "a"
    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as jsonl_fp:
        for idx, item in enumerate(items, start=1):
            messages = _build_messages(args, item)
            if args.print_messages:
                print(messages)

            output_text = _generate_one_vllm(
                llm=llm,
                renderer=renderer,
                messages=messages,
                image_path=item["img_path"],
                sampling_params=sampling_params,
            )

            record = {
                "sample_id": item["sample_id"],
                "img_path": item["img_path"],
                "gt_regions": item["gt_regions"],
                "source": item.get("source", ""),
                "model_id": args.model_id,
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
