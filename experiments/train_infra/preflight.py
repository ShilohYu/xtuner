"""Audit every source row, including records that the reference length filters out."""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer
from xtuner.v1.datasets.collator import glm53_vl_sft_collator
from xtuner.v1.datasets.mllm_tokenize_fn import glm53_vl_tokenize_fn as glm_data


SOURCE = Path("/mnt/shared-storage-user/llmrazor-share/data/train_infra_10_per_dataset/train_infra_localdata.jsonl")
PROCESSOR = "/tmp/GLM-5.3-Flash-4layer-with-processor"
TOKEN_BUDGET = 16777216 // (14 * 2) ** 2


def check_image_geometry(decoded, declared_size):
    # image_wh describes encoded geometry; runtime applies EXIF orientation afterwards.
    assert list(decoded.size) == declared_size, "Encoded image geometry disagrees with annotation"
    return ImageOps.exif_transpose(decoded).size != decoded.size


def check_item(item, image_token_id, image_count, merge_unit):
    ids = torch.as_tensor(item["input_ids"])
    labels = torch.as_tensor(item["labels"])
    assert ids.shape == labels.shape
    assert len(ids) == item["num_tokens"] > 0
    assert int((labels[1:] != -100).sum()) > 0, "No shifted supervision"
    mask = ids == image_token_id
    assert bool((labels[mask] == -100).all()), "Visual tokens supervised"
    if image_count:
        grid = item["image_grid_thw"]
        assert grid.shape == (image_count, 3), "Image count changed"
        patches = int(grid.prod(-1).sum())
        assert patches == sum(item["num_img_tokens"])
        assert item["pixel_values"].shape[0] == patches
        assert int(mask.sum()) == patches // merge_unit
        assert bool(torch.isfinite(item["pixel_values"]).all())
    return int((labels[1:] != -100).sum())


def initialize(length, runtime, fixtures):
    global tokenize, sequence_budget, do_runtime, fixture_dir
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(PROCESSOR, local_files_only=True)
    tokenize = glm_data.Glm53VLTokenizeFunction(
        tokenizer, PROCESSOR, SOURCE.name, max_length=None, max_frames=24,
        max_pixels=TOKEN_BUDGET, llm_pack_weight=-3.2, visual_pack_weight=5.0, debug=True,
    )
    sequence_budget, do_runtime, fixture_dir = length, runtime, fixtures


def audit(entry):
    line, row = entry
    images = [part["image"] for message in row["messages"]
              if isinstance(message.get("content"), list)
              for part in message["content"] if part.get("type") == "image"]
    result = {"source_line": line, "id": row.get("id"), "doc_loc": row.get("doc_loc"),
              "images": len(images), "rotated_images": 0,
              "row_sha256": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    try:
        for image in images:
            path = SOURCE.parent / image["url"]
            assert path.is_file(), f"Missing media: {path}"
            with Image.open(path) as decoded:
                result["rotated_images"] += check_image_geometry(decoded, image["image_wh"])
                if do_runtime:
                    decoded.load()
        tokenize.set_state("cache")
        tokenize.max_length = None
        full = tokenize(row)
        assert full["num_tokens"] > 0
        tokenize.max_length = sequence_budget
        bounded = tokenize(row)
        result.update(full_tokens=full["num_tokens"], cache_tokens=bounded["num_tokens"],
                      reference_complete=full["num_tokens"] <= sequence_budget,
                      reference_filtered=bounded["num_tokens"] == 0)
        if do_runtime:
            # Oversized records are checked intact and marked as outside the reference budget.
            tokenize.max_length = None
            tokenize.set_state("runtime")
            with patch.object(glm_data, "load_image", wraps=glm_data.load_image) as loader:
                item = tokenize(row, media_root=str(SOURCE.parent))
            expected_paths = [str(SOURCE.parent / image["url"]) for image in images]
            assert [call.args[0] for call in loader.call_args_list] == expected_paths, "Image order changed"
            assert item["num_tokens"] == full["num_tokens"], "Cache/runtime length mismatch"
            if images:
                assert item["num_img_tokens"] == full["num_img_tokens"]
            result["supervised_tokens"] = check_item(item, tokenize.processor.image_token_id,
                                                     len(images), tokenize.merge_unit)
            batch = glm53_vl_sft_collator(
                [[item]], pack_max_length=max(sequence_budget, full["num_tokens"]),
                padding_token_idx=tokenize.tokenizer.pad_token_id or 0,
                image_token_id=tokenize.processor.image_token_id, merge_unit=tokenize.merge_unit,
            )[0]
            assert batch["seq_ctx"].input_ids.shape == batch["shifted_labels"].shape
            assert int((batch["shifted_labels"] != -100).sum()) == result["supervised_tokens"]
            result["input_ids_sha256"] = hashlib.sha256(torch.as_tensor(item["input_ids"]).numpy().tobytes()).hexdigest()
            if fixture_dir:
                torch.save(item, Path(fixture_dir) / f"sample_{line:05d}.pt")
            del item, batch
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        tokenize.max_length = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--lines", help="Comma-separated source lines; omitted means the entire dataset")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    assert args.workers > 0 and args.length > 0
    assert not args.fixtures or args.runtime
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = str(args.out / "fixtures") if args.fixtures else None
    if fixtures:
        Path(fixtures).mkdir()
    rows = [(line, json.loads(raw)) for line, raw in enumerate(SOURCE.open(), 1)]
    requested = set(map(int, args.lines.split(","))) if args.lines else None
    if requested:
        rows = [entry for entry in rows if entry[0] in requested]
        assert len(rows) == len(requested)
    repo = Path(__file__).resolve().parents[2]
    manifest = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "worktree": str(repo), "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "arguments": vars(args) | {"out": str(args.out)}, "rows_requested": len(rows),
                "image_token_budget": TOKEN_BUDGET, "processor": PROCESSOR}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(args.length, args.runtime, fixtures)) as pool:
        with (args.out / "samples.jsonl").open("w") as stream:
            for item in pool.map(audit, rows, chunksize=1):
                results.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                if len(results) % 100 == 0 or not item["passed"]:
                    print(json.dumps({"completed": len(results), "last": item}), flush=True)
    summary = {"rows": len(results), "images": sum(r["images"] for r in results),
               "passed": sum(r["passed"] for r in results),
               "failed": sum(not r["passed"] for r in results),
               "reference_filtered": sum(r.get("reference_filtered", False) for r in results),
               "reference_incomplete": sum(not r.get("reference_complete", False) for r in results),
               "max_tokens": max((r.get("full_tokens", 0) for r in results), default=0),
               "elapsed_seconds": time.monotonic() - start,
               "errors": dict(collections.Counter(r["error"] for r in results if not r["passed"]))}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
