"""CLI: offline report demo, CSV report regeneration, and optional CUDA profiling."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from .reporting import write_report, validate


def demo_rows():
    rows = []
    for images in (1, 2):
        for repeat in range(4):
            ttft = 20 * images + repeat
            decode = 12 + repeat
            rows.append(dict(workload="fictional_shapes", sample_id=f"shapes-{images}",
                             repeat=repeat, image_count=images, encoder_ms=12 * images + repeat,
                             ttft_ms=ttft, decode_ms=decode, total_ms=ttft + decode + 1,
                             output_tokens=4, decode_per_token_ms=decode / 3,
                             peak_allocated_gib=1 + images / 10))
    return rows


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in ("repeat", "image_count", "output_tokens"):
            row[name] = int(row[name])
        for name in ("encoder_ms", "ttft_ms", "decode_ms", "total_ms", "peak_allocated_gib"):
            row[name] = float(row[name])
        row["decode_per_token_ms"] = float(row["decode_per_token_ms"]) if row["decode_per_token_ms"] else None
        validate(row)
    return rows


def load_manifest(path):
    path = Path(path)
    samples = json.loads(path.read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must be a nonempty JSON list")
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")
        if not isinstance(sample.get("id"), str) or not sample["id"].strip() or sample["id"] in ids:
            raise ValueError("sample IDs must be unique nonempty strings")
        ids.add(sample["id"])
        for name in ("workload", "prompt"):
            if not isinstance(sample.get(name), str) or not sample[name].strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(sample.get("images"), list) or not sample["images"]:
            raise ValueError("each sample needs at least one local image")
        resolved = []
        for image in sample["images"]:
            if not isinstance(image, str):
                raise ValueError("image paths must be strings")
            location = (path.parent / image).resolve()
            if not location.is_file():
                raise ValueError("an input image does not exist")
            resolved.append(location)
        sample["image_paths"] = resolved
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate explicitly synthetic CSV/JSON/HTML; no model required")
    demo.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("report", help="regenerate a report from a previous toolkit run")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    profile = commands.add_parser("profile", help="measure a cached compatible model on CUDA device 0")
    profile.add_argument("--model", required=True)
    profile.add_argument("--revision")
    profile.add_argument("--manifest", type=Path, required=True)
    profile.add_argument("--out", type=Path, required=True)
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--repeats", type=int, default=3)
    profile.add_argument("--max-new-tokens", type=int, default=16)
    profile.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("output directory exists; choose a new run directory")
        if args.command == "demo":
            rows, metadata = demo_rows(), {"synthetic": True, "model": "none",
                "description": "Invented timings for testing report generation; not hardware measurements"}
        elif args.command == "report":
            rows = read_csv(args.input / "timings.csv")
            metadata = json.loads((args.input / "summary.json").read_text())["metadata"]
        else:
            if args.warmup < 0 or args.repeats < 1 or args.max_new_tokens < 1:
                raise ValueError("invalid warmup, repeats, or token budget")
            samples = load_manifest(args.manifest)
            try:
                from .cuda_backend import CudaProfiler
            except ImportError as exc:
                raise RuntimeError("install the gpu optional dependencies to run live profiling") from exc
            backend = CudaProfiler(args.model, args.revision, args.allow_download)
            rows, metadata = [], dict(backend.metadata)
            metadata.update(warmup=args.warmup, repeats=args.repeats,
                            max_new_tokens=args.max_new_tokens)
            for sample in samples:
                inputs = backend.prepare(sample["image_paths"], sample["prompt"])
                for _ in range(args.warmup):
                    backend.measure(inputs, args.max_new_tokens)
                for repeat in range(args.repeats):
                    row = backend.measure(inputs, args.max_new_tokens)
                    rows.append(dict(workload=sample["workload"], sample_id=sample["id"],
                                     image_count=len(sample["image_paths"]), repeat=repeat, **row))
                del inputs
        metadata["report_created_utc"] = datetime.now(timezone.utc).isoformat()
        result = write_report(args.out, rows, metadata)
        print(f"Report written: {result / 'report.html'}")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
