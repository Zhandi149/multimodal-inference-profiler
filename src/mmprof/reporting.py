"""Validated rows, per-workload summaries, and self-contained HTML reports."""
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics

METRICS = ("encoder_ms", "ttft_ms", "decode_ms", "total_ms", "decode_per_token_ms", "peak_allocated_gib")


def validate(row):
    for name in ("workload", "sample_id"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("repeat", "image_count", "output_tokens"):
        value = row.get(name)
        if type(value) is not int or value < (0 if name == "repeat" else 1):
            raise ValueError(f"invalid {name}")
    for name in METRICS:
        value = row.get(name)
        if name == "decode_per_token_ms" and value is None and row["output_tokens"] == 1:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
    if row["ttft_ms"] + row["decode_ms"] > row["total_ms"] + 0.001:
        raise ValueError("first/last token timestamps exceed generation duration")
    if row["output_tokens"] == 1:
        if row["decode_ms"] != 0 or row["decode_per_token_ms"] is not None:
            raise ValueError("one-token runs have zero decode interval and no per-token estimate")
    elif not math.isclose(row["decode_per_token_ms"], row["decode_ms"] / (row["output_tokens"] - 1),
                          rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("inconsistent per-token decode time")


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize empty measurements")
    identities = set()
    for row in rows:
        validate(row)
        key = row["workload"], row["sample_id"], row["image_count"], row["repeat"]
        if key in identities:
            raise ValueError("duplicate measurement identity")
        identities.add(key)
    groups = {}
    for workload, image_count in sorted({(r["workload"], r["image_count"]) for r in rows}):
        selected = [r for r in rows if (r["workload"], r["image_count"]) == (workload, image_count)]
        values = {"workload": workload, "image_count": image_count, "n": len(selected)}
        for name in METRICS:
            numbers = [r[name] for r in selected if r[name] is not None]
            quartiles = statistics.quantiles(numbers, n=4, method="inclusive") if len(numbers) >= 4 else None
            values[name] = {"median": statistics.median(numbers) if numbers else None,
                            "iqr": quartiles[2] - quartiles[0] if quartiles else None,
                            "n": len(numbers)}
        groups[f"{workload} / images={image_count}"] = values
    return groups


def write_report(destination, rows, metadata):
    groups = summarize(rows)
    if type(metadata.get("synthetic")) is not bool:
        raise ValueError("metadata must explicitly identify synthetic versus measured data")
    # Validate serializability before creating any output directory.
    summary_json = json.dumps({"metadata": metadata, "groups": groups}, indent=2, allow_nan=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(summary_json + "\n")
    label = "SYNTHETIC DEMO — NOT MEASURED PERFORMANCE" if metadata["synthetic"] else "INSTRUMENTED CUDA RUN"
    body = []
    for name, group in groups.items():
        cells = [escape(name), str(group["n"])]
        for metric in METRICS:
            value = group[metric]["median"]
            cells.append(f"{value:.3f}" if value is not None else "n/a")
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    headings = ("Workload / image count", "n", *METRICS)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Inference profile</title>
<style>body{font:16px/1.6 system-ui;margin:40px auto;padding:0 24px;max-width:1200px;color:#173435;background:#f4f7f3}h1{line-height:1.2}.label{background:#e0eae1;padding:14px;border-left:4px solid #226552}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #d6dfd7}pre{white-space:pre-wrap;font-size:12px}.scroll{overflow-x:auto}</style>
<h1>Multimodal inference profile</h1>'''
    html += f'<p class="label">{label}</p><p>Median values per workload and image count. Times are milliseconds; memory is GiB. These instrumented measurements include synchronization overhead.</p>'
    html += '<div class="scroll"><table><tr>' + ''.join(f'<th>{escape(x)}</th>' for x in headings) + '</tr>' + ''.join(body) + '</table></div>'
    html += '<p>Encoder latency is a separate call; it is not a disjoint part of the generation timer. IQR and counts are available in summary.json. Per-token decode latency is unavailable for one-token outputs.</p>'
    html += '<h2>Run metadata</h2><pre>' + escape(json.dumps(metadata, indent=2)) + '</pre></html>'
    (destination / "report.html").write_text(html)
    return destination
