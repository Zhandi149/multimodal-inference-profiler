# Multimodal Inference Profiler · MedGemma Example

A small profiling toolkit for inspecting single-request vision-language inference: time to first token, the decode interval, standalone image-encoder latency, and peak allocated CUDA memory.

Includes a named **MedGemma 4B profiling preset** and a [step-by-step MedGemma guide](docs/medgemma.md). The model is provided by Google; this repository contains the profiling software, synthetic input fixtures, and reporting tools.

**Python CLI · Optional PyTorch / Transformers adapter · CSV + JSON + HTML reports · Offline demo**

Adapted from existing local multimodal profiling utilities into a standalone package. General timing helpers are retained; request handling, validation, reporting, and tests are organized for independent use. No research-specific optimization methods or experimental findings are included.

## Try it without a GPU

From the repository root:

```bash
python3 run.py demo --out outputs/demo
```

Open `outputs/demo/report.html` in a browser. The directory also contains `timings.csv` and `summary.json`.

**The offline demo invents timing values to exercise the reporting pipeline. It does not run a model or measure hardware.** Every generated demo report is prominently marked synthetic. A generated example is included under [examples/synthetic-report/](examples/synthetic-report/).

Regenerate a report from an existing run:

```bash
python3 run.py report --input outputs/demo --out outputs/demo-regenerated
```

Existing output directories are never overwritten; choose a new destination for each run.

## What the code demonstrates

- Separating lightweight measurement/reporting logic from optional model dependencies.
- CUDA-event timing for a standalone image-encoder call.
- Streaming token callbacks with explicit prompt exclusion and output-count checks.
- Greedy batch-one generation with synchronized first/last-token timestamps.
- Per-workload and image-count grouping, medians, IQRs, and validated raw rows.
- Manifest-driven local inputs, model revision metadata, and cached-only model loading by default.
- Self-contained HTML reports with escaped labels and no external scripts or assets.

## Live CUDA profiling

The included adapter targets decoder-only Hugging Face image-text models exposing `get_image_features(pixel_values)`, with Gemma 3 / MedGemma as the intended family. Other architectures need their own adapter; accepting a model ID does not establish compatibility.

Create a virtual environment and install the optional runtime dependencies. Choose the PyTorch build appropriate for your CUDA environment before installing the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[gpu]'
```

Acquire access to the chosen model and cache its files separately. If authentication is needed, set `HF_TOKEN` in your shell; do not place credentials in a manifest or commit them. No weights or datasets are distributed here.

For a cached compatible model:

```bash
python3 run.py profile \
  --preset medgemma-4b \
  --manifest examples/images.json \
  --warmup 1 --repeats 4 --max-new-tokens 16 \
  --out outputs/local-profile
```

The preset selects `google/medgemma-4b-it`. You can instead use `--model MODEL_ID` for a compatible model or local model directory. Use `--revision COMMIT_ID` for a pinned model revision. Model and processor loading are local-only unless you explicitly pass `--allow-download`. The runner uses CUDA device 0 and does not implement multi-GPU dispatch.

The example manifest references locally generated colored-shape images. Add local image paths and prompts to a separate manifest for other workloads. Neither prompts, input paths, generated text, nor image contents are written to reports; sample IDs and workload labels are written, so use non-sensitive labels.

## Measurement boundaries

| Field | Meaning |
| --- | --- |
| `encoder_ms` | CUDA-event duration of a separate `get_image_features` call on this request's pixels. |
| `ttft_ms` | Host elapsed time from `generate()` start to the first generated token callback, synchronized with CUDA. |
| `decode_ms` | Interval between first and last generated-token callbacks. Zero for a one-token output. |
| `total_ms` | Host elapsed time surrounding the generation call and final CUDA synchronization. |
| `decode_per_token_ms` | Decode interval divided by the number of tokens after the first; unavailable for one-token output. |
| `peak_allocated_gib` | Peak PyTorch-allocated CUDA memory during generation, including resident model/input memory. |

Encoder timing and generation timing are **separate measurements**, not additive stages. The toolkit does not label their difference as directly measured prefill latency. Loading, preprocessing, input transfer, and report generation are outside the generation timer. Per-token synchronization perturbs execution, so these results describe instrumented runs rather than uninstrumented serving throughput.

See [measurement notes](docs/measurement.md) for assumptions and limitations.

## Project layout

```text
src/mmprof/
  timing.py          # Library-independent token callback accounting
  compatibility.py   # Signature-aware function invocation
  presets.py         # Named MedGemma model identifier
  gpu_primitives.py  # CUDA-event timing helper
  cuda_backend.py    # Optional model/processor adapter
  reporting.py       # Row validation, summaries, CSV/JSON/HTML output
  cli.py             # Demo, profile, and report commands
examples/            # Invented inputs and prominently labeled synthetic report
tests/              # Offline tests and optional mocked adapter contracts
```

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

There are 12 standard-library tests and 2 optional adapter-contract tests. The latter run with the runtime libraries installed and mock CUDA/model operations; they do not download weights or measure a GPU. The offline tests verify MedGemma preset routing and metadata, prompt exclusion, timestamp consistency, one-token behavior, invalid data rejection, report round trips, output preservation, and HTML escaping.

A [CI template](ci/github-actions.yml.example) runs the lightweight suite on Python 3.10 and 3.12. It is a template rather than an active workflow because the current publishing credentials do not include workflow-write permission.

**Validation status:** offline tests and mocked adapter checks pass in the development environment. No live CUDA profiling run has been performed for this extracted package. The adapter's real model compatibility, timing overhead, and multi-image behavior still need hardware validation.

## Scope

This is software instrumentation, not clinical decision support. It contains no medical datasets, patient records, model weights, unpublished paper details, original research results, or cache-reuse interventions. It does not claim a speedup or clinical accuracy. Model access and dependencies are managed separately. No open-source license is granted; reuse permissions can be discussed separately.

API references: [Hugging Face generation utilities](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils), [Gemma 3 model API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gemma3), and the [MedGemma model card](https://huggingface.co/google/medgemma-4b-it).
