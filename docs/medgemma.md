# MedGemma profiling example

This project includes a named profiling preset for Google's instruction-tuned [MedGemma 4B model](https://huggingface.co/google/medgemma-4b-it). The preset selects `google/medgemma-4b-it`; it does not bundle the model, imply affiliation with Google, or establish clinical performance.

## What this example demonstrates

The toolkit's engineering contribution is inference instrumentation: processor/model integration, current-image encoder timing, streaming token accounting, repeated measurements, CUDA memory statistics, and report generation. It does not train MedGemma or implement a clinical application.

The included input manifest uses a generated four-color image and a pair of identical copies. These are technical smoke-test inputs, not medical images or a quality benchmark.

## 1. Preview the report without a model

```bash
python3 run.py demo --out outputs/report-preview
```

Open `outputs/report-preview/report.html`. Its timings are invented and explicitly labeled synthetic. They are not MedGemma benchmark measurements.

## 2. Prepare a compatible CUDA environment

Install a suitable PyTorch CUDA build for your machine, then install the toolkit and its optional model dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[gpu]'
```

Follow the access instructions on the [official model card](https://huggingface.co/google/medgemma-4b-it) using your own Hugging Face account. If a token is needed, provide `HF_TOKEN` through your environment. Do not commit it.

The toolkit is scoped to the Transformers 4.57 API family. Its adapter targets the Gemma 3-style `get_image_features(pixel_values)` entry point. Other model families and Transformers major versions require separate compatibility checks.

## 3. Run a bounded smoke test

For a model already cached on your machine:

```bash
python3 run.py profile \
  --preset medgemma-4b \
  --manifest examples/images.json \
  --warmup 1 --repeats 1 --max-new-tokens 4 \
  --out outputs/medgemma-smoke
```

The preset is equivalent to `--model google/medgemma-4b-it`. These flags are mutually exclusive. Loading is cached-only by default. Add `--allow-download` only when you intend to download model/processor files and have the necessary access, storage, and runtime resources.

## 4. Collect repeated measurements

After a successful smoke test, increase repeats in a new output directory and pin a revision for reproducibility:

```bash
python3 run.py profile \
  --preset medgemma-4b \
  --revision YOUR_MODEL_COMMIT \
  --manifest examples/images.json \
  --warmup 1 --repeats 4 --max-new-tokens 16 \
  --out outputs/medgemma-repeated
```

Each run creates raw CSV rows, grouped JSON summaries, and an HTML report. Metadata records the selected preset, requested model/revision, available resolved model commit, device, runtime versions, and generation settings. No prompt text, generated answers, image contents, or input paths are saved in these reports.

## Validation status and interpretation

The MedGemma preset is tested through a mocked end-to-end CLI contract. Current-image encoder binding and cached-only loading also have mocked adapter tests. A live MedGemma CUDA smoke test has not been completed for this package.

The separate encoder measurement is not a disjoint component of generation time. Timers and per-token synchronization add overhead. The colored-shape fixture cannot establish medical capability, and the synthetic report cannot establish inference speed. See [measurement notes](measurement.md) for the full contract.
