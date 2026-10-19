# Measurement contract

## Supported operating point

One process, CUDA device 0, a warm decoder-only image-text model, batch size one, greedy generation, one returned sequence, and no beam search. The adapter requires a `get_image_features(pixel_values)` method and a processor compatible with multimodal chat templates. It is intended for the Gemma 3 family; support for every model exposed by an Auto class is not implied.

## Run sequence

1. Validate the local manifest and reject an existing output directory.
2. Load a cached processor/model, record the requested revision and available resolved model commit, then prepare one sample's images and prompt.
3. Run the configured number of warmups for that sample. Warmups are not recorded.
4. Time a separate encoder call using CUDA events on the current sample.
5. Reset peak allocated-memory statistics, execute instrumented generation, and verify token accounting.
6. Repeat measurements, move to the next sample, validate all rows, and create a new output directory.

Model/processor loading, image decoding, preprocessing, and host-to-device transfer are outside the measured generation interval. The encoder call before each generation also warms the device; this is not a cold-start benchmark. Out-of-memory errors abort the live command rather than silently omit failed workloads.

## Token accounting

The first streamer callback is the prompt, including when the prompt contains one token. Every later callback must contain one generated token. The recorded callback count must equal the returned sequence length minus input length. Batched callbacks and count mismatches are rejected rather than assigned misleading per-token durations.

The first-token measurement includes time spent inside generation before the first callback, including image processing within the model. The separately timed encoder is not subtracted to infer a measured prefill phase. The decode interval excludes the first token and any final cleanup after the last token. CPU overhead, callback overhead, and explicit GPU synchronization remain present.

## Memory and summaries

Peak allocated memory is PyTorch's allocator measurement, not total board usage, reserved memory, or incremental memory over the loaded model. It includes resident weights and prepared inputs.

Reports group by workload label and image count. Each group pools all supplied samples/repeats; it is descriptive, not an independent-sample inferential analysis. Medians and inclusive-quartile IQRs are recorded. IQR is null with fewer than four observations. Undefined one-token per-token decode estimates remain null rather than zero. Raw rows are retained for other aggregation choices.

## Reproducibility and validation limits

For a reproducible comparison, pin a model revision, package environment, hardware, input manifest, generation settings, warmup policy, and instrumentation. The metadata records runtime versions, device, requested revision, available resolved model commit, and run settings. It does not fingerprint inputs or capture a complete environment lockfile. Image-token counts and comparisons between model quality levels are not implemented.

The packaged example reports are synthetic. Tests exercise accounting and report semantics, with mocked model/CUDA calls for the optional adapter. These are not substitutes for a live model smoke test or a comparison against an independent GPU profiler.
