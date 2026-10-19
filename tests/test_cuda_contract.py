"""Optional CPU-only adapter contract tests; no model downloads or CUDA execution."""
import importlib.util
import unittest
from unittest.mock import Mock, patch

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "PIL"))


@unittest.skipUnless(HAS_RUNTIME, "optional GPU libraries are not installed")
class AdapterContractTests(unittest.TestCase):
    def test_each_measurement_uses_current_pixels_and_counts_only_output(self):
        import torch
        import mmprof.cuda_backend as backend
        profiler = backend.CudaProfiler.__new__(backend.CudaProfiler)
        profiler.model = Mock()
        seen = []
        profiler.model.get_image_features.side_effect = lambda pixel_values: seen.append(pixel_values)

        def generate(**kwargs):
            tracker = kwargs["streamer"]
            tracker.put(kwargs["input_ids"])
            tracker.put(torch.tensor([11]))
            tracker.put(torch.tensor([12]))
            return torch.cat([kwargs["input_ids"], torch.tensor([[11, 12]])], dim=-1)

        profiler.model.generate.side_effect = generate

        def timed(fn):
            fn()
            return 1.0

        with patch.object(backend, "timed_encoder", side_effect=timed), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=1024**3):
            for pixels in ("first-input", "second-input"):
                row = profiler.measure({"input_ids": torch.tensor([[7]]), "pixel_values": pixels}, 2)
                self.assertEqual(row["output_tokens"], 2)
                self.assertEqual(row["peak_allocated_gib"], 1)
        self.assertEqual(seen, ["first-input", "second-input"])

    def test_model_loading_is_cached_only_by_default(self):
        import torch
        import mmprof.cuda_backend as backend
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config._commit_hash = "fixture-commit"
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "set_device"), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="mock device"), \
             patch.object(backend.AutoProcessor, "from_pretrained") as processor, \
             patch.object(backend.AutoModelForImageTextToText, "from_pretrained", return_value=model) as loader:
            profiler = backend.CudaProfiler("fixture/model", revision="fixture-revision")
        self.assertTrue(processor.call_args.kwargs["local_files_only"])
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertEqual(loader.call_args.kwargs["revision"], "fixture-revision")
        self.assertEqual(profiler.metadata["resolved_commit"], "fixture-commit")


if __name__ == "__main__":
    unittest.main()
