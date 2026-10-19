import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from mmprof.cli import demo_rows, load_manifest, main, read_csv
from mmprof.compatibility import call_with_supported_kwargs
from mmprof.reporting import summarize, write_report
from mmprof.timing import TokenClock


class Tensor:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class TimingTests(unittest.TestCase):
    def test_prompt_is_skipped_even_if_single_token(self):
        times = iter([1.1, 1.3])
        tracker = TokenClock(clock=lambda: next(times))
        tracker.put(Tensor(1))  # prompt, not the first generated token
        tracker.put(Tensor(1))
        tracker.put(Tensor(1))
        result = tracker.finish(1, 1.4, 2)
        self.assertAlmostEqual(result["ttft_ms"], 100)
        self.assertAlmostEqual(result["decode_ms"], 200)
        self.assertAlmostEqual(result["total_ms"], 400)

    def test_one_token_decode_is_undefined_per_token(self):
        tracker = TokenClock(clock=lambda: 2)
        tracker.put(Tensor(10))
        tracker.put(Tensor(1))
        result = tracker.finish(1, 3, 1)
        self.assertEqual(result["decode_ms"], 0)
        self.assertIsNone(result["decode_per_token_ms"])

    def test_mismatch_and_batched_callbacks_are_rejected(self):
        tracker = TokenClock(clock=lambda: 2)
        tracker.put(Tensor(5))
        with self.assertRaises(ValueError):
            tracker.put(Tensor(2))
        tracker.put(Tensor(1))
        with self.assertRaises(ValueError):
            tracker.finish(1, 3, 2)
        with self.assertRaises(ValueError):
            tracker.finish(3, 4, 1)

    def test_signature_adapter_filters_unsupported_arguments(self):
        self.assertEqual(call_with_supported_kwargs(lambda pixel_values: pixel_values,
                                                    {"pixel_values": 5, "unrelated": 8}), 5)


class ReportTests(unittest.TestCase):
    def test_grouping_and_known_quartiles(self):
        groups = summarize(demo_rows())
        self.assertEqual(len(groups), 2)
        one = groups["fictional_shapes / images=1"]
        self.assertEqual(one["ttft_ms"]["median"], 21.5)
        self.assertEqual(one["ttft_ms"]["iqr"], 1.5)
        self.assertIsNone(summarize(demo_rows()[:1])["fictional_shapes / images=1"]["ttft_ms"]["iqr"])

    def test_invalid_values_and_duplicate_measurements(self):
        for field, value in [("ttft_ms", float("nan")), ("total_ms", 0),
                             ("repeat", -1), ("output_tokens", True),
                             ("decode_per_token_ms", 9)]:
            rows = demo_rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                summarize(rows)
        with self.assertRaises(ValueError):
            summarize([demo_rows()[0], demo_rows()[0]])

    def test_csv_roundtrip_and_html_escaping(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = demo_rows()
            rows[0]["workload"] = '<script>alert("x")</script>'
            out = write_report(Path(folder) / "run", rows,
                               {"synthetic": True, "description": "<unsafe>"})
            self.assertEqual(read_csv(out / "timings.csv"), rows)
            html = (out / "report.html").read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('SYNTHETIC DEMO', html)
            with self.assertRaises(FileExistsError):
                write_report(out, rows, {"synthetic": True})

    def test_label_is_required_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "bad"
            with self.assertRaises(ValueError):
                write_report(out, demo_rows(), {})
            self.assertFalse(out.exists())


class CliTests(unittest.TestCase):
    def test_medgemma_preset_routes_model_and_records_metadata(self):
        adapter = types.ModuleType("mmprof.cuda_backend")
        backend = Mock()
        backend.metadata = {"synthetic": False, "requested_model": "google/medgemma-4b-it",
                            "measurement": "mocked adapter contract test"}
        backend.measure.return_value = {k: v for k, v in demo_rows()[0].items()
                                        if k not in ("workload", "sample_id", "repeat", "image_count")}
        adapter.CudaProfiler = Mock(return_value=backend)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.ppm").write_text("P3\n1 1\n255\n255 0 0\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "sample", "workload": "test",
                                            "images": ["image.ppm"], "prompt": "Describe colors"}]))
            with patch.dict(sys.modules, {"mmprof.cuda_backend": adapter}):
                main(["profile", "--preset", "medgemma-4b", "--revision", "fixture-revision",
                      "--manifest", str(manifest), "--out", str(root / "run"),
                      "--repeats", "1", "--warmup", "0"])
            adapter.CudaProfiler.assert_called_once_with("google/medgemma-4b-it", "fixture-revision", False)
            metadata = json.loads((root / "run/summary.json").read_text())["metadata"]
            self.assertEqual(metadata["preset"], "medgemma-4b")
            self.assertEqual(backend.measure.call_count, 1)

    def test_offline_mode_does_not_import_gpu_libraries(self):
        code = "import sys; import mmprof.cli; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_demo_and_report_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            main(["demo", "--out", str(first)])
            main(["report", "--input", str(first), "--out", str(second)])
            self.assertEqual(read_csv(first / "timings.csv"), read_csv(second / "timings.csv"))
            self.assertTrue(json.loads((second / "summary.json").read_text())["metadata"]["synthetic"])
            with self.assertRaises(SystemExit) as caught:
                main(["demo", "--out", str(first)])
            self.assertEqual(caught.exception.code, 1)

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text('[{"id":"a","workload":"test","prompt":"shapes","images":["missing.png"]}]')
            with self.assertRaises(ValueError):
                load_manifest(path)
            image = Path(folder) / "synthetic.ppm"
            image.write_text('P3\n1 1\n255\n255 0 0\n')
            sample = {"id": "a", "workload": "test", "prompt": "shapes", "images": [image.name]}
            path.write_text(json.dumps([sample]))
            self.assertEqual(load_manifest(path)[0]["image_paths"], [image.resolve()])
            path.write_text(json.dumps([sample, sample]))
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
