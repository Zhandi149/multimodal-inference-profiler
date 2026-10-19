"""Optional Hugging Face adapter. Only this module imports GPU dependencies."""
import os
import time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from .timing import TokenClock
from .gpu_primitives import timed_encoder
from .compatibility import call_with_supported_kwargs


class CudaProfiler:
    def __init__(self, model_id, revision=None, allow_download=False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for live profiling; use the offline demo on CPU")
        torch.cuda.set_device(0)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        options = dict(local_files_only=not allow_download, revision=revision, token=os.getenv("HF_TOKEN"))
        self.processor = AutoProcessor.from_pretrained(model_id, **options)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, **options, torch_dtype=self.dtype, attn_implementation="eager")
        self.model.to("cuda:0").eval()
        if not callable(getattr(self.model, "get_image_features", None)):
            raise ValueError("this adapter requires model.get_image_features(pixel_values)")
        self.metadata = {"synthetic": False, "requested_model": model_id,
                         "requested_revision": revision,
                         "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                         "device": torch.cuda.get_device_name(0), "dtype": str(self.dtype),
                         "torch_version": torch.__version__,
                         "transformers_version": __import__("transformers").__version__,
                         "measurement": "synchronized single-request greedy generation; warm model"}

    def prepare(self, image_paths, prompt):
        images = []
        for path in image_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        # Keep token IDs integral; cast only floating-point image inputs.
        return {name: value.to("cuda:0", dtype=self.dtype) if value.is_floating_point()
                else value.to("cuda:0") for name, value in inputs.items()}

    def measure(self, inputs, max_new_tokens):
        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("only batch size one is supported")
        if "pixel_values" not in inputs:
            raise ValueError("processor did not produce image inputs")
        with torch.inference_mode():
            # Bind to this request's pixels, never a prior request's callback.
            encoder_ms = timed_encoder(lambda: call_with_supported_kwargs(
                self.model.get_image_features, {"pixel_values": inputs["pixel_values"]}))
            torch.cuda.reset_peak_memory_stats()
            streamer = TokenClock(synchronize=torch.cuda.synchronize)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         num_beams=1, num_return_sequences=1, use_cache=True,
                                         return_dict_in_generate=False, streamer=streamer)
            torch.cuda.synchronize()
            finish = time.perf_counter()
            count = int(output.shape[-1] - inputs["input_ids"].shape[-1])
            result = streamer.finish(start, finish, count)
            result.update(encoder_ms=encoder_ms,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
            del output
        return result
