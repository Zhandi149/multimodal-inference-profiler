"""Single-request streamer instrumentation, independent of model libraries."""
import math
import time


class TokenClock:
    """Transformers streamer protocol for batch-one greedy generation.

    Transformers first sends the prompt, even for a one-token prompt. Later
    callbacks must each contain exactly one generated token. Synchronization
    is injected so the accounting can be tested without CUDA.
    """
    def __init__(self, synchronize=lambda: None, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.timestamps = []
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if value.numel() != 1:
            raise ValueError("expected one generated token per callback, batch size one")
        self.synchronize()
        self.timestamps.append(self.clock())

    def end(self):
        pass

    def finish(self, start, end, output_tokens):
        if not self.timestamps or len(self.timestamps) != output_tokens:
            raise ValueError("streamed token count does not match generated sequence")
        sequence = [start, *self.timestamps, end]
        if any(not math.isfinite(t) for t in sequence) or any(
                a > b for a, b in zip(sequence, sequence[1:])):
            raise ValueError("timestamps must be finite and monotonic")
        decode = (self.timestamps[-1] - self.timestamps[0]) * 1000
        return {"ttft_ms": (self.timestamps[0] - start) * 1000,
                "decode_ms": decode, "total_ms": (end - start) * 1000,
                "output_tokens": output_tokens,
                "decode_per_token_ms": decode / (output_tokens - 1) if output_tokens > 1 else None}
