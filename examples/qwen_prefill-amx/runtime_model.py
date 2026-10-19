"""Python adapter for the original supplied C++/AMX prefill runtime."""

import torch
from tilefoundry.runtime import RuntimeModule
from kernels import ops
from scheduled import make_scheduled


class QwenPrefill(RuntimeModule):
    """Execute one model through its original native prefill driver.

    The scheduled HIR remains available as ``module`` for inspection. Execution
    uses the supplied native graph, including its existing host operations.
    """

    def __init__(self, config, *, sequence=128, threads=8, capacity=2048):
        super().__init__("QwenPrefill", entry="prefill")
        if sequence < 128 or sequence > 8192 or sequence % 32:
            raise ValueError("sequence must be a multiple of 32 between 128 and 8192")
        self._semantic = make_scheduled(config, sequence)
        self.threads = threads
        self.capacity = max(capacity, sequence)
        self.sequence = sequence
        self.vocab = config["vocab_size"]
        self.ffi = None
        self.loaded = False

    @property
    def module(self):
        return self._semantic

    def load(self, resource, *, placement=None):
        if self.loaded:
            raise RuntimeError("Close the loaded model before loading another")
        self.ffi = ops()
        resource.prepare()
        try:
            vocab = int(
                self.ffi.load_model(
                    str(resource.checkpoint), self.threads, self.capacity, 7
                )
            )
            self.loaded = True
            if vocab != self.vocab:
                raise ValueError(
                    "The checkpoint does not match the selected model vocabulary"
                )
            for name, n, k, path in resource.entries():
                self.ffi.load_weight(name, str(path), n, k)
        except BaseException:
            self.close()
            raise

    def forward(self, tokens):
        if not self.loaded:
            raise RuntimeError("Load the model resource first")
        if (
            tokens.device.type != "cpu"
            or tokens.dtype != torch.int32
            or tokens.ndim != 1
        ):
            raise ValueError("Expected a CPU int32 token vector")
        if tokens.numel() != self.sequence:
            raise ValueError("Token length must equal the module's sequence length")
        logits = torch.empty(self.vocab, dtype=torch.float32)
        self.ffi.prefill(tokens.contiguous(), logits)
        return logits.reshape(1, self.vocab)

    def close(self):
        if self.loaded:
            self.ffi.close_model()
            self.loaded = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
