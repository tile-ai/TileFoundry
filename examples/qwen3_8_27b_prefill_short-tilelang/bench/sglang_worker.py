"""SGLang prefill-only measurement worker (line-oriented JSON protocol).

Run:   CUDA_VISIBLE_DEVICES=<one card> ROOT/sglang-env/bin/python ROOT/bench/sglang_worker.py
       [--model-path $QWEN38_CKPT] [--mem-fraction-static 0.5]

Protocol (stdin -> stdout, exactly one JSON line out per JSON line in; all
logging, including SGLang's own prints, goes to stderr because fd 1 is
re-pointed to fd 2 at start-up and the protocol uses a private dup of fd 1):

  {"cmd":"load"}            -> {"ok":true,"gpu_name":...,"gpu":{...},"backends":{...},"load_s":...}
      optional fields: "model_path", "mem_fraction_static", "extra_server_args"
      (list of SGLang CLI tokens; default empty = SGLang defaults; only for
      diagnostics, the baseline uses none).
  {"cmd":"bench","token_ids":[...],"warmup":W,"iters":N}
                            -> {"ok":true,"T":...,"times_ms":[N],"median_ms","min_ms","max_ms",
                                "argmax":[N],"logits_max_abs_diff":...,"sm_clock_mhz":{...},
                                "recompute_asserts":{...},...}
  {"cmd":"quit"}            -> {"ok":true,"cmd":"quit"} and the process exits.
  Any failure (including a failed recompute assertion) -> {"ok":false,"error":...,"traceback":...}.

Model loading
  sglang.benchmark.one_batch.load_model() (the in-process loader of SGLang's
  bench_one_batch) builds the ModelRunner: weights, memory pools, attention
  backends, mamba/GDN backend, prefill + decode CUDA graphs.  The ServerArgs are
  SGLang defaults except: --disable-radix-cache, --mem-fraction-static (0.5),
  tp_size 1, and decode CUDA-graph max_bs = 1 (what bench_one_batch itself sets).
  Prefill CUDA graph backend is left at SGLang's default ("breakable" on CUDA).

TIMED REGION (exact)
  torch.cuda.synchronize()
  start_event.record(torch.cuda.current_stream())
  out = model_runner.forward(forward_batch)        # ModelRunner.forward
  end_event.record(torch.cuda.current_stream())
  torch.cuda.synchronize()
  forward_ms = start_event.elapsed_time(end_event)

  Inside model_runner.forward(forward_batch) (SGLang 0.5.21 code path, verified
  per iteration by out.can_run_graph == True):
    - ModelRunner._forward_raw bookkeeping (_prepare_eager_forward_batch).
    - _maybe_execute_deferred_mamba_cow_and_clear: zeroes this request's fresh
      GDN state slot (conv + ssm state, 48 layers) on the forward stream.
    - PrefillCudaGraphRunner.execute (breakable CUDA graph, SGLang default):
      load_batch (copy live batch tensors into the static graph buffers),
      FA3 + GDN attention-metadata preparation, then the model forward:
      token embedding (eager), the 64 decoder layers + final RMSNorm (captured
      graph segments for norms / projections / MLP, eager breaks for the FA3
      attention and the GDN conv1d + chunked delta-rule kernels; this is also
      where the K/V of the 16 full-attention layers and the GDN states are
      written to the pools), then the LogitsProcessor: gather of the last
      position's hidden state and the lm_head GEMM -> next_token_logits [1, V].
  So lm_head / logits for the last position ARE inside.  Logits for the other
  T-1 positions are not computed by SGLang (no logprobs requested).

  OUTSIDE the timed region: tokenization (done by the driver), Req creation,
  ScheduleBatch.init_new + prepare_for_extend (KV-slot, req-slot and GDN-slot
  allocation, req_to_token write, sampling-info build), input_ids H2D copy,
  ForwardBatch.init_new (positions, extend_* lens), the NaN poisoning of the
  slots, sampling / argmax, all recompute assertions, and freeing the slots.
  "wall_ms" (perf_counter around forward + the closing synchronize) is reported
  as a cross-check only; the headline numbers are the CUDA-event times.

RECOMPUTE EVIDENCE (asserted for every warm-up and timed iteration)
  - A new Req (unique rid) and a new ScheduleBatch/ForwardBatch every iteration;
    no prefix cache (radix cache disabled, dummy tree cache, prefix_indices
    empty).  Asserted: forward_mode EXTEND, batch_size 1, extend_prefix_lens
    == [0] (cpu list and device tensor), extend_seq_lens == [T], seq_lens ==
    [T], len(input_ids) == len(out_cache_loc) == T, positions == arange(T),
    input_ids == the given token ids.
  - Before the forward the KV slots of this request (out_cache_loc) in all 16
    full-attention layers (K and V) and this request's GDN state slot (conv
    state and fp32 ssm state, all 48 linear layers) are filled with NaN, and
    it is asserted that they hold NaN.  After the forward it is asserted that
    no NaN is left in any of those positions, that every GDN layer's ssm and
    conv state is non-zero (SGLang zeroes the slot first inside the forward,
    so a non-zero state means the GDN kernels wrote it), and that K/V and the
    states agree with the first warm-up iteration (max abs diff reported and
    bounded).  Because layer 63 (full attention) K/V exist for all T tokens,
    all 64 layers ran over all T tokens.
  - next_token_logits are finite and their argmax equals the first warm-up
    iteration's argmax.
  - After freeing, the free counts of the KV allocator, the req pool and the
    GDN state allocator are back to their pre-iteration values (no leak, and
    nothing kept that a later run could reuse).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Keep fd 1 for the protocol only.  Everything else that writes to stdout
# (python prints inside SGLang, C/C++ libraries) is sent to stderr.
# ---------------------------------------------------------------------------
_PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

DEFAULT_MODEL = os.environ.get("QWEN38_CKPT")  # the published checkpoint; see ../paths.py

# Relative tolerance for "state equals the first warm-up state".  The forward
# is expected to be bitwise deterministic; the tolerance only guards against a
# non-deterministic kernel turning the assertion into a flaky one.
STATE_REL_TOL = 1e-2


def log(*a):
    print("[sglang_worker]", *a, file=sys.stderr, flush=True)


def reply(obj):
    _PROTO_OUT.write(json.dumps(obj) + "\n")
    _PROTO_OUT.flush()


# ---------------------------------------------------------------------------
# GPU identity and clocks (NVML preferred, nvidia-smi fallback)
# ---------------------------------------------------------------------------
class GpuInfo:
    def __init__(self, uuid: str):
        self.uuid = uuid
        self.nvml = None
        self.handle = None
        self.index = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.index = int(pynvml.nvmlDeviceGetIndex(self.handle))
        except Exception as e:  # pragma: no cover - fallback path
            log("pynvml unavailable, using nvidia-smi:", repr(e))
            self.nvml = None
            out = self._smi("index,uuid", target=None)
            for line in out.splitlines():
                idx, u = [x.strip() for x in line.split(",")]
                if u == uuid:
                    self.index = int(idx)

    @staticmethod
    def _smi(fields, target):
        import subprocess

        cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        if target is not None:
            cmd += ["-i", str(target)]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    def clocks(self):
        """SM clock now and its max, in MHz."""
        if self.nvml is not None:
            p = self.nvml
            return {
                "sm": int(p.nvmlDeviceGetClockInfo(self.handle, p.NVML_CLOCK_SM)),
                "sm_max": int(p.nvmlDeviceGetMaxClockInfo(self.handle, p.NVML_CLOCK_SM)),
            }
        sm, smax = [int(x) for x in self._smi("clocks.sm,clocks.max.sm", self.index).split(",")]
        return {"sm": sm, "sm_max": smax}

    def sm_clock(self):
        if self.nvml is not None:
            return int(self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM))
        return self.clocks()["sm"]

    def snapshot(self):
        """Clock + other processes on the card (other users share it)."""
        snap = dict(self.clocks())
        if self.nvml is not None:
            p = self.nvml
            try:
                procs = p.nvmlDeviceGetComputeRunningProcesses(self.handle)
                me = os.getpid()
                snap["other_compute_procs"] = sum(1 for x in procs if x.pid != me)
            except Exception:
                snap["other_compute_procs"] = None
            try:
                snap["util_gpu_pct"] = int(p.nvmlDeviceGetUtilizationRates(self.handle).gpu)
            except Exception:
                snap["util_gpu_pct"] = None
            try:
                snap["temp_c"] = int(p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        return snap


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self, model_path: str, mem_fraction_static: float):
        self.model_path = model_path
        self.mem_fraction_static = mem_fraction_static
        self.mr = None  # sglang ModelRunner
        self.gpu = None
        self.rid_counter = 0

    # ----------------------------------------------------------------- load
    def load(self, cmd):
        if self.mr is not None:
            return {"ok": True, "cmd": "load", "already_loaded": True, **self.load_info}
        model_path = cmd.get("model_path", self.model_path)
        mem_frac = float(cmd.get("mem_fraction_static", self.mem_fraction_static))
        t0 = time.perf_counter()

        import torch
        from sglang.benchmark import one_batch as ob
        from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, Phase
        from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
        from sglang.srt.server_args import PortArgs, ServerArgs
        from sglang.srt.utils import configure_logger

        # Same steps as sglang.benchmark.one_batch.main() + latency_test().
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path", model_path,
            "--tp-size", "1",
            "--disable-radix-cache",
            "--mem-fraction-static", str(mem_frac),
        ]
        extra = [str(x) for x in cmd.get("extra_server_args", [])]  # default: none
        argv += extra
        server_args = ServerArgs.from_cli_args(parser.parse_args(argv))
        explicit = server_args.cuda_graph_config
        if isinstance(explicit, CudaGraphConfig):
            explicit = explicit.to_dict()
        graph_config = dict(explicit or {})
        decode = dict(graph_config.get(Phase.DECODE) or {})
        decode["max_bs"] = 1
        graph_config[Phase.DECODE] = decode
        declare_resolution(server_args, "benchmark.one_batch", cuda_graph_config=graph_config)
        server_args.resolve_once()
        cfg = resolving_view(server_args)
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=0, pp_rank=0), gpu_id=0
            ),
        )
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        configure_logger(server_args, prefix=" TP0")

        t_load0 = time.perf_counter()
        bench_runner, _tokenizer = ob.load_model(server_args, port_args, 0, 0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mr = bench_runner.torch_runner
        self.mr = mr
        self.cfg = cfg
        self.ob = ob

        props = torch.cuda.get_device_properties(0)
        uuid = "GPU-" + str(props.uuid)
        self.gpu = GpuInfo(uuid)

        # Pools used by the poisoning / leak checks.
        self.kv_pool = mr.token_to_kv_pool
        self.full_kv = getattr(self.kv_pool, "full_kv_pool", self.kv_pool)
        self.k_bufs = list(self.full_kv.k_buffer)
        self.v_bufs = list(self.full_kv.v_buffer)
        self.req_pool = mr.req_to_token_pool
        self.mamba_pool = getattr(self.req_pool, "mamba_pool", None)
        assert self.mamba_pool is not None, "expected a HybridReqToTokenPool with a mamba pool"
        self.conv_bufs = list(self.mamba_pool.mamba_cache.conv)
        self.ssm_buf = self.mamba_pool.mamba_cache.temporal

        self.load_info = {
            "gpu_name": props.name,
            "gpu": {
                "physical_index": self.gpu.index,
                "uuid": uuid,
                "pci_bus_id": f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_device": "cuda:0",
            },
            "backends": self._describe_backends(),
            "load_s": round(t1 - t0, 3),
            "load_model_s": round(t1 - t_load0, 3),
            "server_args_argv": argv,
            "memory": {
                "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                "max_total_num_tokens": int(mr.max_total_num_tokens),
                "req_pool_size": int(self.req_pool.size),
                "mamba_pool_size": int(self.mamba_pool.size),
            },
            "clock_after_load": self.gpu.snapshot(),
        }
        return {"ok": True, "cmd": "load", **self.load_info}

    def _describe_backends(self):
        import flashinfer
        import sglang
        import torch
        from sglang.srt.layers.attention.linear import gdn_backend as gdn_mod

        mr, cfg = self.mr, self.cfg
        ab = mr.attn_backend
        full_ab = getattr(ab, "full_attn_backend", ab)
        lin_ab = getattr(ab, "linear_attn_backend", None)
        disp = getattr(lin_ab, "kernel_dispatcher", None)
        pcg = mr.prefill_cuda_graph_runner
        dcg = mr.decode_cuda_graph_runner
        cg = cfg.cuda_graph_config
        conv_fn = getattr(gdn_mod, "causal_conv1d_fn", None)

        lin_method = None
        try:
            lm = mr.model.model if hasattr(mr.model, "model") else mr.model
            layer0 = lm.layers[0]
            for name, mod in layer0.named_modules():
                qm = getattr(mod, "quant_method", None)
                if qm is not None and name.endswith(("gate_up_proj", "down_proj")):
                    lin_method = f"layers.0.{name}: {type(qm).__name__}"
                    break
        except Exception as e:
            lin_method = f"unknown ({e!r})"

        return {
            "versions": {
                "sglang": sglang.__version__,
                "torch": torch.__version__,
                "flashinfer": getattr(flashinfer, "__version__", "?"),
            },
            "model_class": type(mr.model).__name__,
            "attention_backend_arg": cfg.attention_backend,
            "attention_backend_class": type(ab).__name__,
            "full_attention_backend_class": f"{type(full_ab).__module__}.{type(full_ab).__name__}",
            "full_attention_fa_impl_ver": getattr(full_ab, "fa_impl_ver", None),
            "prefill_attention_backend_str": getattr(ab, "prefill_attention_backend_str", None),
            "decode_attention_backend_str": getattr(ab, "decode_attention_backend_str", None),
            "linear_attention_backend_class": type(lin_ab).__name__ if lin_ab is not None else None,
            "linear_attn_backend_arg": cfg.linear_attn_backend,
            "gdn_extend_kernel": type(getattr(disp, "extend_kernel", None)).__name__ if disp else None,
            "gdn_decode_kernel": type(getattr(disp, "decode_kernel", None)).__name__ if disp else None,
            "gdn_causal_conv1d_fn": f"{conv_fn.__module__}.{conv_fn.__name__}" if conv_fn else None,
            "mamba_backend_arg": cfg.mamba_backend,
            "mamba_ssm_state_dtype": str(self.ssm_buf.dtype),
            "kv_cache_dtype": str(self.k_bufs[0].dtype),
            "linear_layer_method": lin_method,
            "prefill_cuda_graph": {
                "backend": cg.prefill.backend,
                "runner_class": type(pcg).__name__ if pcg is not None else None,
                "graph_backend_class": type(getattr(pcg, "backend", None)).__name__ if pcg is not None else None,
                "capture_num_tokens": list(getattr(pcg, "capture_num_tokens", []) or []),
                "tc_compiler": cg.prefill.tc_compiler,
                "note": "tc_compiler is only used by tc_piecewise; breakable does not use torch.compile",
            },
            "decode_cuda_graph": {
                "backend": cg.decode.backend,
                "bs": cg.decode.bs,
                "runner_class": type(dcg).__name__ if dcg is not None else None,
                "note": "decode is never run by this worker",
            },
            "enable_torch_compile": bool(cfg.enable_torch_compile),
            "disable_radix_cache": bool(cfg.disable_radix_cache),
            "page_size": cfg.page_size,
            "chunked_prefill_size": cfg.chunked_prefill_size,
            "mem_fraction_static": cfg.mem_fraction_static,
        }

    # ---------------------------------------------------------- batch build
    def _build(self, token_ids):
        """Fresh Req -> ScheduleBatch -> ForwardBatch, prefix length 0.

        Mirrors sglang.benchmark.one_batch.extend() up to (not including)
        model_runner.forward().
        """
        from array import array

        import torch
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.runtime_context import get_schedule
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        mr, ob = self.mr, self.ob
        self.rid_counter += 1
        req = Req(
            rid=f"sglw-{os.getpid()}-{self.rid_counter}",
            origin_input_text="",
            origin_input_ids=array("q", token_ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        assert len(req.prefix_indices) == 0, "new Req must not carry a prefix"
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))

        tree_cache = ob.TreeCacheNamespace(  # no prefix cache at all
            page_size=get_schedule().page_size,
            device=mr.device,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=mr.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        ob._maybe_prepare_mlp_sync_batch(batch, mr)
        if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None
        fb = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
        return req, batch, fb

    def _assert_fresh_extend(self, fb, req, token_ids_dev, T):
        import torch

        ev = {}
        assert fb.forward_mode.is_extend(), f"forward_mode {fb.forward_mode}"
        assert not fb.forward_mode.is_decode()
        assert fb.batch_size == 1, f"batch_size {fb.batch_size}"
        assert list(fb.extend_prefix_lens_cpu) == [0], f"extend_prefix_lens_cpu {fb.extend_prefix_lens_cpu}"
        assert list(fb.extend_seq_lens_cpu) == [T], f"extend_seq_lens_cpu {fb.extend_seq_lens_cpu}"
        assert fb.extend_prefix_lens.tolist() == [0], f"extend_prefix_lens {fb.extend_prefix_lens}"
        assert fb.extend_seq_lens.tolist() == [T], f"extend_seq_lens {fb.extend_seq_lens}"
        assert fb.seq_lens_cpu.tolist() == [T], f"seq_lens_cpu {fb.seq_lens_cpu}"
        assert fb.seq_lens.tolist() == [T]
        assert fb.input_ids.numel() == T and fb.out_cache_loc.numel() == T
        assert torch.equal(fb.input_ids.to(torch.int64), token_ids_dev), "input_ids differ from the given token ids"
        pos = fb.positions
        assert torch.equal(pos.to(torch.int64), torch.arange(T, device=pos.device)), "positions != arange(T)"
        mpos = getattr(fb, "mrope_positions", None)
        if mpos is not None:  # text-only mrope: every section must be 0..T-1
            ar = torch.arange(T, device=mpos.device).expand(mpos.shape[0], T)
            assert torch.equal(mpos.to(torch.int64), ar), "mrope_positions != arange(T)"
        ev["mrope_positions_checked"] = mpos is not None
        assert len(req.prefix_indices) == 0
        assert not fb.return_logprob
        ev["forward_mode"] = fb.forward_mode.name
        ev["extend_prefix_lens"] = list(fb.extend_prefix_lens_cpu)
        ev["extend_seq_lens"] = list(fb.extend_seq_lens_cpu)
        return ev

    # ------------------------------------------------------ poison & check
    def _poison(self, loc, midx):
        nan = float("nan")
        for kb, vb in zip(self.k_bufs, self.v_bufs):
            kb[loc] = nan
            vb[loc] = nan
        for cb in self.conv_bufs:
            cb[:, midx] = nan
        self.ssm_buf[:, midx] = nan

    def _gather(self, loc, midx):
        import torch

        k = torch.stack([kb[loc] for kb in self.k_bufs])  # [16, T, Hkv, D]
        v = torch.stack([vb[loc] for vb in self.v_bufs])
        conv = [cb[:, midx].clone() for cb in self.conv_bufs]  # [48, 1, dim, K-1]
        ssm = self.ssm_buf[:, midx].clone()  # [48, 1, HV, V, K]
        return k, v, conv, ssm

    def _assert_poisoned(self, loc, midx):
        import torch

        k, v, conv, ssm = self._gather(loc, midx)
        ok = bool(torch.isnan(k).all()) and bool(torch.isnan(v).all())
        ok = ok and all(bool(torch.isnan(c).all()) for c in conv) and bool(torch.isnan(ssm).all())
        assert ok, "poisoning failed: slots do not hold NaN before the forward"

    # ---------------------------------------------------------------- bench
    def bench(self, cmd):
        import torch

        if self.mr is None:
            return {"ok": False, "cmd": "bench", "error": "model not loaded; send {\"cmd\":\"load\"} first"}
        token_ids = [int(t) for t in cmd["token_ids"]]
        W = int(cmd.get("warmup", 5))
        N = int(cmd.get("iters", 20))
        T = len(token_ids)
        assert T > 0 and N > 0 and W >= 1, "need T>0, iters>0, warmup>=1 (warm-up 0 is the reference)"
        mr = self.mr
        dev = torch.device("cuda", 0)
        token_ids_dev = torch.tensor(token_ids, dtype=torch.int64, device=dev)

        gpu_before = self.gpu.snapshot()
        free0 = self._free_counts()

        times, walls, argmaxes, cg_used, clocks_iter = [], [], [], [], []
        w_times, w_argmax = [], []
        ref = None  # first warm-up iteration: logits, k, v, conv, ssm, argmax
        lmax = lmin = None  # element-wise running max/min of logits over all iterations
        kv_diff = conv_diff = ssm_diff = logit_diff_ref = 0.0
        kv_bitwise = state_bitwise = logits_bitwise = True
        fresh_ev = None
        n_checked = 0
        mamba_slots_seen, kv_first_locs = set(), set()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for it in range(W + N):
                timed = it >= W
                req, batch, fb = self._build(token_ids)
                ev = self._assert_fresh_extend(fb, req, token_ids_dev, T)
                fresh_ev = fresh_ev or ev
                loc = fb.out_cache_loc.to(torch.int64)
                midx = self.req_pool.get_mamba_indices(fb.req_pool_indices).to(torch.int64)
                assert midx.numel() == 1
                mamba_slots_seen.add(int(midx.item()))
                kv_first_locs.add(int(loc[0].item()))
                self._poison(loc, midx)
                self._assert_poisoned(loc, midx)

                # ---------------- timed region ----------------
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start_ev.record(torch.cuda.current_stream())
                out = mr.forward(fb)
                end_ev.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                # ----------------------------------------------
                ms = start_ev.elapsed_time(end_ev)
                clk = self.gpu.sm_clock()

                logits = out.logits_output.next_token_logits
                assert logits is not None and tuple(logits.shape)[0] == 1, f"logits shape {None if logits is None else logits.shape}"
                logits = logits[0].float().clone()
                assert bool(torch.isfinite(logits).all()), "non-finite logits"
                am = int(torch.argmax(logits).item())

                k, v, conv, ssm = self._gather(loc, midx)
                assert not bool(torch.isnan(k).any()) and not bool(torch.isnan(v).any()), \
                    "NaN left in full-attention KV slots: K/V not written by this forward"
                assert all(not bool(torch.isnan(c).any()) for c in conv), "NaN left in GDN conv state"
                assert not bool(torch.isnan(ssm).any()), "NaN left in GDN ssm state"
                ssm_layer_max = ssm.abs().flatten(1).amax(dim=1)
                conv_layer_max = torch.stack([c.abs().float().flatten(1).amax(dim=1) for c in conv]).flatten()
                assert bool((ssm_layer_max > 0).all()), "a GDN layer left an all-zero ssm state"
                assert bool((conv_layer_max > 0).all()), "a GDN layer left an all-zero conv state"
                assert bool(torch.isfinite(k).all() and torch.isfinite(v).all()), "non-finite K/V"

                if ref is None:
                    ref = {"logits": logits, "k": k, "v": v, "conv": conv, "ssm": ssm, "argmax": am}
                    lmax = logits.clone()
                    lmin = logits.clone()
                else:
                    assert am == ref["argmax"], f"argmax {am} != warm-up argmax {ref['argmax']}"
                    d_logit = (logits - ref["logits"]).abs().max().item()
                    d_k = max((k.float() - ref["k"].float()).abs().max().item(),
                              (v.float() - ref["v"].float()).abs().max().item())
                    d_conv = max((c.float() - rc.float()).abs().max().item() for c, rc in zip(conv, ref["conv"]))
                    d_ssm = (ssm - ref["ssm"]).abs().max().item()
                    kv_scale = max(ref["k"].float().abs().max().item(), ref["v"].float().abs().max().item())
                    assert d_k <= STATE_REL_TOL * kv_scale, f"K/V differ from warm-up: {d_k} (scale {kv_scale})"
                    assert d_ssm <= STATE_REL_TOL * ref["ssm"].abs().max().item(), f"ssm differs from warm-up: {d_ssm}"
                    assert d_conv <= STATE_REL_TOL * max(rc.float().abs().max().item() for rc in ref["conv"]), \
                        f"conv state differs from warm-up: {d_conv}"
                    logit_diff_ref = max(logit_diff_ref, d_logit)
                    kv_diff, conv_diff, ssm_diff = max(kv_diff, d_k), max(conv_diff, d_conv), max(ssm_diff, d_ssm)
                    kv_bitwise &= bool(torch.equal(k, ref["k"]) and torch.equal(v, ref["v"]))
                    state_bitwise &= bool(torch.equal(ssm, ref["ssm"]) and all(
                        torch.equal(c, rc) for c, rc in zip(conv, ref["conv"])))
                    logits_bitwise &= bool(torch.equal(logits, ref["logits"]))
                    torch.maximum(lmax, logits, out=lmax)
                    torch.minimum(lmin, logits, out=lmin)
                n_checked += 1

                if timed:
                    times.append(ms)
                    walls.append((t1 - t0) * 1e3)
                    argmaxes.append(am)
                    cg_used.append(bool(out.can_run_graph))
                    clocks_iter.append(clk)
                else:
                    w_times.append(ms)
                    w_argmax.append(am)

                del out, logits, k, v, conv, ssm
                self._free(req, batch)
                free1 = self._free_counts()
                assert free1 == free0, f"pool leak: before {free0} after {free1}"

        gpu_after = self.gpu.snapshot()
        st = sorted(times)
        med = st[len(st) // 2] if len(st) % 2 else 0.5 * (st[len(st) // 2 - 1] + st[len(st) // 2])
        sw = sorted(walls)
        wmed = sw[len(sw) // 2] if len(sw) % 2 else 0.5 * (sw[len(sw) // 2 - 1] + sw[len(sw) // 2])
        pairwise = float((lmax - lmin).abs().max().item())
        return {
            "ok": True,
            "cmd": "bench",
            "T": T,
            "warmup": W,
            "iters": N,
            "gpu_physical_index": self.gpu.index,
            "gpu_name": self.load_info["gpu_name"],
            "times_ms": [round(x, 4) for x in times],
            "median_ms": round(med, 4),
            "min_ms": round(st[0], 4),
            "max_ms": round(st[-1], 4),
            "mean_ms": round(sum(times) / len(times), 4),
            "range_ms": round(st[-1] - st[0], 4),
            "warmup_times_ms": [round(x, 4) for x in w_times],
            "wall_ms_median": round(wmed, 4),
            "wall_ms": [round(x, 4) for x in walls],
            "argmax": argmaxes,
            "warmup_argmax": w_argmax,
            "logits_max_abs_diff": pairwise,
            "logits_max_abs_diff_vs_first_warmup": logit_diff_ref,
            "logits_bitwise_equal_all_iters": logits_bitwise,
            "prefill_cuda_graph_used": all(cg_used) if cg_used else None,
            "prefill_cuda_graph_used_per_iter": cg_used,
            "sm_clock_mhz": {
                "before": gpu_before["sm"],
                "after": gpu_after["sm"],
                "max": gpu_before["sm_max"],
                "per_iter_after_forward": clocks_iter,
                "per_iter_min": min(clocks_iter),
                "per_iter_max": max(clocks_iter),
            },
            "gpu_state": {"before": gpu_before, "after": gpu_after},
            "recompute_asserts": {
                "all_passed": True,
                "iterations_checked": n_checked,
                "fresh_request_and_batch_each_iteration": True,
                "radix_cache_disabled": bool(self.cfg.disable_radix_cache),
                "tree_cache": "sglang.benchmark.one_batch.TreeCacheNamespace (no prefix matching)",
                **fresh_ev,
                "positions_eq_arange_T": True,
                "input_ids_eq_given_token_ids": True,
                "kv_poison": {
                    "full_attention_layers": len(self.k_bufs),
                    "slots_per_layer": T,
                    "nan_before_forward": True,
                    "nan_left_after_forward": 0,
                    "max_abs_diff_vs_first_warmup": kv_diff,
                    "bitwise_equal_all_iters": kv_bitwise,
                },
                "gdn_state_poison": {
                    "linear_layers": int(self.ssm_buf.shape[0]),
                    "ssm_state_shape_per_layer": list(self.ssm_buf.shape[2:]),
                    "conv_state_shape_per_layer": [list(c.shape[2:]) for c in self.conv_bufs],
                    "nan_before_forward": True,
                    "nan_left_after_forward": 0,
                    "all_layers_nonzero_after_forward": True,
                    "ssm_max_abs_diff_vs_first_warmup": ssm_diff,
                    "conv_max_abs_diff_vs_first_warmup": conv_diff,
                    "bitwise_equal_all_iters": state_bitwise,
                    "mamba_slots_used": sorted(mamba_slots_seen),
                },
                "kv_first_slot_values_seen": len(kv_first_locs),
                "logits_finite_all": True,
                "argmax_equal_first_warmup_all": True,
                "first_warmup_argmax": ref["argmax"],
                "state_rel_tol": STATE_REL_TOL,
                "pool_free_counts_restored_each_iter": True,
                "pool_free_counts": free0,
            },
        }

    def _free_counts(self):
        return {
            "kv_tokens": int(self.mr.token_to_kv_pool_allocator.available_size()),
            "req_slots": int(self.req_pool.available_size()),
            "mamba_slots": int(self.req_pool.mamba_allocator.available_size()),
        }

    def _free(self, req, batch):
        """Give back the KV slots, the GDN state slot and the req slot."""
        self.mr.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        self.req_pool.free_mamba_cache(req)
        self.req_pool.free(req)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", default=DEFAULT_MODEL, required=DEFAULT_MODEL is None,
                    help="the published checkpoint; or set $QWEN38_CKPT")
    ap.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = ap.parse_args()
    worker = Worker(args.model_path, args.mem_fraction_static)
    log(f"ready pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for line in sys.stdin:
        line = line.strip()
        if not line:  # blank lines are not commands and get no reply
            continue
        name = None
        try:
            cmd = json.loads(line)
            name = cmd.get("cmd")
            if name == "load":
                res = worker.load(cmd)
            elif name == "bench":
                res = worker.bench(cmd)
            elif name == "quit":
                reply({"ok": True, "cmd": "quit"})
                break
            else:
                res = {"ok": False, "error": f"unknown cmd {name!r}"}
        except Exception as e:  # one reply per command, even on failure
            res = {
                "ok": False,
                "cmd": name,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            log(res["traceback"])
        reply(res)
    _PROTO_OUT.flush()
    # Skip interpreter teardown of NCCL / CUDA graph state (can hang or spam).
    os._exit(0)


if __name__ == "__main__":
    main()
