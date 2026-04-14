#!/usr/bin/env python3
import argparse, time
import numpy as np

from tinygrad import Tensor, Device
from tinygrad.engine.realize import get_runner
from tinygrad.uop.ops import Ops

def run_case(n:int, seed:int):
  np.random.seed(seed)
  a = np.random.uniform(-2, 2, size=(n, n)).astype(np.float16)
  b = np.random.uniform(-2, 2, size=(n, n)).astype(np.float16)

  probe = Tensor(a).half().matmul(Tensor(b).half())
  sink_asts = [ei.ast for ei in probe.schedule() if ei.ast.op is Ops.SINK]
  if not sink_asts: raise RuntimeError("no SINK ast found")
  runner = get_runner(Device.DEFAULT, sink_asts[-1])
  prg = runner._prg

  wmma_uops = sum(1 for op, _, _, _ in prg.uops if op is Ops.WMMA)
  fused_meta = getattr(prg, "fused_matmul_meta", None) is not None
  hb, fb = getattr(prg, "fused_matmul_hits", 0), getattr(prg, "fused_matmul_fallbacks", 0)

  st = time.perf_counter()
  out = Tensor(a).half().matmul(Tensor(b).half()).realize().numpy()
  elapsed_ms = (time.perf_counter() - st) * 1e3

  ha, fa = getattr(prg, "fused_matmul_hits", 0), getattr(prg, "fused_matmul_fallbacks", 0)
  ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
  max_abs = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))

  print(f"n={n}")
  print(f"  runner={runner.p.name}")
  print(f"  wmma_uops={wmma_uops}")
  print(f"  fused_meta={fused_meta}")
  print(f"  fused_hits={hb}->{ha}  fused_fallbacks={fb}->{fa}")
  print(f"  realize_ms={elapsed_ms:.2f}")
  print(f"  max_abs={max_abs:.6f}")

def main():
  parser = argparse.ArgumentParser(description="Run N x N FP16 matmul and report WMMA/fused stats.")
  parser.add_argument("n", nargs="+", type=int, help="matrix size(s), e.g. 32 64 128")
  parser.add_argument("--seed", type=int, default=0, help="base random seed (default: 0)")
  args = parser.parse_args()

  for i, n in enumerate(args.n):
    if n <= 0: raise ValueError(f"n must be > 0, got {n}")
    run_case(n, args.seed + i)

if __name__ == "__main__":
  main()
