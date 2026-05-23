"""
Side-by-side benchmark: ClangJIT (DEV=CPU) vs native RVV (DEV=CPU:RVV).
Runs each kernel for many iterations, reports kernel-only wall time + GFLOPS.

Two timing modes:
  - "python":  Python-loop walltime (includes tinygrad orchestration overhead per realize)
  - "kernel":  DEBUG=2 kernel-tm output, parsed
The Python-loop time is what a user actually sees; the kernel-time is what the codegen
delivers. Both are interesting.

Usage on the target:
  python3 test/bench_rvv_vs_clang.py  [N=...] [ITERS=...]
"""
import os, sys, time, subprocess, re, json

# Workloads as (label, code-snippet using `a`, `b`, `s`):
WORKLOADS = [
  ("a+b",       "a + b"),
  ("a*b",       "a * b"),
  ("a-b",       "a - b"),
  ("a*2.0",     "a * 2.0"),
  ("a*0.1",     "a * 0.1"),
  ("a+b+a",     "a + b + a"),
  ("(a+b)*a",   "(a + b) * a"),
  ("a*0.1+b*0.2", "a * 0.1 + b * 0.2"),
]
SIZES = [32, 64, 128, 256, 512, 1024]

PY = """
import time, sys
from tinygrad import Tensor

N = int(sys.argv[1]); iters = int(sys.argv[2]); expr = sys.argv[3]
a = Tensor([float(i+1) for i in range(N)]).realize()
b = Tensor([float(i*2-3) for i in range(N)]).realize()
# warmup
_ = eval(expr).realize()
t0 = time.perf_counter()
for _ in range(iters): _ = eval(expr).realize()
t1 = time.perf_counter()
per = (t1-t0)/iters
print(f"{per*1e6:.2f}")
"""

def run(env_extra, args):
  env = os.environ.copy()
  env.update(env_extra)
  res = subprocess.run([sys.executable, "-u", "-c", PY] + list(map(str, args)), env=env, capture_output=True, text=True, timeout=120)
  if res.returncode != 0:
    return None, res.stderr[-500:]
  out = res.stdout.strip().splitlines()[-1]
  try: return float(out), None
  except Exception: return None, f"bad output: {res.stdout!r}"

def iters_for(N):
  # try to hit ~50ms per measurement
  per_iter_estimate_us = max(50, N * 0.1)
  return max(5, int(50_000 / per_iter_estimate_us))

def main():
  N_FILTER = int(os.environ.get("N", 0)) or None
  print(f"{'workload':<14}  {'N':>5}  {'iters':>5}  {'clang_us':>10}  {'rvv_us':>10}  {'speedup':>8}")
  print("-" * 70)
  for N in SIZES:
    if N_FILTER and N != N_FILTER: continue
    iters = iters_for(N)
    for label, expr in WORKLOADS:
      clang, e1 = run({}, [N, iters, expr])
      rvv, e2 = run({"DEV": "CPU:RVV"}, [N, iters, expr])
      if clang is None or rvv is None:
        c = f"{clang:.1f}" if clang is not None else "ERR"
        r = f"{rvv:.1f}" if rvv is not None else "ERR"
        print(f"{label:<14}  {N:>5}  {iters:>5}  {c:>10}  {r:>10}  {'--':>8}")
        if e1: print("  clang err:", e1.replace("\n", " | ")[-150:])
        if e2: print("  rvv err:  ", e2.replace("\n", " | ")[-150:])
        continue
      sp = clang / rvv if rvv > 0 else 0
      arrow = "↑" if sp > 1.05 else ("↓" if sp < 0.95 else "·")
      print(f"{label:<14}  {N:>5}  {iters:>5}  {clang:>10.2f}  {rvv:>10.2f}  {sp:>6.2f}x {arrow}")

if __name__ == "__main__":
  main()
