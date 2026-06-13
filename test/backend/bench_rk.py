import time, os
import numpy as np
from tinygrad import Tensor

def timed(fn, *a):
  t0 = time.perf_counter(); r = fn(*a); dt = time.perf_counter()-t0; return r, dt

def bench_matmul(M, K, N):
  a = Tensor(np.random.randn(M, K).astype(np.float32))
  b = Tensor(np.random.randn(K, N).astype(np.float32))
  t0 = time.perf_counter()
  z = (a @ b).realize()
  dt = time.perf_counter()-t0
  _ = z.numpy()
  print(f"matmul {M}x{K}x{N:<5} realize {dt*1e3:8.2f} ms")

def bench_conv(bs, cin, cout, h, w, k):
  from tinygrad.nn import Conv2d
  x = Tensor(np.random.randn(bs, cin, h, w).astype(np.float32))
  conv = Conv2d(cin, cout, k, bias=False)
  t0 = time.perf_counter()
  z = conv(x).realize()
  dt = time.perf_counter()-t0
  _ = z.numpy()
  print(f"conv bs{bs} {cin}->{cout} {h}x{w} k{k:<3} realize {dt*1e3:8.2f} ms")

if __name__ == "__main__":
  what = os.environ.get("WHAT", "matmul")
  if what == "matmul":
    for s in [4, 8, 16, 32, 64]:
      bench_matmul(s, s, s)
  elif what == "conv":
    for h in [8, 16, 32]:
      bench_conv(1, 3, 4, h, h, 3)
  elif what == "add":
    for s in [16, 64, 256, 1024]:
      a = Tensor(np.random.randn(s, s).astype(np.float32))
      b = Tensor(np.random.randn(s, s).astype(np.float32))
      _, dt = timed(lambda: (a+b).realize())
      print(f"add {s}x{s:<5} realize {dt*1e3:8.2f} ms")
