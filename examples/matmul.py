from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

N = getenv("N", 64)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)

dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None

if __name__ == "__main__":
  a = Tensor.uniform(M, K, low=-0.5, high=0.5, dtype=dtype_in).realize()
  b = Tensor.uniform(K, N, low=-0.5, high=0.5, dtype=dtype_in).realize()
  for i in range(CNT):
    c = a.matmul(b, dtype=acc_dtype).realize()
  print(f"{M}x{K} @ {K}x{N} -> {c.shape}, dtype_in={dtype_in}, acc_dtype={acc_dtype}")
