import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv

# two N x N matmuls separated by an elementwise (ALU) op:  z = ((a @ b) + ADDC) @ d
N = getenv("N", 8)
ADDC = getenv("ADDC", 7.0)

if __name__ == "__main__":
  rng = np.random.default_rng(0)
  a = Tensor((rng.random((N, N), dtype=np.float32) - 0.5)).realize()
  b = Tensor((rng.random((N, N), dtype=np.float32) - 0.5)).realize()
  d = Tensor((rng.random((N, N), dtype=np.float32) - 0.5)).realize()

  x = a.matmul(b)        # matmul #1
  y = x + ADDC           # ALU op separating the two matmuls
  z = y.matmul(d)        # matmul #2
  z.realize()

  ref = ((a.numpy() @ b.numpy()) + ADDC) @ d.numpy()
  np.testing.assert_allclose(z.numpy(), ref, rtol=1e-4, atol=1e-4)
  print("double matmul OK")
