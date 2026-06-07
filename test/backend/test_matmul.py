import unittest
import numpy as np
from tinygrad import Tensor

class TestMatmul(unittest.TestCase):
  def test_tensor_matmul(self):
    n = 12
    a_np = (np.arange(n * n, dtype=np.float32).reshape(n, n) % 7).astype(np.float32)
    b_np = (np.arange(n * n, dtype=np.float32).reshape(n, n) % 5).astype(np.float32)
    a, b = Tensor(a_np), Tensor(b_np)
    z = a.matmul(b).realize()
    np.testing.assert_allclose(z.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
  unittest.main()
