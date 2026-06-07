import unittest
import numpy as np
from tinygrad import Tensor

class TestAdd(unittest.TestCase):
  def test_tensor_add(self):
    shape = (5, 5)
    a_np = (np.arange(np.prod(shape), dtype=np.float16).reshape(shape) % 100).astype(np.float16)
    b_np = np.full(shape, 10, dtype=np.float16)
    a, b = Tensor(a_np), Tensor(b_np)
    z = (a + b).realize()
    np.testing.assert_array_equal(z.numpy(), a_np + b_np)

if __name__ == "__main__":
  unittest.main()
