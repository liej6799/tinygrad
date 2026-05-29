# Example: add three values  z = a + b + c  on the Rockchip NPU backend.
# Run with:
#   DEV=ROCKCHIP DEBUG=6 N=16 python3 test/backend/test_add3_rockchip.py
import os
# default this example to the Rockchip backend unless the caller overrode DEV
os.environ.setdefault("DEV", "ROCKCHIP")

import unittest
from tinygrad import Tensor

class TestAdd3Rockchip(unittest.TestCase):
  def test_tensor_add3(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = Tensor([7, 8, 9])
    z = a + b + c
    z.realize()
    print(z.numpy())
    self.assertEqual(z.numpy().tolist(), [12, 15, 18])

if __name__ == "__main__":
  unittest.main()
