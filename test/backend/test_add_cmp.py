import unittest
import numpy as np
from tinygrad import Tensor, Device, Context
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_rk import compact_loop_to_c

N = getenv("N", 4)

# a non-extractable elementwise kernel (a compare/cast isn't a fusable EW binop) is rendered by the RKRenderer as ONE
# C nested loop straight from the compact uops (compact_loop_to_c) -- no unroll, no per-element scalars:
#   (a+b) > c          -> a single `for` loop with the add and the compare
#   d + ((a+b) > c)    -> the same, one loop
# the loop then runs on the RK runtime (clang), so DEV=RK computes a real result.
class TestAddCmp(unittest.TestCase):
  def _csrc(self, sink):
    with Context(SPEC=0):
      prg = to_program(sink, Device["RK"].renderer)
    uops = list(next(u for u in prg.toposort() if u.op is Ops.LINEAR).src)
    cl = compact_loop_to_c(uops)
    self.assertIsNotNone(cl, "compact_loop_to_c should render the compact loop")
    return "\n".join(cl[0])

  def test_add_cmp_loop(self):
    a, b = Tensor(np.arange(N, dtype=np.float32)), Tensor(np.full(N, 10, dtype=np.float32))
    sink = [s for s in ((a + b) > 12.0).schedule_linear().toposort() if s.op is Ops.SINK][0]
    src = self._csrc(sink)
    self.assertIn("for(int", src)        # it's a structured loop, not N unrolled scalar statements
    self.assertNotIn("acc", src)         # pure elementwise -> no reduce accumulator

  def test_multilevel_loop(self):
    a, b, d = (Tensor(np.full(N, x, dtype=np.float32)) for x in (1, 10, 100))
    sink = [s for s in (d + ((a + b) > 12.0)).schedule_linear().toposort() if s.op is Ops.SINK][0]
    self.assertIn("for(int", self._csrc(sink))

  def test_runs_on_rk(self):
    # the compact loop executes on the RK runtime -> real result
    C = 12.0; a = np.arange(N, dtype=np.float32) % 100; b = np.full(N, 10, dtype=np.float32)
    z = ((Tensor(a, device="RK") + Tensor(b, device="RK")) > C).numpy()
    np.testing.assert_array_equal(z, (a + b) > C)
    d = (np.arange(N, dtype=np.float32) * 3)
    w = (Tensor(d, device="RK") + ((Tensor(a, device="RK") + Tensor(b, device="RK")) > C)).numpy()
    np.testing.assert_allclose(w, d + ((a + b) > C), atol=1e-5)

if __name__ == "__main__":
  unittest.main()
