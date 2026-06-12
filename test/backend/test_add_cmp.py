import unittest, os
os.environ["RK_UNROLL"] = "1"        # the chunk-structure tests need the per-element unrolled view
import numpy as np
from tinygrad import Tensor, Device, Context
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_rk import uops_to_chunks

N = getenv("N", 4)

# the RKRenderer unrolls the uops and chunks them by (dependency level, op category): the arithmetic ALU ops
# (ADD/SUB/MUL/DIV) share one "alu" chunk, every other op gets its own:
#   (a+b) > c          -> s1_alu, s2_cmplt
#   d + ((a+b) > c)    -> s1_alu, s2_cmplt, s3_cast, s4_alu   (add at two levels stays a runnable DAG)
# the unrolled uops are then handed to the uop runtime, so DEV=RK computes a real result.
class TestAddCmp(unittest.TestCase):
  def _chunks(self, sink):
    with Context(SPEC=0):
      prg = to_program(sink, Device["RK"].renderer)
    uops = list(next(u for u in prg.toposort() if u.op is Ops.LINEAR).src)
    return [n for n, _ in uops_to_chunks(uops)[0]]

  def test_add_cmp_chunks(self):
    a, b = Tensor(np.arange(N, dtype=np.float32)), Tensor(np.full(N, 10, dtype=np.float32))
    sink = [s for s in ((a + b) > 12.0).schedule_linear().toposort() if s.op is Ops.SINK][0]
    self.assertEqual(self._chunks(sink), ["s1_alu", "s2_cmplt"])

  def test_multilevel_chunks(self):
    a, b, d = (Tensor(np.full(N, x, dtype=np.float32)) for x in (1, 10, 100))
    sink = [s for s in (d + ((a + b) > 12.0)).schedule_linear().toposort() if s.op is Ops.SINK][0]
    self.assertEqual(self._chunks(sink), ["s1_alu", "s2_cmplt", "s3_cast", "s4_alu"])

  def test_runs_on_rk(self):
    # the unrolled uops execute on the RK runtime -> real result
    C = 12.0; a = np.arange(N, dtype=np.float32) % 100; b = np.full(N, 10, dtype=np.float32)
    z = ((Tensor(a, device="RK") + Tensor(b, device="RK")) > C).numpy()
    np.testing.assert_array_equal(z, (a + b) > C)
    d = (np.arange(N, dtype=np.float32) * 3)
    w = (Tensor(d, device="RK") + ((Tensor(a, device="RK") + Tensor(b, device="RK")) > C)).numpy()
    np.testing.assert_allclose(w, d + ((a + b) > C), atol=1e-5)

if __name__ == "__main__":
  unittest.main()
