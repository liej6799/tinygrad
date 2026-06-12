import unittest, base64, pickle, importlib.util
import numpy as np
from tinygrad import Tensor, Device, Context
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.runtime.ops_rk import uops_to_onnx_staged

# the RKRenderer unrolls the uops and groups them into staged ONNX graphs by (dependency level, op type):
#   (a+b) > c          -> s1_add, s2_less
#   d + ((a+b) > c)    -> s1_add, s2_less, s3_cast, s4_add   (Add at two levels stays a runnable DAG)
class TestAddCmp(unittest.TestCase):
  def _render(self, sink):
    with Context(SPEC=0):
      prg = to_program(sink, Device["RK"].renderer)
    uops = next(u for u in prg.toposort() if u.op is Ops.LINEAR).src
    stages = pickle.loads(base64.b64decode(next(u for u in prg.toposort() if u.op is Ops.SOURCE).arg))
    return list(uops), [n for n, _ in stages]

  def test_add_cmp_stages(self):
    a, b = Tensor(np.arange(4, dtype=np.float32)), Tensor(np.full(4, 10, dtype=np.float32))
    sink = [s for s in ((a + b) > 12.0).schedule_linear().toposort() if s.op is Ops.SINK][0]
    self.assertEqual(self._render(sink)[1], ["s1_add", "s2_less"])

  def test_multilevel_stages(self):
    a, b, d = (Tensor(np.full(2, x, dtype=np.float32)) for x in (1, 10, 100))
    sink = [s for s in (d + ((a + b) > 12.0)).schedule_linear().toposort() if s.op is Ops.SINK][0]
    self.assertEqual(self._render(sink)[1], ["s1_add", "s2_less", "s3_cast", "s4_add"])

  @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "needs onnxruntime")
  def test_chain_reproduces_kernel(self):
    import onnxruntime as ort
    C = 12.0; a = np.arange(4, dtype=np.float32) % 100; b = np.full(4, 10, dtype=np.float32)
    sink = [s for s in ((Tensor(a) + Tensor(b)) > C).schedule_linear().toposort() if s.op is Ops.SINK][0]
    with Context(SPEC=0):
      prg = to_program(sink, Device["RK"].renderer)
    uops = next(u for u in prg.toposort() if u.op is Ops.LINEAR).src
    models, loads, consts, outs = uops_to_onnx_staged(list(uops))
    arrs = {1: a, 2: b}                                   # param 0=out, 1=a, 2=b
    env = {nm: np.array(arrs[p][o], dtype=np.float32) for nm,(p,o) in loads.items()}
    env.update({nm: np.array(v, dtype=np.float32) for nm,v in consts.items()})
    for _, m in models:                                  # run the stages in order, chaining outputs
      feed = {i.name: env[i.name] for i in m.graph.input}
      for o, v in zip([o.name for o in m.graph.output], ort.InferenceSession(m.SerializeToString()).run(None, feed)): env[o] = v
    res = np.array([bool(env[outs[o]]) for o in sorted(outs)])
    np.testing.assert_array_equal(res, (a + b) > C)

if __name__ == "__main__":
  unittest.main()
