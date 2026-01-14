import pickle, time, math, unittest, functools, platform, warnings
import numpy as np
from typing import List, Callable
import torch
from tinygrad.helpers import getenv, IMAGE, DEBUG, CI, Context, TRANSCENDENTAL, AMD_LLVM
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.ops_rockchip import RockchipProgram
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported

if getenv("TINY_BACKEND"):
  import tinygrad.frontend.torch # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

if CI:
  warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 1)
# Only dump tensors when explicitly requested or when debug is very high.
PRINT_TENSORS = getenv("PRINT_TENSORS", 1 if DEBUG >= 3 else 0)

def slow_test(test_func):
  return unittest.skipIf(getenv("SKIP_SLOW_TEST"), "Skipping slow test")(test_func)

def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if getenv("ROCKCHIP", 0) and dtypes.default_float == dtypes.float16:
    atol = max(atol, 2e-3)
    rtol = max(rtol, 8e-3)
  atol = getenv("ROCKCHIP_ATOL", atol)
  rtol = getenv("ROCKCHIP_RTOL", rtol)
  grad_atol = getenv("ROCKCHIP_GRAD_ATOL", grad_atol)
  grad_rtol = getenv("ROCKCHIP_GRAD_RTOL", grad_rtol)
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
  if len(ts) >= 2:
    print("generated torch input", ts[0].detach().cpu().numpy())
    print("generated torch weight", ts[1].detach().cpu().numpy())

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  if mt:=getenv("MOVE_TENSOR", ""):
    for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().cpu().numpy())
  compare("forward pass", ret.numpy(), out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(torch_fxn(*ts).sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI:
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  if PRINT_TENSORS:
    for i, t in enumerate(ts):
      print(f"prepared torch input [{i}] shape {t.shape} dtype {t.dtype}")
      print(t.detach().cpu().numpy())
  return ts, tst

class TestOps(unittest.TestCase):


  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn=None, expected=None, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if getenv("MOCKGPU") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    if tinygrad_fxn is None:
      tinygrad_fxn = torch_fxn
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  # @unittest.skipIf((IMAGE>0), "no conv1d on images")
  # def test_conv1d(self):
  #   for bs in [1,8]:
  #     for cin in [1,3]:
  #       for H in [1,2,5]:
  #         for groups in [1,3] if cin == 3 and H == 5 else [1]:
  #           with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H):
  #             # print("testing conv1d", (bs,cin,11), (6,cin//groups,H))
  #             if ((bs,cin,11), (6,cin//groups,H)) == ((1, 1, 11), (6, 1, 2)):
  #               helper_test_op([(bs,cin,11), (6,cin//groups,H)],
  #                 lambda x,w: torch.nn.functional.conv1d(x,w,groups=groups),
  #                 lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  # def _test_conv2d(self, bs=1, cin=1, cout=6):
  #   for H in [2,3]:
  #     for W in [1,3,5]:
  #       for groups in [1,3] if cin == 3 and cout == 6 and H == 3 and W == 3 else [1]:
  #         with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
  #           helper_test_op([(bs,cin,5,7), (cout,cin//groups,H,W)],
  #             lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
  #             lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)
  # def test_conv2d(self): self._test_conv2d(bs=1, cin=3)

  # def test_gemm_fp16(self):
  #   helper_test_op([(8,8), (8,8)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)
  #   helper_test_op([(9,9), (9,9)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)
  #   helper_test_op([(32,32), (32,32)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)
  #   helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)
  #   # helper_test_op([(256,256), (256,256)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3)

  # def test_9_gemm(self):
  #   helper_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y)


  # def test_cmp_lt_simple(self):
  #   a = Tensor([0.0, 1.0, 2.0])
  #   b = Tensor([2.0, 1.0, 0.0])
  #   np.testing.assert_equal((a < b).realize().numpy(), np.array([True, False, False], dtype=np.bool_))
  #   np.testing.assert_equal((Tensor([-math.inf]) < Tensor([math.inf])).realize().numpy(), np.array([True], dtype=np.bool_))
  #   np.testing.assert_equal((Tensor([math.inf]) < Tensor([math.inf])).realize().numpy(), np.array([False], dtype=np.bool_))
  #   np.testing.assert_equal((Tensor([math.nan]) < Tensor([0.0])).realize().numpy(), np.array([False], dtype=np.bool_))
  #   np.testing.assert_equal((Tensor([0.0]) < Tensor([math.nan])).realize().numpy(), np.array([False], dtype=np.bool_))

#   def test_add(self):
#     helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
#     helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
#     helper_test_op([(), ()], lambda x,y: x+y)

  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y)
    helper_test_op([(), ()], lambda x,y: x*y)

  # def _test_cmp(self, fxn, reverse=True):
  #   # test different dtypes
  #   helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
  #   helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0,1,2], [2,1,0]])
  #   helper_test_op(None, fxn, fxn, forward_only=True, vals=[[True, True, False], [False,True,False]])
  #   # test broadcasting
  #   for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
  #     helper_test_op(shps, fxn, fxn, forward_only=True)
  #   # test cmp with const
  #   helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
  #   if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
  #   # test special floats  # TODO: fix nan
  #   specials = [0.0, 1.0, -1.0, math.inf, -math.inf]#, math.nan]
  #   for s0 in specials:
  #     for s1 in specials:
  #       helper_test_op(None, fxn, fxn, forward_only=True, vals=[[s0], [s1]])
  # def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)



  #### WIP SELU
  # def test_div(self):
  #   helper_test_op([(2,2), (2,2)], lambda x,y: x/y, Tensor.div)
  #   helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
  #   # helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
  #   # helper_test_op([(), ()], lambda x,y: x/y)

  # def test_round(self):
  #   helper_test_op([()], lambda x: x.round(), forward_only=True)
  #   helper_test_op([(45,35)], lambda x: x.round(), forward_only=True)
  #   helper_test_op(None, lambda x: x.round(), vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]], forward_only=True)
  #   helper_test_op(None, lambda x: x.round(), vals=[[2.5, -1.5]], forward_only=True)

  # def test_roundoff(self):
  #   def roundoff_ref(val: float) -> float:
  #     base = math.floor(val)
  #     frac = val - base
  #     base_i = int(base)
  #     if frac > 0.5 or (frac == 0.5 and (base_i & 1)): return base + 1.0
  #     return base

  #   dev = Device["ROCKCHIP"]
  #   prg = RockchipProgram(dev, "roundoff", pickle.dumps([]))
  #   vals = np.array([0.5, 1.4, 1.5, 1.6, 2.5, 3.5, 4.4, 4.5, 4.6, 5.5, 6.49, 6.5, 6.51, 7.5, 8.5, 9.5], dtype=np.float16)
  #   out = prg.roundoff(vals.tolist(), rows=4, cols=4)
  #   expected = np.array([roundoff_ref(float(x)) for x in vals], dtype=np.float16)
  #   np.testing.assert_allclose(np.array(out, dtype=np.float16), expected, atol=1e-3)

  # def test_where(self):
  #   rng = np.random.default_rng(0)
  #   for rows, cols in ((1, 1), (2, 2), (8, 8)):
  #     with self.subTest(rows=rows, cols=cols):
  #       a = rng.uniform(-2.0, 2.0, size=(rows, cols)).astype(np.float16)
  #       b = rng.uniform(-2.0, 2.0, size=(rows, cols)).astype(np.float16)
  #       mask = (np.arange(rows * cols) % 2).reshape(rows, cols).astype(np.bool_)
  #       ta = Tensor(a, dtype=dtypes.float16)
  #       tb = Tensor(b, dtype=dtypes.float16)
  #       tmask = Tensor(mask, dtype=dtypes.bool)
  #       out = tmask.where(ta, tb).realize().numpy()
  #       expected = np.where(mask, a, b)
  #       np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)

  # def test_idiv_shift_rewrite_negative(self):
  #   a = Tensor(-5).idiv(2).item()
  #   b = Tensor(-5).contiguous().idiv(2).item()
  #   self.assertEqual(a, b)
  #   self.assertEqual(Tensor(-1).contiguous().idiv(4).item(), 0)  # NOTE this is trunc-div behaviour

  # def test_cmpne_specials(self):
  #   a_vals = np.array([[0.0, -1.0, math.inf], [math.nan, 2.0, -0.0]], dtype=np.float16)
  #   b_vals = np.array([[0.0, 1.0, math.inf], [math.nan, 0.0, 0.0]], dtype=np.float16)
  #   out = (Tensor(a_vals, dtype=dtypes.float16) != Tensor(b_vals, dtype=dtypes.float16)).realize().numpy()
  #   expected = np.not_equal(a_vals, b_vals)
  #   np.testing.assert_equal(out, expected)

  # def test_relu(self):
  #   helper_test_op([(64,64)], lambda x: x.relu())
  #   helper_test_op([()], lambda x: x.relu())

  # def test_silu(self):
  #   helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.silu, atol=1e-3, rtol=1e-3)
  #   helper_test_op([()], torch.nn.functional.silu, Tensor.silu, atol=1e-3, rtol=1e-3)

  # def test_sigmoid(self):
  #   helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, atol=5e-3, rtol=1e-3)
  #   helper_test_op([()], torch.sigmoid, Tensor.sigmoid, atol=5e-3, rtol=1e-3)

  # def test_abs(self):
  #   helper_test_op([(45,65)], torch.abs, Tensor.abs, atol=1e-3, rtol=1e-3)
  #   helper_test_op([()], torch.abs, Tensor.abs, atol=1e-3, rtol=1e-3)

  # def test_selu(self):
  #   helper_test_op([(2,2)], torch.nn.functional.selu, Tensor.selu, atol=2e-3)
  #   helper_test_op([(45,65)], torch.nn.functional.selu, Tensor.selu, atol=2e-3)
  #   helper_test_op([()], torch.nn.functional.selu, Tensor.selu, atol=2e-3)

  # def test_maximum(self):
  #   helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
  #   helper_test_op([(), ()], torch.maximum, Tensor.maximum)

  # def test_maximum(self):
  #   helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
  #   helper_test_op([(), ()], torch.maximum, Tensor.maximum)

  # def test_minimum(self):
  #   helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
  #   helper_test_op([(), ()], torch.minimum, Tensor.minimum)

  # def test_celu(self):
  #   atol, rtol = 1e-6, 1e-3
  #   if dtypes.default_float == dtypes.float16:
  #     # Rockchip executes exp in float16; allow ~1 ULP differences.
  #     atol = 1e-3
  #   for val in range(1, 5):
  #     helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val), atol=atol, rtol=rtol)
  #     helper_test_op([()], lambda x: torch.nn.functional.celu(x,val), lambda x: x.celu(val), atol=atol, rtol=rtol)

  # def test_swish(self):
  #   helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.swish)
  #   helper_test_op([()], torch.nn.functional.silu, Tensor.swish)

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
