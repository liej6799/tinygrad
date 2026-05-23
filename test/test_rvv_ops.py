"""
Mini test_ops.py for the RVV backend. No numpy/torch — uses pure Python lists for golden reference.
Covers what the RVV framework path currently supports:
  - elementwise f32 ops via fully-unrolled (N<=8) kernels
  - elementwise f32 ops via loop kernels (N>=8)
  - scalar broadcast constants (whole numbers, fractions, negatives)
  - nested expressions (chained ALUs)

Run on the target board:
  DEV=CPU:RVV python3 -u test/test_rvv_ops.py
"""
import math, sys, time, unittest
from tinygrad import Tensor, Device

ATOL = 1e-4
RTOL = 1e-4

def golden_unary(xs, f): return [f(x) for x in xs]
def golden_binary(xs, ys, f): return [f(x, y) for x, y in zip(xs, ys)]
def golden_scalar(xs, s, f): return [f(x, s) for x in xs]

def allclose(got, exp, atol=ATOL, rtol=RTOL):
  if len(got) != len(exp): return False, f"length mismatch: {len(got)} vs {len(exp)}"
  for i, (g, e) in enumerate(zip(got, exp)):
    if math.isnan(g) and math.isnan(e): continue
    diff = abs(g - e)
    tol = atol + rtol * abs(e)
    if diff > tol:
      return False, f"[{i}] got={g} exp={e} diff={diff} tol={tol}"
  return True, ""

def t(vals): return Tensor(list(vals)).realize()

class TestRVVOps(unittest.TestCase):
  SIZES = [4, 8, 16, 32, 64, 128]

  # ---- elementwise binary ----
  def test_add(self):
    for N in self.SIZES:
      a_vals = [float(i+1) for i in range(N)]
      b_vals = [float(i*2-3) for i in range(N)]
      got = (t(a_vals) + t(b_vals)).tolist()
      ok, msg = allclose(got, golden_binary(a_vals, b_vals, lambda x,y: x+y))
      self.assertTrue(ok, f"N={N}: {msg}")

  def test_mul(self):
    for N in self.SIZES:
      a_vals = [float(i+1)*0.5 for i in range(N)]
      b_vals = [float(i)-3.0 for i in range(N)]
      got = (t(a_vals) * t(b_vals)).tolist()
      ok, msg = allclose(got, golden_binary(a_vals, b_vals, lambda x,y: x*y))
      self.assertTrue(ok, f"N={N}: {msg}")

  def test_sub(self):
    for N in self.SIZES:
      a_vals = [float(i+1) for i in range(N)]
      b_vals = [float(i*3) for i in range(N)]
      got = (t(a_vals) - t(b_vals)).tolist()
      ok, msg = allclose(got, golden_binary(a_vals, b_vals, lambda x,y: x-y))
      self.assertTrue(ok, f"N={N}: {msg}")

  # ---- scalar broadcast ----
  def test_add_scalar(self):
    for N in self.SIZES:
      for s in [0.0, 1.0, 0.5, 0.1, -2.5, 3.14159]:
        a_vals = [float(i+1) for i in range(N)]
        got = (t(a_vals) + s).tolist()
        ok, msg = allclose(got, golden_scalar(a_vals, s, lambda x,y: x+y))
        self.assertTrue(ok, f"N={N} s={s}: {msg}")

  def test_mul_scalar(self):
    for N in self.SIZES:
      for s in [0.0, 1.0, 2.0, 0.5, 0.1, -1.0, 3.14159, -2.5]:
        a_vals = [float(i+1) for i in range(N)]
        got = (t(a_vals) * s).tolist()
        ok, msg = allclose(got, golden_scalar(a_vals, s, lambda x,y: x*y))
        self.assertTrue(ok, f"N={N} s={s}: {msg}")

  # ---- chained / nested expressions ----
  def test_chain_a_plus_b_plus_a(self):
    for N in self.SIZES:
      a_vals = [float(i+1) for i in range(N)]
      b_vals = [float(i*2) for i in range(N)]
      a, b = t(a_vals), t(b_vals)
      got = (a+b+a).tolist()
      exp = [x+y+x for x,y in zip(a_vals, b_vals)]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"N={N}: {msg}")

  def test_chain_aplusb_times_a(self):
    for N in self.SIZES:
      a_vals = [float(i+1) for i in range(N)]
      b_vals = [float(i*2) for i in range(N)]
      a, b = t(a_vals), t(b_vals)
      got = ((a+b)*a).tolist()
      exp = [(x+y)*x for x,y in zip(a_vals, b_vals)]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"N={N}: {msg}")

  def test_chain_three_way(self):
    for N in self.SIZES:
      a_vals = [float(i+1) for i in range(N)]
      b_vals = [float(i*2) for i in range(N)]
      a, b = t(a_vals), t(b_vals)
      got = (((a+b)*a)+b).tolist()
      exp = [((x+y)*x)+y for x,y in zip(a_vals, b_vals)]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"N={N}: {msg}")

  def test_mixed_const_and_tensor(self):
    for N in self.SIZES:
      a_vals = [float(i+1)*0.5 for i in range(N)]
      b_vals = [float(i*2)+1.0 for i in range(N)]
      a, b = t(a_vals), t(b_vals)
      got = (a*0.1 + b*0.2).tolist()
      exp = [x*0.1 + y*0.2 for x,y in zip(a_vals, b_vals)]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"N={N}: {msg}")

  # ---- edge cases ----
  def test_zero_const(self):
    a = t([1.0, 2.0, 3.0, 4.0])
    self.assertEqual((a * 0.0).tolist(), [0.0, 0.0, 0.0, 0.0])
    self.assertEqual((a + 0.0).tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_self_arithmetic(self):
    for N in [4, 32]:
      a_vals = [float(i+1) for i in range(N)]
      a = t(a_vals)
      got = (a + a).tolist()
      exp = [x + x for x in a_vals]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"a+a N={N}: {msg}")
      got = (a * a).tolist()
      exp = [x * x for x in a_vals]
      ok, msg = allclose(got, exp)
      self.assertTrue(ok, f"a*a N={N}: {msg}")

if __name__ == "__main__":
  print(f"device: {Device['CPU'].__class__.__name__}, renderer: {type(Device['CPU'].renderer).__name__}")
  print()
  unittest.main(verbosity=2)
