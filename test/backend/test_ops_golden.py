#!/usr/bin/env python3
import argparse, math, pickle, random
from pathlib import Path
from typing import Any

CASES = [
  ("add", "add", [(8, 8), (8, 8)], False),
  ("mul_broadcast", "mul", [(4, 3, 5), (1, 3, 1)], False),
  ("div", "div", [(16,), (16,)], False),
  ("matmul", "matmul", [(8, 6), (6, 5)], False),
  ("relu", "relu", [(9, 7)], False),
  ("exp", "exp", [(7, 5)], False),
  ("sum_axis", "sum_axis", [(4, 5, 6)], False),
  ("mean_axis", "mean_axis", [(4, 5, 6)], False),
  ("max_axis", "max_axis", [(4, 5, 6)], True),
  ("softmax", "softmax", [(6, 10)], False),
  ("conv2d", "conv2d", [(2, 3, 8, 8), (4, 3, 3, 3)], False),
  ("avg_pool2d", "avg_pool2d", [(2, 3, 8, 8)], False),
]

def make_data(shape: tuple[int, ...], rng: random.Random) -> Any:
  if len(shape) == 0: return rng.uniform(-2.0, 2.0)
  return [make_data(shape[1:], rng) for _ in range(shape[0])]

def op_tiny(name, xs):
  from tinygrad import Tensor
  if name == "add": return xs[0] + xs[1]
  if name == "mul": return xs[0] * xs[1]
  if name == "div": return xs[0] / (xs[1].abs() + 0.25)
  if name == "matmul": return xs[0].matmul(xs[1])
  if name == "relu": return xs[0].relu()
  if name == "exp": return (xs[0] * 0.25).exp()
  if name == "sum_axis": return xs[0].sum(axis=1)
  if name == "mean_axis": return xs[0].mean(axis=2)
  if name == "max_axis": return xs[0].max(axis=1)
  if name == "softmax": return xs[0].softmax(axis=1)
  if name == "conv2d": return xs[0].conv2d(xs[1], padding=1).relu()
  if name == "avg_pool2d": return xs[0].avg_pool2d(kernel_size=(2, 2))
  raise ValueError(name)

def op_torch(name, xs):
  import torch
  if name == "add": return xs[0] + xs[1]
  if name == "mul": return xs[0] * xs[1]
  if name == "div": return xs[0] / (xs[1].abs() + 0.25)
  if name == "matmul": return xs[0].matmul(xs[1])
  if name == "relu": return xs[0].relu()
  if name == "exp": return (xs[0] * 0.25).exp()
  if name == "sum_axis": return xs[0].sum(dim=1)
  if name == "mean_axis": return xs[0].mean(dim=2)
  if name == "max_axis": return xs[0].max(dim=1).values
  if name == "softmax": return xs[0].softmax(dim=1)
  if name == "conv2d": return torch.nn.functional.conv2d(xs[0], xs[1], padding=1).relu()
  if name == "avg_pool2d": return torch.nn.functional.avg_pool2d(xs[0], kernel_size=(2, 2))
  raise ValueError(name)

def flatten(x):
  if isinstance(x, (list, tuple)):
    for y in x: yield from flatten(y)
  else:
    yield x

def shape_of(x):
  return [len(x), *shape_of(x[0])] if isinstance(x, list) else []

def assert_close(name, got, exp, atol=1e-4, rtol=1e-3):
  if shape_of(got) != shape_of(exp): raise AssertionError(f"{name}: shape {shape_of(got)} != {shape_of(exp)}")
  for i, (a, b) in enumerate(zip(flatten(got), flatten(exp))):
    if math.isnan(a) and math.isnan(b): continue
    if abs(a-b) > atol + rtol*abs(b): raise AssertionError(f"{name}: mismatch at flat index {i}: got {a}, expected {b}")

def generate(path: Path):
  import torch
  fixtures = []
  for idx, (case_name, op_name, shapes, forward_only) in enumerate(CASES):
    rng = random.Random(1337 + idx)
    inputs = [make_data(shape, rng) for shape in shapes]
    ts = [torch.tensor(x, dtype=torch.float32, requires_grad=not forward_only) for x in inputs]
    out = op_torch(op_name, ts)
    grads = None
    if not forward_only:
      grads = [g.detach().cpu().tolist() for g in torch.autograd.grad(out.sum(), ts)]
    fixtures.append({"case": case_name, "op": op_name, "inputs": inputs, "expected": out.detach().cpu().tolist(), "grads": grads})
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f: pickle.dump(fixtures, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(f"wrote {len(fixtures)} fixtures to {path}")

def replay(path: Path, forward_only=False):
  from tinygrad import Tensor
  with path.open("rb") as f: fixtures = pickle.load(f)
  for fx in fixtures:
    xs = [Tensor(x) for x in fx["inputs"]]
    got = op_tiny(fx["op"], xs).realize().tolist()
    assert_close(f"{fx['case']} forward", got, fx["expected"])
    if fx["grads"] is not None and not forward_only:
      grads = op_tiny(fx["op"], xs).sum().gradient(*xs)
      for i, (g, exp) in enumerate(zip(grads, fx["grads"])):
        assert_close(f"{fx['case']} grad {i}", g.realize().tolist(), exp)
    print(f"ok {fx['case']}")
  print(f"OK: replayed {len(fixtures)} golden op cases")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["generate", "replay"])
  parser.add_argument("--path", default="/tmp/tinygrad_ops_golden.pkl")
  parser.add_argument("--forward-only", action="store_true")
  args = parser.parse_args()
  generate(Path(args.path)) if args.mode == "generate" else replay(Path(args.path), args.forward_only)
