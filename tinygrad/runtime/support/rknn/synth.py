"""Synthesize a runnable .rknn from tinygrad UOps (toolkit-free).

This is the tinygrad-side port of rknn-decode's `helpers.rknn_synth`: it reads the
fully-lowered element-wise UOp graph that reaches `RKRenderer.render`, recovers the
element count N and the op, and emits the .rknn (FlatBuffer body + NPU
register-command stream + container) from scratch via the vendored builders
(`_rknn_flatbuf` + `_rc_template_gen`, which depend only on `flatbuffers`).

Scope: a single fp16/fp32 element-wise op `z = a OP b` (OP in Add/Sub/Mul/Div) over
N elements, in either the scalar full-unroll form
  STORE(INDEX(out,k), OP(LOAD(INDEX(a,k)), LOAD(INDEX(b,k))))
or tinygrad's vectorized UPCAST form
  STORE(CAST(INDEX(out)), STACK(OP(GEP(LOAD(CAST(INDEX(a)))), GEP(LOAD(...))), ...)).
Anything else (reductions/matmul, >2 inputs) raises -- the caller can fall back.
"""
from tinygrad.uop.ops import Ops, UOp
from . import _rknn_flatbuf as _fb


def _index_param(u:UOp):
  """(CAST of) INDEX(PARAM, idx) -> the PARAM. CAST = the upcast vec-ptr cast."""
  if u.op is Ops.CAST: u = u.src[0]
  if u.op is not Ops.INDEX or u.src[0].op is not Ops.PARAM:
    raise ValueError("expected (CAST of) INDEX(PARAM, ...)")
  return u.src[0]


def _is_neg(u:UOp):
  """MUL(x, CONST(-1)) or NEG(x) -> x, else None (tinygrad lowers a-b to a + (-b))."""
  if u.op is Ops.NEG: return u.src[0]
  if u.op is Ops.MUL and len(u.src) == 2:
    for i in (0, 1):
      if u.src[i].op is Ops.CONST and u.src[i].arg == -1: return u.src[1 - i]
  return None


def _operand_param(u:UOp):
  """ALU operand -> input PARAM. Scalar LOAD(INDEX(p)) or vectorized GEP(LOAD(...))."""
  if u.op is Ops.GEP: u = u.src[0]
  if u.op is not Ops.LOAD:
    raise ValueError("operand must be LOAD(INDEX(PARAM)) (scalar) or GEP(LOAD(...)) (vectorized)")
  return _index_param(u.src[0])


def _resolve_op(alu:UOp):
  """Map the STORE value op to (op_name, (operand_a, operand_b)), undoing tinygrad's
  lowering of Sub -> a + (-b) and Div -> a * (1/b) back to native NPU Sub/Div tiles."""
  if alu.op is Ops.ADD and len(alu.src) == 2:
    # a - b lowers to ADD(a, MUL(b, -1)) / ADD(MUL(a,-1), b)
    for i in (0, 1):
      if (other := _is_neg(alu.src[i])) is not None: return "Sub", (alu.src[1 - i], other)
    return "Add", (alu.src[0], alu.src[1])
  if alu.op is Ops.MUL and len(alu.src) == 2:
    # a / b lowers to MUL(a, RECIPROCAL(b))
    for i in (0, 1):
      if alu.src[i].op is Ops.RECIPROCAL: return "Div", (alu.src[1 - i], alu.src[i].src[0])
    return "Mul", (alu.src[0], alu.src[1])
  if alu.op is Ops.FDIV and len(alu.src) == 2: return "Div", (alu.src[0], alu.src[1])
  if alu.op is Ops.SUB and len(alu.src) == 2: return "Sub", (alu.src[0], alu.src[1])
  raise ValueError(f"unsupported element-wise op {alu.op}")


def analyze_elementwise(uops:list[UOp]):
  """Validate a 2-input element-wise op graph; return (N, op_name). Raises otherwise.

  Works for the scalar, fully-unrolled, and loop+UPCAST forms uniformly: N is the
  output PARAM's pointer size (`float.ptr(N)`); the op + the two distinct input
  PARAMs come from the STORE's value chain (peeling STACK/GEP, and undoing
  Sub->a+(-b) / Div->a*(1/b) lowering). No unrolling needed.
  """
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores: raise ValueError("no STORE in uops")
  out_params, in_params, ops = set(), set(), set()
  for st in stores:
    out = _index_param(st.src[0])
    out_params.add(out.arg)
    val = st.src[1]
    alu = val.src[0] if val.op is Ops.STACK else val       # peel STACK -> one lane's op
    op_name, operands = _resolve_op(alu)
    ops.add(op_name)
    for operand in operands: in_params.add(_operand_param(operand).arg)
  if len(ops) != 1: raise ValueError(f"all stores must use the same op, got {ops}")
  if len(out_params) != 1: raise ValueError(f"expected one output PARAM, got {out_params}")
  if len(in_params) != 2: raise ValueError(f"expected two input PARAMs, got {in_params}")
  # N = number of elements of the output buffer = the output PARAM's pointer size.
  out_param = next(u for u in uops if u.op is Ops.PARAM and u.arg == next(iter(out_params)))
  N = out_param.dtype.size
  if not isinstance(N, int) or N < 1: raise ValueError(f"could not recover element count (ptr size={N})")
  return N, next(iter(ops))


def uops_to_rknn(uops:list[UOp]) -> bytes:
  """Build a runnable .rknn (bytes) from a tinygrad element-wise op UOp graph."""
  N, op_name = analyze_elementwise(uops)
  body = _fb.build_body(N, 2, ops=[op_name])              # FlatBuffer + regcmd, from scratch
  return bytes(_fb.assemble_rknn(body, 1, N, 2))          # + 64-byte header + JSON trailer


# --------------------------------------------------------------------------- #
# matmul (reduction) -> K-step MULACC decomposition
# --------------------------------------------------------------------------- #
def _is_matmul(uops:list[UOp]) -> bool:
  """A reduction whose store value contains a*b products (LOAD(in1)*LOAD(in2))."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1: return False
  # find a MUL of two loads coming from two *different* input PARAMs
  muls = [u for u in uops if u.op is Ops.MUL and len(u.src) == 2]
  for m in muls:
    try:
      pa, pb = _operand_param(m.src[0]), _operand_param(m.src[1])
    except ValueError:
      continue
    if pa.op is Ops.PARAM and pb.op is Ops.PARAM and pa.arg != pb.arg and pa.arg != 0 and pb.arg != 0:
      return any(u.op is Ops.RANGE for u in uops) or any(u.op is Ops.MULACC for u in uops) \
        or len([x for x in muls]) > 1
  return False


def analyze_matmul(uops:list[UOp]):
  """Recover (M, K, N) for out[M,N] = a[M,K] @ b[K,N] from the PARAM pointer sizes.

  out=M*N, a=M*K, b=K*N  ->  K=sqrt(a*b/out), M=a/K, N=b/K. Raises if not a matmul."""
  if not _is_matmul(uops): raise ValueError("not a matmul reduction")
  params = {u.arg: u for u in uops if u.op is Ops.PARAM}
  if set(params) != {0, 1, 2}: raise ValueError(f"matmul expects PARAMs 0,1,2; got {sorted(params)}")
  so, sa, sb = (params[i].dtype.size for i in (0, 1, 2))
  if not all(isinstance(s, int) and s > 0 for s in (so, sa, sb)): raise ValueError("bad ptr sizes")
  import math
  K = round(math.sqrt((sa * sb) / so))
  if K <= 0 or sa % K or sb % K: raise ValueError(f"cannot factor (out={so},a={sa},b={sb})")
  M, N = sa // K, sb // K
  if M * N != so or M * K != sa or K * N != sb: raise ValueError(f"shape solve failed M={M} K={K} N={N}")
  return M, K, N


def matmul_step_rknn(MN:int) -> bytes:
  """One MULACC step model: out = (col * row) + acc, MN element-wise, 3 inputs.

  Built from the chain ["Mul","Add"] (Mul tiles + Add tiles in one body), exactly the
  rknn-decode matmul decomposition. The K accumulation steps all reuse this one model.
  The reported shape is a square when MN is a perfect square (else [1,MN]) -- this is
  cosmetic for the runtime but matches rknn-decode's chain_to_rknn convention."""
  import math
  s = math.isqrt(MN)
  rows, cols = (s, s) if s * s == MN else (1, MN)
  body = _fb.build_body(MN, 3, ops=["Mul", "Add"])
  return bytes(_fb.assemble_rknn(body, rows, cols, 3))
