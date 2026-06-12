import inspect, functools, base64, pickle
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import dedup, cpu_profile, NULL_ALLOW_COPYOUT

def loop_unrolling(sink:UOp):
  # fully unroll every RANGE so the stream is range-free (every op explicit), mirrors VLIWRenderer.loop_unrolling
  if not (rng:=[x for x in sink.toposort() if x.op is Ops.RANGE]): return None
  return UOp.sink(*[s for i in range(rng[0].vmax+1) for s in sink.substitute({rng[0]:rng[0].const_like(i)}).src], arg=sink.arg)
rk_prepare = PatternMatcher([(UPat(Ops.SINK, name="sink"), loop_unrolling)])+symbolic

# element-wise uops expressed as ONNX nodes (SHL/SHR use BitShift, handled separately)
ONNX_OP = {Ops.ADD:"Add", Ops.SUB:"Sub", Ops.MUL:"Mul", Ops.FDIV:"Div", Ops.CDIV:"Div", Ops.CMOD:"Mod", Ops.MAX:"Max",
           Ops.CMPLT:"Less", Ops.CMPNE:"Equal", Ops.OR:"BitwiseOr", Ops.XOR:"BitwiseXor", Ops.AND:"BitwiseAnd",
           Ops.WHERE:"Where", Ops.SQRT:"Sqrt", Ops.RECIPROCAL:"Reciprocal", Ops.NEG:"Neg", Ops.SIN:"Sin"}
ONNX_SHIFT = {Ops.SHL:"LEFT", Ops.SHR:"RIGHT"}
ONNX_DT = {dtypes.float:1, dtypes.uint:12, dtypes.ulong:13, dtypes.half:10, dtypes.double:11, dtypes.int:6, dtypes.long:7, dtypes.bool:9}  # onnx TensorProto codes

def uops_to_onnx_staged(uops:list[UOp]):
  # group by (dependency level, op type) so the result is a runnable DAG even when the same op type
  # appears at several levels (e.g. d + ((a+b) > c)). handles element-wise math, bit ops and threefry.
  from onnx import helper
  idx = {u:i for i,u in enumerate(uops)}
  val, vdt, off, pof, lvl = {}, {}, {}, {}, {}
  groups, loads, consts, outs = {}, {}, {}, {}             # (level,op_type) -> [(node,ins,out)]
  def node(nm, dt, op, ins, L, **attrs):
    vdt[nm] = dt; groups.setdefault((L, op), []).append((helper.make_node(op, ins, [nm], name=nm, **attrs), ins, nm))
  def emit(u, op, **attrs):
    val[u]=nm=f"u{idx[u]}"; lvl[u]=L=1+max([lvl.get(s, 0) for s in u.src], default=0); node(nm, ONNX_DT[u.dtype.scalar()], op, [val[s] for s in u.src], L, **attrs)  # noqa: E501
  def kconst(dt, py): nm=f"k{len(consts)}"; vdt[nm]=dt; consts[nm]=py; return nm
  for u in uops:
    if u.op is Ops.CONST:
      if not dtypes.is_float(u.dtype): off[u]=int(u.arg)                                                    # may be an index offset...
      val[u]=f"c{idx[u]}"; vdt[val[u]]=ONNX_DT[u.dtype.scalar()]; consts[val[u]]=u.arg; lvl[u]=0            # ...or a value operand
    elif u.op is Ops.INDEX: pof[u], off[u] = u.src[0], off[u.src[1]]
    elif u.op is Ops.CAST:
      if u.src[0] in pof: pof[u], off[u] = pof[u.src[0]], off[u.src[0]]                                     # index cast -> passthrough
      elif u.src[0] in val and u.dtype.scalar() != u.src[0].dtype.scalar(): emit(u, "Cast", to=ONNX_DT[u.dtype.scalar()])  # value cast
      elif u.src[0] in val: val[u] = val[u.src[0]]; lvl[u] = lvl.get(u.src[0], 0)
    elif u.op is Ops.LOAD:
      ix=u.src[0]; nm=f"data{pof[ix].arg}_{off[ix]}"; val[u]=nm; vdt[nm]=ONNX_DT[u.dtype.scalar()]; loads[nm]=(pof[ix].arg, off[ix]); lvl[u]=0
    elif u.op in ONNX_SHIFT: emit(u, "BitShift", direction=ONNX_SHIFT[u.op])
    elif u.op in ONNX_OP: emit(u, ONNX_OP[u.op])
    elif u.op is Ops.BITCAST:    # uint32 -> float32 for normalized [1,2) values (threefry): 1 + (x & 0x7fffff)/2^23
      if (u.src[0].dtype.scalar(), u.dtype.scalar()) != (dtypes.uint, dtypes.float):
        raise NotImplementedError(f"RK ONNX converter only bitcasts uint32->float32, got {u.src[0].dtype}->{u.dtype}")
      x, L = val[u.src[0]], lvl.get(u.src[0], 0); i = idx[u]
      node(f"u{i}_m", ONNX_DT[dtypes.uint], "BitwiseAnd", [x, kconst(ONNX_DT[dtypes.uint], 0x7fffff)], L+1)
      node(f"u{i}_c", ONNX_DT[dtypes.float], "Cast", [f"u{i}_m"], L+2, to=ONNX_DT[dtypes.float])
      node(f"u{i}_t", ONNX_DT[dtypes.float], "Mul", [f"u{i}_c", kconst(ONNX_DT[dtypes.float], 2.0**-23)], L+3)
      val[u]=f"u{i}"; lvl[u]=L+4; node(f"u{i}", ONNX_DT[dtypes.float], "Add", [kconst(ONNX_DT[dtypes.float], 1.0), f"u{i}_t"], L+4)
    elif u.op is Ops.EXP2:       # 2^x  (ONNX has no Exp2)
      val[u]=f"u{idx[u]}"; lvl[u]=L=1+lvl.get(u.src[0],0); node(val[u], ONNX_DT[u.dtype.scalar()], "Pow", [kconst(ONNX_DT[u.dtype.scalar()],2.0), val[u.src[0]]], L)  # noqa: E501
    elif u.op is Ops.LOG2:       # log2(x) = ln(x) * (1/ln2)   (ONNX has no Log2)
      x, L, i = val[u.src[0]], lvl.get(u.src[0],0), idx[u]
      node(f"u{i}_l", ONNX_DT[u.dtype.scalar()], "Log", [x], L+1)
      val[u]=f"u{i}"; lvl[u]=L+2; node(f"u{i}", ONNX_DT[u.dtype.scalar()], "Mul", [f"u{i}_l", kconst(ONNX_DT[u.dtype.scalar()], 1.4426950408889634)], L+2)  # noqa: E501
    elif u.op is Ops.TRUNC:      # truncate toward zero via an int round-trip (ONNX has no Trunc)
      x, L = val[u.src[0]], lvl.get(u.src[0], 0); i = idx[u]
      node(f"u{i}_i", ONNX_DT[dtypes.long], "Cast", [x], L+1, to=ONNX_DT[dtypes.long])
      val[u]=f"u{i}"; lvl[u]=L+2; node(f"u{i}", ONNX_DT[u.dtype.scalar()], "Cast", [f"u{i}_i"], L+2, to=ONNX_DT[u.dtype.scalar()])
    elif u.op is Ops.STORE: outs[off[u.src[0]]] = val[u.src[1]]
    elif u.op not in (Ops.PARAM, Ops.SINK, Ops.GROUP):     # pure dataflow only; accumulators (DEFINE_REG/AFTER) etc. can't be ONNX nodes
      raise NotImplementedError(f"RK ONNX converter can't express {u.op.name} (only element-wise + unrolled reductions)")
  sv = lambda nm: helper.make_tensor_value_info(nm, vdt[nm], [])
  models = []
  for (L, op_type) in sorted(groups):
    items = groups[(L, op_type)]; produced = {o for _,_,o in items}; used = {i for _,ins,_ in items for i in ins}
    g = helper.make_graph([n for n,_,_ in items], f"s{L}_{op_type.lower()}",
                          [sv(x) for x in sorted(used-produced)], [sv(o) for _,_,o in items])
    models.append((f"s{L}_{op_type.lower()}", helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 18)])))
  return models, loads, consts, outs

class RKRenderer(Renderer):
  has_local = False
  supports_float4 = False     # scalar stream -> no GEP/STACK to handle in the ONNX conversion
  full_unroll_reduces = True  # force reductions into a pure add-tree (no accumulator) so they are ONNX-expressible
  # keep transcendentals native (don't let them decompose into float<->int bitcast, which ONNX can't express)
  code_for_op = {Ops.ADD:"+", Ops.SUB:"-", Ops.MUL:"*", Ops.FDIV:"/", Ops.CMPLT:"<", Ops.CMPNE:"!=", Ops.MAX:"max",
                 Ops.SIN:"sin", Ops.SQRT:"sqrt", Ops.EXP2:"exp2", Ops.LOG2:"log2", Ops.RECIPROCAL:"recip"}
  pre_matcher = rk_prepare
  def render(self, uops:list[UOp]) -> str:
    import onnx
    # group the kernel into ONNX graphs by (dependency level, op type) so the stages form a runnable DAG
    models, _, _, _ = uops_to_onnx_staged(uops)
    print(f"RKRenderer: grouped the kernel into {len(models)} staged ONNX graphs")
    for fname, m in models:
      print(f"===== {fname}.onnx =====\n" + onnx.helper.printable_graph(m.graph))
    return base64.b64encode(pickle.dumps([(n, m.SerializeToString()) for n,m in models])).decode()

class RKCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)
RKRenderer.compiler = RKCompiler()

class RKProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs): self.device, self.name = device, name
  def __call__(self, *bufs, wait=False, **kw):
    with cpu_profile(self.name, self.device): return 1e-3

class RKAllocator(Allocator['RKDevice']):
  def _alloc(self, size, options): return bytearray(size)
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src):
    if not NULL_ALLOW_COPYOUT: raise RuntimeError("no copyout on RK")

class RKDevice(Compiled):
  def __init__(self, device:str):
    renderers = [RKRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl] for r in m.__dict__.values()
                                if inspect.isclass(r) and issubclass(r, Renderer)]
    super().__init__(device, RKAllocator(self), dedup(renderers), functools.partial(RKProgram, device))
