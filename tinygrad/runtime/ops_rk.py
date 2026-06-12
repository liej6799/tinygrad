import inspect, functools, base64, pickle
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import dedup, cpu_profile, DEBUG, getenv

# unrolling the kernel into per-element chunks is only needed for the dataflow/ONNX/NPU view; it explodes large
# reductions (matmul/conv). default off -> RK runs the compact looped kernel on CPU. RK_UNROLL=1 to get chunks.
RK_UNROLL = getenv("RK_UNROLL", 0)
# a big unrolled kernel (mnist-size conv) produces chunks of thousands of nodes; RK_SUBMIT caps the nodes per
# submit so each chunk is split into several bounded submits (0 = one submit per whole chunk).
RK_SUBMIT = getenv("RK_SUBMIT", 0)


def loop_unrolling(sink:UOp):
  # fully unroll every RANGE so the stream is range-free (every op explicit), mirrors VLIWRenderer.loop_unrolling.
  # single topological pass: only range-dependent nodes get per-index copies (the rest are shared), avoiding the
  # vmax+1 full-graph substitute() walks that re-traverse the whole sink each iteration.
  topo = list(sink.toposort())
  if not (rng:=[x for x in topo if x.op is Ops.RANGE]): return None
  r, copies = rng[0], {}
  for u in topo:
    if u is r: copies[u] = [u.const_like(i) for i in range(u.vmax+1)]
    elif u is not sink and any(s in copies for s in u.src):
      copies[u] = [UOp(u.op, u.dtype, tuple(copies[s][i] if s in copies else s for s in u.src), u.arg, u.tag) for i in range(r.vmax+1)]
  return UOp.sink(*[copies[s][i] if s in copies else s for i in range(r.vmax+1) for s in sink.src], arg=sink.arg)
rk_prepare = PatternMatcher([(UPat(Ops.SINK, name="sink"), loop_unrolling)])+symbolic

# the arithmetic ALU ops share one execution unit -> they go in a single "alu" chunk per level. every other op
# (compare, bitwise, shift, cast, transcendental, ...) gets its own chunk.
ALU_GROUP = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.CDIV, Ops.FLOORDIV}

def uops_to_chunks(uops:list[UOp]):
  # the chunking is a pure uop-graph operation: assign each compute uop a dependency level and group by
  # (level, op category, dtype). returns chunks = [(name, [uops])] in execution order, plus the I/O mapping.
  off, pof, lvl, isval = {}, {}, {}, {}
  groups, loads, consts, outs = {}, {}, {}, {}
  for u in uops:
    if u.op is Ops.CONST:
      if not dtypes.is_float(u.dtype): off[u] = int(u.arg)
      consts[u] = u.arg; lvl[u] = 0; isval[u] = True
    elif u.op is Ops.INDEX: pof[u], off[u] = u.src[0], off[u.src[1]]
    elif u.op is Ops.CAST and u.src[0] in pof: pof[u], off[u] = pof[u.src[0]], off[u.src[0]]   # index cast
    elif u.op is Ops.LOAD: ix=u.src[0]; loads[u]=(pof[ix].arg, off[ix]); lvl[u]=0; isval[u]=True
    elif u.op is Ops.STORE: outs[off[u.src[0]]] = u.src[1]
    elif u.op in (Ops.PARAM, Ops.SINK, Ops.GROUP): pass
    else:                                                                                       # a compute op
      lvl[u] = L = 1 + max([lvl.get(s, 0) for s in u.src if isval.get(s)], default=0); isval[u] = True
      cat = "alu" if u.op in ALU_GROUP else u.op.name                                           # ALU ops share a chunk
      groups.setdefault((L, cat, u.dtype), []).append(u)                                        # ...split by dtype to stay homogeneous
  chunks = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2])))
  return [(f"s{L}_{cat.lower()}", us) for (L, cat, _), us in chunks], loads, consts, outs

class RKRenderer(Renderer):
  has_local = False
  supports_float4 = not RK_UNROLL          # scalar stream (for the per-element chunk view) only when unrolling
  full_unroll_reduces = bool(RK_UNROLL)     # reductions -> pure add-tree (chunk-expressible) only when unrolling
  # native ops must be a subset of what the uop runtime (python_alu) can execute; keep transcendentals native
  # so they don't decompose into float<->int bitcast (which ONNX can't express). (no FDIV -> uses RECIPROCAL+MUL)
  code_for_op = {Ops.ADD:"+", Ops.SUB:"-", Ops.MUL:"*", Ops.CMPLT:"<", Ops.CMPNE:"!=", Ops.MAX:"max",
                 Ops.SIN:"sin", Ops.SQRT:"sqrt", Ops.EXP2:"exp2", Ops.LOG2:"log2", Ops.RECIPROCAL:"recip"}
  pre_matcher = rk_prepare if RK_UNROLL else None
  def render(self, uops:list[UOp]) -> str:
    idx = {u:i for i,u in enumerate(uops)}
    if RK_UNROLL:
      # break the unrolled uops into chunks and hand the chunk DAG to the runtime, not the full per-element uop
      # stream. the runtime compiles the DAG to C (clang) and runs it. each chunk = same op at the same level.
      chunks, loads, consts, outs = uops_to_chunks(uops)
      nid:dict[UOp,int] = {}
      for _, us in chunks:
        for u in us: nid[u] = len(nid)
      # a src ref is (0,buf,off,dtype) for a load, (1,value,dtype) for a const, or (2,node_id,dtype) for a prior node.
      # carrying the dtype keeps the stream consistently typed so the emitted C stays correct (e.g. uint64 << uint).
      ref = lambda s: (0, loads[s][0], loads[s][1], s.dtype.scalar()) if s in loads else \
                      (1, consts[s], s.dtype.scalar()) if s in consts else (2, nid[s], s.dtype.scalar())
      # split each chunk into RK_SUBMIT-sized submits (same flat order -> node ids stay == runtime position).
      # nodes in a chunk are all the same level, so they have no inter-dependency and slice safely.
      prog = [[(u.op, [ref(s) for s in u.src], u.dtype.scalar()) for u in us[i:i+(RK_SUBMIT or len(us) or 1)]]
              for _, us in chunks for i in range(0, len(us), RK_SUBMIT or len(us) or 1)]
      out_list = [(o, ref(v), v.dtype.scalar()) for o, v in sorted(outs.items())]
      if DEBUG >= 6:
        dref = lambda s: f"in{loads[s][0]}[{loads[s][1]}]" if s in loads else f"={consts[s]}" if s in consts else f"%{idx[s]}"
        print(f"RKRenderer: {len(uops)} unrolled uops -> {len(chunks)} chunks ({len(nid)} nodes) in {len(prog)} submits")
        for name, us in chunks:
          print(f"  {name}:")
          for u in us: print(f"    %{idx[u]:<3d} = {u.op.name}(" + ", ".join(dref(s) for s in u.src) + ")")
        print("  output: " + ", ".join(f"out[{o}]=%{idx[v]}" for o, v in sorted(outs.items())))
      return base64.b64encode(pickle.dumps(("chunks", prog, out_list, len(nid)))).decode()
    if DEBUG >= 6: print(f"RKRenderer: {len(uops)} uops (compact loops; set RK_UNROLL=1 to unroll into chunks)")
    # the uops are a complete program -> hand them straight to the uop runtime
    lops = [(u.op, u.dtype, [] if u.op is Ops.SPECIAL else [idx[s] for s in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(("uops", lops))).decode()

class RKCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)
RKRenderer.compiler = RKCompiler()

CT = {dtypes.float:"float", dtypes.half:"_Float16", dtypes.double:"double", dtypes.bool:"unsigned char", dtypes.int:"int",
      dtypes.uint:"unsigned int", dtypes.long:"long long", dtypes.ulong:"unsigned long long"}
C_BIN = {Ops.ADD:"+", Ops.SUB:"-", Ops.MUL:"*", Ops.FDIV:"/", Ops.CDIV:"/", Ops.CMOD:"%", Ops.CMPLT:"<",
         Ops.CMPNE:"!=", Ops.CMPEQ:"==", Ops.AND:"&", Ops.OR:"|", Ops.XOR:"^", Ops.SHL:"<<", Ops.SHR:">>"}
C_FN = {Ops.SQRT:"sqrt", Ops.SIN:"sin", Ops.EXP2:"exp2", Ops.LOG2:"log2"}   # __builtin_<fn>[f]

def chunks_to_c(prog, out_list, nnodes:int) -> tuple[list[str], int]:
  # render the chunk DAG as one C program PER SUBMIT (each chunk/RK_SUBMIT slice -> its own `void k(B,S)`), run in
  # sequence. node values live in a shared scratch S (8 bytes/node): a node reads its sources from buffers/consts/S
  # and writes its result back to S; the last submit also performs the output stores. raises on anything C can't express.
  srcdt = lambda r: r[3] if r[0] == 0 else r[2]
  def ct(dt):
    if dt not in CT: raise NotImplementedError(f"RK clang chunk: no C type for {dt}")
    return CT[dt]
  def lit(v, dt):
    if dt in (dtypes.float, dtypes.half): return f"{float(v)!r}f"
    if dt is dtypes.double: return repr(float(v))
    if not isinstance(v, int): raise NotImplementedError(f"RK clang chunk: non-int const {v!r} for {dt}")
    if dt is dtypes.uint: return f"{v&0xffffffff}u"
    if dt is dtypes.ulong: return f"{v&0xffffffffffffffff}ull"
    if dt is dtypes.bool: return "1" if v else "0"
    return str(int(v))
  # a load reads a buffer, a const is a literal, a prior node is read from scratch at its 8-byte slot
  def expr(r): return f"(({ct(r[3])}*)b[{r[1]}])[{r[2]}]" if r[0] == 0 else lit(r[1], r[2]) if r[0] == 1 else f"(*({ct(r[2])}*)(s+{r[1]}*8))"
  def rhs_of(op, srcs, dt):
    c, es, fsuf = ct(dt), [expr(s) for s in srcs], ('f' if dt in (dtypes.float, dtypes.half) else '')
    if op is Ops.CAST: return f"({c})({es[0]})"
    if op is Ops.BITCAST: return f"({{ {ct(srcdt(srcs[0]))} _s=({es[0]}); {c} _d; __builtin_memcpy(&_d,&_s,sizeof(_d)); _d; }})"
    if op is Ops.NEG: return f"-({es[0]})"
    if op is Ops.RECIPROCAL: return f"(({c})1)/({es[0]})"
    if op in C_FN: return f"__builtin_{C_FN[op]}{fsuf}({es[0]})"
    if op is Ops.MAX: return f"(({es[0]})>({es[1]})?({es[0]}):({es[1]}))"
    if op is Ops.WHERE: return f"(({es[0]})?({es[1]}):({es[2]}))"
    if op is Ops.MULACC: return f"({es[0]})*({es[1]})+({es[2]})"
    if op is Ops.FLOORDIV and not fsuf: return f"({{ {c} _a=({es[0]}),_b=({es[1]}),_m=((_a%_b)+_b)%_b; (_a-_m)/_b; }})"
    if op is Ops.FLOORMOD and not fsuf: return f"({{ {c} _a=({es[0]}),_b=({es[1]}); ((_a%_b)+_b)%_b; }})"
    if op is Ops.FLOORDIV: return f"__builtin_floor{fsuf}(({es[0]})/({es[1]}))"
    if op is Ops.FLOORMOD: return f"({{ {c} _a=({es[0]}),_b=({es[1]}); _a-_b*__builtin_floor{fsuf}(_a/_b); }})"
    if op in C_BIN: return f"(({es[0]}){C_BIN[op]}({es[1]}))"
    raise NotImplementedError(f"RK clang chunk: unsupported op {op}")
  nbufs = max([1] + [s[1]+1 for sub in prog for _,srcs,_ in sub for s in srcs if s[0] == 0] + [r[1]+1 for _,r,_ in out_list if r[0] == 0])
  store_lines = [f"(({ct(dt)}*)b[0])[{o}] = {expr(r)};" for o, r, dt in out_list]
  submits, gid = [], 0
  for si, sub in enumerate(prog or [[]]):
    lines = [f"*({ct(dt)}*)(s+{(gid:=gid+1)-1}*8) = {rhs_of(op, srcs, dt)};" for op, srcs, dt in sub]
    if si == len(prog)-1 or not prog: lines += store_lines                     # output stores ride in the last submit
    submits.append(f"void k{si}(void** B, void* S){{ char** b=(char**)B; char* s=(char*)S;\n" + "\n".join(lines) + "\n}\n")
  return submits, nbufs

class RKProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs):
    self.device, self.name, self.prog = device, name, pickle.loads(lib)
    self._fxns:list = []
    if self.prog[0] == "chunks":                                # the chunk DAG runs as a sequence of compiled C submits
      submits, self._nbufs = chunks_to_c(self.prog[1], self.prog[2], self.prog[3])
      self._scratch = memoryview(bytearray(max(1, self.prog[3]) * 8))   # shared 8-byte/node value store across submits
      if DEBUG >= 6:
        for si, src in enumerate(submits):
          print(f"RKProgram {self.name}: submit {si+1}/{len(submits)} -> clang ({len(src.splitlines())-2} lines)")
          print(src, end="")
      # each submit is its own C function k0,k1,...; compiled together (one clang call) but dispatched separately
      self._lib = self._compile("\n".join(submits))
      self._fxns = [self._lib[f"k{si}"] for si in range(len(submits))]
  def _compile(self, src:str):
    import ctypes, subprocess, tempfile, os
    fd, so = tempfile.mkstemp(suffix=".so"); os.close(fd)
    subprocess.run([getenv("CC", "clang"), "-O0", "-shared", "-fPIC", "-x", "c", "-", "-lm", "-o", so],
                   input=src.encode(), check=True, stderr=subprocess.PIPE)
    return ctypes.CDLL(so)
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals:tuple[int,...]=(), wait=False, **kw):
    import ctypes
    from tinygrad.helpers import mv_address
    with cpu_profile(self.name, self.device):
      if self._fxns:                                            # submit each compiled C kernel in turn, chaining via scratch
        B = (ctypes.c_void_p*self._nbufs)(*[mv_address(bufs[i]) for i in range(self._nbufs)])
        S = ctypes.c_void_p(mv_address(self._scratch))
        for fxn in self._fxns: fxn(B, S)
      else:                                                     # compact looped uops -> run on the uop interpreter
        import io, contextlib
        from tinygrad.runtime.ops_python import PythonProgram
        with contextlib.redirect_stdout(io.StringIO()):
          PythonProgram(self.name, pickle.dumps(self.prog[1]))(*bufs, global_size=global_size, local_size=local_size, vals=vals)
    return None

class RKAllocator(Allocator['RKDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RKDevice(Compiled):
  def __init__(self, device:str):
    renderers = [RKRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl] for r in m.__dict__.values()
                                if inspect.isclass(r) and issubclass(r, Renderer)]
    super().__init__(device, RKAllocator(self), dedup(renderers), functools.partial(RKProgram, device))
