import inspect, functools, base64, pickle
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AxisType, graph_rewrite
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import dedup, cpu_profile, DEBUG, getenv

# unrolling the kernel into per-element chunks is only needed for the dataflow/ONNX/NPU view; it explodes large
# reductions (matmul/conv). default off -> RK runs the compact looped kernel on CPU. RK_UNROLL=1 to get chunks.
RK_UNROLL = getenv("RK_UNROLL", 0)

# the matmul reductions are pulled out of the graph as a structured GEMM (M,K,N) before any unrolling, so only the
# elementwise/epilogue work is unrolled into chunks. each extracted GEMM writes straight to the output buffer C; its
# result value is replaced by a LOAD back from C[m,n] so the epilogue reads it. meta collected here, consumed in render.
_RK_GEMM:list[tuple] = []

def extract_matmul(sink:UOp):
  # find the contraction accumulator loop (kept as a raw M,K,N reduce by structured_reduce). extract M,K,N + the affine
  # A/B/C index maps, then replace the loop's result value with LOAD(C[m,n]); the K loop goes dead and only M,N remain.
  topo = list(sink.toposort())
  kr = [u for u in topo if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
  if len(kr) != 1: return None                                                       # only single-reduce GEMMs
  K = kr[0]
  prod = [u for u in topo if u.op is Ops.MUL and u.dtype is dtypes.float and all(s.op is Ops.LOAD for s in u.src)]
  if len(prod) != 1: return None                                                     # MUL(LOAD A, LOAD B) is the product
  la, lb = prod[0].src
  ia, ib = la.src[0].src[1], lb.src[0].src[1]                                         # the index expressions into A,B
  rng_of = lambda e: {u for u in e.toposort() if u.op is Ops.RANGE}
  ra, rb = rng_of(ia), rng_of(ib)
  if K not in ra or K not in rb or len(ra) != 2 or len(rb) != 2: return None
  M, N = (ra-{K}).pop(), (rb-{K}).pop()                                               # A indexes (M,K); B indexes (K,N)
  # the GEMM result is the value read out of the accumulator after the K loop (not the in-loop reload)
  res = [u for u in topo if u.op is Ops.LOAD and u.src[0].src[0].op is Ops.AFTER and
         any(s.op is Ops.END and K in s.src for s in u.src[0].src[0].src)]
  if len(res) != 1: return None
  res = res[0]
  # the single output store this result flows into -> gives the output buffer C and its (m,n) index map
  st = [u for u in topo if u.op is Ops.STORE and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM and res in u.src[1].toposort()]
  if len(st) != 1: return None
  cidx = st[0].src[0]
  iv = lambda e, vm: graph_rewrite(e.substitute({r: r.const_like(v) for r, v in vm.items()}), symbolic).arg
  ba, bb, cb = iv(ia,{M:0,K:0}), iv(ib,{K:0,N:0}), iv(cidx.src[1],{M:0,N:0})
  am, ak = iv(ia,{M:1,K:0})-ba, iv(ia,{M:0,K:1})-ba
  bk, bn = iv(ib,{K:1,N:0})-bb, iv(ib,{K:0,N:1})-bb
  cm, cn = iv(cidx.src[1],{M:1,N:0})-cb, iv(cidx.src[1],{M:0,N:1})-cb
  pa, pb = la.src[0].src[0], lb.src[0].src[0]                                         # the A,B param uops
  _RK_GEMM.append((M.vmax+1, K.vmax+1, N.vmax+1, pa.arg, pb.arg, ba, am, ak, bb, bk, bn, cidx.src[0].arg, cb, cm, cn))
  out = sink.substitute({res: UOp(Ops.LOAD, res.dtype, (cidx,))})                     # read the gemm output back from C[m,n]
  # A,B no longer have any load after extraction; keep their params reachable (as NOOPs) so they stay in globals
  return out.replace(src=out.src + tuple(UOp(Ops.NOOP, dtypes.void, (p,)) for p in (pa, pb)))

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

def rk_sink(sink:UOp):
  if (ext:=extract_matmul(sink)) is not None: return ext   # pull the GEMM out first, then unroll only what's left
  return loop_unrolling(sink)
rk_prepare = PatternMatcher([(UPat(Ops.SINK, name="sink"), rk_sink)])+symbolic

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
    elif u.op in (Ops.PARAM, Ops.SINK, Ops.GROUP, Ops.NOOP): pass
    else:                                                                                       # a compute op
      lvl[u] = L = 1 + max([lvl.get(s, 0) for s in u.src if isval.get(s)], default=0); isval[u] = True
      cat = "alu" if u.op in ALU_GROUP else u.op.name                                           # ALU ops share a chunk
      groups.setdefault((L, cat, u.dtype), []).append(u)                                        # ...split by dtype to stay homogeneous
  chunks = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2])))
  return [(f"s{L}_{cat.lower()}", us) for (L, cat, _), us in chunks], loads, consts, outs

class RKRenderer(Renderer):
  has_local = False
  supports_float4 = not RK_UNROLL          # scalar stream (for the per-element chunk view) only when unrolling
  structured_reduce = bool(RK_UNROLL)       # keep matmul as raw M,K,N ranges so extract_matmul can pull it out pre-unroll
  full_unroll_reduces = bool(RK_UNROLL)     # any remaining (non-matmul) reduction -> add-tree via loop_unrolling
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
      # GEMMs were pulled out before unrolling (extract_matmul) and write straight to C; they run as the first submits.
      # the epilogue/elementwise chunks then stay scalar and read C[m,n] back as an ordinary load.
      gemms = list(_RK_GEMM); _RK_GEMM.clear()
      prog:list = [("mm",)+g for g in gemms]
      nid:dict[UOp,int] = {}
      for _, us in chunks:
        for u in us: nid[u] = len(nid)
      # a src ref is (0,buf,off,dtype) for a load, (1,value,dtype) for a const, or (2,node_id,dtype) for a prior node.
      # carrying the dtype keeps the stream consistently typed so the emitted C stays correct (e.g. uint64 << uint).
      ref = lambda s: (0, loads[s][0], loads[s][1], s.dtype.scalar()) if s in loads else \
                      (1, consts[s], s.dtype.scalar()) if s in consts else (2, nid[s], s.dtype.scalar())
      # each chunk becomes one scalar submit (gemm submits first -> node ids stay == runtime scratch position).
      prog += [[(u.op, [ref(s) for s in u.src], u.dtype.scalar()) for u in us] for _, us in chunks]
      out_list = [(o, ref(v), v.dtype.scalar()) for o, v in sorted(outs.items())]
      if DEBUG >= 6:
        dref = lambda s: f"in{loads[s][0]}[{loads[s][1]}]" if s in loads else f"={consts[s]}" if s in consts else f"%{idx[s]}"
        print(f"RKRenderer: {len(uops)} unrolled uops -> {len(chunks)} chunks ({len(nid)} nodes) in {len(prog)} submits")
        for g in gemms: print(f"  GEMM submit (extracted pre-unroll): M={g[0]} K={g[1]} N={g[2]} -> C[{g[11]}]")
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
  is_mm = lambda sub: isinstance(sub, tuple) and sub[0] == "mm"
  def mm_lines(M, K, N, bufA, bufB, ba, am, ak, bb, bk, bn, bufC, cb, cm, cn):
    # the structured GEMM (M,K,N from the original ranges): one triple loop over the affine A/B index maps, writing
    # each output straight to C via its affine output map. this is the only place the contraction is expanded.
    return [f"const float* A=(const float*)b[{bufA}]; const float* Bm=(const float*)b[{bufB}]; float* C=(float*)b[{bufC}];",
            f"for(int m=0;m<{M};m++)for(int n=0;n<{N};n++){{ float acc=0;",
            f"for(int kk=0;kk<{K};kk++) acc+=A[{ba}+{am}*m+{ak}*kk]*Bm[{bb}+{bk}*kk+{bn}*n];",
            f"C[{cb}+{cm}*m+{cn}*n]=acc; }}"]
  nbufs = max([1] + [s[1]+1 for sub in prog if not is_mm(sub) for _,srcs,_ in sub for s in srcs if s[0] == 0] +
              [r[1]+1 for _,r,_ in out_list if r[0] == 0] + [b+1 for sub in prog if is_mm(sub) for b in (sub[4], sub[5], sub[12])])
  store_lines = [f"(({ct(dt)}*)b[0])[{o}] = {expr(r)};" for o, r, dt in out_list]
  submits, gid = [], 0
  for si, sub in enumerate(prog or [[]]):
    if is_mm(sub): lines = mm_lines(*sub[1:])
    else: lines = [f"*({ct(dt)}*)(s+{(gid:=gid+1)-1}*8) = {rhs_of(op, srcs, dt)};" for op, srcs, dt in sub]
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
