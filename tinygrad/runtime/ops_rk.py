import inspect, functools, base64, pickle
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AxisType, graph_rewrite
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import dedup, cpu_profile, DEBUG, getenv

# the structured path: matmul/elementwise are extracted from the uop graph (extract_matmul / extract_elementwise /
# extract_bias_vector) and run via clang/NPU; the leftover loop renders straight to C (compact_loop_to_c). only an op
# compact_loop_to_c can't render to C falls back to the slow per-uop PythonProgram interpreter.

# the matmul reductions are pulled out of the graph as a structured GEMM (M,K,N); each extracted GEMM runs on the NPU
# MAC (vendor gemm.py) and writes straight to the output buffer C; its result value is replaced by a LOAD back from
# C[m,n] so the epilogue reads it. meta collected here, consumed in render.
_RK_GEMM:list[tuple] = []
# the elementwise epilogue is *also* pulled out structurally (extract_elementwise) -> an EW chain descriptor run on the
# NPU DPU-EW (chain_add) with a clang CPU fallback (ew_lines), or, when it's exactly C+const / C+bias[n], fused into the GEMM.
_RK_EW:list[dict] = []
# a  C[m,n] + bias[n]  epilogue (per-column bias broadcast over rows) is pulled out as a VECTOR fused into the GEMM
# via run_gemm augmentation ([x|1]@[W;b], pure matmul). collected here per kernel as (out_buf, bias_param, base, stride, N).
_RK_BIAS:list[tuple] = []

@functools.cache
def _gemm():
  # import the vendor GEMM module (gemm.py) by path. import-safe: it opens the NPU only on the first run_gemm call.
  import importlib.util
  spec = importlib.util.spec_from_file_location("rk_gemm", getenv("RK_GEMM", "/data/rkt/examples/gemm.py"))
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  return mod

@functools.cache
def _chain_add():
  # import the vendor DPU-EW module (chain_add.py) by path; like _gemm it opens the NPU only on the first submit.
  import importlib.util
  spec = importlib.util.spec_from_file_location("rk_chain_add", getenv("RK_CHAIN_ADD", "/data/rkt/examples/chain_add.py"))
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  return mod

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

EW_BINOPS = {Ops.ADD:"ADD", Ops.MUL:"MUL", Ops.SUB:"SUB", Ops.FDIV:"DIV"}   # extractable EW binops (must be in ew_lines' EW_C)

def extract_elementwise(sink:UOp):
  # the ALU analogue of extract_matmul: detect a pure elementwise kernel (NO reduce) whose single STORE value is a
  # DAG of EW binops over affine LOADs/consts, and return it as a STRUCTURED task graph over the loop RANGE(s) --
  # rendered as one structured clang loop (ew_lines), WITHOUT unrolling. None if it isn't a representable contiguous EW graph.
  topo = list(sink.toposort())
  if any(u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE for u in topo): return None   # has a reduce -> not pure EW
  stores = [u for u in topo if u.op is Ops.STORE]
  if len(stores) != 1 or stores[0].src[0].op is not Ops.INDEX or stores[0].src[0].src[0].op is not Ops.PARAM: return None
  if not dtypes.is_float(stores[0].src[1].dtype): return None              # the NPU EW / clang ew_lines are fp16/fp32 only -> non-float EW stays a C loop
  rngs = [u for u in topo if u.op is Ops.RANGE]
  if not rngs: return None
  n = 1
  for r in rngs: n *= r.vmax+1
  # row-major flatten strides; a leaf is only tile-expressible if its affine index map matches them (contiguous).
  strides, acc = [], 1
  for r in reversed(rngs): strides.insert(0, acc); acc *= r.vmax+1
  iv = lambda e, vm: graph_rewrite(e.substitute({r: r.const_like(v) for r, v in vm.items()}), symbolic).arg
  def affine(idx):
    base = iv(idx, {r:0 for r in rngs})
    return base, [iv(idx, {**{rr:0 for rr in rngs}, r:1})-base for r in rngs]
  def leaf(u):
    # ('const', value) or ('buf', bufid, base) for a contiguous affine load; else None (not a tile leaf)
    if u.op is Ops.CONST: return ("const", u.arg)
    if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
      base, coeffs = affine(u.src[0].src[1])
      return ("buf", u.src[0].src[0].arg, base) if coeffs == strides else None
    return None
  # walk the store value into a task DAG: each EW binop -> one task (op, src_ref, ew_ref). a ref is
  # ("leaf", i) for an input operand (contiguous load or broadcast const) or ("node", k) for task k's result.
  # shared subexpressions collapse to a single task, so the whole ALU graph is one structured loop.
  leaves, leaf_idx, tasks, node_of = [], {}, [], {}
  def leaf_ref(lf):
    if lf not in leaf_idx: leaf_idx[lf] = len(leaves); leaves.append(lf)
    return ("leaf", leaf_idx[lf])
  def build(v):
    if (lf:=leaf(v)) is not None: return leaf_ref(lf)
    if v in node_of: return ("node", node_of[v])                                       # reuse a shared subexpr
    if v.op not in EW_BINOPS or len(v.src) != 2: return None                            # not an EW op -> not extractable
    if (ra:=build(v.src[0])) is None or (rb:=build(v.src[1])) is None: return None
    node_of[v] = len(tasks); tasks.append((EW_BINOPS[v.op], ra, rb))                    # children first -> topo order
    return ("node", node_of[v])
  obase, ocoeffs = affine(stores[0].src[0].src[1])
  # one regcmd block per task in a shared 1KB buffer -> cap the graph at the toolkit-free limit, else fall back to scalar
  if ocoeffs != strides or (root:=build(stores[0].src[1])) is None or root[0] != "node" or len(tasks) > 6: return None
  return {"n": n, "out": (stores[0].src[0].src[0].arg, obase), "tasks": tasks, "leaves": leaves}

def extract_bias_vector(sink:UOp):
  # after extract_matmul, detect a pure  C[m,n] + bias[n]  epilogue (per-column bias, broadcast over rows): the store
  # value is ADD(LOAD C[m,n] in place, LOAD bias[<contiguous axis only>]). record it as a bias VECTOR for the GEMM
  # (run_gemm augments [x|1]@[W;b]) and drop the store. None if it isn't exactly that shape.
  topo = list(sink.toposort())
  stores = [u for u in topo if u.op is Ops.STORE]
  if len(stores) != 1: return None
  st = stores[0]; cidx = st.src[0]
  if cidx.op is not Ops.INDEX or cidx.src[0].op is not Ops.PARAM or st.src[1].op is not Ops.ADD or len(st.src[1].src) != 2: return None
  selfl = [s for s in st.src[1].src if s.op is Ops.LOAD and s.src[0] is cidx]                       # in-place GEMM output read
  biasl = [s for s in st.src[1].src if s.op is Ops.LOAD and s.src[0].op is Ops.INDEX and s.src[0] is not cidx and
           s.src[0].src[0].op is Ops.PARAM]
  if len(selfl) != 1 or len(biasl) != 1: return None
  bidx = biasl[0].src[0].src[1]                                                                     # bias index expression
  orng = {u for u in cidx.src[1].toposort() if u.op is Ops.RANGE}                                   # the (M,N) output ranges
  brng = {u for u in bidx.toposort() if u.op is Ops.RANGE}
  if len(brng) != 1 or not brng <= orng: return None                                                # bias indexes exactly one axis
  ncol = next(iter(brng))
  iv = lambda e, vm: graph_rewrite(e.substitute({r: r.const_like(v) for r, v in vm.items()}), symbolic).arg
  z = {r: 0 for r in orng}
  if iv(cidx.src[1], {**z, ncol: 1}) - iv(cidx.src[1], z) != 1: return None                          # must be the contiguous (column) axis
  base, stride = iv(bidx, z), iv(bidx, {**z, ncol: 1}) - iv(bidx, z)
  _RK_BIAS.append((cidx.src[0].arg, biasl[0].src[0].src[0].arg, base, stride, ncol.vmax + 1))
  return UOp.sink(*[UOp(Ops.NOOP, dtypes.void, (p,)) for p in dedup([u for u in topo if u.op is Ops.PARAM])], arg=sink.arg)

def rk_sink(sink:UOp):
  s, mm = sink, extract_matmul(sink)                       # pull the GEMM out first (structured M,K,N), then...
  if mm is not None: s = mm
  bv = extract_bias_vector(s) if mm is not None else None  # ...a  C[m,n]+bias[n]  epilogue -> a per-column bias VECTOR on the GEMM
  if bv is not None: s = bv
  ew = extract_elementwise(s) if bv is None else None      # ...else pull the elementwise epilogue out structurally too (no unroll)
  if ew is not None:
    _RK_EW.append(ew)
    # the epilogue is now a descriptor; drop its store/ranges but keep every param reachable so globals stays correct
    s = UOp.sink(*[UOp(Ops.NOOP, dtypes.void, (p,)) for p in dedup([u for u in s.toposort() if u.op is Ops.PARAM])], arg=s.arg)
  if mm is None and ew is None: return None                # nothing structured -> leave the sink compact; render emits a C loop
  return s                                                 # something extracted; re-apply, then render the leftover as a C loop
rk_prepare = PatternMatcher([(UPat(Ops.SINK, name="sink"), rk_sink)])+symbolic

class RKRenderer(Renderer):
  has_local = False
  supports_float4 = False                   # scalar stream: compact_loop_to_c renders scalar loads/stores, not vec4
  structured_reduce = True                   # keep matmul as raw M,K,N ranges so extract_matmul can pull it out
  full_unroll_reduces = False               # keep non-matmul reductions as register-accumulator loops -> compact_loop_to_c
  # native ops must be a subset of what the uop runtime (python_alu) can execute; keep transcendentals native
  # so they don't decompose into float<->int bitcast (which ONNX can't express). (no FDIV -> uses RECIPROCAL+MUL)
  code_for_op = {Ops.ADD:"+", Ops.SUB:"-", Ops.MUL:"*", Ops.CMPLT:"<", Ops.CMPNE:"!=", Ops.MAX:"max",
                 Ops.SIN:"sin", Ops.SQRT:"sqrt", Ops.EXP2:"exp2", Ops.LOG2:"log2", Ops.RECIPROCAL:"recip"}
  pre_matcher = rk_prepare
  def render(self, uops:list[UOp]) -> str:
    idx = {u:i for i,u in enumerate(uops)}
    if not any(u.op is Ops.RANGE for u in uops):   # fully extracted: no leftover compute -- only structured GEMMs + EWs.
      # GEMMs (extract_matmul) write straight to C; the EW (extract_elementwise / extract_bias_vector) is a descriptor.
      gemms = list(_RK_GEMM); _RK_GEMM.clear(); ews = list(_RK_EW); _RK_EW.clear()
      # the GEMM runs on the NPU MAC (vendor gemm.py). an epilogue that is exactly  C + const  FUSES into the GEMM's DPU
      # bias unit (run_gemm bias=); a  C + bias[n]  -> a per-column bias VECTOR (augmentation). detect & drop those EWs;
      # any other EW runs on the NPU DPU-EW (chain_add), with a clang fallback (ew_lines, fp32) if the submit fails.
      bias:dict = {}
      for ew in (ews if gemms else []):
        t, lv = ew["tasks"], ew["leaves"]
        if len(t) == 1 and t[0][0] == "ADD":
          bl = [lv[r[1]] for r in t[0][1:3] if r[0] == "leaf" and lv[r[1]][0] == "buf"]      # buffer operand(s)
          cl = [lv[r[1]][1] for r in t[0][1:3] if r[0] == "leaf" and lv[r[1]][0] == "const"] # const operand(s)
          # one operand is the GEMM output C (read & written back in place), the other a broadcast const
          if len(bl) == 1 and len(cl) == 1 and ew["out"] == (bl[0][1], bl[0][2]) and any(bl[0][1] == g[11] for g in gemms):
            bias[bl[0][1]] = cl[0]
      ews = [ew for ew in ews if ew["out"][0] not in bias]                                    # fused const epilogues removed
      vbias = {ob: (bp, base, stride, nn) for (ob, bp, base, stride, nn) in _RK_BIAS}; _RK_BIAS.clear()
      # per-GEMM bias carried to the runtime: a float (-> DPU BS ALU) or a (param,base,stride,N) spec (-> augmentation) or None
      npu_gemm = [g + (bias.get(g[11], vbias.get(g[11])),) for g in gemms]
      # the leftover EW(s) run on the NPU DPU-EW (chain_add) FIRST, with a structured clang loop (ew_lines, fp32) as the
      # CPU fallback if the submit fails (e.g. the gemm->EW session-state limit, §8.3, ENXIOs a post-GEMM EW).
      prog:list = [("ew", e) for e in ews]
      if DEBUG >= 6:
        for g in gemms: print(f"  GEMM (M={g[0]} K={g[1]} N={g[2]} -> C[{g[11]}]" +
                              (f" + {bias[g[11]]} fused (DPU BS)" if g[11] in bias else
                               f" + bias[{g[2]}] fused (augment buf{vbias[g[11]][0]})" if g[11] in vbias else "") + ") on NPU MAC")
      return base64.b64encode(pickle.dumps(("chunks", prog, npu_gemm))).decode()
    if DEBUG >= 6:
      print(f"RKRenderer: {len(uops)} compact uops:")
      for u in uops:
        print(f"  %{idx[u]:<3d} = {u.op.name:12s} {u.dtype}" + (" src=("+", ".join(f'%{idx[s]}' for s in u.src)+")" if u.src else "") +
              ("" if u.arg is None else f" arg={u.arg}"))
    if (cl:=compact_loop_to_c(uops)) is not None:
      # render the leftover compact loop straight to clang C (no unroll, no interpreter). a GEMM extracted before it
      # (matmul + a non-extractable epilogue, e.g. (a@b).relu()) still runs on the NPU first: carry the gemms and append
      # the cloop as the last clang submit. no extractable EW reaches here (an extractable epilogue leaves no RANGE).
      gemms = list(_RK_GEMM); _RK_GEMM.clear(); ews = list(_RK_EW); _RK_EW.clear(); _RK_BIAS.clear()
      npu_gemm = [g+(None,) for g in gemms]
      prog:list = [("ew", e) for e in ews] + [("cloop", cl[0], cl[1])]
      if DEBUG >= 3: print(f"RKRenderer: {len(uops)} compact uops -> clang C loop (CPU)"+(f", after {len(gemms)} NPU GEMM" if gemms else "")+":\n"+"\n".join(cl[0]))
      return base64.b64encode(pickle.dumps(("chunks", prog, npu_gemm))).decode()
    # not renderable to C -> hand the uops straight to the (slow) uop interpreter
    if DEBUG >= 3: print(f"RKRenderer: {len(uops)} compact uops not renderable to C -> uop interpreter (CPU)")
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

def chunks_to_c(gemms, prog) -> tuple[list[str], list[str], int]:
  # render the clang CPU fallbacks: one `void gN(void** B)` per NPU GEMM (mm_lines) and one `void kN(void** B)` per EW/cloop
  # submit (ew_lines / the compact loop body). a g/k fxn runs only if its NPU path fails; each writes straight to its output(s).
  is_ew = lambda sub: isinstance(sub, tuple) and sub[0] == "ew"
  is_cloop = lambda sub: isinstance(sub, tuple) and sub[0] == "cloop"     # ("cloop", body_lines, nbufs): a compact loop
  EW_C = {"ADD":"+", "MUL":"*", "SUB":"-", "DIV":"/"}
  def mm_lines(M, K, N, bufA, bufB, ba, am, ak, bb, bk, bn, bufC, cb, cm, cn, bias):
    # fp32 matmul  C[m,n] = sum_k A[m,k]*B[k,n] (+bias)  -- the same affine A/B/C index maps the NPU GEMM uses.
    bt = "" if bias is None else f"+((float*)b[{bias[0]}])[{bias[1]}+{bias[2]}*n]" if isinstance(bias, tuple) else f"+{float(bias)}f"
    return [f"for(int m=0;m<{M};m++)for(int n=0;n<{N};n++){{", "  float acc=0;",
            f"  for(int k=0;k<{K};k++)acc+=((float*)b[{bufA}])[{ba}+{am}*m+{ak}*k]*((float*)b[{bufB}])[{bb}+{bk}*k+{bn}*n];",
            f"  ((float*)b[{bufC}])[{cb}+{cm}*m+{cn}*n]=acc{bt};", "}"]
  def ew_lines(d):
    # the structured elementwise graph (extract_elementwise) as one C loop over n: each task is a binop over leaf
    # buffers/consts or a prior task, writing the root straight to the output. no unroll -- the loop IS the range.
    leaves, tasks, (ob, obase) = d["leaves"], d["tasks"], d["out"]
    refc = lambda r: (f"((float*)b[{leaves[r[1]][1]}])[{leaves[r[1]][2]}+i]" if leaves[r[1]][0] == "buf" else f"{float(leaves[r[1]][1])}f") \
                     if r[0] == "leaf" else f"t{r[1]}"
    body = [f"for(int i=0;i<{d['n']};i++){{"]
    for k, (op, ra, rb) in enumerate(tasks):
      body.append(f"  float t{k} = " + (f"fmaxf({refc(ra)},{refc(rb)})" if op == "MAX" else f"({refc(ra)}{EW_C[op]}{refc(rb)})") + ";")
    return body + [f"  ((float*)b[{ob}])[{obase}+i] = t{len(tasks)-1};", "}"]
  gemm_bufs = [b for g in gemms for b in (g[3], g[4], g[11])] + [g[15][0] for g in gemms if isinstance(g[15], tuple)]  # A,B,C (+bias buf)
  ew_bufs = [b for sub in prog if is_ew(sub) for b in [sub[1]["out"][0]] + [lf[1] for lf in sub[1]["leaves"] if lf[0] == "buf"]]
  nbufs = max([1] + [b+1 for b in gemm_bufs + ew_bufs] + [sub[2] for sub in prog if is_cloop(sub)])
  wrap = lambda nm, lines: f"void {nm}(void** B){{ char** b=(char**)B;\n" + "\n".join(lines) + "\n}\n"
  gsubs = [wrap(f"g{gi}", mm_lines(*g)) for gi, g in enumerate(gemms)]
  submits = [wrap(f"k{si}", ew_lines(sub[1]) if is_ew(sub) else sub[1] if is_cloop(sub) else []) for si, sub in enumerate(prog)]
  return gsubs, submits, nbufs

def compact_loop_to_c(uops:list[UOp]):
  # render a COMPACT looped sink -- elementwise AND reductions (via the register-accumulator form: DEFINE_REG + a
  # zero-store, an in-reduce-loop update store, then an output store) -- straight to one C function with nested
  # for-loops, by walking the uops in linearized order (RANGE->`for{`, END->`}`, STORE->assignment). directly from the
  # uops: NO unroll, NO per-element scalars, NO uop interpreter. returns ([submit], nbufs) or None if an op can't render.
  if not any(u.op is Ops.RANGE for u in uops): return None
  bufs = {u: u.arg for u in uops if u.op is Ops.PARAM}
  regs = {u: f"acc{i}" for i, u in enumerate(u for u in uops if u.op is Ops.DEFINE_REG)}
  rng:dict = {}
  def ct2(dt):
    if (s:=dt.scalar()) not in CT: raise NotImplementedError(s)
    return CT[s]
  def memref(ix):                                               # INDEX uop -> ("reg",name) or ("buf",bufid,index_uop)
    p = ix.src[0]
    while p.op in (Ops.CAST, Ops.AFTER): p = p.src[0]           # AFTER/CAST are dependency/ptr wrappers -> the base
    if p.op is Ops.DEFINE_REG: return ("reg", regs[p])
    if p.op is Ops.PARAM: return ("buf", bufs[p], ix.src[1])
    raise NotImplementedError(f"index base {p.op}")
  def ce(u):                                                    # render a value / index uop to a C expression
    if u.op is Ops.CONST:
      if u.dtype is dtypes.bool: return "1" if u.arg else "0"
      return (f"{float(u.arg)!r}f" if u.dtype.scalar() in (dtypes.float, dtypes.half) else repr(float(u.arg))) \
             if dtypes.is_float(u.dtype) else str(int(u.arg))
    if u.op is Ops.RANGE: return rng[u]
    if u.op is Ops.CAST: return f"({ct2(u.dtype)})({ce(u.src[0])})"
    if u.op is Ops.LOAD:
      mr = memref(u.src[0]); return mr[1] if mr[0] == "reg" else f"(({ct2(u.dtype)}*)b[{mr[1]}])[{ce(mr[2])}]"
    es, op, f, c = [ce(s) for s in u.src], u.op, ('f' if u.dtype.scalar() in (dtypes.float, dtypes.half) else ''), ct2(u.dtype)
    if op is Ops.MAX: return f"(({es[0]})>({es[1]})?({es[0]}):({es[1]}))"
    if op is Ops.WHERE: return f"(({es[0]})?({es[1]}):({es[2]}))"
    if op is Ops.NEG: return f"(-({es[0]}))"
    if op is Ops.RECIPROCAL: return f"((({c})1)/({es[0]}))"
    if op is Ops.MULACC: return f"(({es[0]})*({es[1]})+({es[2]}))"
    if op is Ops.BITCAST: return f"({{ {ct2(u.src[0].dtype)} _s=({es[0]}); {c} _d; __builtin_memcpy(&_d,&_s,sizeof(_d)); _d; }})"
    if op is Ops.FLOORDIV: return f"__builtin_floor{f}(({es[0]})/({es[1]}))" if f else \
                                  f"({{ {c} _a=({es[0]}),_b=({es[1]}),_m=((_a%_b)+_b)%_b; (_a-_m)/_b; }})"
    if op is Ops.FLOORMOD: return f"({{ {c} _a=({es[0]}),_b=({es[1]}); _a-_b*__builtin_floor{f}(_a/_b); }})" if f else \
                                  f"({{ {c} _a=({es[0]}),_b=({es[1]}); ((_a%_b)+_b)%_b; }})"
    if op in C_FN: return f"__builtin_{C_FN[op]}{f}({es[0]})"
    if op in C_BIN: return f"(({es[0]}){C_BIN[op]}({es[1]}))"
    raise NotImplementedError(f"RK cloop: op {op}")
  regdt:dict = {}
  lines, depth = [], 1
  try:
    for u in uops:                                              # walk in linearized order: ranges open/close loops
      if u.op is Ops.RANGE:
        rng[u] = f"r{len(rng)}"; lines.append("  "*depth + f"for(int {rng[u]}=0;{rng[u]}<{u.vmax+1};{rng[u]}++){{"); depth += 1
      elif u.op is Ops.END: depth -= 1; lines.append("  "*depth + "}")
      elif u.op is Ops.STORE:
        mr = memref(u.src[0])
        if mr[0] == "reg":
          p = u.src[0].src[0]
          while p.op in (Ops.CAST, Ops.AFTER): p = p.src[0]
          regdt[p] = u.src[1].dtype; dest = mr[1]
        else: dest = f"(({ct2(u.src[1].dtype)}*)b[{mr[1]}])[{ce(mr[2])}]"
        lines.append("  "*depth + f"{dest} = {ce(u.src[1])};")
  except (NotImplementedError, KeyError): return None
  return [f"{ct2(regdt.get(r, dtypes.float))} {n};" for r, n in regs.items()] + lines, max(bufs.values())+1   # (decls + loop body), nbufs

class RKProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs):
    self.device, self.name, self.prog = device, name, pickle.loads(lib)
    self._fxns:list = []; self._gemm_fxns:list = []
    self._npu_gemm = self.prog[2] if self.prog[0] == "chunks" else None    # GEMMs to run on the NPU MAC, or None
    if self.prog[0] == "chunks":                                # clang CPU fallbacks: one per NPU GEMM + one per EW/cloop submit
      gsubs, submits, self._nbufs = chunks_to_c(self._npu_gemm, self.prog[1])
      if DEBUG >= 6:
        for si, src in enumerate(gsubs + submits):
          print(f"RKProgram {self.name}: CPU-fallback fxn {si+1}/{len(gsubs)+len(submits)} -> clang ({len(src.splitlines())-2} lines)")
          print(src, end="")
      # each fallback is its own C function g0,g1,.../k0,k1,...; compiled together (one clang call) but dispatched separately
      self._lib = self._compile("\n".join(gsubs + submits))
      self._gemm_fxns = [self._lib[f"g{gi}"] for gi in range(len(gsubs))]
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
      if self.prog[0] == "chunks":                              # NPU GEMMs + EW (chain_add), each falling back to its clang fxn
        import numpy as np
        B = (ctypes.c_void_p*self._nbufs)(*[mv_address(bufs[i]) for i in range(self._nbufs)])
        for i, (M, K, N, bufA, bufB, ba, am, ak, bb, bk, bn, bufC, cb, cm, cn, bias) in enumerate(self._npu_gemm):
          try:                                                  # GEMM on the NPU MAC (vendor gemm.py) FIRST -> writes C
            A = np.frombuffer(bufs[bufA], dtype=np.float32)[ba:ba+M*K].reshape(M, K)   # standard row-major A[M,K]@B[K,N]
            Bm = np.frombuffer(bufs[bufB], dtype=np.float32)[bb:bb+K*N].reshape(K, N)
            bv = bias
            if isinstance(bias, tuple):                                                # (param,base,stride,N) -> read bias VECTOR
              bp, base, stride, NN = bias; bv = np.frombuffer(bufs[bp], dtype=np.float32)[base:base+stride*NN:stride or 1]
            if DEBUG >= 2: print(f"\033[35m*** RK NPU\033[0m  GEMM MAC M={M} K={K} N={N}  C[{cb}:] <- buf{bufA} @ buf{bufB}" +
                                 (f" + bias[{N}] (fused augment)" if np.ndim(bv) else f" + {bv} (fused DPU bias)" if bv is not None else ""))
            C = _gemm().run_gemm(M, N, K, A.astype(np.float16), Bm.astype(np.float16), bias=bv)
            np.frombuffer(bufs[bufC], dtype=np.float32)[cb:cb+M*N] = np.asarray(C, dtype=np.float32).reshape(-1)
          except (OSError, RuntimeError) as e:                  # NPU submit failed (e.g. ENXIO) -> recompute the GEMM on CPU (clang)
            if DEBUG >= 2: print(f"\033[33m*** RK NPU GEMM failed ({e}) -> CPU clang fallback\033[0m")
            self._gemm_fxns[i](B)
        for i, sub in enumerate(self.prog[1]):                  # then each EW epilogue: NPU DPU-EW (chain_add) first, clang fallback
          if sub[0] == "ew":
            d = sub[1]; n = d["n"]; ob, obase = d["out"]
            try:
              lf2v = lambda lf: np.frombuffer(bufs[lf[1]], dtype=np.float32)[lf[2]:lf[2]+n] if lf[0] == "buf" else float(lf[1])
              if DEBUG >= 2: print(f"\033[35m*** RK NPU\033[0m  EW chain ({len(d['tasks'])} op) n={n} -> buf{ob}[{obase}:]")
              res = _chain_add().cached_graph(n, d["tasks"]).run([lf2v(lf) for lf in d["leaves"]])
              np.frombuffer(bufs[ob], dtype=np.float32)[obase:obase+n] = np.asarray(res, dtype=np.float32)
            except (OSError, RuntimeError) as e:                # NPU EW submit failed -> recompute the epilogue on CPU (clang)
              if DEBUG >= 2: print(f"\033[33m*** RK NPU EW failed ({e}) -> CPU clang fallback\033[0m")
              self._fxns[i](B)
          else: self._fxns[i](B)                                # cloop: clang only (no NPU path), reads any C written above
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
