import ctypes, functools, os, pathlib
from tinygrad.device import Allocator, Compiled, Compiler
from tinygrad.helpers import cpu_profile, mv_address
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp

RKNNRT_PATH = os.getenv("RKNNRT_PATH", "/data/rk3588/rknn-header/librknnrt.so")
RKNN_MODEL = os.getenv("RKNN_MODEL", "/data/test/add_10x10.rknn")
RKNN_TENSOR_FLOAT32, RKNN_TENSOR_FLOAT16 = 0, 1
RKNN_QUERY_IN_OUT_NUM, RKNN_QUERY_INPUT_ATTR, RKNN_QUERY_OUTPUT_ATTR = 0, 1, 2
RKNN_NPU_CORE_0_1_2 = 7
RKNN_MAX_DIMS, RKNN_MAX_NAME_LEN = 16, 256

class RKNNInputOutputNum(ctypes.Structure):
  _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]

class RKNNTensorAttr(ctypes.Structure):
  _fields_ = [("index", ctypes.c_uint32), ("n_dims", ctypes.c_uint32), ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
              ("name", ctypes.c_char * RKNN_MAX_NAME_LEN), ("n_elems", ctypes.c_uint32), ("size", ctypes.c_uint32),
              ("fmt", ctypes.c_int), ("type", ctypes.c_int), ("qnt_type", ctypes.c_int), ("fl", ctypes.c_int8),
              ("zp", ctypes.c_int32), ("scale", ctypes.c_float), ("w_stride", ctypes.c_uint32),
              ("size_with_stride", ctypes.c_uint32), ("pass_through", ctypes.c_uint8), ("h_stride", ctypes.c_uint32)]

class RKNNInput(ctypes.Structure):
  _fields_ = [("index", ctypes.c_uint32), ("buf", ctypes.c_void_p), ("size", ctypes.c_uint32), ("pass_through", ctypes.c_uint8),
              ("type", ctypes.c_int), ("fmt", ctypes.c_int)]

class RKNNOutput(ctypes.Structure):
  _fields_ = [("want_float", ctypes.c_uint8), ("is_prealloc", ctypes.c_uint8), ("index", ctypes.c_uint32),
              ("buf", ctypes.c_void_p), ("size", ctypes.c_uint32)]

def _check(ret:int, name:str):
  if ret != 0: raise RuntimeError(f"{name} failed: {ret}")

def _addr(buf) -> int: return mv_address(memoryview(buf).cast("B"))

def _float_type(size:int, n_elems:int, name:str) -> int:
  if size == n_elems * 2: return RKNN_TENSOR_FLOAT16
  if size == n_elems * 4: return RKNN_TENSOR_FLOAT32
  raise AssertionError(f"{name} wants float16/float32 buffer for {n_elems} elems, got {size} bytes")

class RKNNBlob(bytes):
  def __new__(cls, data:bytes, path:str, meta:dict|None=None):
    ret = bytes.__new__(cls, data)
    ret.path = path
    ret.meta = meta            # e.g. {"kind":"matmul","M":..,"K":..,"N":..} for the MULACC decomposition
    return ret
  def __repr__(self): return f"<RKNN model {len(self)} bytes from {self.path}>"
  __str__ = __repr__

def _load_rknn():
  lib = ctypes.CDLL(RKNNRT_PATH)
  lib.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
  lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
  lib.rknn_query.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
  lib.rknn_inputs_set.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(RKNNInput)]
  lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
  lib.rknn_outputs_get.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(RKNNOutput), ctypes.c_void_p]
  lib.rknn_outputs_release.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(RKNNOutput)]
  lib.rknn_destroy.argtypes = [ctypes.c_uint64]
  return lib

class RKCompiler(Compiler):
  # keep the RKNNBlob subclass (don't downcast to plain bytes) so .meta survives to RKProgram
  def compile(self, src:bytes) -> bytes: return src if isinstance(src, RKNNBlob) else bytes(src)

class RKRenderer(Renderer):
  compiler = RKCompiler()
  has_local = False
  device = "RK"
  def render(self, uops:list[UOp]) -> bytes:
    # Synthesize a .rknn straight from the UOps (toolkit-free), then RKProgram loads it
    # into the official rknn runtime. This is the rknn-decode uops->.rknn path vendored
    # under runtime/support/rknn. RKNN_STATIC=1 (or a synth failure with RKNN_MODEL set)
    # falls back to a prebuilt static model for debugging.
    from tinygrad.runtime.support.rknn import synth
    if not int(os.getenv("RKNN_STATIC", "0")):
      try: return RKNNBlob(synth.uops_to_rknn(uops), "<synth>")
      except Exception: pass
      try:
        # matmul reduction: emit one MULACC step model + carry (M,K,N); RKProgram runs the
        # K-step element-wise decomposition `acc = A[:,k]*B[k,:] + acc` on the rknn runtime.
        M, K, N = synth.analyze_matmul(uops)
        return RKNNBlob(synth.matmul_step_rknn(M*N), "<synth:matmul>", {"kind":"matmul", "M":M, "K":K, "N":N})
      except Exception as e:
        if not RKNN_MODEL or not pathlib.Path(RKNN_MODEL).exists(): raise
        if int(os.getenv("DEBUG", "0")): print(f"RKRenderer: synth failed ({e}); using {RKNN_MODEL}")
    path = pathlib.Path(RKNN_MODEL)
    return RKNNBlob(path.read_bytes(), str(path))
def _rknn_init(lib, model_bytes:bytes):
  ctx = ctypes.c_uint64(0)
  model = ctypes.create_string_buffer(model_bytes)
  _check(lib.rknn_init(ctypes.byref(ctx), ctypes.cast(model, ctypes.c_void_p), len(model_bytes), 0, None), "rknn_init")
  _check(lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2), "rknn_set_core_mask")
  return ctx

def _rknn_infer(lib, ctx, in_addrs_sizes, out_buf):
  """Run one inference: in_addrs_sizes is [(addr,size,fmt,type), ...]; result -> out_buf bytes."""
  io = RKNNInputOutputNum()
  _check(lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io), ctypes.sizeof(io)), "RKNN_QUERY_IN_OUT_NUM")
  assert io.n_output == 1, "RKNN model expects one output"
  assert io.n_input == len(in_addrs_sizes), f"model expects {io.n_input} inputs, got {len(in_addrs_sizes)}"
  in_attrs = (RKNNTensorAttr * io.n_input)()
  for i in range(io.n_input):
    in_attrs[i].index = i
    _check(lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(in_attrs[i]), ctypes.sizeof(RKNNTensorAttr)), "RKNN_QUERY_INPUT_ATTR")
  inputs = (RKNNInput * io.n_input)()
  for i, (addr, size) in enumerate(in_addrs_sizes):
    input_type = _float_type(size, in_attrs[i].n_elems, f"RKNN input {i}")
    inputs[i] = RKNNInput(i, addr, size, 0, input_type, in_attrs[i].fmt)
  _check(lib.rknn_inputs_set(ctx, io.n_input, inputs), "rknn_inputs_set")
  _check(lib.rknn_run(ctx, None), "rknn_run")
  out_attr = RKNNTensorAttr(); out_attr.index = 0
  _check(lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(out_attr), ctypes.sizeof(RKNNTensorAttr)), "RKNN_QUERY_OUTPUT_ATTR")
  output_type = _float_type(len(out_buf), out_attr.n_elems, "RKNN output")
  output = RKNNOutput(output_type == RKNN_TENSOR_FLOAT32, 0, 0, None, 0)
  _check(lib.rknn_outputs_get(ctx, 1, ctypes.byref(output), None), "rknn_outputs_get")
  try: ctypes.memmove(_addr(out_buf), output.buf, len(out_buf))
  finally: _check(lib.rknn_outputs_release(ctx, 1, ctypes.byref(output)), "rknn_outputs_release")

class RKProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs):
    self.device, self.name, self.lib = device, name, lib
    self.meta = getattr(lib, "meta", None)
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(),
               wait=False, **kw):
    with cpu_profile(self.name, self.device):
      lib = _load_rknn()
      if self.meta is not None and self.meta.get("kind") == "matmul":
        self._run_matmul(lib, bufs)
      else:
        ctx = _rknn_init(lib, self.lib)
        try: _rknn_infer(lib, ctx, [(_addr(b), len(b)) for b in bufs[1:]], bufs[0])
        finally: _check(lib.rknn_destroy(ctx), "rknn_destroy")
    return 1e-3 if wait else None

  def _run_matmul(self, lib, bufs):
    # out[M,N] = a[M,K] @ b[K,N], run as K MULACC steps acc = A[:,k]*B[k,:] + acc on the
    # official rknn runtime -- the rknn-decode matmul decomposition. The K steps reuse one
    # loaded model (a single load_rknn + init_runtime). The multi-input chain model is fed
    # via the toolkit inference() entry of the official runtime (it normalizes the 3 inputs);
    # the single-op element-wise path uses the faster raw librknnrt C API.
    import numpy as np, os, tempfile, shutil
    from rknn.api import RKNN
    M, K, N = self.meta["M"], self.meta["K"], self.meta["N"]
    out_buf, a_buf, b_buf = bufs[0], bufs[1], bufs[2]
    out_fp32 = len(out_buf) == M*N*4
    A = np.frombuffer(bytes(a_buf), dtype=(np.float32 if len(a_buf)==M*K*4 else np.float16)).astype(np.float16).reshape(M, K)
    B = np.frombuffer(bytes(b_buf), dtype=(np.float32 if len(b_buf)==K*N*4 else np.float16)).astype(np.float16).reshape(K, N)
    acc = np.zeros(M*N, dtype=np.float16)
    work = tempfile.mkdtemp(prefix="rk_mm_", dir=("/dev/shm" if os.path.isdir("/dev/shm") else None))
    path = os.path.join(work, "step.rknn")
    with open(path, "wb") as f: f.write(bytes(self.lib))
    rk = RKNN(verbose=False)
    try:
      _check(rk.load_rknn(path), "load_rknn")
      # single core (mask 0): the chained multi-tile MULACC model is data-dependent across
      # tiles and is corrupted by the 3-core split, so don't use RKNN_NPU_CORE_0_1_2 here.
      _check(rk.init_runtime(target=os.getenv("RKNN_TARGET", "rk3588"), core_mask=0), "init_runtime")
      for k in range(K):
        col = np.repeat(A[:, k], N).astype(np.float16)   # A[:,k] broadcast over N cols
        row = np.tile(B[k], M).astype(np.float16)         # B[k,:] tiled over M rows
        acc = np.asarray(rk.inference(inputs=[col, row, acc])[0]).reshape(-1)[:M*N].astype(np.float16)
    finally:
      rk.release(); shutil.rmtree(work, ignore_errors=True)
    res = np.ascontiguousarray(acc.astype(np.float32) if out_fp32 else acc)
    ctypes.memmove(_addr(out_buf), _addr(res), len(out_buf))

class RKAllocator(Allocator['RKDevice']):
  def _alloc(self, size, options): return bytearray(size)
  def _copyin(self, dest, src:memoryview): ctypes.memmove(_addr(dest), mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src): ctypes.memmove(mv_address(dest), _addr(src), dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return memoryview(buf)[offset:offset+size]

class RKDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, RKAllocator(self), [RKRenderer], functools.partial(RKProgram, device))
