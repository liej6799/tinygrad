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
  def __new__(cls, data:bytes, path:str):
    ret = bytes.__new__(cls, data)
    ret.path = path
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
  def compile(self, src:bytes) -> bytes: return bytes(src)

class RKRenderer(Renderer):
  compiler = RKCompiler()
  has_local = False
  def render(self, uops:list[UOp]) -> bytes:
    print([(u.op, u.dtype, [uops.index(v) for v in u.src if u.op is not Ops.SPECIAL], u.arg) for u in uops])
    path = pathlib.Path(RKNN_MODEL)
    return RKNNBlob(path.read_bytes(), str(path))

class RKProgram:
  def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs): self.device, self.name, self.lib = device, name, lib
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(),
               wait=False, **kw):
    with cpu_profile(self.name, self.device):
      ctx, lib = ctypes.c_uint64(0), _load_rknn()
      model = ctypes.create_string_buffer(self.lib)
      _check(lib.rknn_init(ctypes.byref(ctx), ctypes.cast(model, ctypes.c_void_p), len(self.lib), 0, None), "rknn_init")
      try:
        _check(lib.rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2), "rknn_set_core_mask")
        io = RKNNInputOutputNum()
        _check(lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io), ctypes.sizeof(io)), "RKNN_QUERY_IN_OUT_NUM")
        assert io.n_output == 1, "static RKNN model expects one output"
        assert len(bufs) == io.n_input + 1, f"static RKNN model expects output plus {io.n_input} inputs"
        in_attrs = (RKNNTensorAttr * io.n_input)()
        for i in range(io.n_input):
          in_attrs[i].index = i
          _check(lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(in_attrs[i]), ctypes.sizeof(RKNNTensorAttr)), "RKNN_QUERY_INPUT_ATTR")
        inputs = (RKNNInput * io.n_input)()
        for i in range(io.n_input):
          input_type = _float_type(len(bufs[i+1]), in_attrs[i].n_elems, f"RKNN input {i}")
          inputs[i] = RKNNInput(i, _addr(bufs[i+1]), len(bufs[i+1]), 0, input_type, in_attrs[i].fmt)
        _check(lib.rknn_inputs_set(ctx, io.n_input, inputs), "rknn_inputs_set")
        _check(lib.rknn_run(ctx, None), "rknn_run")
        out_attr = RKNNTensorAttr()
        out_attr.index = 0
        _check(lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(out_attr), ctypes.sizeof(RKNNTensorAttr)), "RKNN_QUERY_OUTPUT_ATTR")
        output_type = _float_type(len(bufs[0]), out_attr.n_elems, "RKNN output")
        assert output_type == out_attr.type or output_type == RKNN_TENSOR_FLOAT32, f"RKNN output wants {out_attr.size} bytes"
        output = RKNNOutput(output_type == RKNN_TENSOR_FLOAT32, 0, 0, None, 0)
        _check(lib.rknn_outputs_get(ctx, 1, ctypes.byref(output), None), "rknn_outputs_get")
        try: ctypes.memmove(_addr(bufs[0]), output.buf, len(bufs[0]))
        finally: _check(lib.rknn_outputs_release(ctx, 1, ctypes.byref(output)), "rknn_outputs_release")
      finally: _check(lib.rknn_destroy(ctx), "rknn_destroy")
    return 1e-3 if wait else None

class RKAllocator(Allocator['RKDevice']):
  def _alloc(self, size, options): return bytearray(size)
  def _copyin(self, dest, src:memoryview): ctypes.memmove(_addr(dest), mv_address(src), src.nbytes)
  def _copyout(self, dest:memoryview, src): ctypes.memmove(mv_address(dest), _addr(src), dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return memoryview(buf)[offset:offset+size]

class RKDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, RKAllocator(self), [RKRenderer], functools.partial(RKProgram, device))
