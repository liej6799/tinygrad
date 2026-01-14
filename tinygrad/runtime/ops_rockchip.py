# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING, cast
import pickle, base64, itertools, time, struct, sys, functools, array, ctypes, mmap, os
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate, float_to_fp16, float_to_bf16, float_to_fp8, fp8_to_float
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, EMULATE, mv_address, to_mv
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair, BufferSpec
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.runtime.autogen import rockchip as rk

def storage_fmt_for_dtype(dtype: DType): return 'H' if dtype == dtypes.bfloat16 else 'B' if dtype in dtypes.fp8s else dtype.fmt

def to_storage_scalar(x, dtype: DType):
  if dtype == dtypes.half: return float_to_fp16(x)
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF
  if dtype in dtypes.fp8s: return float_to_fp8(float(x), dtype)
  return x

def from_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]
  if dtype in dtypes.fp8s: return fp8_to_float(int(x), dtype)
  return x

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return from_storage_scalar(m[i], dtype)

def load(inp, j, dtype: DType):
  if len(inp) == 2: return [_load(m, x+j if x is not None else None, dtype) if gate else default for (m,x,gate),default in zip(*inp)]
  return [_load(m, x+j if x is not None else None, dtype) for m,x,_ in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = to_storage_scalar(v, dtype)

# here are the models for the WMMA instruction on the different hardware
def generic_wmma_helper(inp, warp_size, WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
  for cc, tinp, num in zip(("A", "B", "C"), inp, (NUM_A, NUM_B, NUM_C)):
    assert len(tinp) == num, f"{cc} must have {num} elements per thread, it has {len(tinp)}"
    assert len(flatten(tinp)) == num * warp_size, f"WMMA must have {num * warp_size} total elements for {cc} in WMMA"
  assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
  out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
  for goff in range(0, warp_size, WARP_THREADS):
    for lane_id in range(WARP_THREADS):
      for elem_idx in range(NUM_C): # calculate new muls and add to acc
        (c_i, c_j) = c_map(lane_id, elem_idx)
        out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
  return out

class RockchipProgram:
  def reg(self, val, shift, mask):
    return ((val) << shift) & mask
  def emit_raw(self, target, reg, value):
    # Pack the values into a 64-bit integer as per hardware spec
    target = target + 0x1
    packed_value = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF)
    self.q.append(packed_value)
  def boilerplate(self):
    self.q = []
    burst_len = 0xF
    conv_mode = 0
    output_mode  = 0x2
    flying_mode = 0x1 # bypass CNA, directly to DPU (0x0 for default)
    channel = 7
    dataout_width = 5
    dataout_height = 0

    precision_int8 = 0
    precision_float16 = 2
    precision_int32 = 4
    precision_float32 = 5

    ew_cvt_type = 0
    ew_data_mode = 1
    ew_data_size = 2
    ew_relu_bypass = 1
    ew_lut_bypass = 1
    ew_alu_algo = 2
    ew_op_src = 1
    erdma_data_size_16bit=2

    self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(conv_mode, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__MASK) |
        self.reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
        self.reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(ew_lut_bypass, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        self.reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
        self.reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))

  def submit(self):
    self.q.append(0x2001000178495044), # 63
    self.q.append(0x0081000000180008), # 72
    tasks = ctypes.cast(self.device.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task* 128)).contents
    regcmd = ctypes.cast(self.device.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * 128)).contents
    for i in range(len(self.q)):
      regcmd[i] = self.q[i]

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(self.q)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = self.device.cmd_buf.meta.dma_addr

    submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=self.device.task_buf.meta.obj_addr,   # Placeholder, would be actual address in real code
            regcfg_obj_addr=0,
            task_base_addr=0,
            user_data=0,
            core_mask=1,
            fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
            )
    )
    res = rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl,
            __payload=submit_res
    )
    print(res)

  def __init__(self, dev:'RockchipDevice', name:str, lib:bytes):
    self.uops: list[tuple[Ops, DType, list[int], Any]] = pickle.loads(lib)
    self.device = dev
    self.q = []

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == Ops.END}
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[int, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, src_values, src_dtypes)
        if uop is Ops.END:
          i = srcs[1]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        if uop is Ops.STORE:
          for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
            for (m,o,g),v in zip(src_values[0], val):
              if g: _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if uop is Ops.AFTER: values[i] = src_values[0]
        elif uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType), dtype
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if uop is Ops.DEFINE_REG:
            # REGs are per thread
            values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.DEFINE_GLOBAL else pbufs.pop(0)
            values[i] = [buf.cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          values[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': values[i] = [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': values[i] = [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: values[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
              if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          values[i] = src_values[0]
        elif uop is Ops.RANGE:
          if i not in values: values[i] = [0] * warp_size
          else:
            for j in range(len(values[i])):
              values[i][j] += 1
          if values[i][0] == src_values[0][0]:
            del values[i]
            i = loop_ends[i] + 1
            continue
        elif uop is Ops.VECTORIZE: values[i] = src_values
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()),
                               *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
          values[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
          values[i] = [from_storage_scalar(x, dtype.scalar()) for x in values[i]]
        elif uop is Ops.CAST:
          values[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in src_values[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            values[i] = [load([src_values[i][j] if i != 0 and src_dtypes[i].count > 1 else src_values[i] \
                               for i in range(len(src_values))], j, dtype.scalar()) for j in range(dtype.count)]
          else:
            values[i] = load(src_values, 0, dtype)
        elif uop is Ops.GEP: values[i] = src_values[0][get_single_element(arg)]
        elif uop is Ops.WMMA:
          first_src_dtype = self.uops[srcs[0]][1]
          assert isinstance(first_src_dtype, DType) # mypy
          dims, dtype_in, device, threads = arg[1], first_src_dtype.scalar(), arg[4], arg[5]
          wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
          # TODO: refactor these to a shared TensorCoreLayout
          if device == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            values[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif device == "AMD" and threads == 64:
            def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            values[i] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
          elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            values[i] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif device == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            values[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if dims == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              values[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,32):
              def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
              values[i] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {arg}")
          elif device == "INTEL":
            # A (16 elements on 8 threads)
            def a_elem(x, k, row, goff): return x[k%2+row*2][goff+k//2]
            # B (16 elements on 8 threads)
            def b_elem(x, col, k, goff): return x[k][goff+col]
            # C, D (8 elements on 8 threads)
            def c_map(lane, elem): return (lane, elem)
            values[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CPU":
            def elem(x, col, row, _): return x[col+row][0] # k is always 0
            def c_map(lane, elem): return (elem%16, elem//16)
            values[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
          else: raise NotImplementedError(f"unimplemented tensor core {arg}")
        elif uop in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {uop}"
          assert all_same([dtype] + src_dtypes) or uop in {*GroupOp.Comparison, Ops.WHERE}, f"dtype mismatch on {uop}"
          if len(src_values) >= 2:
            self.boilerplate()

            self.input_buf = self.device._gpu_alloc(len(src_values[0]), 0)
            self.weight_buf = self.device._gpu_alloc(len(src_values[1]), 0)
            self.output_buf = self.device._gpu_alloc(len(src_values[0]), 0)

            src = memoryview(array.array('i', src_values[0]))
            ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
            src2 = memoryview(array.array('i', src_values[1]))
            ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)

            self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
                self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                         rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
              self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT,
                       rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
              self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT,
                       rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))

            self.submit()

            dst = memoryview(bytearray(self.output_buf.size))
            ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
            print('dst', list(dst))
            print('src', list(src))
            print('src2', list(src2))
            print([exec_alu(uop, dtype, p) for p in zip(*src_values)])

          values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
        assert i in values, (uop, dtype, srcs, arg)
        i += 1
    return time.perf_counter() - st

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  code_for_op = python_alu
  def render(self, uops:list[UOp]) -> str:
    # the value of SPECIAL comes from local/global_size, not form its source
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src if u.op is not Ops.SPECIAL], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class RockchipRegisterAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, src_addr, dest_addr, src_size):
    ctypes.memmove(dest_addr, src_addr, src_size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self._do_copy(src.va_addr, mv_address(dest), src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    return to_mv(ctypes.cast(int, src.va_addr), src.size)

class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RockchipDevice(Compiled):
  def _gpu_alloc(self, size:int, flags) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)

  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.cmd_buf = self._gpu_alloc(1024, 0)
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING)

    compilers = CompilerSet([CompilerPair(RockchipRenderer, RockchipCompiler)])
    super().__init__(device, RockchipAllocator(self), compilers, functools.partial(RockchipProgram, self))
