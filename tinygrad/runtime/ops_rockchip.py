# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
import array
import ctypes
import functools
import mmap
import os
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, struct, sys
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, mv_address, to_mv
from tinygrad.device import BufferSpec, Compiled, Compiler, Allocator
from tinygrad.codegen.opt import tc
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.uop.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk

def _load(m, i):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]

def load(inp, j=0):
  if len(inp) == 2: return [_load(m, x+j if x is not None else None) if gate else default for (m,x,gate),default in zip(*inp)]
  return [_load(m, x+j if x is not None else None) for m,x,_ in inp[0]]

def _store(m, i, v):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  def render(self, uops:list[UOp]) -> str:
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()



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


    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), RockchipCompiler(), functools.partial(RockchipProgram, self))
    

class RockchipProgram:

  def reg(self, val, shift, mask):
    return ((val) << shift) & mask;
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

   # ALU OPS IS HERE
    ew_cvt_type = 0
    ew_data_mode = 1
    ew_data_size = 2
    ew_relu_bypass = 1
    ew_lut_bypass = 1
    ew_alu_algo = 2
    # 4'd0: Max; 
    # 4'd1: Min; 
    # 4'd2: Add; 
    # 4'd3: Div; 
    # 4'd4: Minus; 
    # 4'd5: Abs; 
    # 4'd6: Neg; 
    # 4'd7: Floor; 
    # 4'd8: Ceil. 
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

  def __init__(self, dev:RockchipDevice, name:str, lib:bytes):
    self.uops: list[tuple[Ops, DType|None, list[int], Any]] = pickle.loads(lib)
    self.device = dev
    self.q = []
    print('enter init')

  

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      ul: dict[int, Any] = {}
      dl: dict[int, DType] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      loop_ends: dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.STORE}
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is Ops.ENDRANGE:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop is Ops.STORE:
          for j,val in enumerate(inp[1] if dtp[1].count > 1 else [inp[1]]):
            for (m,o,g),v in zip(inp[0], val):
              if g: _store(m, o+j, v)
          i += 1
          continue
        if uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert dtype.fmt is not None and isinstance(dtype, PtrDType)
          if TYPE_CHECKING or sys.version_info < (3, 12): assert dtype.fmt != "e"
          if uop is Ops.DEFINE_REG:
            # REGs are per thread
            ul[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(dtype.fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.DEFINE_GLOBAL else pbufs.pop(0)
            ul[i] = [buf.cast(dtype.fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0][0] == 'g': ul[i] = [idxs[2-int(arg[0][-1])]] * warp_size
          elif arg[0][0] == 'l': ul[i] = [x[2-int(arg[0][-1])] for x in warp]
        elif uop is Ops.CONST: ul[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(dtp[0], ImageDType):
            for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
              if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*dtp[0].shape[1]*4))
          else:
            for m,o in zip(inp[0], inp[1]): ret.append((m,o))
          ul[i] = [(m,o,g) for (m,o),g in zip(ret, inp[2] if len(inp) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          ul[i] = inp[0]
        elif uop is Ops.RANGE:
          if i not in ul: ul[i] = [0] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[0][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is Ops.VECTORIZE: ul[i] = inp
        elif uop is Ops.BITCAST:
          assert dtp[0].fmt and dtype.fmt
          pack_format, unpack_format = str(warp_size) + dtp[0].fmt, str(warp_size) + dtype.fmt
          ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
        elif uop is Ops.CAST:
          ul[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in inp[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            ul[i] = [load([inp[i][j] if i != 0 and dtp[i].count > 1 else inp[i] for i in range(len(inp))], j) for j in range(dtype.count)]
          else:
            ul[i] = load(inp)
        elif uop is Ops.GEP: ul[i] = inp[0][get_single_element(arg)]
      
        elif uop in GroupOp.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {uop}"
          assert all_same([dtype] + dtp) or uop in {Ops.CMPNE, Ops.CMPLT, Ops.WHERE}, f"dtype mismatch on {uop}"
          self.boilerplate()

          self.input_buf = self.device._gpu_alloc(len(inp[0]), 0)
          self.weight_buf = self.device._gpu_alloc(len(inp[1]), 0)
          self.output_buf = self.device._gpu_alloc(len(inp[0]), 0)
     
          src = memoryview(array.array('i', inp[0]))
          ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
          src2 = memoryview(array.array('i', inp[1]))
          ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)

          self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 
              self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
          self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
            self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
          self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
            self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
          
          self.submit()

    
          dst = memoryview(bytearray(self.output_buf.size))
          ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
          print('dst', list(dst))
          print('src', list(src))
          print('src2', list(src2))
          print([exec_alu(uop, dtype, p) for p in zip(*inp)])
     

          ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]
    
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

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

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)
