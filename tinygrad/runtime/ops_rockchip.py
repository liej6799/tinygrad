import platform
from types import SimpleNamespace
from tinygrad.helpers import init_c_var, from_mv, init_c_struct_t, getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.ops import Ops, UOp
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import BumpAllocator, HCQArgsState, HCQCompiled, HCQAllocatorBase, HCQBuffer, HWQueue, CLikeArgsState, HCQSignal, HCQProgram, FileIOInterface

import ctypes as ct 
import os, ctypes, functools, mmap, struct, array, math, sys

import os, mmap, functools
import fcntl
from typing import Any, cast, ClassVar
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport
from tinygrad.runtime.autogen import libc
from tinygrad.device import Compiled, ProfileEvent, BufferSpec, CPUProgram, PROFILE
import ctypes


class RDNACodegen:
  def __init__(self):
    print('RDNACodegen')
    self.rops: list[any] = []
  def lower_define_global(self, u: UOp):
    print('lower_define_global', u)
  def lower_special(self, u: UOp):
    print('lower_special', u)
  def lower_const(self, u: UOp):
    print('lower_const', u)
  def lower_add(self, u: UOp):
    print('lower_add', u)
  def lower_mul(self, u: UOp):
    print('lower_mul', u)
  def lower_index(self, u: UOp):
    print('lower_index', u)
  def lower_load(self, u: UOp):
    print('lower_load', u)
  def lower_store(self, u: UOp):
    print('lower_store', u)
  def lower_gep(self, u: UOp):
    print('lower_gep', u)
  def lower_const(self, u: UOp):
    print('lower_const', u)
  def lower_sink(self, u: UOp):
    print('lower_sink', u)
  


  def lower(self, uops:list[UOp]):
    for i,u in enumerate(uops):
      getattr(self, f'lower_{u.op.name.lower()}')(u)




class RockchipRenderer(Renderer):
  device = "ROCKCHIP" 
  def render(self, uops:list) -> str:
    codegen = RDNACodegen()
    codegen.lower(uops)
    print('renderer', uops)
    return "123"


class RockchipDevice(HCQCompiled):

  def __init__(self, device:str=""):
    print('rockchip device')
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.dma_ctl = FileIOInterface(f"/dev/dma_heap/system", os.O_RDWR)

    self.cmd_buf = self._gpu_alloc(1024, 0)
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING)

    renderer = RockchipRenderer()
    compiler = RockchipCompiler()
    runtime = functools.partial(RockchipProgram, self)

    super().__init__(device, RockchipAllocator(self), renderer, compiler, runtime, RockchipSignal, RockchipComputeQueue, RockchipCopyQueue)

  def _gpu_alloc(self, size:int, flags) -> HCQBuffer:
     
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)    
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)

    print('obj_addr', self.fd_ctl)
    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)


  def synchronize(self):
    """Synchronize the device."""
    print('rockchip synchronize')
    # In a full implementation, this would wait for all pending operations
    pass

  def _alloc_signal(self, value:int=0) -> 'RockchipSignal':
    """Allocate a signal with the given initial value."""
    return RockchipSignal(value=value)

  def signal_t(self) -> type['RockchipSignal']:
    """Return the signal type for this device."""
    return RockchipSignal


class RockchipSignal(HCQSignal):
  def __init__(self, base_addr:int|None=None, **kwargs):
    print('rockchip signal')
    super().__init__(base_addr, **kwargs, timestamp_divider=1000, dev_t=RockchipDevice)

  def __del__(self):
    pass

  def _sleep(self, time_spent_waiting_ms:int):
    # Resonable to sleep for long workloads (which take more than 2s) and only timeline signals.
    pass

  def wait(self, value:int, timeout:int=1000):
    """Wait for the signal to reach the specified value."""
    print(f'rockchip signal wait for {value}')
    # In a full implementation, this would wait for the hardware signal
    return self

  def _set_value(self, value:int):
    """Set the signal value."""
    print(f'rockchip signal set to {value}')
    # In a full implementation, this would set the hardware signal value
    return self 
  
class RockchipProgram(HCQProgram):
  def __init__(self, dev:RockchipDevice, name:str, lib:bytes):
    print('rockchip program')
    self.dev, self.name, self.lib = dev, name, lib
    self.buf_info, self.consts_info = [], []

    self.buf_info.append(SimpleNamespace(offset=0 ))
    self.buf_info.append(SimpleNamespace(offset=16 ))
    self.buf_info.append(SimpleNamespace(offset=32))


    super().__init__(RockchipArgsState, self.dev, self.name, kernargs_alloc_size=8)

class RockchipArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:RockchipProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)
    # Store the buffers and values for easy access
    self.bufs = bufs
    self.vals = vals

    self.bufs_info = []
    

    self.output = self.bufs[0].meta.dma_addr
    self.input = self.bufs[1].meta.dma_addr
    self.weights = self.bufs[2].meta.dma_addr

    for i, b in enumerate(bufs):
      self.bufs_info.append(SimpleNamespace(dma_addr = b.meta.dma_addr))



class RockchipComputeQueue(HWQueue):
  def memory_barrier(self):
    return self

  def wait(self, signal:RockchipSignal, value=0):
    return self

  def __init__(self, *args, **kwargs):
    print('rockchip compute queue', args, kwargs)
    super().__init__(*args, **kwargs)

  def _build_gpu_command(self, dev:RockchipDevice):
    

    regcmd_ptr = ctypes.cast(dev.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * (1024 // 8)))
    regcmd = regcmd_ptr.contents
    
    tasks_ptr = ctypes.cast(dev.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task))
    tasks = tasks_ptr.contents

    for i in range(len(self._q)):
      regcmd[i] = self._q[i]

    regcmd_dma = dev.cmd_buf.meta.dma_addr

    # Start of Selection
    # Define a ctypes Structure for rknpu_task if not already defined
    print('regcmd_dma', int(regcmd_dma))
  
    # Create and populate the rknpu_task instance
    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(npu_regs)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_dma

    return dev.task_buf.meta.obj_addr



  def _submit(self, dev:RockchipDevice):
    print('enter submit')
    # Process the queued commands and submit to hardware

    tasks = ctypes.cast(dev.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task* 128)).contents
    regcmd = ctypes.cast(dev.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * 128)).contents

    for i in range(len(self._q)):
      regcmd[i] = self._q[i]

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(self._q)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = dev.cmd_buf.meta.dma_addr

    submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=dev.task_buf.meta.obj_addr,   # Placeholder, would be actual address in real code
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

    res = rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd_ctl,   
            __payload=submit_res
    )
    print('res', res)    

    return self


  def signal(self, signal:RockchipSignal, value=0):
    print('signal', signal, value)
    return self


    
    # Return as ctypes 64-bit unsigned integer type
    return ctypes.c_uint64(result)
  def timestamp(self, signal:RockchipSignal): 
    print('timestamp')
    return self.signal(signal, 0)

  def exec(self, prg:RockchipProgram, args_state:RockchipArgsState, global_size, local_size):
    print('exec123')
    self.bind_args_state(args_state)
    def reg(val, shift, mask):
      return ((val) << shift) & mask;

    def emit_raw(target, reg, value):
        # Pack the values into a 64-bit integer as per hardware spec
        target = target + 0x1
        print('target', hex(target), 'reg', hex(reg), 'value', hex(value))
        packed_value = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF)
        print('packed_value', hex(packed_value))
        self.q(packed_value)



    def NPUOP(op, value, reg):
      print('op', hex(op), 'value', hex(value), 'reg', hex(reg))
      # Ensure inputs are within bit ranges by masking
      op_masked = op & 0xffff        # 16 bits
      value_masked = value & 0xffffffff  # 32 bits
      reg_masked = reg & 0xffff      # 16 bits
      
      # Construct the 64-bit integer as unsigned long long
      result = (ctypes.c_uint64(op_masked).value << 48) | \
              (ctypes.c_uint64(value_masked).value << 16) | \
              ctypes.c_uint64(reg_masked).value
      return result



    print('rockchip exec', global_size, local_size, prg)
    print("HCQBuffer values:", args_state.bufs)
    print("HCQBuffer values:", args_state.vals)
    
    # Access individual buffer properties
    for i, buf in enumerate(args_state.bufs):
        print(f"Buffer {i}: va_addr=0x{buf.va_addr:x}, size={buf.size}, meta={buf.meta.dma_addr}")
    # input_dma = args_state.bufs[0].meta.dma_addr
    # weights_dma = args_state.bufs[1].meta.dma_addr
    # output_dma = args_state.bufs[2].meta.dma_addr
    # Access buffer values (virtual addresses) for hardware commands
    buffer_addrs = [buf.va_addr for buf in args_state.bufs]
    print(f"Buffer addresses: {[f'0x{addr:x}' for addr in buffer_addrs]}")
    
    # Queue the execution command with buffer addresses

    # Sequence of hardware commands for execution
    print('output', args_state.output)
    print('prg', args_state.ptr)

    burst_len = 0xF
    conv_mode = rk.direct_convolution
    output_mode  = 0x2
    flying_mode = 0x1 # bypass CNA, directly to DPU (0x0 for default)
    channel= 7 # max is 7 channel
    dataout_width = 5
    dataout_height = 0

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

    emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
    reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
    reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
    reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

    emit_raw(rk.DPU, rk.DPU_FEATURE_MODE_CFG,
    reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
    reg(conv_mode, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__MASK) |
    reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
    reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    emit_raw(rk.DPU, rk.DPU_DATA_FORMAT,
    reg(rk.precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
    reg(rk.precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
    reg(rk.precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    emit_raw(rk.DPU, rk.DPU_DATA_CUBE_CHANNEL,
    reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
    reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))

    emit_raw(rk.DPU, rk.DPU_DATA_CUBE_WIDTH,
    reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))

    emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 
    reg(args_state.bufs_info[0].dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))

    emit_raw(rk.DPU, rk.DPU_EW_CFG,
    reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
    reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
    reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
    reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
    reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
    reg(ew_lut_bypass, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
    reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))


    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
    reg(args_state.bufs_info[1].dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
    reg(args_state.bufs_info[2].dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))

    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
    reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
    reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
    reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))

    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
    reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
    reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
    reg(3, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
    reg(3, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
    reg(conv_mode, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__MASK) |
    reg(flying_mode, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK) |
    
    reg(burst_len, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) | 
    reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__MASK))
    
    self.q(0x2001000178495044), # 63

    self.q(0x0081000000180008), # 72
    
    return self

  def bind(self, dev:RockchipDevice):
    """Bind the queue to a specific device for optimized execution."""
    self.binded_device = dev
    return self
    
class RockchipCopyQueue(HWQueue):
  def __init__(self, max_copy_size=0x40000000):
    super().__init__()

  def copy(self, dest, src, copy_size):
    """Enqueue a copy command."""
    print(f'rockchip copy {dest:x} <- {src:x} ({copy_size})')
    # Queue the copy command
   
    return self

  def _submit(self, dev:RockchipDevice):
    print('enter copy submit')


    return self

class RockchipCompiler(Compiler):
  def __init__(self):
    super().__init__("compile_rdna")
    pass
  def compile(self, src:str) -> bytes:
    pass
  def disassemble(self, lib:bytes):
    pass


class RockchipBuffer:
  def __init__(self, buf:Any, size:int):
    self.buf, self.size = buf, size

class RockchipAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, src_addr, dest_addr, src_size):
    ctypes.memmove(dest_addr, src_addr, src_size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()
    self._do_copy(src.va_addr, mv_address(dest), src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(cast(int, src.va_addr), src.size)

