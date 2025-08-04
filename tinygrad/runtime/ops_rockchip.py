import platform
from tinygrad.helpers import init_c_var, from_mv, init_c_struct_t, getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
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
class RockchipRenderer(Renderer):
  device = "ROCKCHIP" 
  def render(self, uops:list) -> str:
    print('renderer')
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

    super().__init__(RockchipArgsState, self.dev, self.name, kernargs_alloc_size=1)

class RockchipArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:RockchipProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)
    # Store the buffers and values for easy access
    self.bufs = bufs
    self.vals = vals
    print('rockchip args state', self.bufs, self.vals)
    
    self.output = self.bufs[0].meta.dma_addr
    self.input = self.bufs[1].meta.dma_addr
    self.weights = self.bufs[2].meta.dma_addr
    self.dev = self.prg.dev
    print('rockchip args state', self.output, self.input, self.weights)
    



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

  def timestamp(self, signal:RockchipSignal): 
    print('timestamp')
    return self.signal(signal, 0)

  def exec(self, prg:RockchipProgram, args_state:RockchipArgsState, global_size, local_size):
    print('exec123')
    self.bind_args_state(args_state)

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

    self.q(0x10010000000e4004), # 0
    self.q(0x20010000000e5004), # 1
    self.q(0x1001000001e5400c), # 2
    self.q(0x1001480000024010), # 3
    self.q(0x1001000000004014), # 4
    self.q(0x1001000000004020), # 5
    self.q(0x1001000000c04024), # 6
    self.q(0x1001000000094030), # 7
    self.q(0x1001000000004034), # 8
    self.q(0x1001000000004038), # 9
    self.q(0x100100070007403c), # 10
    self.q(0x1001000000534040), # 11
    self.q(0x1001000000004044), # 12
    self.q(0x1001000000004048), # 13
    self.q(0x100100000000404c), # 14
    self.q(0x1001000000024050), # 15
    self.q(0x1001000000004054), # 16
    self.q(0x1001000000074058), # 17
    self.q(0x100100000009405c), # 18
    self.q(0x1001000000534060), # 19
    self.q(0x1001000000004064), # 20
    self.q(0x1001000000004068), # 21
    self.q(0x100100000000406c), # 22
    self.q(0x1001108202c04070), # 23
    self.q(0x1001000000004074), # 24
    self.q(0x1001000000014078), # 25
    self.q(0x100100000000407c), # 26
    self.q(0x1001000000004080), # 27
    self.q(0x1001000100014084), # 28
    self.q(0x1001000000004088), # 29
    self.q(0x1001000000004090), # 30
    self.q(0x1001000000004094), # 31
    self.q(0x1001000000004098), # 32
    self.q(0x100100000000409c), # 33
    self.q(0x10010000000040a0), # 34
    self.q(0x10010000000040a4), # 35
    self.q(0x10010000000040a8), # 36
    self.q(0x10010000000040ac), # 37
    self.q(0x1001000000c040c0), # 38
    self.q(0x10010000000040c4), # 39
    self.q(0x1001000000004100), # 40
    self.q(0x1001000000004104), # 41
    self.q(0x1001000000004108), # 42
    self.q(0x100100000000410c), # 43
    self.q(0x1001000000004110), # 44
    self.q(0x1001000000004114), # 45
    self.q(0x1001000000004118), # 46
    self.q(0x100100000000411c), # 47
    self.q(0x1001000000004120), # 48
    self.q(0x1001000000004124), # 49
    self.q(0x1001000000004128), # 50
    self.q(0x100100000000412c), # 51
    self.q(0x200100000009500c), # 52
    self.q(0x2001000000005010), # 53
    self.q(0x2001000000075014), # 54
    self.q(0x2001000000005018), # 55
    self.q(0x200100000000501c), # 56
    self.q(0x2001000000005020), # 57
    self.q(0x2001000000005028), # 58
    self.q(0x200100000000502c), # 59
    self.q(0x2001400000085034), # 60
    self.q(0x2001000000005038), # 61
    self.q(0x2001000000c05040), # 62
    self.q(0x2001000178495044), # 63
    self.q(0x2001000000005048), # 64
    self.q(0x200100000020504c), # 65
    self.q(0x2001000000005064), # 66
    self.q(0x2001010101015068), # 67
    self.q(0x200100000020506c), # 68
    self.q(0x0000000000000000), # 69
    self.q(0x0101000000000014), # 70
    self.q(0x0041000000000000), # 71
    self.q(0x0081000000180008), # 72
    
    self._q[55] = self._q[55] |  ((args_state.input & 0xFFFFFFFF) << 16)
    self._q[61] = self._q[61] | ((args_state.weights & 0xFFFFFFFF) << 16)
    self._q[5] = self._q[5] | ((args_state.output & 0xFFFFFFFF) << 16)


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

