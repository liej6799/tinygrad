import platform
from tinygrad.helpers import init_c_var, from_mv, init_c_struct_t, getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.runtime.autogen import rockchip as rk
import ctypes as ct 
import os, mmap, functools
import fcntl
from typing import Any, cast, ClassVar
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport
from tinygrad.runtime.autogen import libc
class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  def render(self, uops:list) -> str: 
    print('renderer')
    return ""

class RockchipBuffer:
  def __init__(self, va_addr:int, fd:int, size:int, offset:int=0):
    print('rockchip buffer')
    self.va_addr, self.size, self.offset = va_addr, size, offset



class RockchipDevice(Compiled):

  def __init__(self, device:str=""):
    print('rockchip device')
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.dma_ctl = FileIOInterface(f"/dev/dma_heap/system", os.O_RDWR)
    
    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), Compiler(), functools.partial(RockchipProgram, self))


class RockchipProgram(Compiled):
  def __init__(self, dev:MetalDevice, name:str, lib:bytes):
    print('rockchip program')
    self.dev, self.name, self.lib = dev, name, lib

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    print('call', global_size, local_size)
    return 1e-4


class RockchipAllocator(Allocator):
  def __init__(self, dev:RockchipDevice):
    self.dev = dev
    super().__init__()    
  dev = None

  def _alloc(self, size, options): 
    import os
    print('alloc', size)    
    # mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.dev.fd_ctl, size=size, flags=rk.RKNPU_MEM_NON_CACHEABLE)
    # mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.dev.fd_ctl, handle=mem_create.handle, offset=0)
    # va_addr = self.dev.fd_ctl.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)
    
    #buf_data = rk.struct_dma_heap_allocation_data()
    
    # ct.memset(ct.byref(buf_data), 0, ct.sizeof(buf_data))

    ret = rk.DMA_HEAP_IOCTL_ALLOC(self.dev.dma_ctl, len = size, fd_flags = os.O_CLOEXEC | os.O_RDWR)
  
    #self.dev.fd_ctl.mmap(None, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, buf_data.fd, 0)
    va_addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, ret.fd, 0)

    return RockchipBuffer(va_addr, ret.fd, size, offset=0)
    pass
  def _copyin(self, dest, src:memoryview): 
    print('copyin')
    # uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW  
    # print(my_uint64)
    rk.DMA_BUF_IOCTL_SYNC(self.dev.dma_ctl, data =ct.c_uint64( rk.DMA_BUF_SYNC_START | rk.DMA_BUF_SYNC_RW))
    
    # ct.memmove(dest, from_mv(src), src.nbytes)
    
  def _copyout(self, dest:memoryview, src): 
    print('copyout')
    rk.DMA_BUF_IOCTL_SYNC(self.dev.dma_ctl, data =ct.c_uint64( rk.DMA_BUF_SYNC_END | rk.DMA_BUF_SYNC_RW))
    # rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.dev.fd_ctl, obj_addr=src, flags=rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    # ct.memmove(src, from_mv(dest), dest.nbytes)


  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): 
    print('transfer')
    pass    

class RockchipGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

