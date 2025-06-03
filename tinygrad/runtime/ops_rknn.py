import platform
from tinygrad.helpers import init_c_var, from_mv, init_c_struct_t, getenv
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.autogen import rknn
import ctypes as ct 
import os, mmap, functools
import fcntl
from typing import Any, cast, ClassVar
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport
from tinygrad.runtime.autogen import libc
class RKNNRenderer(Renderer):
  device = "RKNN"
  def render(self, uops:list) -> str: 
    print('renderer')
    return ""

class RKNNBuffer:
  def __init__(self, buf:Any, virt_addr:int, size:int, offset:int=0):
    self.buf, self.virt_addr, self.size, self.offset = buf, virt_addr, size, offset


class RKNNDevice(Compiled):

  def __init__(self, device:str="", ctx=None):
    print('RKNN device')
    self.ctx = ct.c_ulong()

    super().__init__(device, RKNNAllocator(self), RKNNRenderer(), Compiler(), functools.partial(RKNNProgram, self))


class RKNNProgram:
  def __init__(self, dev:RKNNDevice, name:str, lib:bytes):
    print('rknn program')

    MatmulKernelArray =  rk.struct_ggml_rknpu2_matmul_kernel * rk.GGML_RKNPU2_MAX_MATMUL_KERNELS
    self.matmul_kernels = MatmulKernelArray()

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    
    print(self.matmul_kernels[0].matmul_info)

    print('call', global_size, local_size)
    return 1e-4


class RKNNAllocator(Allocator):
  def __init__(self, dev:RKNNDevice):
    self.dev = dev
    super().__init__()    
  dev = None

  def _alloc(self, size, options): 
    buf = rknn.rknn_create_mem(self.dev.ctx, size)
    return RKNNBuffer(buf, buf.contents.virt_addr, size)  # Placeholder for actual buffer allocation logic)
    pass
  def _copyin(self, dest:RKNNBuffer, src:memoryview): 
    ct.memmove(dest.virt_addr, from_mv(src), src.nbytes)
    
  def _copyout(self, dest:memoryview, src:RKNNBuffer): 
    ct.memmove(from_mv(src), dest.virt_addr, dest.size)


  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): 
    print('transfer')
    pass    

class RKNNGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

