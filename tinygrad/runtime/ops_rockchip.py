import platform
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.runtime.autogen import rockchip as rk
import ctypes as ct 
import os, mmap
import fcntl
from tinygrad.runtime.support.hcq import HCQBuffer
from typing import Any, cast, ClassVar
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport
class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  def render(self, uops:list) -> str: return ""

class rknn_sdk_version(ct.Structure):
    _fields_ = [("version", ct.c_char * 64)]
    

class RockchipProgram:
  def __init__(self, name:str, lib:bytes): 
   
    pass
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    return 1e-4

class RockchipDevice(Compiled):
  def get_struct(self, argp, stype):
    return ct.cast(ct.c_void_p(argp), ct.POINTER(stype)).contents


  def __init__(self, device:str): 

    # self.ctx = ct.c_ulong(0)
    # self.ctx_ptr = ct.pointer(self.ctx)

    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    hw_version = rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, type=rk.RKNPU_GET_DRV_VERSION, flags=rk.RKNPU_GET_HW_VERSION)
    # version = rk.DRM_IOCTL_VERSION(self.fd_ctl)
    print("HW Version: ", hw_version.value)

    # fd = os.open(f"/dev/dri/card1", os.O_RDWR)
    
    
    # Hook LIBC
    # libc = ct.CDLL(ct.util.find_library("c"))
    # processor = platform.processor()
    # IOCTL_SYSCALL = {"aarch64": 0x1d, "x86_64":16}[processor]

    # print(fd_ctl)

    # ret = fcntl.ioctl(fd, 3221775424, my_struct)
    # print(my_struct.flags)
    # print(my_struct.value)
    # # hook ioctl

    # fd = os.open("/dev/dri/card1", os.O_RDWR)
    # # ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
    # data = self.get_struct( rk.struct_rknpu_mem_create)

    
    # import fcntl
    # # ret = libc.syscall(IOCTL_SYSCALL, ct.c_int(fd), rk.IOCTL_RKNPU_MEM_CREATE, data)
 
    
    # # rk.DRM_IOCTL_RKNPU_ACTION(fd_ctl, type=rk.RKNPU_ACT_RESET, value = ct.addressof(my_struct), sizebytes=ct.sizeof(my_struct))

    # with open('/root/tinygrad/extra/rockchip/mobilenet_v1.rknn', 'rb') as f:
    #   file_data = f.read()
    #   rk.rknn_init(self.ctx_ptr, ct.cast(ct.create_string_buffer(file_data + b'\0'), ct.c_char_p), len(file_data), 0, None)
      
    #   sdk_ver = rk.rknn_sdk_version()
    #   rk.rknn_query(self.ctx, rk.RKNN_QUERY_SDK_VERSION, ct.byref(sdk_ver), ct.sizeof(sdk_ver))
    #   print(f"SDK Version: {sdk_ver.api_version}")
    #   rk.rknn_run(self.ctx, None)

    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), Compiler(), RockchipProgram, RockchipGraph)

class RockchipTextureInfo:
  def __init__(self, pitch:int, real_stride:int, desc:list[int], ibo:list[int]):
    self.pitch, self.real_stride, self.desc, self.ibo = pitch, real_stride, desc, ibo

class RockchipAllocator(LRUAllocator):
  def __init__(self, dev:RockchipDevice):
    self.dev = dev
    super().__init__()  
  dev = None
  def _alloc(self, size, options): 
    print(size)

    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.dev.fd_ctl, size=size, flags= rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.dev.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.dev.fd_ctl.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)
    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_map)

  def _do_copy(self, src_addr, dest_addr, src_size, real_size, src_stride, dest_stride, dest_off=0, src_off=0):
    while src_off < src_size:
      ct.memmove(dest_addr+dest_off, src_addr+src_off, real_size)
      src_off, dest_off = src_off+src_stride, dest_off+dest_stride

  def _copyin(self, dest, src:memoryview): 
    stride, pitch = (src.nbytes, src.nbytes) if (ti:=cast(RockchipTextureInfo, dest.texture_info)) is None else (ti.real_stride, ti.pitch)
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes, stride, stride, pitch)

  def _copyout(self, dest:memoryview, src): 
    self.dev.synchronize()
    stride, pitch = (src.size, src.size) if (ti:=cast(RockchipTextureInfo, src.texture_info)) is None else (ti.real_stride, ti.pitch)
    self._do_copy(src.va_addr, mv_address(dest), src.size, stride, pitch, stride)

  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): 
    print('transfer')
    pass    

class RockchipGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

