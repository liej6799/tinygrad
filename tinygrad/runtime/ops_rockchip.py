import platform
from tinygrad.device import Compiled, Compiler, LRUAllocator, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.runtime.autogen import rockchip as rk
import ctypes as ct 
import os

from tinygrad.runtime.support.hcq import FileIOInterface

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

    self.ctx = ct.c_ulong(0)
    self.ctx_ptr = ct.pointer(self.ctx)

    fd_ctl = os.open(f"/dev/dri/card1", os.O_RDWR)
    
    # Hook LIBC
    libc = ct.CDLL(ct.util.find_library("c"))
    # processor = platform.processor()
    # IOCTL_SYSCALL = {"aarch64": 0x1d, "x86_64":16}[processor]

    # hook ioctl

    # fd = os.open("/dev/dri/card1", os.O_RDWR)
    # # ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
    # data = self.get_struct(, rk.struct_rknpu_mem_create)

    

    # ret = libc.syscall(IOCTL_SYSCALL, ct.c_int(fd), rk.IOCTL_RKNPU_MEM_CREATE, data)

    # rk.IOCTL_RKNPU_ACTION(fd_ctl, flags=rk.RKNPU_GET_HW_VERSION)


    print(fd_ctl)


    # with open('/root/tinygrad/extra/rockchip/mobilenet_v1.rknn', 'rb') as f:
    #   file_data = f.read()
    #   rk.rknn_init(self.ctx_ptr, ct.cast(ct.create_string_buffer(file_data + b'\0'), ct.c_char_p), len(file_data), 0, None)
      
    #   sdk_ver = rk.rknn_sdk_version()
    #   rk.rknn_query(self.ctx, rk.RKNN_QUERY_SDK_VERSION, ct.byref(sdk_ver), ct.sizeof(sdk_ver))
    #   print(f"SDK Version: {sdk_ver.api_version}")
    #   rk.rknn_run(self.ctx, None)

    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), Compiler(), RockchipProgram, RockchipGraph)


class RockchipAllocator(LRUAllocator):
  def __init__(self, dev:RockchipDevice):
    self.dev = dev
    super().__init__()  
  dev = None
  def _alloc(self, size, options): 
    # return rk.rknn_create_mem(self.dev.ctx, size)
    pass
  def _copyin(self, dest, src:memoryview): 
    print(src)

    # io_num = rk.rknn_input_output_num()
    # rk.rknn_query(self.dev.ctx, rk.RKNN_QUERY_IN_OUT_NUM, ct.byref(io_num), ct.sizeof(io_num))

    # # input_attrs = rk.rknn_tensor_attr(3)
    # input_attrs = (rk.rknn_tensor_attr * (io_num.n_input))()
    # ct.memset(ct.addressof(input_attrs), 0, io_num.n_input * ct.sizeof(rk.rknn_tensor_attr))

    # for i in range(io_num.n_input):
    #   input_attrs[i].index = i
    #   rk.rknn_query(self.dev.ctx, rk.RKNN_QUERY_INPUT_ATTR, ct.addressof(input_attrs[i]), ct.sizeof(rk.rknn_tensor_attr))

    # rk.rknn_set_io_mem(self.dev.ctx, input_attrs, ct.addressof(input_attrs[0]))
    
    #print('copyin', ct.addressof(input_attrs))
    # print('copyin', ct.addressof(input_attrs[0]))
    pass
  def _copyout(self, dest:memoryview, src): 
    print('copyout')
    pass
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): 
    print('transfer')
    pass    

class RockchipGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

