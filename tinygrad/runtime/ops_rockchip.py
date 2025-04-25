from tinygrad.device import Compiled, Compiler, Renderer, Allocator
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.runtime.autogen import rockchip as rk
import ctypes as ct 

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  def render(self, uops:list) -> str: return ""

class rknn_sdk_version(ct.Structure):
    _fields_ = [("version", ct.c_char * 64)]
    

class RockchipProgram:
  def __init__(self, name:str, lib:bytes): 
    ctx = ct.c_ulong(0)
    ctx_ptr = ct.pointer(ctx)
  
    with open('/root/tinygrad/extra/rockchip/mobilenet_v1.rknn', 'rb') as f:
      file_data = f.read()
      buffer = ct.create_string_buffer(file_data + b'\0')
      size = len(file_data)
      char_ptr = ct.cast(buffer, ct.c_char_p)
      rk.rknn_init(ctx_ptr, char_ptr, size, 0, None)

      sdk_ver = rk.rknn_sdk_version()
      rk.rknn_query(ctx, rk.RKNN_QUERY_SDK_VERSION, ct.byref(sdk_ver), ct.sizeof(sdk_ver))
      # SDK Verion 2.3.0 (c949ad889d@2024-11-07T11:35:33)

      # rk.rknn_query(ctx, rk.RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num))


      # Return SDK Version but in ASCII number.

      # print(ctx)
    # sdk_ptr = ct.pointer(rk.rknn_sdk_version)
    # rk.rknn_query(ctx, rk.RKNN_QUERY_SDK_VERSION, sdk_ptr, 1)
    
    # print(ctx.)
    pass
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    return 1e-4

class RockchipAllocator(Allocator):
  dev = None
  def _alloc(self, size, options): 
    # print("_alloc", size, options.image)	
    pass
  def _copyin(self, dest, src:memoryview): 

    pass
  def _copyout(self, dest:memoryview, src): pass
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): pass

class RockchipGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3


class RockchipDevice(Compiled):
  def __init__(self, device:str): 
    
    super().__init__(device, RockchipAllocator(), RockchipRenderer(), Compiler(), RockchipProgram, RockchipGraph)
