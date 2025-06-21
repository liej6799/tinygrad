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
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport, OSX
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQSignal, HCQProgram, FileIOInterface
from tinygrad.helpers import from_mv, getenv, round_up, mv_address, to_mv, cpu_objdump, DEBUG
import ctypes, os, mmap, tempfile, pathlib, array, functools, threading, contextlib, sys, subprocess, struct
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
import time
import pickle, base64, itertools, time, struct, sys
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.runtime.ops_cpu import ClangJITCompiler


class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:list[str]|None=None, objdump_tool='objdump'):
    self.args = ['-shared'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output([getenv("CC", 'clang'), *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      print('output_file', output_file)
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)
  
class RKNNCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:list[str]|None=None, objdump_tool='objdump'):
    self.args = ['-shared'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output([getenv("CC", 'clang'), *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-L/usr/lib/librknnrt.so', '-lstdc++', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)


class RKNNRenderer(ClangRenderer):
  device = "RKNN"

class RKNNBuffer:
  def __init__(self, buf:Any, virt_addr:int, size:int, malloc, offset:int=0):
    self.buf, self.virt_addr, self.size, self.malloc, self.offset = buf, virt_addr, size, malloc, offset




class RKNNDevice(Compiled):
  devices: ClassVar[list[HCQCompiled]] = []
  signal_pages: ClassVar[list[Any]] = []
  signal_pool: ClassVar[list[int]] = []
  
  def __init__(self, device:str=""):

    print('RKNN device')
    self.ctx = ct.c_ulong()
    self.custom_ctx = ct.c_ulong()
    #/root/dev/tinygrad/extra/rockchip/mobilenet_v1.rknn

    py_str = '/root/dev/rk3588/rknn-demo/dual_residual_custom.rknn'

    # Convert to bytes
    byte_str = py_str.encode('utf-8')

    # Create c_char_p
    c_str = ctypes.c_char_p(byte_str)


    rknn.rknn_init(self.custom_ctx, c_str, 0, 0, None)

    ROW_A = 1
    COL_A = 32
    COL_B = 32



    info = rknn.struct_rknn_matmul_info_t()
    ct.memset(ct.byref(info), 0, ct.sizeof(rknn.struct_rknn_matmul_info_t))
    info.M = ROW_A
    info.K = COL_A
    info.N = COL_B
    info.type = rk.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
    info.B_layout = 0
    info.AC_layout = 0

    self.io_attr = rknn.struct__rknn_matmul_io_attr()
    ct.memset(ct.byref(self.io_attr), 0, ct.sizeof(rknn.struct__rknn_matmul_io_attr))
    rknn.rknn_matmul_create(self.ctx, info, self.io_attr)
    rknn.rknn_matmul_set_core_mask(self.ctx, rknn.RKNN_NPU_CORE_0_1_2)

    super().__init__(device, MallocAllocator, RKNNRenderer(), ClangJITCompiler(),  functools.partial(RKNNProgram, self))


class RKNNProgram:
  def __init__(self, dev:RKNNDevice, name:str, lib:bytes):
    print('name', name) 
    self.dev = dev
    liba = ClangJITCompiler().compile("""
    #include <stdio.h>
    #include "rknn_api.h"
    #include "rknn_custom_op.h"

    int compute_custom_sigmoid_float32(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                                    rknn_custom_op_tensor* outputs, uint32_t n_outputs)
{

    return 0;
}
    """)
    print('liba', liba)
    print('name', name)
    from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
    # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
    # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
    self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)
    self.mem.write(lib)


    custom_op = (rknn.rknn_custom_op * 1)()
    ct.memset(ct.byref(custom_op), 0, ct.sizeof(rknn.rknn_custom_op))   

    print('size: ',  ct.sizeof(rknn.rknn_tensor_attr))

    self.io_num = rknn.rknn_input_output_num()
    rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_IN_OUT_NUM, ct.byref(self.io_num), ct.sizeof(self.io_num))

    custom_string = rknn.rknn_custom_string()
    rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_CUSTOM_STRING, ct.byref(custom_string), ct.sizeof(custom_string))
    print(custom_string)

    self.input_attrs= (rknn.rknn_tensor_attr * self.io_num.n_input)()
    ct.memset(self.input_attrs, 0, self.io_num.n_input * ct.sizeof(rknn.rknn_tensor_attr))

    output_attrs= (rknn.rknn_tensor_attr * self.io_num.n_input)()
    ct.memset(output_attrs, 0, self.io_num.n_input * ct.sizeof(rknn.rknn_tensor_attr))

    self.inputs = (rknn.rknn_input * self.io_num.n_input)()
    ct.memset(self.inputs, 0, self.io_num.n_input * ct.sizeof(rknn.rknn_input))
    self.outputs = (rknn.rknn_output * self.io_num.n_output)()
    ct.memset(self.outputs, 0, self.io_num.n_output * ct.sizeof(rknn.rknn_output))


    for i in range(self.io_num.n_input):
      self.input_attrs[i].index = i
      output_attrs[i].index = i
      ret = rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_INPUT_ATTR, ct.byref(self.input_attrs[i]), ct.sizeof(rknn.rknn_tensor_attr))
      ret = rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_OUTPUT_ATTR, ct.byref(output_attrs[i]), ct.sizeof(rknn.rknn_tensor_attr))

    
    # Copy string into op_type buffer
    dest_ptr = ct.cast(custom_op[0].op_type, ct.c_char_p)
    libc.strncpy(dest_ptr, b"cstDualResidual", 256 - 1)
    custom_op[0].version = 1
    custom_op[0].target  = rknn.RKNN_TARGET_TYPE_CPU

    self.fxn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(rknn.struct__rknn_custom_op_context), ctypes.POINTER(rknn.struct__rknn_custom_op_tensor), ctypes.c_uint32, ctypes.POINTER(rknn.struct__rknn_custom_op_tensor), ctypes.c_uint32)(mv_address(self.mem))
    custom_op[0].compute = self.fxn
    reg_custom_op = rknn.rknn_register_custom_ops(self.dev.custom_ctx, custom_op, 1)

    print('reg_custom_op', reg_custom_op)

   
      
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    # print([i.malloc for i in bufs])
    # args = list([i.malloc for i in bufs]) 
    # print('args: ', args)
    # print("run cpu", self.fxn)
    # print(self.fxn(*args))
    # args = list(bufs) 
    # print('args: ', args)
    # print("run cpu", self.fxn)
    # print(self.fxn(*bufs))
    print('local_size', local_size)
    print('global_size', global_size)
    print('vals', vals)

    args = list(bufs)

    for i in range(10):
      print(args[1][i])


    # rknn.rknn_matmul_set_io_mem(self.dev.ctx, bufs[2].buf, self.dev.io_attr.A)
    # rknn.rknn_matmul_set_io_mem(self.dev.ctx, bufs[1].buf, self.dev.io_attr.B)
    # rknn.rknn_matmul_set_io_mem(self.dev.ctx, bufs[0].buf, self.dev.io_attr.C)
    # rknn.rknn_matmul_run(self.dev.ctx)

    # for i in range(10):

    #   print(bufs[1].malloc)


#  # try input fake 
#     print('self.io_num.n_input', self.input_attrs[0].n_dims)
#     input_data = (ct.c_void_p * self.io_num.n_input)()
#     for i in range(self.io_num.n_input):
#     #  buf = ct.create_string_buffer(self.input_attrs[i].n_elems * ct.sizeof(ct.c_uint16))
#       for j in range(self.input_attrs[0].n_dims):
        
#         buf = (ctypes.c_float * 1)(1.0)
        
#         print('buf', buf)
#         input_data[i][j] = ct.cast(buf, ct.c_void_p)
    size = 16777216

    # Define a ctypes array of floats
    FloatArray = ctypes.c_float * size

    # Allocate and initialize 'test' array
    test = FloatArray()
    for i in range(size):
        test[i] = 10.0

    # Allocate 'data' array
    data = FloatArray()

    # Copy memory from 'test' to 'data' (like memcpy)
    ctypes.memmove(data, test, ctypes.sizeof(FloatArray))


    byte_data = ctypes.cast(data, ctypes.c_void_p)

    print(f"First float value: {data[0]:.6f}")

    # input_data[0] = (args[1])
    # input_data[1] = (args[2])

    for i in range(self.io_num.n_input):
      self.inputs[i].index = i
      self.inputs[i].pass_through = 0
      self.inputs[i].type = rk.RKNN_TENSOR_FLOAT32
      self.inputs[i].fmt = rknn.RKNN_TENSOR_UNDEFINED
      self.inputs[i].size = size
      self.inputs[i].buf = byte_data

    print(' self.input_attrs[i].fmt',  self.input_attrs[i].fmt)
    print('self.inputs[i].size', self.inputs[i].size)
    rknn.rknn_inputs_set(self.dev.custom_ctx, self.io_num.n_input, self.inputs)
    rknn.rknn_run(self.dev.custom_ctx, None)

  # io_num = rknn.rknn_input_output_num()
  #   rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_IN_OUT_NUM, ct.byref(io_num), ct.sizeof(io_num))

    perf_run = rknn.rknn_perf_run()
    ret = rknn.rknn_query(self.dev.custom_ctx, rknn.RKNN_QUERY_PERF_RUN, ct.byref(perf_run), ct.sizeof(perf_run));

    print('run_duration', perf_run.run_duration)
    for i in range(self.io_num.n_output):
      self.outputs[i].index = i
      self.outputs[i].want_float = 1
      self.outputs[i].is_prealloc = 0

    rknn.rknn_outputs_get(self.dev.custom_ctx, self.io_num.n_output, self.outputs, None)

    print(self.inputs[0].buf)
    print(self.outputs[0].buf)
    # Cast void_ptr to c_char_p (pointer to char)
    float_ptr_1 = ct.cast(self.inputs[0].buf, ct.POINTER(ct.c_float))
    float_ptr_2 = ct.cast(self.outputs[0].buf, ct.POINTER(ct.c_float))


    print('value:', float_ptr_1[0])
    print('value:', float_ptr_2[0])





class RKNNAllocator(Allocator):
  def __init__(self, dev:RKNNDevice):
    self.dev = dev
    super().__init__()    
  dev = None
  def _as_buffer(self, src:RKNNBuffer) -> memoryview: return to_mv(src.virt_addr, src.size)
  def _alloc_aligned(self, size:int, alignment:int):
    buffer = (ctypes.c_uint8 * (size + alignment))()
    offset = round_up(ctypes.addressof(buffer), alignment) - ctypes.addressof(buffer)
    return (ctypes.c_uint8 * size).from_buffer(buffer, offset)
  def _alloc(self, size, options): 
    buf = rknn.rknn_create_mem(self.dev.ctx, size)
    alignment = 0x1000 if size >= 0x1000 else 0x20
    malloc = (ctypes.c_uint8 * size).from_address(options.external_ptr) if options.external_ptr else self._alloc_aligned(size, alignment)

    print('options', options)

    return RKNNBuffer(buf, buf.contents.virt_addr, size, malloc)  # Placeholder for actual buffer allocation logic)
    
  def _copyin(self, dest:RKNNBuffer, src:memoryview): 
    print('copyin')
    ct.memmove(dest.virt_addr, from_mv(src), src.nbytes)
    
  def _copyout(self, dest:memoryview, src:RKNNBuffer): 
    print('copyout')
    ct.memmove(from_mv(src), dest.virt_addr, dest.size)

  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): 
    print('transfer')
    pass    

class RKNNGraph(MultiGraphRunner):
  def __call__(self, input_rawbuffers, var_vals, wait=False) -> float|None: return 1e-3

