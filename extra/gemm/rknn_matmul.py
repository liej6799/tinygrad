import os
os.environ["ROCKCHIP"] = "1"
import time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_rknn import RKNNDevice, RKNNAllocator, RKNNProgram
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
import ctypes as ct
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.autogen import rknn as rknn

N = getenv("N", 32)
LID = 2

device = RKNNDevice("RKNN")
rknnalloc = MallocAllocator

ROW_A = 1
COL_A = 32
COL_B = 32

a = rknnalloc.alloc(device.io_attr.A.size * 2)
b = rknnalloc.alloc(device.io_attr.B.size * 2)
c = rknnalloc.alloc(device.io_attr.C.size)

A = np.empty((ROW_A, COL_A), dtype=np.float32)
B = np.empty((COL_A, COL_B), dtype=np.float32)
C = np.empty((ROW_A, COL_B), dtype=np.float32)

A.fill(10.0)
B.fill(10.0)
print("data: ", A[0][0])

# copyin
rknnalloc._copyin(a, memoryview(bytearray(A)))
rknnalloc._copyin(b, memoryview(bytearray(B)))

compiler = device.compiler
prog = RKNNProgram(device, "cstDualResidual", compiler.compile("""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
#include "rknn_custom_op.h"

#include <math.h>

int cstDualResidual(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                                    rknn_custom_op_tensor* outputs, uint32_t n_outputs)
{
   
  unsigned char*      in_ptr_0  = (unsigned char*)inputs[0].mem.virt_addr + inputs[0].mem.offset;
  unsigned char*      in_ptr_1   = (unsigned char*)inputs[1].mem.virt_addr + inputs[1].mem.offset;
  unsigned char*      out_ptr_0  = (unsigned char*)outputs[0].mem.virt_addr + outputs[0].mem.offset;
  unsigned char*      out_ptr_1  = (unsigned char*)outputs[1].mem.virt_addr + outputs[1].mem.offset;
  const float*        in_data_0  = (const float*)in_ptr_0;
  const float*        in_data_1  = (const float*)in_ptr_1;
  float*              out_data_0 = (float*)out_ptr_0;
  float*              out_data_1 = (float*)out_ptr_1;


    const auto out_elems = outputs[0].attr.n_elems; 
    for (size_t idx=0; idx<out_elems;idx++) {
      float val0 = *(in_data_0+idx);
      float val1 = *(in_data_1+idx);

      *(out_data_0+idx) = val0 + 5.0f;
    }
    return 0;
    }
"""))

# print(a, b, c)
# args = (c, a, b)
prog(c, a, b)



# rknnalloc._copyout(c,flat_mv(C.data))

# print(C)
# prog = RKNNProgram(device, "test", RKNNCompiler().compile(f"""
# #include <vector123>
# #include <arm_neon.h>
# #include <rknn_matmul_api.h>
# void test(rknn_matmul_io_attr_t io_attr, rknn_matmul_info_t info) {{
#   printf("awd")
# }}
# """))

# prog(a, b, c, global_size=[N//(8*4), N//(8*4*LID), 1], local_size=[32, LID, 1], wait=True)

# for i in range(1):
#   rknn.rknn_matmul_run(device.ctx)
#   rknnalloc._copyout(c,flat_mv(C.data))
#   print(C)
