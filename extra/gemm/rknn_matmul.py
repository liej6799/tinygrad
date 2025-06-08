import os
os.environ["ROCKCHIP"] = "1"
import time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_rknn import RKNNDevice, RKNNAllocator, RKNNProgram
import ctypes as ct
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.autogen import rknn as rknn

N = getenv("N", 32)
LID = 2

device = RKNNDevice("RKNN")
rknnalloc = RKNNAllocator(dev=device)

ROW_A = 1
COL_A = 32
COL_B = 32

a = rknnalloc.alloc(device.io_attr.A.size)
b = rknnalloc.alloc(device.io_attr.B.size)
c = rknnalloc.alloc(device.io_attr.C.size)

A = np.random.rand(ROW_A, COL_A).astype(np.float16)
B = np.random.rand(COL_A, COL_B).astype(np.float16)
C = np.empty((ROW_A, COL_B), dtype=np.float32)

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
   
 unsigned char*      in_ptr   = (unsigned char*)inputs[0].mem.virt_addr + inputs[0].mem.offset;
    unsigned char*      out_ptr  = (unsigned char*)outputs[0].mem.virt_addr + outputs[0].mem.offset;
    const float*        in_data  = (const float*)in_ptr;
    float*              out_data = (float*)out_ptr;
    int inside  = 1;

        out_data[0] = 1.0f;


    return 0;
}
"""))

# print(a, b, c)
# args = (c, a, b)
prog(c, a, b)



rknnalloc._copyout(c,flat_mv(C.data))

print(C)
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
