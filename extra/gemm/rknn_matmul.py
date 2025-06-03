import os
os.environ["ROCKCHIP"] = "1"
import time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_rockchip import RockchipDevice
import ctypes as ct
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.autogen import rknn as rknn
N = getenv("N", 32)
LID = 2


ROW_A = 1
COL_A = 32
COL_B = 32

ctx = ct.c_ulong()
info = rknn.struct_rknn_matmul_info_t()
ct.memset(ct.byref(info), 0, ct.sizeof(rknn.struct_rknn_matmul_info_t))
info.M = ROW_A
info.K = COL_A
info.N = COL_B
info.type = rk.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
info.B_layout = 0
info.AC_layout = 0

io_attr = rknn.struct__rknn_matmul_io_attr()
ct.memset(ct.byref(io_attr), 0, ct.sizeof(rknn.struct__rknn_matmul_io_attr))

rknn.rknn_matmul_create(ctx, info, io_attr)
rknn.rknn_matmul_set_core_mask(ctx, rknn.RKNN_NPU_CORE_AUTO)

RKNN_Tensor_Mem_Ptr  = ct.POINTER(rknn.struct__rknn_tensor_memory)
MatsArray = RKNN_Tensor_Mem_Ptr * 3
mats = MatsArray()

mats[0] = rknn.rknn_create_mem(ctx, io_attr.A.size)
mats[1] = rknn.rknn_create_mem(ctx, io_attr.B.size)
mats[2] = rknn.rknn_create_mem(ctx, io_attr.C.size)

rknn.rknn_matmul_set_io_mem(ctx, mats[2], io_attr.C)

A = np.random.rand(ROW_A, COL_A).astype(np.float16)
B = np.random.rand(COL_A, COL_B).astype(np.float16)
C = np.zeros((ROW_A, COL_B), dtype=np.float32)

dest_ptr = mats[0].contents.virt_addr

ct.memmove(mats[0].contents.virt_addr, ct.c_void_p(A.ctypes.data), io_attr.A.size)
ct.memmove(mats[1].contents.virt_addr, ct.c_void_p(B.ctypes.data), io_attr.B.size)

rknn.rknn_matmul_set_io_mem(ctx, mats[0], io_attr.A)
rknn.rknn_matmul_set_io_mem(ctx, mats[1], io_attr.B)

for i in range(1):
  rknn.rknn_matmul_run(ctx)
  ct.memmove(ct.c_void_p(C.ctypes.data), mats[2].contents.virt_addr , io_attr.C.size)
  print(C)

# rknn.rknn_matmul_create()
# if (rknn_matmul_create(&ctx, &info, &io_attr))
#     return false;

device = RockchipDevice("ROCKCHIP")
print(device)