import os
os.environ["ROCKCHIP"] = "1"
import time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_rknn import RKNNDevice, RKNNAllocator
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

rknn.rknn_matmul_create(device.ctx, info, io_attr)
rknn.rknn_matmul_set_core_mask(device.ctx, rknn.RKNN_NPU_CORE_0_1_2)

a = rknnalloc.alloc(io_attr.A.size)
b = rknnalloc.alloc(io_attr.B.size)
c = rknnalloc.alloc(io_attr.C.size)

A = np.random.rand(ROW_A, COL_A).astype(np.float16)
B = np.random.rand(COL_A, COL_B).astype(np.float16)
C = np.empty((ROW_A, COL_B), dtype=np.float32)

# copyin
rknnalloc._copyin(a, memoryview(bytearray(A)))
rknnalloc._copyin(b, memoryview(bytearray(B)))

#copyin 270959098978304 b"\xbc8\xad;p5\xc9;L;\xf4:t4A8<9F9B)\x845\xdd:\xe1*p6\xde;&5'6C9-9\x07;q\x1c\x830\xa9;^$B1\xc71\x00:\xc5-\xb1308\xa60" 64

# ct.memmove(mats[0].contents.virt_addr, ct.c_void_p(A.ctypes.data), io_attr.A.size)
# print(mats[0].contents.virt_addr, ct.c_void_p(A.ctypes.data), io_attr.A.size)
# ct.memmove(mats[1].contents.virt_addr, ct.c_void_p(B.ctypes.data), io_attr.B.size)

rknn.rknn_matmul_set_io_mem(device.ctx, a.buf, io_attr.A)
rknn.rknn_matmul_set_io_mem(device.ctx, b.buf, io_attr.B)
rknn.rknn_matmul_set_io_mem(device.ctx, c.buf, io_attr.C)


for i in range(1):
  rknn.rknn_matmul_run(device.ctx)
  print(C.tobytes(), ct.c_void_p(C.ctypes.data), c.virt_addr , io_attr.C.size)

  #ct.memmove(ct.c_void_p(C.ctypes.data), c.virt_addr, io_attr.C.size)
  rknnalloc._copyout(c,flat_mv(C.data))
  print(C)

# rknn.rknn_matmul_create()
# if (rknn_matmul_create(&ctx, &info, &io_attr))
#     return false;
