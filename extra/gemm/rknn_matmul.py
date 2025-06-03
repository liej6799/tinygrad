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

a = rknnalloc.alloc(device.io_attr.A.size)
b = rknnalloc.alloc(device.io_attr.B.size)
c = rknnalloc.alloc(device.io_attr.C.size)

A = np.random.rand(ROW_A, COL_A).astype(np.float16)
B = np.random.rand(COL_A, COL_B).astype(np.float16)
C = np.empty((ROW_A, COL_B), dtype=np.float32)

# copyin
rknnalloc._copyin(a, memoryview(bytearray(A)))
rknnalloc._copyin(b, memoryview(bytearray(B)))

rknn.rknn_matmul_set_io_mem(device.ctx, a.buf, device.io_attr.A)
rknn.rknn_matmul_set_io_mem(device.ctx, b.buf, device.io_attr.B)
rknn.rknn_matmul_set_io_mem(device.ctx, c.buf, device.io_attr.C)

for i in range(1):
  rknn.rknn_matmul_run(device.ctx)
  rknnalloc._copyout(c,flat_mv(C.data))
  print(C)
