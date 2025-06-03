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
from tinygrad.runtime.support.hcq import FileIOInterface
import os, mmap, functools
N = getenv("N", 32)
LID = 2

def NPUOP(op, value, reg):
    op_part = (ct.c_uint64(op & 0xffff).value) << 48
    value_part = (ct.c_uint64(value & 0xffffffff).value) << 16
    reg_part = ct.c_uint64(reg & 0xffff).value
    return ct.c_uint64(op_part | value_part | reg_part).value

ROW_A = 1
COL_A = 32
COL_B = 32

M = ROW_A
K = COL_A
N = COL_B

buf1 = (ct.c_ubyte * 256)()
buf2 = (ct.c_ubyte * 256)()
buf3 = (ct.c_ubyte * 256)()

ct.memset(ct.byref(buf1), 0, ct.sizeof(buf1))
ct.memset(ct.byref(buf2), 0, ct.sizeof(buf2))
ct.memset(ct.byref(buf3), 0, ct.sizeof(buf3))

fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)

regcmd_dma = ct.c_uint64()
regcmd_obj = ct.c_uint64()
regcmd_handle = ct.c_uint32()

mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=1024, flags= 0 | rk.RKNPU_MEM_NON_CACHEABLE)
mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=mem_create.handle, offset=0)
regcmd_map = fd_ctl.mmap(None, 1024, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)

regcmd_dma = mem_create.dma_addr
regcmd_obj = mem_create.obj_addr
regcmd_handle = mem_create.handle

tasks_dma = ct.c_uint64()
tasks_obj = ct.c_uint64()
tasks_handle = ct.c_uint32()

mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=1024, flags= rk.RKNPU_MEM_KERNEL_MAPPING | rk.RKNPU_MEM_NON_CACHEABLE)
mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=mem_create.handle, offset=0)
tasks_map = fd_ctl.mmap(None, 1024, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)

tasks_dma = mem_create.dma_addr
tasks_obj = mem_create.obj_addr
tasks_handle = mem_create.handle


input_dma = ct.c_uint64()
input_obj = ct.c_uint64()
input_handle = ct.c_uint32()
size = ROW_A * COL_A * ct.sizeof(ct.c_uint16)

mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=size, flags= 0 | rk.RKNPU_MEM_NON_CACHEABLE)
mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=mem_create.handle, offset=0)
input_map = fd_ctl.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)

input_dma = mem_create.dma_addr
input_obj = mem_create.obj_addr
input_handle = mem_create.handle

weights_dma = ct.c_uint64()
weights_obj = ct.c_uint64()
weights_handle = ct.c_uint32()
size = COL_A * COL_B * ct.sizeof(ct.c_uint16)

mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=size, flags= 0 | rk.RKNPU_MEM_NON_CACHEABLE)
mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=mem_create.handle, offset=0)
weights_map = fd_ctl.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)

weights_dma = mem_create.dma_addr
weights_obj = mem_create.obj_addr
weights_handle = mem_create.handle


output_dma = ct.c_uint64()
output_obj = ct.c_uint64()
output_handle = ct.c_uint32()
size = COL_A * COL_B * ct.sizeof(ct.c_uint16)

mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(fd_ctl, size=size, flags= 0 | rk.RKNPU_MEM_NON_CACHEABLE)
mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(fd_ctl, handle=mem_create.handle, offset=0)
output_map = fd_ctl.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset=mem_map.offset)

output_dma = mem_create.dma_addr
output_obj = mem_create.obj_addr
output_handle = mem_create.handle



print(regcmd_dma, regcmd_obj, regcmd_handle, regcmd_map)
print(tasks_dma, tasks_obj, tasks_handle, tasks_map)
print(input_dma, input_obj, input_handle, input_map)
print(weights_dma, weights_obj, weights_handle, weights_map)
print(output_dma, output_obj, output_handle, output_map)

res = rk.DRM_IOCTL_RKNPU_ACTION(fd_ctl, flags= rk.RKNPU_ACT_RESET)
print(res.value)

cna_desc = rk.struct_npu_cna_desc()
core_desc = rk.struct_npu_core_desc()
dpu_desc = rk.struct_npu_dpu_desc()

fd_bytes = ct.c_uint()
fd_banks = ct.c_uint()
weight_banks = ct.c_uint()
surf_stride = ct.c_int()

cna_desc.conv_mode = rk.direct_convolution
cna_desc.in_precision = rk.precision_float16
cna_desc.proc_precision = rk.precision_float16

cna_desc.kernel_groups = 0
cna_desc.feature_grains = M+1
cna_desc.conv_x_stride = 1
cna_desc.conv_y_stride = 1

cna_desc.datain_width = 1
cna_desc.datain_height = M
cna_desc.datain_channel = K
cna_desc.dataout_width = 1
cna_desc.dataout_height = M
cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height

cna_desc.weight_width = 1
cna_desc.weight_height = 1
cna_desc.weight_kernels = N
cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height * cna_desc.datain_channel * ct.sizeof(ct.c_uint16)
cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels  

fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * ct.sizeof(ct.c_uint16)
fd_banks = (fd_bytes // rk.NPU_CBUF_BANK_SIZE)
if ((fd_bytes % rk.NPU_CBUF_BANK_SIZE) == 0):
    fd_banks = fd_banks
else:
    fd_banks = fd_banks + 1

weight_banks = (cna_desc.weight_bytes // rk.NPU_CBUF_BANK_SIZE)
if (cna_desc.weight_bytes % rk.NPU_CBUF_BANK_SIZE) == 0:
    weight_banks = weight_banks
else:
    weight_banks = weight_banks + 1

if ((fd_banks) > rk.NPU_CBUF_BANKS-1):
    raise RuntimeError(f"Error -1")
else:
    if (cna_desc.weight_bytes_per_kernel <= rk.NPU_CBUF_BANK_SIZE):
        weight_banks = rk.NPU_CBUF_BANKS - fd_banks
    else:
        raise RuntimeError(f"Error -2")

cna_desc.weight_bank = weight_banks
cna_desc.data_bank = fd_banks
cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) // 32
if (((cna_desc.datain_width * cna_desc.datain_channel) % 32) == 0):
    cna_desc.data_entries = cna_desc.data_entries
else:
    cna_desc.data_entries = cna_desc.data_entries + 1

cna_desc.data_sign = 0x1 
cna_desc.cvt_type  = 0x1 
cna_desc.cvt_bypass = 0x1 
cna_desc.cvt_scale0 = 0x1 
cna_desc.cvt_scale1 = 0x1 
cna_desc.cvt_scale2 = 0x1 
cna_desc.cvt_scale3 = 0x1 
cna_desc.fc_skip_en = 0 
cna_desc.data_offset = 0x0 
cna_desc.pad_left = 0 
cna_desc.pad_top = 0 
cna_desc.feature_base_addr = input_dma 
cna_desc.weight_offset = 0 
cna_desc.weight_burst_len = 0xf 
cna_desc.data_burst_len = 0xf 
cna_desc.line_stride = cna_desc.datain_width * 4 
surf_stride = cna_desc.line_stride * ((cna_desc.datain_height // 4)-1) 
if (surf_stride < 0):
    surf_stride = surf_stride + 1
else:
    surf_stride = surf_stride

cna_desc.surf_stride = surf_stride 
cna_desc.dma_width = cna_desc.datain_width 
cna_desc.dma_height = cna_desc.datain_height 
cna_desc.dma_channel = cna_desc.datain_channel 
cna_desc.decompress_addr0 = weights_dma 

core_desc.proc_precision = rk.precision_float16
core_desc.qd_en = 1
core_desc.dataout_height = cna_desc.dataout_height - 1
core_desc.dataout_width = cna_desc.dataout_width - 1
core_desc.dataout_channel = cna_desc.weight_kernels -1


dpu_desc.burst_len = 0xf  
dpu_desc.conv_mode = rk.direct_convolution  
dpu_desc.output_mode = 0x2  
dpu_desc.flying_mode = 0x0  
dpu_desc.out_precision = rk.precision_float16
dpu_desc.in_precision = rk.precision_float16  
dpu_desc.proc_precision = rk.precision_float16  
dpu_desc.dst_base_addr = output_dma  
dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width  
dpu_desc.width = core_desc.dataout_width   
dpu_desc.height = core_desc.dataout_height  
dpu_desc.channel = core_desc.dataout_channel  
dpu_desc.bs_bypass = 1  
dpu_desc.bs_alu_bypass = 1  
dpu_desc.bs_mul_bypass = 1  
dpu_desc.bs_relu_bypass = 1  
dpu_desc.bn_bypass =1  
dpu_desc.bn_alu_bypass = 1  
dpu_desc.bn_mul_bypass = 1  
dpu_desc.bn_relu_bypass = 1  
dpu_desc.ew_bypass =1  
dpu_desc.ew_op_bypass =1  
dpu_desc.ew_lut_bypass =1  
dpu_desc.ew_op_cvt_bypass =1  
dpu_desc.ew_relu_bypass=1  
dpu_desc.fp32tofp16_en = 1 & 0x1  
dpu_desc.out_cvt_scale =1  

dpu_desc.size_e_2 = 1  
dpu_desc.size_e_1 = 1  
dpu_desc.size_e_0 = 1  

dpu_desc.od_bypass = 1  
dpu_desc.width_wdma = core_desc.dataout_width  
dpu_desc.height_wdma = core_desc.dataout_height  
dpu_desc.channel_wdma = core_desc.dataout_channel  
dpu_desc.surf_add = dpu_desc.dst_surf_stride * 2  

NPUOP(rk.OP_REG_DPU, 0xE, rk.DPU_S_POINTER)

# cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 32
# if ((((cna_desc.datain_width * cna_desc.datain_channel) % 32) == 0)):
#     cna_desc.data_entries = cna_desc.data_entries
# else:
#     cna_desc.data_entries = cna_desc.data_entries + 1
    
# dv = rk.struct_drm_version()
# ct.memset(ct.byref(dv), 0, ct.sizeof(dv))
# dv.name = buf1
# dv.name_len = ct.sizeof(buf1)
# dv.date = buf2
# dv.date_len = ct.sizeof(buf2)
# dv.desc = buf3
# dv.desc_len = ct.sizeof(buf3)

# res = rk.DRM_IOCTL_VERSION(fd_ctl)
# print(res.name.contents.value)

# ctx = ct.c_ulong()
# info = rknn.struct_rknn_matmul_info_t()
# ct.memset(ct.byref(info), 0, ct.sizeof(rknn.struct_rknn_matmul_info_t))
# info.M = ROW_A
# info.K = COL_A
# info.N = COL_B
# info.type = rk.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
# info.B_layout = 0
# info.AC_layout = 0

# io_attr = rknn.struct__rknn_matmul_io_attr()
# ct.memset(ct.byref(io_attr), 0, ct.sizeof(rknn.struct__rknn_matmul_io_attr))

# rknn.rknn_matmul_create(ctx, info, io_attr)
# rknn.rknn_matmul_set_core_mask(ctx, rknn.RKNN_NPU_CORE_AUTO)

# RKNN_Tensor_Mem_Ptr  = ct.POINTER(rknn.struct__rknn_tensor_memory)
# MatsArray = RKNN_Tensor_Mem_Ptr * 3
# mats = MatsArray()

# mats[0] = rknn.rknn_create_mem(ctx, io_attr.A.size)
# mats[1] = rknn.rknn_create_mem(ctx, io_attr.B.size)
# mats[2] = rknn.rknn_create_mem(ctx, io_attr.C.size)

# rknn.rknn_matmul_set_io_mem(ctx, mats[2], io_attr.C)

# A = np.random.rand(ROW_A, COL_A).astype(np.float16)
# B = np.random.rand(COL_A, COL_B).astype(np.float16)
# C = np.zeros((ROW_A, COL_B), dtype=np.float32)

# dest_ptr = mats[0].contents.virt_addr

# ct.memmove(mats[0].contents.virt_addr, ct.c_void_p(A.ctypes.data), io_attr.A.size)
# ct.memmove(mats[1].contents.virt_addr, ct.c_void_p(B.ctypes.data), io_attr.B.size)

# rknn.rknn_matmul_set_io_mem(ctx, mats[0], io_attr.A)
# rknn.rknn_matmul_set_io_mem(ctx, mats[1], io_attr.B)

# for i in range(1):
#   rknn.rknn_matmul_run(ctx)
#   ct.memmove(ct.c_void_p(C.ctypes.data), mats[2].contents.virt_addr , io_attr.C.size)
#   print(C)

# # rknn.rknn_matmul_create()
# # if (rknn_matmul_create(&ctx, &info, &io_attr))
# #     return false 

# device = RockchipDevice("ROCKCHIP")
# print(device)


