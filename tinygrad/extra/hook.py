import os
print("from import")

import ctypes, ctypes.util
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.extra.run import get_struct, format_struct
@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
def _mmap(addr, length, prot, flags, fd, offset):
  mmap_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
  orig_mmap = mmap_type(ctypes.addressof(orig_mmap_mv))
  ret = orig_mmap(addr, length, prot, flags, fd, offset)
  # ll = os.readlink(f"/proc/self/fd/{fd}") if fd >= 0 else ""
  print(f"mmap {addr=}, {length=}, {prot=}, {flags=}, {fd=}, {offset=} {ret=}")
  return ret

#install_hook(libc.ioctl, ioctl)
#orig_mmap_mv = install_hook(libc.mmap, _mmap)
print("import done")
import mmap

alloc_sizes = {}
mmaped = {}

def handle_ioctl(fd, request, argp, ret):
  fn = os.readlink(f"/proc/self/fd/{fd}")
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  
  if fn == "/dev/dri/card1":
    
    if nr == 64: # 0x40 + 0x00
        st = get_struct(argp, rk.struct_rknpu_action)
        print(ret, "struct_rknpu_action", format_struct(st))   

    elif nr == 65: # 0x40 + 0x01
        st = get_struct(argp, rk.struct_rknpu_submit)
        print(ret, "struct_rknpu_submit", format_struct(st))      

    elif nr == 66: # 0x40 + 0x02 
        st = get_struct(argp, rk.struct_rknpu_mem_create)
        print(ret, "struct_rknpu_mem_create", format_struct(st))

    elif nr == 67: # 0x40 + 0x03
        st = get_struct(argp, rk.struct_rknpu_mem_map)
        print(ret, "struct_rknpu_mem_map", format_struct(st))        
      
    elif nr == 68: # 0x40 + 0x04     
        st = get_struct(argp, rk.struct_rknpu_mem_destroy)
        print(ret, "struct_rknpu_mem_destroy", format_struct(st))

    elif nr == 69: # 0x40 + 0x05    
        st = get_struct(argp, rk.struct_rknpu_mem_sync)
        print(ret, "struct_rknpu_mem_sync", format_struct(st))

    elif nr == 0: # 0x00
        st = get_struct(argp, rk.struct_drm_version)
        print(ret, "struct_drm_version", format_struct(st))    
    elif nr == 1: # 0x01
        st = get_struct(argp, rk.struct_drm_unique)
        print(ret, "struct_drm_unique", format_struct(st))

    elif nr == 10: # 0x0a
        st = get_struct(argp, rk.struct_drm_gem_flink)
        print(ret, "struct_drm_gem_flink", format_struct(st))

    elif nr == 45: # 0x2d
        st = get_struct(argp, rk.struct_drm_prime_handle)
        print(ret, "struct_drm_prime_handle", format_struct(st))        
    else:
        print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=} {argp=}", fn)
               
#   if fn == "/dev/ion":
#     if nr == 0:
#       st = get_struct(argp, qcom_dsp.struct_ion_allocation_data)
#       print(ret, "ION_IOC_ALLOC", format_struct(st))
#       alloc_sizes[st.handle] = st.len
#     elif nr == 1:
#       st = get_struct(argp, qcom_dsp.struct_ion_handle_data)
#       print(ret, "ION_IOC_FREE", format_struct(st))
#       if st.handle in alloc_sizes: del alloc_sizes[st.handle]
#       if st.handle in mmaped: del mmaped[st.handle]
#     elif nr == 2:
#       st = get_struct(argp, qcom_dsp.struct_ion_fd_data)
#       print(ret, "ION_IOC_MAP", format_struct(st))
#       mmaped[st.handle] = mmap.mmap(st.fd, alloc_sizes[st.handle])
#   elif fn == "/dev/adsprpc-smd":
#     assert chr(itype) == 'R'
#     if nr == 8:
#       st = ctypes.c_uint32.from_address(argp)
#       print(ret, "FASTRPC_IOCTL_GETINFO", st.value)
#     elif nr == 2:
#       st = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_mmap)
#       print(ret, "FASTRPC_IOCTL_MMAP", format_struct(st))
#     elif nr == 1:
#       # https://research.checkpoint.com/2021/pwn2own-qualcomm-dsp/
#       st = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke)
#       print(ret, "FASTRPC_IOCTL_INVOKE", format_struct(st))
#       # 0xFF000000 = Method index and attribute (the highest byte)
#       # 0x00FF0000 = Number of input arguments
#       # 0x0000FF00 = Number of output arguments
#       # 0x000000F0 = Number of input handles
#       # 0x0000000F = Number of output handles

#       method = (st.sc>>24) & 0xFF
#       in_args = (st.sc>>16) & 0xFF
#       out_args = (st.sc>>8) & 0xFF
#       in_h = (st.sc>>4) & 0xF
#       out_h = (st.sc>>0) & 0xF
#       print(f"\tm:{method} ia:{in_args} oa:{out_args} ih:{in_h} oh:{out_h}")
#       """
#       if in_args or out_args:
#         for arg in range(in_args+out_args):
#           print(arg, format_struct(st.pra[arg]))
#           if st.pra[arg].buf.pv is not None:
#             ww = to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)
#             hexdump(to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)[:0x40])
#       """
#     elif nr == 6:
#       print(ret, "FASTRPC_IOCTL_INIT", format_struct(ini:=get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_init)))
#       print(os.readlink(f"/proc/self/fd/{ini.filefd}"))
#       # print(bytearray(to_mv(ini.file, ini.filelen)))
#     elif nr == 7:
#       print(ret, "FASTRPC_IOCTL_INVOKE_ATTRS", format_struct(ini:=get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke_attrs)))
#     elif nr == 12: print(ret, "FASTRPC_IOCTL_CONTROL", format_struct(get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_control)))
#     elif nr == 4:
#       st_fd = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke_fd)
#       st = st_fd.inv
#       print(ret, "FASTRPC_IOCTL_INVOKE_FD", format_struct(st))

#       method = (st.sc>>24) & 0xFF
#       in_args = (st.sc>>16) & 0xFF
#       out_args = (st.sc>>8) & 0xFF
#       in_h = (st.sc>>4) & 0xF
#       out_h = (st.sc>>0) & 0xF
#       print(f"\tm:{method} ia:{in_args} oa:{out_args} ih:{in_h} oh:{out_h}")

#       if st.sc in [0x2030200, 0x3040300]:
#         for handle, mapped in mmaped.items():
#           print(f" buffer {handle} {alloc_sizes[handle]:X}")
#           with open(f"/tmp/buf_{st.sc:X}_{handle}_{alloc_sizes[handle]:X}", "wb") as f: f.write(mapped)
#     else:
#       print(f"{ret} UNPARSED {nr}")
#   else:
#     print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=}", fn)

