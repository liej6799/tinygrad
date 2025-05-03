
# mypy: ignore-errors
#!/usr/bin/env python3
import os, ctypes, ctypes.util, struct, platform, time
def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")   

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field in s._fields_:
    dat = getattr(s, field[0])
    if isinstance(dat, int): sdats.append(f"{field[0]}:0x{dat:X}")
    elif hasattr(dat, "_fields_"): sdats.append((field[0], format_struct(dat)))
    elif field[0] == "PADDING_0": pass
    else: sdats.append(f"{field[0]}:{dat}")
  return sdats
