"""
Standalone test for the RVV ISA backend's encoding layer.

Builds a hand-coded `vadd` kernel (c[i] = a[i] + b[i]) using only the
low-level enc_* helpers from tinygrad.renderer.isa.rvv, JITs it into
executable memory, and calls it via ctypes. No tinygrad UOp pipeline,
no isel — this is a direct proof that the encoders produce executable
RVV machine code on real hardware.

Run on the target board:
  python3 test/test_rvv_kernel.py
"""
import ctypes, mmap, struct
from tinygrad.renderer.isa.rvv import (enc_vsetvli, enc_vload, enc_vstore, enc_vop, enc_I, enc_R, enc_B,
                                       vtype, SEW32, LMUL_1)

# Register indices (psABI):
#   x0  zero
#   x1  ra
#   x5  t0
#   x6  t1
#   x10 a0  (arg0 / return value)
#   x11 a1  (arg1)
#   x12 a2  (arg2)
#   x13 a3  (arg3)

def build_vadd_kernel() -> bytes:
  """RV64 + RVV1.0 kernel: void vadd_f32(float* a0_dst, float* a1_a, float* a2_b, size_t a3_n)"""
  code = bytearray()
  loop_off = len(code)
  # vsetvli t0, a3, e32, m1, ta, ma  ; t0 = min(VLMAX, a3)  (also sets vl)
  code += enc_vsetvli(rd=5, rs1=13, vtypei=vtype(SEW32, LMUL_1))
  # vle32.v v0, (a1)                  ; v0 = *a1
  code += enc_vload(nf=0, mew=0, mop=0, vm=1, lumop=0, rs1=11, width=6, vd=0)
  # vle32.v v1, (a2)                  ; v1 = *a2
  code += enc_vload(nf=0, mew=0, mop=0, vm=1, lumop=0, rs1=12, width=6, vd=1)
  # vfadd.vv v2, v0, v1               ; v2 = v0 + v1   (funct6=0x00, funct3=1=OPFVV)
  code += enc_vop(funct6=0x00, vm=1, vs2=0, vs1_or_rs1_or_imm=1, funct3=1, vd=2)
  # vse32.v v2, (a0)                  ; *a0 = v2
  code += enc_vstore(nf=0, mew=0, mop=0, vm=1, sumop=0, rs1=10, width=6, vs3=2)
  # slli t1, t0, 2                    ; t1 = vl * sizeof(float)   (SLLI: imm12 holds shamt, funct3=1)
  code += enc_I(imm=2, rs1=5, funct3=0x1, rd=6, opcode=0x13)
  # add a0, a0, t1                    ; advance dst pointer
  code += enc_R(funct7=0x00, rs2=6, rs1=10, funct3=0, rd=10, opcode=0x33)
  # add a1, a1, t1                    ; advance src-a pointer
  code += enc_R(funct7=0x00, rs2=6, rs1=11, funct3=0, rd=11, opcode=0x33)
  # add a2, a2, t1                    ; advance src-b pointer
  code += enc_R(funct7=0x00, rs2=6, rs1=12, funct3=0, rd=12, opcode=0x33)
  # sub a3, a3, t0                    ; n -= vl
  code += enc_R(funct7=0x20, rs2=5, rs1=13, funct3=0, rd=13, opcode=0x33)
  # bnez a3, loop  ==  bne a3, zero, <off>   (funct3=1 for BNE)
  branch_off = loop_off - len(code)
  code += enc_B(imm=branch_off, rs2=0, rs1=13, funct3=0x1, opcode=0x63)
  # ret  ==  jalr x0, 0(ra)
  code += enc_I(imm=0, rs1=1, funct3=0, rd=0, opcode=0x67)
  return bytes(code)

def jit_call(code:bytes, dst, a, b, n):
  # Allocate RWX page and copy code.
  page_size = mmap.PAGESIZE
  size = ((len(code) + page_size - 1) // page_size) * page_size
  mem = mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
  mem.write(code)
  addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
  # Invalidate i-cache: RISC-V requires fence.i after writing code. libgcc_s.so.1 provides __clear_cache.
  libgcc = ctypes.CDLL("libgcc_s.so.1")
  libgcc["__clear_cache"](ctypes.c_void_p(addr), ctypes.c_void_p(addr + size))
  # Call as: void fn(float*, float*, float*, size_t)
  fn_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)
  fn = fn_t(addr)
  fn(dst.ctypes.data, a.ctypes.data, b.ctypes.data, n)

def jit_call_arrays(code:bytes, a_vals:list[float], b_vals:list[float]) -> list[float]:
  n = len(a_vals)
  assert len(b_vals) == n
  arr_t = ctypes.c_float * n
  a_arr = arr_t(*a_vals)
  b_arr = arr_t(*b_vals)
  dst_arr = arr_t()
  page_size = mmap.PAGESIZE
  size = ((len(code) + page_size - 1) // page_size) * page_size
  mem = mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
  mem.write(code)
  addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
  ctypes.CDLL("libgcc_s.so.1")["__clear_cache"](ctypes.c_void_p(addr), ctypes.c_void_p(addr + size))
  fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)(addr)
  fn(ctypes.addressof(dst_arr), ctypes.addressof(a_arr), ctypes.addressof(b_arr), n)
  return list(dst_arr)

if __name__ == "__main__":
  code = build_vadd_kernel()
  print(f"kernel size: {len(code)} bytes ({len(code)//4} instructions)")
  print("hex dump:", code.hex())

  n = 1024
  a_vals = [float(i) for i in range(n)]
  b_vals = [i * 0.5 for i in range(n)]
  expected = [a_vals[i] + b_vals[i] for i in range(n)]

  dst = jit_call_arrays(code, a_vals, b_vals)

  ok = all(abs(dst[i] - expected[i]) < 1e-6 for i in range(n))
  if ok:
    print(f"PASS: vadd of {n} f32 values matches expected output")
    print(f"  dst[0:8] = {dst[0:8]}")
    print(f"  exp[0:8] = {expected[0:8]}")
  else:
    print("FAIL")
    print(f"  dst[0:8] = {dst[0:8]}")
    print(f"  exp[0:8] = {expected[0:8]}")
    diff_idx = [i for i in range(n) if abs(dst[i] - expected[i]) >= 1e-6]
    print(f"  {len(diff_idx)} mismatches, first at index {diff_idx[0] if diff_idx else 'n/a'}")
