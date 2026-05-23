"""
4-way i-unrolled RVV matmul. Processes 4 rows of C simultaneously, sharing
each B-row load across 4 vfmacc.vf ops (4x B-reuse, 4 accumulator vregs).
Requires M % 4 == 0.

Registers used:
  a0=C, a1=A, a2=B, a3=M, a4=K, a5=N  (args)
  t0=i (block-of-4 counter)
  t1=j
  t2=vl
  t3=B cursor
  t4..t7  (=x29..x31, x7)  A row base ptrs for the 4 rows
  Wait — only have t0..t6 in the standard temps. Need more.
  Plan: keep i=0..M/4 in t0. A row bases as a6/a7/s2/s3 (idx 16,17,18,19).
        C row bases as s4/s5/s6/s7 (idx 20,21,22,23).
        scratch a_cursors as s8..s11 (idx 24..27).
  We clobber s* (callee-saved) — but since we have no caller-side state to preserve in this test, and we save/restore nothing, that's fine here.
"""
import ctypes, mmap, time
from tinygrad.renderer.isa.rvv import (enc_vsetvli, enc_vload, enc_vstore, enc_vop, enc_I, enc_R, enc_B,
                                       vtype, SEW32, LMUL_1)

ZERO, RA, SP = 0, 1, 2
T0, T1, T2, T3 = 5, 6, 7, 28
A0, A1, A2, A3, A4, A5 = 10, 11, 12, 13, 14, 15
A6, A7 = 16, 17
S2, S3, S4, S5, S6, S7, S8, S9, S10, S11 = 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
T4, T5, T6 = 29, 30, 31
FA0, FA1, FA2, FA3 = 10, 11, 12, 13   # scalar A values
V0, V1, V2, V3, V4 = 0, 1, 2, 3, 4    # 4 accumulators (v0-v3) + B row (v4)

from tinygrad.renderer.isa.rvv import enc_S
def build_matmul_u4() -> bytes:
  c = bytearray()
  # --- prologue: save callee-saved regs we clobber (s2..s11 = 10 regs * 8 = 80 bytes) ---
  saved_regs = [S2, S3, S4, S5, S6, S7, S8, S9, S10, S11]
  c += enc_I(-80, SP, 0, SP, 0x13)        # addi sp, sp, -80
  for i, r in enumerate(saved_regs):
    c += enc_S(i*8, r, SP, 3, 0x23)       # sd r, i*8(sp)
  # ---
  # t6 = N*4 (B/C row stride bytes); s11 = K*4 (A row stride bytes)
  c += enc_I(2, A5, 1, T6, 0x13)                   # slli t6, a5, 2
  c += enc_I(2, A4, 1, S11, 0x13)                  # slli s11, a4, 2
  # Initial A row base pointers for the 4 rows: a6..a7,s2,s3 = A, A+K*4, A+2*K*4, A+3*K*4
  c += enc_R(0x00, ZERO, A1, 0, A6, 0x33)          # add a6, a1, zero
  c += enc_R(0x00, S11, A6, 0, A7, 0x33)           # add a7, a6, s11
  c += enc_R(0x00, S11, A7, 0, S2, 0x33)           # add s2, a7, s11
  c += enc_R(0x00, S11, S2, 0, S3, 0x33)           # add s3, s2, s11
  # Initial C row base pointers: s4..s7 = C, C+N*4, C+2*N*4, C+3*N*4
  c += enc_R(0x00, ZERO, A0, 0, S4, 0x33)          # add s4, a0, zero
  c += enc_R(0x00, T6, S4, 0, S5, 0x33)            # add s5, s4, t6
  c += enc_R(0x00, T6, S5, 0, S6, 0x33)            # add s6, s5, t6
  c += enc_R(0x00, T6, S6, 0, S7, 0x33)            # add s7, s6, t6
  # i = 0  (block index, 0..M/4)
  c += enc_I(0, ZERO, 0, T0, 0x13)                 # t0 = 0
  i_loop = len(c)
  # j = 0
  c += enc_I(0, ZERO, 0, T1, 0x13)                 # t1 = 0
  j_loop = len(c)
  # vl = vsetvli(N - j)
  c += enc_R(0x20, T1, A5, 0, T2, 0x33)            # sub t2, a5, t1
  c += enc_vsetvli(T2, T2, vtype(SEW32, LMUL_1))
  # acc0..3 = 0
  for v in (V0, V1, V2, V3): c += enc_vop(0x17, 1, 0, 0, 3, v)
  # b_cursor (t3) = a2 + j*4
  c += enc_I(2, T1, 1, T3, 0x13)                   # slli t3, t1, 2
  c += enc_R(0x00, T3, A2, 0, T3, 0x33)            # add t3, a2, t3
  # A cursors (per row): s8..s11 = a6, a7, s2, s3 (will be bumped each k iter)
  c += enc_R(0x00, ZERO, A6, 0, S8, 0x33)
  c += enc_R(0x00, ZERO, A7, 0, S9, 0x33)
  c += enc_R(0x00, ZERO, S2, 0, S10, 0x33)
  # s11 is K*4 stride; use t4 for the 4th A cursor instead
  c += enc_R(0x00, ZERO, S3, 0, T4, 0x33)
  # k down-counter in t5
  c += enc_R(0x00, ZERO, A4, 0, T5, 0x33)          # t5 = K
  k_loop = len(c)
  # Load 4 A scalars
  c += enc_I(0, S8, 2, FA0, 0x07)                  # flw fa0, 0(s8)
  c += enc_I(0, S9, 2, FA1, 0x07)                  # flw fa1, 0(s9)
  c += enc_I(0, S10, 2, FA2, 0x07)                 # flw fa2, 0(s10)
  c += enc_I(0, T4, 2, FA3, 0x07)                  # flw fa3, 0(t4)
  # Load 1 B row -> v4
  c += enc_vload(0, 0, 0, 1, 0, T3, 6, V4)         # vle32.v v4, (t3)
  # 4 vfmacc.vf: accN += faN * v4
  c += enc_vop(0x2C, 1, V4, FA0, 5, V0)
  c += enc_vop(0x2C, 1, V4, FA1, 5, V1)
  c += enc_vop(0x2C, 1, V4, FA2, 5, V2)
  c += enc_vop(0x2C, 1, V4, FA3, 5, V3)
  # Bump A cursors by 4 (next k); bump B by t6 (next row).
  c += enc_I(4, S8, 0, S8, 0x13)
  c += enc_I(4, S9, 0, S9, 0x13)
  c += enc_I(4, S10, 0, S10, 0x13)
  c += enc_I(4, T4, 0, T4, 0x13)
  c += enc_R(0x00, T6, T3, 0, T3, 0x33)            # t3 += N*4
  # k--; if k != 0 -> k_loop
  c += enc_I(-1, T5, 0, T5, 0x13)
  c += enc_B(k_loop - len(c), ZERO, T5, 1, 0x63)
  # Store 4 accumulators to C[i..i+3, j..]
  # addr_i = c_row_base + j*4
  c += enc_I(2, T1, 1, T5, 0x13)                   # t5 = j*4
  for v, cbase in ((V0, S4), (V1, S5), (V2, S6), (V3, S7)):
    c += enc_R(0x00, T5, cbase, 0, T4, 0x33)       # t4 = cbase + j*4
    c += enc_vstore(0, 0, 0, 1, 0, T4, 6, v)
  # j += vl
  c += enc_R(0x00, T2, T1, 0, T1, 0x33)
  c += enc_B(j_loop - len(c), A5, T1, 4, 0x63)     # blt t1, a5, j_loop
  # Advance A/C row bases by 4*K*4 / 4*N*4 (=stride * 4)
  for ptr in (A6, A7, S2, S3):
    c += enc_R(0x00, S11, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, S11, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, S11, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, S11, ptr, 0, ptr, 0x33)
  for ptr in (S4, S5, S6, S7):
    c += enc_R(0x00, T6, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, T6, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, T6, ptr, 0, ptr, 0x33)
    c += enc_R(0x00, T6, ptr, 0, ptr, 0x33)
  # i++; (i is the block index, runs from 0..M/4 — but we don't divide M, we use a loop counter t0 incremented by 4 and compare against M)
  c += enc_I(4, T0, 0, T0, 0x13)
  c += enc_B(i_loop - len(c), A3, T0, 4, 0x63)
  # --- epilogue ---
  for i, r in enumerate(saved_regs):
    c += enc_I(i*8, SP, 3, r, 0x03)       # ld r, i*8(sp)
  c += enc_I(80, SP, 0, SP, 0x13)         # addi sp, sp, 80
  c += enc_I(0, RA, 0, ZERO, 0x67)        # ret
  return bytes(c)

def make_jit(code:bytes):
  page = mmap.PAGESIZE
  size = ((len(code) + page - 1) // page) * page
  mem = mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
  mem.write(code)
  addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
  ctypes.CDLL("libgcc_s.so.1")["__clear_cache"](ctypes.c_void_p(addr), ctypes.c_void_p(addr + size))
  fn_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
  return mem, fn_t(addr)

def naive(A, B, M, K, N):
  C = [0.0] * (M * N)
  for i in range(M):
    for j in range(N):
      s = 0.0
      for k in range(K): s += A[i*K + k] * B[k*N + j]
      C[i*N + j] = s
  return C

if __name__ == "__main__":
  code = build_matmul_u4()
  print(f"kernel: {len(code)} bytes ({len(code)//4} instructions)")
  mem, fn = make_jit(code)
  # Correctness (requires M % 4 == 0)
  for M, K, N in [(4,4,4), (4,4,8), (4,1,9), (8,3,5), (4,17,19), (16,16,16)]:
    A = [(i*0.1 + 1.0) for i in range(M*K)]
    B = [(i*0.07 + 0.3) for i in range(K*N)]
    exp = naive(A, B, M, K, N)
    A_arr = (ctypes.c_float * (M*K))(*A); B_arr = (ctypes.c_float * (K*N))(*B); C_arr = (ctypes.c_float * (M*N))()
    fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    got = list(C_arr)
    ok = all(abs(got[i] - exp[i]) < max(1e-3, abs(exp[i])*1e-5) for i in range(M*N))
    print(f"  {M}x{K} @ {K}x{N}: {'PASS' if ok else 'FAIL'}")
    if not ok:
      for q in range(min(8, M*N)): print(f"    [{q}] got={got[q]}, exp={exp[q]}")
      raise SystemExit(1)
  print("--- benchmark ---")
  for N in [64, 128, 256, 512]:
    M = K = N  # multiples of 4
    A_arr = (ctypes.c_float * (M*K))(); B_arr = (ctypes.c_float * (K*N))(); C_arr = (ctypes.c_float * (M*N))()
    for i in range(M*K): A_arr[i] = i * 0.001
    for i in range(K*N): B_arr[i] = i * 0.002
    fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    iters = max(1, 100_000_000 // (2*M*K*N))
    t0 = time.perf_counter()
    for _ in range(iters):
      fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    t1 = time.perf_counter()
    per = (t1-t0)/iters
    gf = 2*M*K*N/per/1e9
    print(f"  N={N:4d}  {iters:5d} iters  {per*1e6:9.1f} us/iter  {gf:6.2f} GFLOPS", flush=True)
