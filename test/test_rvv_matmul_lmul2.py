"""
LMUL=2 variant of the 4-way i-unrolled matmul. Each vector register group
contains v_n + v_(n+1), giving 16 f32 lanes per instruction on VLEN=256.

Register groups (spec §3.4.2 — even-aligned, occupy 2 physical regs):
  v0  = acc row 0  (occupies v0,v1)
  v2  = acc row 1  (occupies v2,v3)
  v4  = acc row 2  (occupies v4,v5)
  v6  = acc row 3  (occupies v6,v7)
  v8  = B row      (occupies v8,v9)

Per inner-k iteration on the X60 (VLEN=256, e32, m2): one vle32 fetches 16
floats; four vfmacc.vf execute as 16-wide ops. If the macc unit has 1
vfmacc/cycle throughput at LMUL=2, we'd see up to 2x over LMUL=1.
"""
import ctypes, mmap, time
from tinygrad.renderer.isa.rvv import (enc_vsetvli, enc_vload, enc_vstore, enc_vop, enc_I, enc_R, enc_B, enc_S,
                                       vtype, SEW32, LMUL_2)

ZERO, RA, SP = 0, 1, 2
T0, T1, T2, T3, T4, T5, T6 = 5, 6, 7, 28, 29, 30, 31
A0, A1, A2, A3, A4, A5, A6, A7 = 10, 11, 12, 13, 14, 15, 16, 17
S2, S3, S4, S5, S6, S7, S8, S9, S10, S11 = 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
FA0, FA1, FA2, FA3 = 10, 11, 12, 13
# LMUL=2: 4 accumulator register *groups* + 1 B-row group, all even-aligned
V_ACC0, V_ACC1, V_ACC2, V_ACC3, V_B = 0, 2, 4, 6, 8

def build_matmul_u4_lmul2() -> bytes:
  c = bytearray()
  saved_regs = [S2, S3, S4, S5, S6, S7, S8, S9, S10, S11]
  c += enc_I(-80, SP, 0, SP, 0x13)
  for i, r in enumerate(saved_regs): c += enc_S(i*8, r, SP, 3, 0x23)
  # Strides: t6 = N*4, s11 = K*4
  c += enc_I(2, A5, 1, T6, 0x13)
  c += enc_I(2, A4, 1, S11, 0x13)
  # A row bases for 4 rows
  c += enc_R(0x00, ZERO, A1, 0, A6, 0x33)
  c += enc_R(0x00, S11, A6, 0, A7, 0x33)
  c += enc_R(0x00, S11, A7, 0, S2, 0x33)
  c += enc_R(0x00, S11, S2, 0, S3, 0x33)
  # C row bases for 4 rows
  c += enc_R(0x00, ZERO, A0, 0, S4, 0x33)
  c += enc_R(0x00, T6, S4, 0, S5, 0x33)
  c += enc_R(0x00, T6, S5, 0, S6, 0x33)
  c += enc_R(0x00, T6, S6, 0, S7, 0x33)
  c += enc_I(0, ZERO, 0, T0, 0x13)
  i_loop = len(c)
  c += enc_I(0, ZERO, 0, T1, 0x13)
  j_loop = len(c)
  # vsetvli with LMUL=2 — the only line that differs from the LMUL=1 variant
  c += enc_R(0x20, T1, A5, 0, T2, 0x33)
  c += enc_vsetvli(T2, T2, vtype(SEW32, LMUL_2))
  # Zero accumulators (each is a 16-wide group)
  for v in (V_ACC0, V_ACC1, V_ACC2, V_ACC3): c += enc_vop(0x17, 1, 0, 0, 3, v)
  # B cursor
  c += enc_I(2, T1, 1, T3, 0x13)
  c += enc_R(0x00, T3, A2, 0, T3, 0x33)
  # A cursors
  c += enc_R(0x00, ZERO, A6, 0, S8, 0x33)
  c += enc_R(0x00, ZERO, A7, 0, S9, 0x33)
  c += enc_R(0x00, ZERO, S2, 0, S10, 0x33)
  c += enc_R(0x00, ZERO, S3, 0, T4, 0x33)
  c += enc_R(0x00, ZERO, A4, 0, T5, 0x33)
  k_loop = len(c)
  c += enc_I(0, S8, 2, FA0, 0x07)
  c += enc_I(0, S9, 2, FA1, 0x07)
  c += enc_I(0, S10, 2, FA2, 0x07)
  c += enc_I(0, T4, 2, FA3, 0x07)
  c += enc_vload(0, 0, 0, 1, 0, T3, 6, V_B)
  c += enc_vop(0x2C, 1, V_B, FA0, 5, V_ACC0)
  c += enc_vop(0x2C, 1, V_B, FA1, 5, V_ACC1)
  c += enc_vop(0x2C, 1, V_B, FA2, 5, V_ACC2)
  c += enc_vop(0x2C, 1, V_B, FA3, 5, V_ACC3)
  c += enc_I(4, S8, 0, S8, 0x13)
  c += enc_I(4, S9, 0, S9, 0x13)
  c += enc_I(4, S10, 0, S10, 0x13)
  c += enc_I(4, T4, 0, T4, 0x13)
  c += enc_R(0x00, T6, T3, 0, T3, 0x33)
  c += enc_I(-1, T5, 0, T5, 0x13)
  c += enc_B(k_loop - len(c), ZERO, T5, 1, 0x63)
  # Stores
  c += enc_I(2, T1, 1, T5, 0x13)
  for v, cbase in ((V_ACC0, S4), (V_ACC1, S5), (V_ACC2, S6), (V_ACC3, S7)):
    c += enc_R(0x00, T5, cbase, 0, T4, 0x33)
    c += enc_vstore(0, 0, 0, 1, 0, T4, 6, v)
  c += enc_R(0x00, T2, T1, 0, T1, 0x33)
  c += enc_B(j_loop - len(c), A5, T1, 4, 0x63)
  for ptr in (A6, A7, S2, S3):
    for _ in range(4): c += enc_R(0x00, S11, ptr, 0, ptr, 0x33)
  for ptr in (S4, S5, S6, S7):
    for _ in range(4): c += enc_R(0x00, T6, ptr, 0, ptr, 0x33)
  c += enc_I(4, T0, 0, T0, 0x13)
  c += enc_B(i_loop - len(c), A3, T0, 4, 0x63)
  for i, r in enumerate(saved_regs): c += enc_I(i*8, SP, 3, r, 0x03)
  c += enc_I(80, SP, 0, SP, 0x13)
  c += enc_I(0, RA, 0, ZERO, 0x67)
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
  code = build_matmul_u4_lmul2()
  print(f"kernel: {len(code)} bytes ({len(code)//4} instructions)")
  mem, fn = make_jit(code)
  for M, K, N in [(4,4,4), (4,4,16), (8,3,5), (4,17,19), (16,16,16), (16,16,32)]:
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
    M = K = N
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
