"""
Hand-coded RVV matmul kernel + benchmark.

Algorithm (one row of C at a time, strip-mined over N):
  for i in [0, M):
    for j in [0, N) step vl:           # vl chosen by vsetvli
      acc = 0  (vreg)
      for k in [0, K):
        a_ik     = A[i*K + k]          (scalar broadcast)
        b_k_jvec = B[k*N + j..j+vl]
        acc     += a_ik * b_k_jvec     (vfmacc.vf)
      C[i*N + j..j+vl] = acc

Registers used:
  a0=C, a1=A, a2=B, a3=M, a4=K, a5=N   (args)
  t0=i, t1=j, t2=vl, t3=k              (loop counters)
  t4, t5 = scratch (pointer math)
  fa0    = scalar A[i,k]
  v0     = accumulator
  v1     = scratch (B row vector)
"""
import ctypes, mmap, time
from tinygrad.renderer.isa.rvv import (enc_vsetvli, enc_vload, enc_vstore, enc_vop, enc_I, enc_R, enc_B,
                                       vtype, SEW32, LMUL_1)

# psABI register indices
ZERO, RA = 0, 1
T0, T1, T2, T3, T4, T5, T6 = 5, 6, 7, 28, 29, 30, 31
A0, A1, A2, A3, A4, A5 = 10, 11, 12, 13, 14, 15
FA0 = 10  # f10
V0, V1 = 0, 1

def build_matmul_kernel() -> bytes:
  """Optimized: index math hoisted; inner k-loop is just flw / vle32 / vfmacc.vf + 2 ptr bumps + branch."""
  # Extra registers: T6 = K*4 (B row stride bytes), S1 (=x9) = N*4 (C/B col stride bytes... no, used as scratch).
  S1 = 9
  c = bytearray()
  # Precompute byte strides: t6 = N*4, s1 = K*4
  c += enc_I(2, A5, 1, T6, 0x13)         # slli t6, a5, 2     ; t6 = N*4
  c += enc_I(2, A4, 1, S1, 0x13)         # slli s1, a4, 2     ; s1 = K*4 (unused if K==N, kept for clarity)
  # i = 0;  a_row_ptr (t4 init at i_loop entry); c_row_ptr (t5 init at i_loop entry)
  c += enc_I(0, ZERO, 0, T0, 0x13)       # t0 = 0  (i)
  # t4 = a1 (current A row base); t5 = a0 (current C row base)
  c += enc_R(0x00, ZERO, A1, 0, T4, 0x33)  # add t4, a1, zero
  c += enc_R(0x00, ZERO, A0, 0, T5, 0x33)  # add t5, a0, zero
  i_loop = len(c)
  # j = 0
  c += enc_I(0, ZERO, 0, T1, 0x13)       # t1 = 0  (j)
  j_loop = len(c)
  # remaining = N - j  -> t2
  c += enc_R(0x20, T1, A5, 0, T2, 0x33)
  c += enc_vsetvli(T2, T2, vtype(SEW32, LMUL_1))
  # acc = 0
  c += enc_vop(0x17, 1, 0, 0, 3, V0)
  # B-pointer for this (j,k=0): t3 = a2 + j*4   (j*4 = slli)
  c += enc_I(2, T1, 1, T3, 0x13)         # slli t3, t1, 2
  c += enc_R(0x00, T3, A2, 0, T3, 0x33)  # add  t3, a2, t3    ; t3 = &B[0,j]
  # k counter: use a6 (idx 16) as down-counter starting at K
  c += enc_R(0x00, ZERO, A4, 0, 16, 0x33)  # add a6, a4, zero
  # a_ptr (for A[i,k]) = t4 (already at A[i,0])  — but we need a separate cursor since t4 must persist for next i
  # Use a7 (idx 17) as moving A cursor
  c += enc_R(0x00, ZERO, T4, 0, 17, 0x33)  # add a7, t4, zero
  k_loop = len(c)
  c += enc_I(0, 17, 2, FA0, 0x07)        # flw fa0, 0(a7)
  c += enc_vload(0, 0, 0, 1, 0, T3, 6, V1)  # vle32.v v1, (t3)
  c += enc_vop(0x2C, 1, V1, FA0, 5, V0)   # vfmacc.vf v0, fa0, v1
  c += enc_I(4, 17, 0, 17, 0x13)         # addi a7, a7, 4     ; A cursor +=1 elem
  c += enc_R(0x00, T6, T3, 0, T3, 0x33)  # add  t3, t3, t6    ; B cursor += N*4 bytes (next row)
  c += enc_I(-1, 16, 0, 16, 0x13)        # addi a6, a6, -1    ; k--
  c += enc_B(k_loop - len(c), ZERO, 16, 1, 0x63)  # bnez a6, k_loop
  # vse32.v v0, 0(t5+j*4)   - compute output addr
  # use t3 as temp
  c += enc_I(2, T1, 1, T3, 0x13)         # slli t3, t1, 2
  c += enc_R(0x00, T3, T5, 0, T3, 0x33)  # add  t3, t5, t3    ; &C[i,j]
  c += enc_vstore(0, 0, 0, 1, 0, T3, 6, V0)
  # j += vl ; if j < N -> j_loop
  c += enc_R(0x00, T2, T1, 0, T1, 0x33)  # add t1, t1, t2
  c += enc_B(j_loop - len(c), A5, T1, 4, 0x63)  # blt t1, a5, j_loop
  # advance A and C row pointers; i++
  c += enc_R(0x00, S1, T4, 0, T4, 0x33)  # t4 += K*4
  c += enc_R(0x00, T6, T5, 0, T5, 0x33)  # t5 += N*4
  c += enc_I(1, T0, 0, T0, 0x13)         # i++
  c += enc_B(i_loop - len(c), A3, T0, 4, 0x63)  # blt t0, a3, i_loop
  c += enc_I(0, RA, 0, ZERO, 0x67)       # ret
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

def naive_matmul(A, B, M, K, N):
  C = [0.0] * (M * N)
  for i in range(M):
    for j in range(N):
      s = 0.0
      for k in range(K):
        s += A[i*K + k] * B[k*N + j]
      C[i*N + j] = s
  return C

def test_correctness():
  code = build_matmul_kernel()
  print(f"matmul kernel: {len(code)} bytes ({len(code)//4} instructions)")
  _, fn = make_jit(code)
  for M, K, N in [(1,1,1), (1,1,8), (4,4,4), (3,5,7), (8,8,8), (13,17,19)]:
    A = [(i*0.1 + 1.0) for i in range(M*K)]
    B = [(i*0.07 + 0.3) for i in range(K*N)]
    expected = naive_matmul(A, B, M, K, N)
    A_arr = (ctypes.c_float * (M*K))(*A)
    B_arr = (ctypes.c_float * (K*N))(*B)
    C_arr = (ctypes.c_float * (M*N))()
    fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    got = list(C_arr)
    ok = all(abs(got[i] - expected[i]) < max(1e-3, abs(expected[i]) * 1e-5) for i in range(M*N))
    print(f"  {M}x{K} @ {K}x{N}: {'PASS' if ok else 'FAIL'}")
    if not ok:
      for i in range(min(8, M*N)):
        print(f"    [{i}] got={got[i]}, expected={expected[i]}")
      return False
  return True

def benchmark():
  code = build_matmul_kernel()
  _, fn = make_jit(code)
  for N in [64, 128, 256, 512]:
    M = K = N
    A_arr = (ctypes.c_float * (M*K))()
    B_arr = (ctypes.c_float * (K*N))()
    C_arr = (ctypes.c_float * (M*N))()
    for i in range(M*K): A_arr[i] = i * 0.001
    for i in range(K*N): B_arr[i] = i * 0.002
    # warmup
    fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    # measure
    iters = max(1, 200_000_000 // (2 * M * K * N))
    t0 = time.perf_counter()
    for _ in range(iters):
      fn(ctypes.addressof(C_arr), ctypes.addressof(A_arr), ctypes.addressof(B_arr), M, K, N)
    t1 = time.perf_counter()
    flops = 2 * M * K * N
    per_iter_s = (t1 - t0) / iters
    gflops = flops / per_iter_s / 1e9
    print(f"  N={N:4d}  {iters:5d} iters  {per_iter_s*1e6:9.1f} us/iter  {gflops:6.2f} GFLOPS")

if __name__ == "__main__":
  if test_correctness():
    print("--- benchmark ---")
    benchmark()
