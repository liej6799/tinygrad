# flake8: noqa: E702
# RISC-V (RV64GC + V) native ISA backend for tinygrad.
#
# STATUS: SCAFFOLD. Encoding helpers and op enum are complete enough for a
# minimal float32 matmul kernel; instruction selection is a stub. Until the
# isel_matcher is filled in, RVVRenderer.render() will refuse to emit code
# for anything beyond a trivial smoke test.
#
# References:
#   - RISC-V ISA Manual Vol I (unprivileged): https://github.com/riscv/riscv-isa-manual
#   - RVV 1.0 spec (PDF):                    https://docs.riscv.org/reference/isa/extensions/vector/_attachments/riscv-v-spec.pdf
#     (text-extracted copy in repo root as rvv-spec.txt; §refs in this file point to it)
#   - psABI:                                 https://github.com/riscv-non-isa/riscv-elf-psabi-doc
#
# Section references used below:
#   §5    Instruction Formats / Operand encoding
#   §6    Configuration-Setting (vsetvli/vsetivli/vsetvl)
#   §7    Vector Loads and Stores
#   §13   Vector Floating-Point Instructions
#   §19   Vector Instruction Listing (funct6 table)

import struct
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.uop import FastEnum, auto, Ops
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, GroupOp
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
from tinygrad.helpers import getenv, CPU_COUNT, Target

def _f32_to_bits(v:float) -> int: return struct.unpack("<I", struct.pack("<f", float(v)))[0]

# ***** RVV instruction formats *****
#
# All base RV instructions are 32 bits, little-endian. The lowest two bits of
# the opcode are 0b11 (16-bit "compressed" RVC encoding uses 0b00/0b01/0b10).
# We never emit RVC — it's harder to encode and gives ~0 perf benefit for hot
# matmul kernels.
#
# Format summary (RV base):
#   R-type:  funct7[31:25] rs2[24:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]
#   I-type:  imm[31:20]                rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]   (signed 12-bit imm)
#   S-type:  imm[31:25] rs2[24:20] rs1[19:15] funct3[14:12] imm[11:7]    opcode[6:0]
#   B-type:  imm[12|10:5] rs2 rs1 funct3 imm[4:1|11] opcode                            (PC-relative branch)
#   U-type:  imm[31:12] rd opcode                                                      (upper-20-bit immediate)
#   J-type:  imm[20|10:1|11|19:12] rd opcode                                           (PC-relative jump)
#
# RVV adds vector instructions that share the encoding space but use distinct
# opcodes (OP-V=0x57) and `vm` (mask) + `funct6` fields. Vector instructions
# form variants by operand source: OPIVV (vv), OPIVX (vx), OPIVI (vi),
# OPFVV (vv float), OPFVF (vf scalar-float-broadcast), etc.

def _check_imm12(v:int) -> int:
  if not (-2048 <= v <= 2047): raise ValueError(f"imm12 out of range: {v}")
  return v & 0xFFF

def _check_imm5(v:int) -> int:
  if not (-16 <= v <= 15): raise ValueError(f"imm5 out of range: {v}")
  return v & 0x1F

# ----- Base RV (RV32I/RV64I) encoders -----

def enc_R(funct7:int, rs2:int, rs1:int, funct3:int, rd:int, opcode:int) -> bytes:
  w = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
  return struct.pack("<I", w)

def enc_I(imm:int, rs1:int, funct3:int, rd:int, opcode:int) -> bytes:
  w = (_check_imm12(imm) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
  return struct.pack("<I", w)

def enc_S(imm:int, rs2:int, rs1:int, funct3:int, opcode:int) -> bytes:
  i = _check_imm12(imm)
  w = ((i >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((i & 0x1F) << 7) | opcode
  return struct.pack("<I", w)

def enc_B(imm:int, rs2:int, rs1:int, funct3:int, opcode:int) -> bytes:
  # imm is in bytes, must be even, signed 13-bit range
  if imm & 1: raise ValueError(f"branch imm not 2-byte aligned: {imm}")
  if not (-(1<<12) <= imm <= (1<<12)-1): raise ValueError(f"branch imm out of range: {imm}")
  i = imm & 0x1FFE
  w = (((i>>12)&1)<<31) | (((i>>5)&0x3F)<<25) | (rs2<<20) | (rs1<<15) | (funct3<<12) | (((i>>1)&0xF)<<8) | (((i>>11)&1)<<7) | opcode
  return struct.pack("<I", w)

def enc_U(imm:int, rd:int, opcode:int) -> bytes:
  # imm is the upper 20 bits, given as already-shifted value (so caller passes the value to put in bits 31:12)
  w = ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode
  return struct.pack("<I", w)

def enc_J(imm:int, rd:int, opcode:int) -> bytes:
  if imm & 1: raise ValueError(f"jal imm not 2-byte aligned: {imm}")
  if not (-(1<<20) <= imm <= (1<<20)-1): raise ValueError(f"jal imm out of range: {imm}")
  i = imm & 0x1FFFFE
  w = (((i>>20)&1)<<31) | (((i>>1)&0x3FF)<<21) | (((i>>11)&1)<<20) | (((i>>12)&0xFF)<<12) | (rd << 7) | opcode
  return struct.pack("<I", w)

# ----- RVV vector encoders -----
#
# All RVV instructions use opcode 0x57 (OP-V). Major formats:
#   OPIVV (funct3=0): vector-vector integer/logical
#   OPFVV (funct3=1): vector-vector float
#   OPMVV (funct3=2): vector-vector mask/reduction
#   OPIVI (funct3=3): vector-immediate
#   OPIVX (funct3=4): vector-scalar (x integer reg)
#   OPFVF (funct3=5): vector-scalar (f float reg)
#   OPMVX (funct3=6): vector-scalar mask
# Plus vector loads/stores using opcode 0x07 (FLW-class) / 0x27 (FSW-class).
#
# vsetvli (and vsetivli) use opcode 0x57 with funct3=7, special encoding.

OPV = 0x57
OPLOAD_FP = 0x07
OPSTORE_FP = 0x27

def enc_vsetvli(rd:int, rs1:int, vtypei:int) -> bytes:
  # §6: vsetvli encoding — bit31=0, bits30:20=zimm[10:0] (vtypei), 19:15=rs1, 14:12=111, 11:7=rd, 6:0=0x57
  # vtypei layout: bits[7]=vma, [6]=vta, [5:3]=vsew, [2:0]=vlmul
  if vtypei >> 11: raise ValueError(f"vtypei out of range: {vtypei}")
  w = (0 << 31) | (vtypei << 20) | (rs1 << 15) | (7 << 12) | (rd << 7) | OPV
  return struct.pack("<I", w)

def enc_vsetivli(rd:int, uimm:int, vtypei:int) -> bytes:
  # §6: vsetivli — bits31:30=11, bits29:20=zimm[9:0] (10-bit vtypei), 19:15=uimm[4:0], 14:12=111, 11:7=rd, 6:0=0x57
  if not (0 <= uimm < 32): raise ValueError(f"vsetivli uimm out of range: {uimm}")
  if vtypei >> 10: raise ValueError(f"vtypei out of range: {vtypei}")
  w = (0b11 << 30) | (vtypei << 20) | (uimm << 15) | (7 << 12) | (rd << 7) | OPV
  return struct.pack("<I", w)

def enc_vop(funct6:int, vm:int, vs2:int, vs1_or_rs1_or_imm:int, funct3:int, vd:int) -> bytes:
  w = (funct6 << 26) | ((vm & 1) << 25) | (vs2 << 20) | (vs1_or_rs1_or_imm << 15) | (funct3 << 12) | (vd << 7) | OPV
  return struct.pack("<I", w)

def enc_vload(nf:int, mew:int, mop:int, vm:int, lumop:int, rs1:int, width:int, vd:int) -> bytes:
  # unit-stride vector load: VL{e8,e16,e32,e64}.v
  # funct7-ish layout: [31:29]=nf, [28]=mew, [27:26]=mop, [25]=vm, [24:20]=lumop, [19:15]=rs1, [14:12]=width, [11:7]=vd, opcode=0x07
  w = (nf<<29) | (mew<<28) | (mop<<26) | ((vm&1)<<25) | (lumop<<20) | (rs1<<15) | (width<<12) | (vd<<7) | OPLOAD_FP
  return struct.pack("<I", w)

def enc_vstore(nf:int, mew:int, mop:int, vm:int, sumop:int, rs1:int, width:int, vs3:int) -> bytes:
  w = (nf<<29) | (mew<<28) | (mop<<26) | ((vm&1)<<25) | (sumop<<20) | (rs1<<15) | (width<<12) | (vs3<<7) | OPSTORE_FP
  return struct.pack("<I", w)

# ----- vtype field encoding -----
# SEW: selected element width. 0=8, 1=16, 2=32, 3=64.
# LMUL: vector register grouping. 0=m1, 1=m2, 2=m4, 3=m8, 5=mf8, 6=mf4, 7=mf2.
# vta: tail-agnostic (1) vs tail-undisturbed (0).
# vma: mask-agnostic (1) vs mask-undisturbed (0).
def vtype(sew:int, lmul:int, vta:int=1, vma:int=1) -> int:
  return ((vma&1)<<7) | ((vta&1)<<6) | ((sew&7)<<3) | (lmul&7)

SEW8, SEW16, SEW32, SEW64 = 0, 1, 2, 3
LMUL_F8, LMUL_F4, LMUL_F2 = 5, 6, 7
LMUL_1, LMUL_2, LMUL_4, LMUL_8 = 0, 1, 2, 3

# ***** RVV ops enum *****

class RVVOps(FastEnum):
  # pseudo-ops
  LABEL = auto(); FRAME_INDEX = auto()
  # scalar GPR ops (RV64I subset we need)
  ADD = auto(); ADDI = auto(); SUB = auto(); MUL = auto()
  AND = auto(); ANDI = auto(); OR = auto(); ORI = auto(); XOR = auto(); XORI = auto()
  SLL = auto(); SLLI = auto(); SRL = auto(); SRLI = auto(); SRA = auto(); SRAI = auto()
  LUI = auto(); AUIPC = auto()
  LD = auto(); LW = auto(); LH = auto(); LB = auto(); LWU = auto(); LHU = auto(); LBU = auto()
  SD = auto(); SW = auto(); SH = auto(); SB = auto()
  # branch / jump
  BEQ = auto(); BNE = auto(); BLT = auto(); BGE = auto(); BLTU = auto(); BGEU = auto()
  JAL = auto(); JALR = auto()
  # scalar float (F + D)
  FLW = auto(); FLD = auto(); FSW = auto(); FSD = auto()
  FADD_S = auto(); FSUB_S = auto(); FMUL_S = auto(); FDIV_S = auto(); FSQRT_S = auto()
  FMIN_S = auto(); FMAX_S = auto()
  # scalar float casts
  FCVT_W_S = auto(); FCVT_S_W = auto()        # f32 <-> i32 (signed)
  FMV_W_X = auto(); FMV_X_W = auto()
  # ----- RVV -----
  VSETVLI = auto(); VSETIVLI = auto()
  # vector loads/stores (unit stride)
  VLE8_V = auto(); VLE16_V = auto(); VLE32_V = auto(); VLE64_V = auto()
  VSE8_V = auto(); VSE16_V = auto(); VSE32_V = auto(); VSE64_V = auto()
  # vector integer arith (.vv / .vx / .vi where applicable)
  VADD_VV = auto(); VADD_VX = auto(); VADD_VI = auto()
  VSUB_VV = auto(); VSUB_VX = auto()
  VMUL_VV = auto(); VMUL_VX = auto()
  # vector float arith
  VFADD_VV = auto(); VFADD_VF = auto()
  VFSUB_VV = auto(); VFSUB_VF = auto(); VFRSUB_VF = auto()   # vfrsub: vd = rs1_scalar - vs2
  VFMUL_VV = auto(); VFMUL_VF = auto()
  VFDIV_VV = auto(); VFDIV_VF = auto(); VFRDIV_VF = auto()   # vfrdiv: vd = rs1_scalar / vs2
  VFMACC_VV = auto(); VFMACC_VF = auto()  # vd[i] += vs1[i]*vs2[i]
  VFMSAC_VV = auto()
  VFMIN_VV = auto(); VFMAX_VV = auto()
  VFSQRT_V = auto()
  # vector casts (VFUNARY0 sub-ops via vs1 field)
  VFCVT_X_F_V = auto()    # f32 -> i32 (truncate, rtz)
  VFCVT_F_X_V = auto()    # i32 -> f32 (signed)
  # vector moves / broadcasts
  VMV_V_V = auto(); VMV_V_X = auto(); VMV_V_I = auto(); VMV_V_F = auto()
  # reduction (single-pass) — §14
  VFREDOSUM_VS = auto(); VFREDUSUM_VS = auto(); VFREDMAX_VS = auto(); VFREDMIN_VS = auto()
  # scalar-from-vector move (vfmv.f.s — §16.2)
  VFMV_F_S = auto(); VFMV_S_F = auto()
  # pseudo lane extract: vfmv.f.s for lane0; vslidedown.vi v31, v, lane + vfmv.f.s for lane>0.
  VEXTRACT_F32_0 = auto(); VEXTRACT_F32_1 = auto(); VEXTRACT_F32_2 = auto(); VEXTRACT_F32_3 = auto()
  # return
  RET = auto()

# ***** RISC-V registers *****
# ABI names from psABI. We list the index next to the canonical name.
#   x0 = zero, x1 = ra (return addr), x2 = sp, x3 = gp, x4 = tp
#   x5-7 = t0-t2 (temps), x8 = s0/fp, x9 = s1, x10-17 = a0-a7 (args/ret)
#   x18-27 = s2-s11 (saved), x28-31 = t3-t6 (temps)
#   f0-7 = ft0-7, f8-9 = fs0-1, f10-17 = fa0-7, f18-27 = fs2-11, f28-31 = ft8-11
#   v0-v31 = vector regs (v0 is also the mask register)

# scalar GPRs
ZERO = Register("zero", 0)
RA = Register("ra", 1); SP = Register("sp", 2); GP = Register("gp", 3); TP = Register("tp", 4)
T0, T1, T2 = Register("t0", 5), Register("t1", 6), Register("t2", 7)
FP = Register("fp", 8); S1 = Register("s1", 9)
A0, A1, A2, A3, A4, A5, A6, A7 = (Register(f"a{i}", 10+i) for i in range(8))
S2, S3, S4, S5, S6, S7, S8, S9, S10, S11 = (Register(f"s{i+2}", 18+i) for i in range(10))
T3, T4, T5, T6 = (Register(f"t{i+3}", 28+i) for i in range(4))
GPR = (ZERO, RA, SP, GP, TP, T0, T1, T2, FP, S1, A0, A1, A2, A3, A4, A5, A6, A7,
       S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, T3, T4, T5, T6)
# GPRs available for general allocation (excluding zero/ra/sp/gp/tp/fp). regalloc should treat saved (s*) as callee-saved.
WGPR = (T0, T1, T2, S1, A0, A1, A2, A3, A4, A5, A6, A7, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, T3, T4, T5, T6)

# scalar FPRs
FT = tuple(Register(f"ft{i}", i) for i in range(8))            # f0-f7
FS0_1 = (Register("fs0", 8), Register("fs1", 9))               # f8-f9
FA = tuple(Register(f"fa{i}", 10+i) for i in range(8))         # f10-f17
FS2_11 = tuple(Register(f"fs{i+2}", 18+i) for i in range(10))  # f18-f27
FT8_11 = tuple(Register(f"ft{i+8}", 28+i) for i in range(4))   # f28-f31
FPR = FT + FS0_1 + FA + FS2_11 + FT8_11
WFPR = FPR  # all FPRs are writable

# vector regs; v0 is reserved as mask register; v31 is reserved as pseudo-op scratch.
VREG = tuple(Register(f"v{i}", i) for i in range(32))
WVREG = VREG[1:31]  # exclude v0 (mask register) and v31 (scratch) from general allocation

# psABI: s0-s11 are callee-saved GPRs, fs0-fs11 are callee-saved FPRs.
CALLEE_SAVED = (FP, S1) + tuple(GPR[i] for i in range(18, 28)) + FS0_1 + FS2_11

# ***** RVV legalization (extra_matcher) *****
# STUB: float16/bfloat16/fp8 lowering, integer-size promotion.
# For now we only support float32 scalar+vector and int32/int64 scalar.
extra_matcher = PatternMatcher([])

# ***** RVV pre-instruction-selection *****
# Re-vectorize STACK(<nested ALU over GEPs and CONSTs>, ...) -> vector ALU on underlying vectors.
# Children that contain only GEP(_, i) leaves and CONSTs are rewritten by replacing GEP(v, i) with v
# (the underlying vector). CONSTs are kept as scalars in the rewritten tree; the isel patterns below
# detect "vec ALU scalar CONST" and emit .vf variants with a LUI+FMV.W.X broadcast.
def _revec(expr:UOp, i:int, vec_dt:DType) -> UOp|None:
  if expr.op is Ops.GEP:
    return expr.src[0] if expr.arg == (i,) else None
  if expr.op is Ops.CONST:
    return expr  # keep scalar; downstream .vf isel handles the broadcast
  if expr.op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.FDIV) and len(expr.src) == 2:
    a = _revec(expr.src[0], i, vec_dt)
    b = _revec(expr.src[1], i, vec_dt)
    if a is None or b is None: return None
    # At least one side must be a vector for the result to be vector
    if a.dtype.count == 1 and b.dtype.count == 1: return None
    return expr.replace(dtype=vec_dt, src=(a, b))
  # Unary ops over vectors: SQRT, CAST (only the f32<->i32 case here).
  if expr.op is Ops.SQRT and len(expr.src) == 1:
    a = _revec(expr.src[0], i, vec_dt)
    if a is None or a.dtype.count == 1: return None
    return expr.replace(dtype=vec_dt, src=(a,))
  if expr.op is Ops.CAST and len(expr.src) == 1:
    a = _revec(expr.src[0], i, expr.src[0].dtype.scalar().vec(vec_dt.count))
    if a is None or a.dtype.count == 1: return None
    return expr.replace(dtype=vec_dt, src=(a,))
  # Scalar LOAD<INDEX<base, idx>> at lane i -> a vec LOAD reading N consecutive elements.
  # The per-lane idx structure is one of:
  #   lane 0: idx = X                        (any expression — possibly a single CONST=0,
  #                                            or a dynamic expression like SHL<RANGE, 2>)
  #   lane i: idx = ADD(X, CONST(i))         (X is the SAME expression as lane 0's idx)
  # The "underlying" vec LOAD then reads N elements starting at base[X]. All N lanes
  # construct the SAME LOAD UOp (UOps are interned by structural equality), so
  # _all_nested's identity check passes.
  #
  # The new address is wrapped in a CAST<ptr, ptr> so the existing pre_isel CAST<ptr, ptr>
  # rewrite converts it to NOOP<base, X> (n_src=2), which keeps `base` as a real child
  # of the NOOP. Without the CAST wrapper the isel INDEX<base, CONST=0> matcher would
  # fire on a bare INDEX with CONST=0 idx and produce NOOP<empty_src> (it replaces op
  # on base directly), swallowing the PARAM/leaf so abi() never tags it.
  if expr.op is Ops.LOAD and len(expr.src) == 1 and expr.dtype.count == 1:
    addr = expr.src[0]
    if addr.op is Ops.INDEX and len(addr.src) == 2:
      base, idx = addr.src[0], addr.src[1]
      if i == 0:
        base_idx = idx
      elif idx.op is Ops.CONST and idx.arg == i:
        # fully-unrolled small kernel: lane i has bare CONST=i, base_idx = CONST(0)
        base_idx = UOp.const(idx.dtype, 0)
      elif idx.op is Ops.ADD and len(idx.src) == 2:
        # loop kernel: lane i has ADD(X, CONST(i)) or symmetric — strip the +i
        a, b = idx.src
        if a.op is Ops.CONST and a.arg == i: base_idx = b
        elif b.op is Ops.CONST and b.arg == i: base_idx = a
        else: return None
      else: return None
      new_idx = UOp(Ops.INDEX, base.dtype, (base, base_idx))   # PtrDType, preserves srcs
      casted = UOp(Ops.CAST, new_idx.dtype, (new_idx,))         # CAST<ptr, ptr> -> NOOP via pre_isel
      return UOp(Ops.LOAD, expr.dtype.vec(vec_dt.count), (casted,))
    return None
  return None

def _all_nested(stack:UOp) -> UOp|None:
  if stack.op is not Ops.STACK or not stack.src: return None
  ref = _revec(stack.src[0], 0, stack.dtype)
  if ref is None: return None
  # All children must rewrite to the SAME vector expression (dedup-identity).
  for i, s in enumerate(stack.src):
    if _revec(s, i, stack.dtype) is not ref: return None
  return ref

pre_isel_matcher = PatternMatcher([
  # CAST between two PtrDType is a no-op
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0].replace(op=Ops.NOOP) if isinstance(x.dtype, PtrDType) and isinstance(x.src[0].dtype, PtrDType) else None),
  # noop of a noop
  (UPat(Ops.NOOP, src=(UPat(Ops.NOOP),), name="x"), lambda x: x.replace(src=x.src[0].src)),
  # NEG -> SUB(0, x). Lets re-vectorization and the SUB(const,vec)->vfrsub.vf isel handle it.
  (UPat(Ops.NEG, name="x"), lambda x: UOp(Ops.SUB, x.dtype, (x.const_like(0),) + x.src)),
  # Reduction: detect ADD-tree over all lanes of one vector -> vfredusum chain.
  # Must come before STACK re-vectorize so scalar ADDs get folded before any STACK rewrite.
  (UPat(Ops.ADD, name="x"), lambda x: _detect_reduction(x)),
  # Re-vectorize STACK(<nested ALU over GEPs, CONSTs>, ...)
  (UPat(Ops.STACK, name="x"), lambda x: _all_nested(x)),
])

# ***** RVV instruction selection *****

def def_reg(dt:DType, reg:Register|None=None) -> UOp:
  return UOp(Ops.DEFINE_REG, dt, tag=None if reg is None else (reg,))

def imm(dt:DType, v:int) -> UOp: return UOp.const(dt, v).rtag()

def _f32_const_to_freg(c:UOp) -> UOp:
  """Materialize an arbitrary f32 constant in a fresh float register.
  Returns the FMV.W.X INS UOp. Chain: LUI hi20 [+ ADDI lo12 with sign-correction] -> FMV.W.X.
  Handles any 32-bit float bit pattern, including sign-bit-set (negative) and lo12 != 0."""
  bits = _f32_to_bits(c.arg) & 0xFFFFFFFF
  lo12 = bits & 0xFFF
  hi20 = (bits >> 12) & 0xFFFFF
  # ADDI's imm[11:0] is sign-extended. If bit 11 is set in lo12, it would subtract — bump hi20 by 1
  # and use a negative ADDI imm so LUI(hi20)+ADDI(negative) reconstructs the desired bit pattern.
  if lo12 & 0x800:
    hi20 = (hi20 + 1) & 0xFFFFF
    addi_imm = lo12 - 0x1000   # in [-2048, -1]
  else:
    addi_imm = lo12             # in [0, 2047]
  # LUI itself sign-extends its 20-bit immediate to 64 bits on RV64. The low 32 bits are still correct,
  # which is all FMV.W.X reads. So negative hi20 (top bit set) doesn't need special handling.
  lui = UOp(Ops.INS, dtypes.uint64, (UOp.const(dtypes.int32, hi20).rtag(),), RVVOps.LUI)
  intreg = lui if addi_imm == 0 else UOp(Ops.INS, dtypes.uint64, (lui, UOp.const(dtypes.int32, addi_imm).rtag()), RVVOps.ADDI)
  return UOp(Ops.INS, dtypes.float32, (intreg,), RVVOps.FMV_W_X)

def _emit_vf(x:UOp, v:UOp, c:UOp, vf_op) -> UOp:
  """Emit `<materialize const into f-reg>; <vf_op> vd, v, ft` for any float CONST."""
  return x.ins(vf_op, src=(v, _f32_const_to_freg(c)))

def _broadcast_f32_to_vec(c:UOp, vec_dt:DType) -> UOp:
  """Emit a vector broadcast of a scalar float CONST. Chain ends in VMV_V_F INS of the vector dtype."""
  return UOp(Ops.INS, vec_dt, (_f32_const_to_freg(c),), RVVOps.VMV_V_F)

def _is_single_use(ctx:IselContext, parent:UOp, src:UOp) -> bool:
  """True if src is used exactly once (as a src of parent)."""
  return len(ctx.uses[src]) == parent.src.count(src) == 1

def _emit_vfmacc(ctx:IselContext, x:UOp) -> UOp|None:
  """Detect ADD(MUL(a,b), c) (or symmetric) on vec f32 where all of a/b/c are vectors,
  and fuse into vfmacc.vv. Semantics: vfmacc.vv vd, vs1, vs2  =>  vd[i] = vs1[i]*vs2[i] + vd[i].
  Skips when any operand is a scalar CONST (would need vfmacc.vf, not yet implemented)."""
  if x.dtype.count <= 1 or x.dtype.scalar() is not dtypes.float32: return None
  for mul_idx in (0, 1):
    mul = x.src[mul_idx]
    other = x.src[1 - mul_idx]
    if mul.op is not Ops.MUL or mul.dtype != x.dtype or len(mul.src) != 2: continue
    # All three operands must already be vectors (not scalar CONSTs)
    if any(s.dtype.count != x.dtype.count for s in (mul.src[0], mul.src[1], other)): continue
    if not _is_single_use(ctx, x, mul): continue
    return x.ins(RVVOps.VFMACC_VV, src=(other, mul.src[0], mul.src[1]))
  return None

def _collect_reduction_geps(expr:UOp) -> list[UOp]|None:
  """Walk a scalar ADD-tree, collecting all GEP leaves. Returns the list or None if the tree
  isn't a pure ADD-over-GEPs."""
  if expr.op is Ops.GEP and expr.dtype.count == 1: return [expr]
  if expr.op is Ops.ADD and len(expr.src) == 2 and expr.dtype.count == 1:
    l = _collect_reduction_geps(expr.src[0])
    r = _collect_reduction_geps(expr.src[1])
    if l is None or r is None: return None
    return l + r
  return None

def _detect_reduction(expr:UOp) -> UOp|None:
  """If expr is `sum(GEP(v, i) for i in range(v.dtype.count))` (ADD-tree over ALL lanes of one
  vector AND nothing else), emit a vfredusum chain. The strictness ensures we don't fire on
  partial subtrees during a larger reduction (which would create a malformed graph mixing
  vfredusum with the per-lane FADD_S chain that's already in place)."""
  geps = _collect_reduction_geps(expr)
  if geps is None: return None
  vec = geps[0].src[0]
  if vec.dtype.count <= 1: return None
  if len(geps) != vec.dtype.count: return None         # must cover EVERY lane
  if not all(g.src[0] is vec for g in geps): return None
  if sorted(g.arg[0] for g in geps) != list(range(vec.dtype.count)): return None
  # Emit: vmv.v.i v_init, 0 ; vfredusum.vs v_red, vec, v_init ; vfmv.f.s fd, v_red
  v_init = UOp(Ops.INS, vec.dtype, (UOp.const(dtypes.int32, 0).rtag(),), RVVOps.VMV_V_I)
  v_red  = UOp(Ops.INS, vec.dtype, (vec, v_init), RVVOps.VFREDUSUM_VS)
  return UOp(Ops.INS, expr.dtype, (v_red,), RVVOps.VFMV_F_S)

def _scalarize_gep_load(x:UOp) -> UOp|None:
  """Lower GEP(LOAD(vec_ptr), lane) to a scalar LOAD from the same base plus lane.
  Tinygrad uses this in small matmul kernels: load a short row vector, then GEP
  individual lanes for scalar FMUL/FADD. Raw GEP is not an ISA op, so remove it
  before linear regalloc."""
  if x.op is not Ops.GEP or len(x.arg) != 1 or x.src[0].op is not Ops.LOAD or not (1 < x.src[0].dtype.count <= 2): return None
  lane = x.arg[0]
  addr = x.src[0].src[0]
  if addr.op is Ops.CAST and isinstance(addr.dtype, PtrDType) and isinstance(addr.src[0].dtype, PtrDType): addr = addr.src[0]
  if lane == 0: return addr.load(dtype=x.dtype)
  if addr.op is Ops.INDEX and len(addr.src) == 2:
    return UOp(Ops.INDEX, addr.dtype, (addr.src[0], addr.src[1] + UOp.const(dtypes.int, lane))).load(dtype=x.dtype)
  return addr.index(UOp.const(dtypes.int, lane)).load(dtype=x.dtype)

def _resolve_addr(a:UOp) -> UOp:
  """Given the `addr` src of a LOAD/STORE — possibly wrapped in NOOP(ptr, elem_idx) — return a UOp
  whose register holds the actual byte address. If elem_idx is literal 0, returns ptr unchanged.
  Otherwise emits SLLI (idx * sizeof(elem)) + ADD (ptr + offset).  For f32 we scale by 2 bits.
  Assumes the LOAD/STORE is for f32 (sizeof=4 -> shift left by 2)."""
  if a.op is not Ops.NOOP or len(a.src) < 2: return a
  ptr, idx = a.src[0], a.src[1]
  # detect literal-zero offset: ADDI rd, zero, 0
  if (idx.op is Ops.INS and idx.arg is RVVOps.ADDI and len(idx.src) == 2
      and idx.src[1].op is Ops.CONST and idx.src[1].arg == 0): return ptr
  # scale idx by sizeof(f32) = 4 via SLLI by 2
  scaled = UOp(Ops.INS, dtypes.uint64, (idx, UOp.const(dtypes.int32, 2).rtag()), RVVOps.SLLI)
  return UOp(Ops.INS, dtypes.uint64, (ptr, scaled), RVVOps.ADD)

def _scalar_mem_addr(base:UOp, idx:UOp, itemsize:int=4) -> tuple[UOp, UOp]:
  """Return (base_reg, imm_off) for a scalar memory op. Static indices use the imm field
  (with byte scaling); dynamic element indices are converted to byte addresses with SLLI+ADD."""
  shift_amt = {1:0, 2:1, 4:2, 8:3}[itemsize]
  if idx.op is Ops.CONST: return base, UOp.const(dtypes.int32, idx.arg * itemsize).rtag()
  if shift_amt == 0:
    return UOp(Ops.INS, dtypes.uint64, (base, idx), RVVOps.ADD), UOp.const(dtypes.int32, 0).rtag()
  scaled = UOp(Ops.INS, dtypes.uint64, (idx, UOp.const(dtypes.int32, shift_amt).rtag()), RVVOps.SLLI)
  return UOp(Ops.INS, dtypes.uint64, (base, scaled), RVVOps.ADD), UOp.const(dtypes.int32, 0).rtag()

def _extract_f32_lane(x:UOp) -> UOp|None:
  if x.op is not Ops.GEP or len(x.arg) != 1 or x.src[0].dtype.count <= 1 or x.src[0].dtype.scalar() is not dtypes.float32: return None
  return x.ins((RVVOps.VEXTRACT_F32_0, RVVOps.VEXTRACT_F32_1, RVVOps.VEXTRACT_F32_2, RVVOps.VEXTRACT_F32_3)[x.arg[0]], src=(x.src[0],)) \
    if 0 <= x.arg[0] <= 3 else None

# RISC-V psABI: integer args in a0-a7, float args in fa0-fa7. Up to 8 of each in regs; rest on stack.
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  if isinstance(x.tag, tuple): return None
  i = ctx.func_args.index(x)
  if i >= 8: raise NotImplementedError(f"RVV: >8 args (stack args not supported yet, got arg #{i})")
  # All our kernel args so far are pointers or ints, go to a0..a7
  return x.replace(tag=((A0, A1, A2, A3, A4, A5, A6, A7)[i],))

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  if x.op is Ops.DEFINE_REG and x.tag is not None: return None
  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and (not x.tag or (hasattr(x.tag[0], '_cons') and x.tag[0]._cons)): return None
  if isinstance(x.tag, tuple): return x.replace(tag=(ctx.vreg(x.tag),))
  # pick a register pool based on dtype
  if isinstance(x.dtype, PtrDType) or x.dtype in dtypes.ints + (dtypes.bool,):
    return x.replace(tag=(ctx.vreg(WGPR),))
  if x.dtype.count > 1 or x.dtype in dtypes.floats:
    # use vregs for vectors, fregs for scalar floats
    return x.replace(tag=(ctx.vreg(WVREG if x.dtype.count > 1 else WFPR),))
  return None

isel_matcher = PatternMatcher([
  # SINK -> RET (callee-saved regs added as srcs so regalloc spills any it touches into the prologue)
  (UPat(Ops.SINK, name="x"), lambda x:
   x.replace(src=(x.ins(RVVOps.RET, src=x.src + tuple(def_reg(dtypes.uint64, r) for r in CALLEE_SAVED if isinstance(r.index, int) and r.index < 32)),))
   if not x.src or x.src[0].arg is not RVVOps.RET else None),
  # Function args -> ABI-constrained DEFINE_REG
  (UPat((Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL), name="x"), abi),
  # COPY (register-to-register move) — used by regalloc to reconcile register constraints.
  # For ints/pointers, "addi rd, rs, 0" is the canonical mov pseudo. For vectors, vmv.v.v. For floats: fsgnj.s fd, fs, fs (alias fmv.s) — not yet emitted but reachable via VMV_V_V style.
  (UPat(Ops.COPY, dtype=dtypes.ints+(dtypes.bool,), name="x"), lambda x: x.ins(RVVOps.ADDI, src=(x.src[0], imm(dtypes.int32, 0)))),
  (UPat(Ops.COPY, name="x"), lambda x: x.ins(RVVOps.ADDI, src=(x.src[0], imm(dtypes.int32, 0))) if isinstance(x.dtype, PtrDType) else (
     x.ins(RVVOps.VMV_V_V, src=(x.src[0],)) if x.dtype.count > 1 else None)),
  # Scalar LOAD/STORE: pick width-correct instruction. SD=8B for ptr/i64, SW=4B for i32, FSW=4B float.
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("disp"))), UPat.var("val")), name="x"),
   lambda base, disp, val, x:
     x.ins(RVVOps.SD, src=(val, *_scalar_mem_addr(base, disp, 8))) if isinstance(val.dtype, PtrDType) or val.dtype in (dtypes.int64, dtypes.uint64) else (
     x.ins(RVVOps.SW, src=(val, *_scalar_mem_addr(base, disp, 4))) if val.dtype in (dtypes.int32, dtypes.uint32) else (
     x.ins(RVVOps.SB, src=(val, *_scalar_mem_addr(base, disp, 1))) if val.dtype is dtypes.bool else (
     x.ins(RVVOps.FSW, src=(val, *_scalar_mem_addr(base, disp, 4))) if val.dtype is dtypes.float32 else None)))),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("disp"))),), name="x"),
   lambda base, disp, x:
     x.ins(RVVOps.LD, src=_scalar_mem_addr(base, disp, 8)) if isinstance(x.dtype, PtrDType) or x.dtype in (dtypes.int64, dtypes.uint64) else (
     x.ins(RVVOps.LW, src=_scalar_mem_addr(base, disp, 4)) if x.dtype in (dtypes.int32, dtypes.uint32) else (
     x.ins(RVVOps.LBU, src=_scalar_mem_addr(base, disp, 1)) if x.dtype is dtypes.bool else (
     x.ins(RVVOps.FLW, src=_scalar_mem_addr(base, disp, 4)) if x.dtype is dtypes.float32 else None)))),
  # small int constant -> ADDI(zero, imm). Larger constants would need LUI+ADDI (not yet).
  (UPat.cvar("x", dtypes.ints+(dtypes.bool,)), lambda x:
   x.ins(RVVOps.ADDI, src=(def_reg(dtypes.uint64, ZERO), imm(x.dtype, x.arg))) if not x.tag and -2048 <= int(x.arg) <= 2047 else None),
  # INDEX(ptr, scalar_idx) -> ptr + idx*itemsize.  For now only handle CONST 0 (no-offset)
  (UPat(Ops.INDEX, name="x"), lambda x:
   x.src[0].replace(op=Ops.NOOP) if x.src[1].op is Ops.CONST and x.src[1].arg == 0 else None),
  # Ops.RANGE: tag the loop counter with a vreg (WGPR); the actual init/cmp/jmp sequence is emitted in post_regalloc.
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(tag=(ctx.vreg(WGPR),)) if not isinstance(x.tag, tuple) else None),
  # Scalar int ops needed for address arithmetic inside loop bodies.
  (UPat(Ops.SHL, src=(UPat.var("a"), UPat.cvar("c", dtypes.ints)), name="x"),
   lambda a,c,x: x.ins(RVVOps.SLLI, src=(a, UOp.const(dtypes.int32, c.arg).rtag())) if x.dtype.count == 1 else None),
  # int ADD with const (e.g. stack-pointer adjustments emitted by regalloc prologue)
  (UPat(Ops.ADD, src=(UPat.var("a"), UPat.cvar("c", dtypes.ints)), name="x"),
   lambda a,c,x: x.ins(RVVOps.ADDI, src=(a, UOp.const(dtypes.int32, c.arg).rtag())) if x.dtype.count == 1 and -2048 <= int(c.arg) <= 2047 else None),
  # int SUB with const -> ADDI with negated imm
  (UPat(Ops.SUB, src=(UPat.var("a"), UPat.cvar("c", dtypes.ints)), name="x"),
   lambda a,c,x: x.ins(RVVOps.ADDI, src=(a, UOp.const(dtypes.int32, -int(c.arg)).rtag())) if x.dtype.count == 1 and -2047 <= int(c.arg) <= 2048 else None),
  (UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a,b,x: x.ins(RVVOps.ADD) if x.dtype.count == 1 and x.dtype in dtypes.ints else None),
  (UPat(Ops.SUB, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a,b,x: x.ins(RVVOps.SUB) if x.dtype.count == 1 and x.dtype in dtypes.ints else None),
  (UPat(Ops.MUL, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a,b,x: x.ins(RVVOps.MUL) if x.dtype.count == 1 and x.dtype in dtypes.ints else None),
  # Scalar lane extraction from vector values left over after pre-isel (small matmul/reductions).
  (UPat(Ops.GEP, name="x"), lambda x: _extract_f32_lane(x)),
  # Scalar f32 binary ALU with a CONST operand: load const to an f-reg first.
  # These must come BEFORE the generic scalar f32 patterns or else they get masked.
  (UPat(Ops.SUB, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FSUB_S, src=(_f32_const_to_freg(c), v)) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.SUB, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FSUB_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.ADD, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FADD_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.ADD, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FADD_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.MUL, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FMUL_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.MUL, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FMUL_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  # Scalar f32 ALU needed by reductions and small matmul kernels.
  (UPat(Ops.ADD, name="x"), lambda x:
   x.ins(RVVOps.FADD_S) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.MUL, name="x"), lambda x:
   x.ins(RVVOps.FMUL_S) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.SUB, name="x"), lambda x:
   x.ins(RVVOps.FSUB_S) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.FDIV, name="x"), lambda x:
   x.ins(RVVOps.FDIV_S) if x.dtype is dtypes.float32 else None),
  # Scalar f32 binary ALU with a CONST operand: load const to an f-reg first.
  (UPat(Ops.SUB, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FSUB_S, src=(_f32_const_to_freg(c), v)) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.SUB, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FSUB_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.ADD, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FADD_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.ADD, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FADD_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.MUL, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda c, v, x: x.ins(RVVOps.FMUL_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  (UPat(Ops.MUL, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda c, v, x: x.ins(RVVOps.FMUL_S, src=(v, _f32_const_to_freg(c))) if x.dtype is dtypes.float32 else None),
  # max via WHERE(CMPLT(a,b), b, a) -> fmax.s ; min via WHERE(CMPLT(a,b), a, b) -> fmin.s
  (UPat(Ops.WHERE, src=(UPat(Ops.CMPLT, src=(UPat.var("a"), UPat.var("b"))), UPat.var("c"), UPat.var("d")), name="x"),
   lambda a,b,c,d,x: (x.ins(RVVOps.FMAX_S, src=(a, b)) if (c is b and d is a) else
                      (x.ins(RVVOps.FMIN_S, src=(a, b)) if (c is a and d is b) else None))
                     if x.dtype is dtypes.float32 else None),
  # vec(N) fmax/fmin: vfmax.vv / vfmin.vv (same pattern but on vector dtypes)
  (UPat(Ops.WHERE, src=(UPat(Ops.CMPLT, src=(UPat.var("a"), UPat.var("b"))), UPat.var("c"), UPat.var("d")), name="x"),
   lambda a,b,c,d,x: (x.ins(RVVOps.VFMAX_VV, src=(a, b)) if (c is b and d is a) else
                      (x.ins(RVVOps.VFMIN_VV, src=(a, b)) if (c is a and d is b) else None))
                     if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None),
  # scalar sqrt + casts
  (UPat(Ops.SQRT, name="x"), lambda x:
   x.ins(RVVOps.FSQRT_S) if x.dtype is dtypes.float32 else
   (x.ins(RVVOps.VFSQRT_V) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None)),
  (UPat(Ops.CAST, name="x"), lambda x:
   x.ins(RVVOps.FCVT_W_S) if x.dtype is dtypes.int32 and x.src[0].dtype is dtypes.float32 else
   (x.ins(RVVOps.FCVT_S_W) if x.dtype is dtypes.float32 and x.src[0].dtype is dtypes.int32 else
    (x.ins(RVVOps.VFCVT_X_F_V) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.int32 and x.src[0].dtype.scalar() is dtypes.float32 else
     (x.ins(RVVOps.VFCVT_F_X_V) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 and x.src[0].dtype.scalar() is dtypes.int32 else None)))),
  # vec(N) LOAD f32/i32 -> VLE32_V. Element-size-based (32-bit) op handles both float32 and int32.
  # Unwraps NOOP(ptr, idx) to compute the byte address.
  (UPat(Ops.LOAD, src=(UPat.var("a"),), name="x"), lambda a,x:
   x.ins(RVVOps.VLE32_V, src=(_resolve_addr(a),))
   if x.dtype.count > 1 and x.dtype.scalar() in (dtypes.float32, dtypes.int32) else None),
  # vec(N) ADD/MUL f32 with one scalar CONST operand -> .vf variant.
  # Constant is materialized into a fresh f-reg via LUI [+ ADDI w/ sign correction] + FMV.W.X.
  # Handles ANY 32-bit float bit pattern (including negative and lo12 != 0).
  (UPat(Ops.MUL, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFMUL_VF) if x.dtype.count > 1 else None),
  (UPat(Ops.MUL, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFMUL_VF) if x.dtype.count > 1 else None),
  (UPat(Ops.ADD, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFADD_VF) if x.dtype.count > 1 else None),
  (UPat(Ops.ADD, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFADD_VF) if x.dtype.count > 1 else None),
  # SUB with scalar CONST on either side: vec-const -> vfsub.vf ; const-vec -> vfrsub.vf
  (UPat(Ops.SUB, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFSUB_VF) if x.dtype.count > 1 else None),
  (UPat(Ops.SUB, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFRSUB_VF) if x.dtype.count > 1 else None),
  # FDIV with scalar CONST: vec/const -> vfdiv.vf ; const/vec -> vfrdiv.vf
  (UPat(Ops.FDIV, src=(UPat.var("v"), UPat.cvar("c", dtypes.float32)), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFDIV_VF) if x.dtype.count > 1 else None),
  (UPat(Ops.FDIV, src=(UPat.cvar("c", dtypes.float32), UPat.var("v")), name="x"),
   lambda v,c,x: _emit_vf(x, v, c, RVVOps.VFRDIV_VF) if x.dtype.count > 1 else None),
  # vec(N) FMA fusion: ADD(MUL(a,b), c) -> vfmacc.vv (only when MUL is single-use).
  (UPat(Ops.ADD, name="x"), _emit_vfmacc),
  # vec(N) ADD/MUL/SUB f32 -> .vv (both sources are vectors). vm=1 baked into encoding.
  (UPat(Ops.ADD, name="x"), lambda x:
   x.ins(RVVOps.VFADD_VV) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None),
  (UPat(Ops.MUL, name="x"), lambda x:
   x.ins(RVVOps.VFMUL_VV) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None),
  (UPat(Ops.SUB, name="x"), lambda x:
   x.ins(RVVOps.VFSUB_VV) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None),
  (UPat(Ops.FDIV, name="x"), lambda x:
   x.ins(RVVOps.VFDIV_VV) if x.dtype.count > 1 and x.dtype.scalar() is dtypes.float32 else None),
  # STORE(ptr, scalar_const_f32): broadcast the scalar to a vec then VSE32_V (e.g. zero-fill kernels).
  # The target buffer's element count is taken from the ptr's PtrDType.
  (UPat.var("a").store(UPat.cvar("c", dtypes.float32), name="x"), lambda a,c,x:
   x.ins(RVVOps.VSE32_V, src=(_broadcast_f32_to_vec(c, dtypes.float32.vec(a.dtype.size if isinstance(a.dtype, PtrDType) else 4)),
                              _resolve_addr(a)))),
  # STORE vec(N) f32/i32 -> VSE32_V. Element-size-based (32-bit) op handles both float32 and int32.
  # Resolves NOOP(ptr, idx) and swaps srcs (orig: ptr, val) -> (val, addr).
  (UPat.var("a").store(UPat.var("b"), name="x"), lambda a,b,x:
   x.ins(RVVOps.VSE32_V, src=(b, _resolve_addr(a)))
   if b.dtype.count > 1 and b.dtype.scalar() in (dtypes.float32, dtypes.int32) else None),
  # alloc vregs to anything that defines a value
  (UPat((Ops.INS, Ops.DEFINE_REG), name="x"), alloc_vregs),
])

# ***** RVV post-register-allocation *****
# Lower RANGE/END after regalloc (when the counter has a real register).
# Pattern: RANGE(limit) -> [acc=0; LABEL_HEAD; bge acc, limit, LABEL_OUT]
#          END(_, range_acc) -> [acc += 1; jal LABEL_HEAD; LABEL_OUT]
def _lower_range(ctx, x:UOp):
  label = "_".join(str(i) for i in x.arg[:-1]) if isinstance(x.arg, tuple) else str(id(x))
  # acc = ADDI rd, zero, 0  (rd is the regalloc'd counter, taken from x.src[1:] as "after" srcs)
  acc = x.ins(RVVOps.ADDI, src=(def_reg(dtypes.uint64, ZERO), UOp.const(dtypes.int32, 0).rtag()) + x.src[1:])
  head = UOp(Ops.INS, arg=RVVOps.LABEL, tag=f".LOOP_{label}")
  # bge acc, limit, LABEL_OUT
  bge = UOp(Ops.INS, arg=RVVOps.BGE, src=(acc, x.src[0]), tag=f".LOOP_OUT_{label}")
  ctx.loop_label[acc] = label
  return (acc, [acc, head, bge])

def _lower_end(ctx, x:UOp):
  acc = x.src[1]
  label = ctx.loop_label[acc]
  # acc++ ; jal zero, LABEL_HEAD ; LABEL_OUT
  inc = acc.ins(RVVOps.ADDI, src=(acc, UOp.const(dtypes.int32, 1).rtag()))
  jal = UOp(Ops.INS, dtype=dtypes.uint64, src=(), arg=RVVOps.JAL, tag=f".LOOP_{label}")
  out = UOp(Ops.INS, arg=RVVOps.LABEL, tag=f".LOOP_OUT_{label}")
  return (jal, [inc, jal, out])

def _strip_two_addr(ctx, x:UOp):
  """For two-address INS ops: strip the first src (= accumulator, coalesced with vd). If regalloc
  failed to coalesce (vd reg != src[0] reg), prepend a COPY to put src[0] in vd. Mirrors x86.py."""
  if x.arg not in RVVRenderer.TWO_ADDRESS: return None
  nx = x.replace(src=x.src[1:])
  if x.reg != x.src[0].reg: return (nx, [ctx.ren.copy(x.src[0], x.reg), nx])
  return (nx, [nx])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), _lower_range),
  (UPat(Ops.END, name="x"), _lower_end),
  (UPat(Ops.INS, name="x"), _strip_two_addr),
])

# ***** RVV instruction encoding *****
# Returns the bytes for an instruction UOp. UOps with srcs are expected to
# have all srcs already register-allocated (each src has u.reg set).
#
# For the scaffold, we expose the encoding table. As isel patterns get added,
# they will produce Ops.INS UOps with these args and srcs in the expected
# order. Order convention used here: (rd, rs1, rs2, [imm]) for scalar;
# (vd, vs1, vs2, [vm]) for vector; (vd, rs1, [vm]) for unit-stride load.

def _r(u:UOp, i:int) -> int: return u.src[i].reg.index  # type: ignore[union-attr]
def _idx(u:UOp) -> int: return u.reg.index              # type: ignore[union-attr]
def _imm(u:UOp, i:int) -> int: return u.src[i].arg

def _enc_vextract_f32(x:UOp, lane:int) -> bytes:
  if lane == 0: return enc_vop(0x10, 1, _r(x,0), 0, 1, _idx(x))  # vfmv.f.s fd, vs2
  # vslidedown.vi v31, vs2, lane ; vfmv.f.s fd, v31. v31 is reserved scratch.
  return enc_vop(0x0F, 1, _r(x,0), lane, 3, 31) + enc_vop(0x10, 1, 31, 0, 1, _idx(x))

# Each entry takes the Ops.INS UOp `x` and returns its encoded bytes.
# Branch/jump targets are encoded with placeholder 0; render() fixes them up.
# --- Encoding convention ---
# Each lambda takes an Ops.INS UOp `x` where:
#   x.reg     = destination register (rd, vd)
#   x.src     = operand list, ordered as how it appears in assembly (after rd)
#               R-type:  src=(rs1, rs2)        — e.g. ADD rd, rs1, rs2
#               I-type:  src=(rs1, imm_uop)    — e.g. ADDI rd, rs1, imm
#               U-type:  src=(imm_uop,)        — LUI rd, imm
#               S-type:  src=(rs2_val, rs1_base, imm_uop)  — SW rs2, imm(rs1)
#               B-type:  src=(rs1, rs2), x.tag=label string  — BNE rs1, rs2, label
#               vector:  src ordered as the spec mnemonic, ending with vm immediate

encodings = {
  # --- RV64I scalar (R-type funct7|rs2|rs1|funct3|rd|opcode) ---
  RVVOps.ADD:  lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x0, _idx(x), 0x33),
  RVVOps.SUB:  lambda x: enc_R(0x20, _r(x,1), _r(x,0), 0x0, _idx(x), 0x33),
  RVVOps.MUL:  lambda x: enc_R(0x01, _r(x,1), _r(x,0), 0x0, _idx(x), 0x33),
  RVVOps.AND:  lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x7, _idx(x), 0x33),
  RVVOps.OR:   lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x6, _idx(x), 0x33),
  RVVOps.XOR:  lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x4, _idx(x), 0x33),
  RVVOps.SLL:  lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x1, _idx(x), 0x33),
  RVVOps.SRL:  lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x5, _idx(x), 0x33),
  RVVOps.SRA:  lambda x: enc_R(0x20, _r(x,1), _r(x,0), 0x5, _idx(x), 0x33),
  # I-type: src=(rs1, imm)
  RVVOps.ADDI: lambda x: enc_I(_imm(x,1), _r(x,0), 0x0, _idx(x), 0x13),
  RVVOps.ANDI: lambda x: enc_I(_imm(x,1), _r(x,0), 0x7, _idx(x), 0x13),
  RVVOps.ORI:  lambda x: enc_I(_imm(x,1), _r(x,0), 0x6, _idx(x), 0x13),
  RVVOps.XORI: lambda x: enc_I(_imm(x,1), _r(x,0), 0x4, _idx(x), 0x13),
  RVVOps.SLLI: lambda x: enc_I(_imm(x,1)&0x3F, _r(x,0), 0x1, _idx(x), 0x13),
  RVVOps.SRLI: lambda x: enc_I(_imm(x,1)&0x3F, _r(x,0), 0x5, _idx(x), 0x13),
  RVVOps.SRAI: lambda x: enc_I(0x400|(_imm(x,1)&0x3F), _r(x,0), 0x5, _idx(x), 0x13),
  # U-type: src=(imm,)
  RVVOps.LUI:   lambda x: enc_U(_imm(x,0), _idx(x), 0x37),
  RVVOps.AUIPC: lambda x: enc_U(_imm(x,0), _idx(x), 0x17),
  # loads (I-type): src=(rs1_base, imm)
  RVVOps.LB:  lambda x: enc_I(_imm(x,1), _r(x,0), 0x0, _idx(x), 0x03),
  RVVOps.LH:  lambda x: enc_I(_imm(x,1), _r(x,0), 0x1, _idx(x), 0x03),
  RVVOps.LW:  lambda x: enc_I(_imm(x,1), _r(x,0), 0x2, _idx(x), 0x03),
  RVVOps.LD:  lambda x: enc_I(_imm(x,1), _r(x,0), 0x3, _idx(x), 0x03),
  RVVOps.LBU: lambda x: enc_I(_imm(x,1), _r(x,0), 0x4, _idx(x), 0x03),
  RVVOps.LHU: lambda x: enc_I(_imm(x,1), _r(x,0), 0x5, _idx(x), 0x03),
  RVVOps.LWU: lambda x: enc_I(_imm(x,1), _r(x,0), 0x6, _idx(x), 0x03),
  # stores (S-type): src=(rs2_val, rs1_base, imm)
  RVVOps.SB: lambda x: enc_S(_imm(x,2), _r(x,0), _r(x,1), 0x0, 0x23),
  RVVOps.SH: lambda x: enc_S(_imm(x,2), _r(x,0), _r(x,1), 0x1, 0x23),
  RVVOps.SW: lambda x: enc_S(_imm(x,2), _r(x,0), _r(x,1), 0x2, 0x23),
  RVVOps.SD: lambda x: enc_S(_imm(x,2), _r(x,0), _r(x,1), 0x3, 0x23),
  # branches (B-type, target via x.tag fixed up in render): src=(rs1, rs2)
  RVVOps.BEQ:  lambda x: enc_B(0, _r(x,1), _r(x,0), 0x0, 0x63),
  RVVOps.BNE:  lambda x: enc_B(0, _r(x,1), _r(x,0), 0x1, 0x63),
  RVVOps.BLT:  lambda x: enc_B(0, _r(x,1), _r(x,0), 0x4, 0x63),
  RVVOps.BGE:  lambda x: enc_B(0, _r(x,1), _r(x,0), 0x5, 0x63),
  RVVOps.BLTU: lambda x: enc_B(0, _r(x,1), _r(x,0), 0x6, 0x63),
  RVVOps.BGEU: lambda x: enc_B(0, _r(x,1), _r(x,0), 0x7, 0x63),
  # JAL: when used as an unconditional jump (tag is a label string), rd=0 (write to x0/zero — discarded).
  RVVOps.JAL:  lambda x: enc_J(0, 0 if isinstance(x.tag, str) else _idx(x), 0x6F),
  RVVOps.JALR: lambda x: enc_I(_imm(x,1), _r(x,0), 0x0, _idx(x), 0x67),
  # scalar float (F): src=(rs1, rs2)
  # scalar float (F): src=(rs1, rs2). funct3 in {7=DYN rounding, 0=fmin, 1=fmax}
  RVVOps.FMIN_S: lambda x: enc_R(0x14, _r(x,1), _r(x,0), 0x0, _idx(x), 0x53),
  RVVOps.FMAX_S: lambda x: enc_R(0x14, _r(x,1), _r(x,0), 0x1, _idx(x), 0x53),
  # scalar f32 casts (rs2 field selects the variant). FCVT.W.S funct7=0x60 rs2=0. FCVT.S.W funct7=0x68 rs2=0.
  # FCVT.W.S uses funct3=1 (RTZ) to truncate toward zero, matching tinygrad/Python/C/NumPy f32->i32 semantics
  # (DYN reads fcsr.frm which defaults to RNE on this hardware — would round 1.5→2 instead of 1.5→1).
  # FCVT.S.W uses funct3=7 (DYN) since i32->f32 is exact for values fitting in the mantissa (no rounding needed).
  RVVOps.FCVT_W_S: lambda x: enc_R(0x60, 0, _r(x,0), 0x1, _idx(x), 0x53),
  RVVOps.FCVT_S_W: lambda x: enc_R(0x68, 0, _r(x,0), 0x7, _idx(x), 0x53),
  RVVOps.FADD_S: lambda x: enc_R(0x00, _r(x,1), _r(x,0), 0x7, _idx(x), 0x53),
  RVVOps.FSUB_S: lambda x: enc_R(0x04, _r(x,1), _r(x,0), 0x7, _idx(x), 0x53),
  RVVOps.FMUL_S: lambda x: enc_R(0x08, _r(x,1), _r(x,0), 0x7, _idx(x), 0x53),
  RVVOps.FDIV_S: lambda x: enc_R(0x0C, _r(x,1), _r(x,0), 0x7, _idx(x), 0x53),
  # FSQRT.S: funct7=0x2C, rs2=0, funct3=7 (dyn rm). src=(rs1,)
  RVVOps.FSQRT_S: lambda x: enc_R(0x2C, 0, _r(x,0), 0x7, _idx(x), 0x53),
  RVVOps.FLW:    lambda x: enc_I(_imm(x,1), _r(x,0), 0x2, _idx(x), 0x07),
  RVVOps.FSW:    lambda x: enc_S(_imm(x,2), _r(x,0), _r(x,1), 0x2, 0x27),
  RVVOps.FMV_W_X: lambda x: enc_R(0x78, 0, _r(x,0), 0x0, _idx(x), 0x53),
  RVVOps.FMV_X_W: lambda x: enc_R(0x70, 0, _r(x,0), 0x0, _idx(x), 0x53),
  # --- RVV ---
  # vsetvli: src=(rs1=avl_reg, vtypei_const), dest = rd
  RVVOps.VSETVLI:  lambda x: enc_vsetvli(_idx(x), _r(x,0), _imm(x,1)),
  RVVOps.VSETIVLI: lambda x: enc_vsetivli(_idx(x), _imm(x,0), _imm(x,1)),
  # unit-stride loads: src=(rs1_base,); vm=1 baked in for unmasked
  RVVOps.VLE8_V:  lambda x: enc_vload(0, 0, 0, 1, 0, _r(x,0), 0, _idx(x)),
  RVVOps.VLE16_V: lambda x: enc_vload(0, 0, 0, 1, 0, _r(x,0), 5, _idx(x)),
  RVVOps.VLE32_V: lambda x: enc_vload(0, 0, 0, 1, 0, _r(x,0), 6, _idx(x)),
  RVVOps.VLE64_V: lambda x: enc_vload(0, 0, 0, 1, 0, _r(x,0), 7, _idx(x)),
  # unit-stride stores: src=(vs3_val, rs1_base); vm=1 baked in
  RVVOps.VSE8_V:  lambda x: enc_vstore(0, 0, 0, 1, 0, _r(x,1), 0, _r(x,0)),
  RVVOps.VSE16_V: lambda x: enc_vstore(0, 0, 0, 1, 0, _r(x,1), 5, _r(x,0)),
  RVVOps.VSE32_V: lambda x: enc_vstore(0, 0, 0, 1, 0, _r(x,1), 6, _r(x,0)),
  RVVOps.VSE64_V: lambda x: enc_vstore(0, 0, 0, 1, 0, _r(x,1), 7, _r(x,0)),
  # vector float arith: src=(vs2, vs1); vm=1 baked in
  RVVOps.VFADD_VV:  lambda x: enc_vop(0x00, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFSUB_VV:  lambda x: enc_vop(0x02, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFMUL_VV:  lambda x: enc_vop(0x24, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFDIV_VV:  lambda x: enc_vop(0x20, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFMACC_VV: lambda x: enc_vop(0x2C, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFMIN_VV:  lambda x: enc_vop(0x04, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFMAX_VV:  lambda x: enc_vop(0x06, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  # VFUNARY1 unary float ops: funct6=0x13, vs1 field selects sub-op. vfsqrt.v: vs1=0b00000.
  RVVOps.VFSQRT_V: lambda x: enc_vop(0x13, 1, _r(x,0), 0, 1, _idx(x)),
  # VFUNARY0 conversions: funct6=0x12, vs1=0b00111 for vfcvt.rtz.x.f.v ; vs1=0b00011 for vfcvt.f.x.v.
  RVVOps.VFCVT_X_F_V: lambda x: enc_vop(0x12, 1, _r(x,0), 0b00111, 1, _idx(x)),
  RVVOps.VFCVT_F_X_V: lambda x: enc_vop(0x12, 1, _r(x,0), 0b00011, 1, _idx(x)),
  RVVOps.VFADD_VF:  lambda x: enc_vop(0x00, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFSUB_VF:  lambda x: enc_vop(0x02, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFRSUB_VF: lambda x: enc_vop(0x27, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFDIV_VF:  lambda x: enc_vop(0x20, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFRDIV_VF: lambda x: enc_vop(0x21, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFMUL_VF:  lambda x: enc_vop(0x24, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  RVVOps.VFMACC_VF: lambda x: enc_vop(0x2C, 1, _r(x,0), _r(x,1), 5, _idx(x)),
  # vector integer arith
  RVVOps.VADD_VV: lambda x: enc_vop(0x00, 1, _r(x,0), _r(x,1), 0, _idx(x)),
  RVVOps.VSUB_VV: lambda x: enc_vop(0x02, 1, _r(x,0), _r(x,1), 0, _idx(x)),
  RVVOps.VMUL_VV: lambda x: enc_vop(0x25, 1, _r(x,0), _r(x,1), 2, _idx(x)),
  # vector moves — §11.16; funct6=0x17, vs2=v0 reserved.
  RVVOps.VMV_V_V: lambda x: enc_vop(0x17, 1, 0, _r(x,0), 0, _idx(x)),
  RVVOps.VMV_V_X: lambda x: enc_vop(0x17, 1, 0, _r(x,0), 4, _idx(x)),
  RVVOps.VMV_V_I: lambda x: enc_vop(0x17, 1, 0, _imm(x,0) & 0x1F, 3, _idx(x)),  # vmv.v.i vd, imm5
  RVVOps.VMV_V_F: lambda x: enc_vop(0x17, 1, 0, _r(x,0), 5, _idx(x)),
  # reductions — §14. src=(vs2_input_vec, vs1_init_vec); vd[0] holds the reduced scalar.
  RVVOps.VFREDUSUM_VS: lambda x: enc_vop(0x01, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFREDOSUM_VS: lambda x: enc_vop(0x03, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFREDMAX_VS:  lambda x: enc_vop(0x07, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  RVVOps.VFREDMIN_VS:  lambda x: enc_vop(0x05, 1, _r(x,0), _r(x,1), 1, _idx(x)),
  # vfmv.f.s — §16.2
  RVVOps.VFMV_F_S: lambda x: enc_vop(0x10, 1, _r(x,0), 0, 1, _idx(x)),
  RVVOps.VFMV_S_F: lambda x: enc_vop(0x10, 1, 0, _r(x,0), 5, _idx(x)),
  RVVOps.VEXTRACT_F32_0: lambda x: _enc_vextract_f32(x, 0),
  RVVOps.VEXTRACT_F32_1: lambda x: _enc_vextract_f32(x, 1),
  RVVOps.VEXTRACT_F32_2: lambda x: _enc_vextract_f32(x, 2),
  RVVOps.VEXTRACT_F32_3: lambda x: _enc_vextract_f32(x, 3),
  # ret = jalr x0, ra, 0
  RVVOps.RET: lambda x: enc_I(0, 1, 0, 0, 0x67),
}

# ***** RVVRenderer class *****

class RVVRenderer(ISARenderer):
  device = "CPU"
  has_local = False
  has_threads = bool(getenv("THREADS", 1))
  global_max = (CPU_COUNT.value, 0, 0)
  extra_matcher = extra_matcher
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.FDIV, Ops.CMPLT, Ops.CMPEQ)}

  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_cpu import RVVCompiler
    self.compiler = RVVCompiler()

  # Two-address ops: vd is both source AND destination (the accumulator). vfmacc.vv: vd += vs1*vs2.
  TWO_ADDRESS = {RVVOps.VFMACC_VV, RVVOps.VFMACC_VF}
  def is_two_address(self, x:UOp) -> bool: return x.arg in RVVRenderer.TWO_ADDRESS
  def stack_pointer(self) -> UOp: return UOp(Ops.DEFINE_REG, dtypes.uint64, tag=(SP,))

  # Regalloc hooks. Follow the X86 pattern: build a UOp expressing the operation and re-run isel on it.
  def copy(self, x:UOp, reg:Register) -> UOp:
    dt = dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype
    ret = isel_matcher.rewrite(UOp(Ops.COPY, dt, (x,), tag=reg))
    assert ret is not None, f"RVV: no COPY pattern for dtype={dt}"
    return ret.replace(dtype=x.dtype)

  def spill(self, disp:UOp, x:UOp) -> UOp:
    nx = x.replace(dtype=dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype)
    ret = isel_matcher.rewrite(self.stack_pointer().index(disp).store(nx))
    assert ret is not None, f"RVV: no spill pattern for dtype={nx.dtype}"
    return ret.replace(src=tuple(s if s is not nx else x for s in ret.src))

  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    ndt = dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype
    ret = isel_matcher.rewrite(self.stack_pointer().index(disp).load(dtype=ndt, tag=reg))
    assert ret is not None, f"RVV: no fill pattern for dtype={ndt}"
    return ret.replace(dtype=x.dtype)

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    # Minimal text form for debugging; not a full assembler.
    lines = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is RVVOps.LABEL: lines.append(f"{u.tag}:"); continue
      lines.append(f"    {str(u.arg).removeprefix('RVVOps.').lower():12s} " +
                   ", ".join([str(u.reg)] + [str(s.reg if s.reg else s.arg) for s in u.src]))
    return "\n".join(lines)

  def render(self, uops:list[UOp]) -> str:
    if not any(u.op is Ops.INS for u in uops):
      raise NotImplementedError("RVV instruction selection is not implemented yet — isel_matcher in renderer/isa/rvv.py is empty. "
                                "The scaffold (ops enum, register file, encodings, render-time relocation fixup) is in place; "
                                "next step is to fill in isel patterns mapping UOps (LOAD/STORE/ALU/REDUCE/RANGE) to RVVOps.")
    targets: dict[str, int] = {}
    branches: list[tuple[int, UOp]] = []  # (byte offset of instruction start, branch uop)
    binary = bytearray()
    # Detect any vector op and emit a single vsetivli prelude — sufficient for fully-unrolled vec kernels.
    # For loop-based vector kernels (future work) we'll need to interleave vsetvli inside the j-loop.
    VEC_OPS = {RVVOps.VLE32_V, RVVOps.VSE32_V, RVVOps.VFADD_VV, RVVOps.VFSUB_VV, RVVOps.VFMUL_VV, RVVOps.VFDIV_VV,
               RVVOps.VFMACC_VV, RVVOps.VFMACC_VF, RVVOps.VFADD_VF, RVVOps.VFSUB_VF, RVVOps.VFRSUB_VF, RVVOps.VFMUL_VF,
               RVVOps.VFDIV_VF, RVVOps.VFRDIV_VF, RVVOps.VADD_VV, RVVOps.VSUB_VV, RVVOps.VMUL_VV,
               RVVOps.VMV_V_V, RVVOps.VMV_V_X, RVVOps.VMV_V_I, RVVOps.VMV_V_F, RVVOps.VFREDUSUM_VS,
               RVVOps.VFSQRT_V, RVVOps.VFCVT_X_F_V, RVVOps.VFCVT_F_X_V}
    vec_uops = [u for u in uops if u.op is Ops.INS and u.arg in VEC_OPS]
    if vec_uops:
      # Find the first vec LOAD/STORE/ALU and use its dtype.count as the AVL. f32 / m1 for now.
      vl_count = next((u.dtype.count for u in vec_uops if u.dtype is not dtypes.void and u.dtype.count > 1), 4)
      if vl_count <= 31:
        # vsetivli zero, vl_count, e32, m1, ta, ma  (rd=0 = don't write vl back to a reg)
        binary.extend(enc_vsetivli(rd=0, uimm=vl_count, vtypei=vtype(SEW32, LMUL_1)))
      else:
        # vsetivli's uimm5 caps at 31. For larger AVL we materialize it in t0 and use vsetvli.
        if vl_count > 2047: raise NotImplementedError(f"RVV: prelude vl > 2047 (got {vl_count}); needs LUI+ADDI")
        binary.extend(enc_I(vl_count, 0, 0, 5, 0x13))                            # addi t0, zero, vl_count
        binary.extend(enc_vsetvli(rd=0, rs1=5, vtypei=vtype(SEW32, LMUL_1)))     # vsetvli zero, t0, e32, m1, ta, ma
    for u in uops:
      if u.op is not Ops.INS: continue
      if u.arg is RVVOps.LABEL: targets[u.tag] = len(binary); continue
      if u.arg not in encodings: raise RuntimeError(f"RVV: no encoding for {u.arg}")
      if u.arg in (RVVOps.BEQ, RVVOps.BNE, RVVOps.BLT, RVVOps.BGE, RVVOps.BLTU, RVVOps.BGEU, RVVOps.JAL): branches.append((len(binary), u))
      binary.extend(encodings[u.arg](u))
    # fix up branch/jal offsets
    for off, u in branches:
      tgt = targets[u.tag]
      delta = tgt - off
      if u.arg is RVVOps.JAL: binary[off:off+4] = enc_J(delta, 0 if isinstance(u.tag, str) else _idx(u), 0x6F)
      else:
        funct3 = {RVVOps.BEQ:0,RVVOps.BNE:1,RVVOps.BLT:4,RVVOps.BGE:5,RVVOps.BLTU:6,RVVOps.BGEU:7}[u.arg]
        binary[off:off+4] = enc_B(delta, _r(u,1), _r(u,0), funct3, 0x63)
    return binary.hex()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16, dtypes.float16)}
