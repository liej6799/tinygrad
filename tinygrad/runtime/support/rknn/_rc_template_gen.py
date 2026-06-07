"""Algorithmic regcmd template generator for chained element-wise RKNN models.

Each supported RC template (n_inputs = 2..64) is:

    [OFF[n] zero bytes]                      alignment lead
    PREFIX[n] words                          header + cascade + chains + descriptors
    6 tiles x ( for j in 0..n_adds-1:        CANON register blocks
        block(71 if j<n_adds-1 else 69) + gap )
    TRAILING[n] words                        tail task descriptors / zero pad

PREFIX is generated algorithmically by _build_prefix(n).  CANON contains 52 DPU +
17 DPU_RDMA register writes.  A 71-block adds a 2-word PC preamble.  Patched
fields (those _rc_patch_block overwrites at build time) are stored as 0.
"""
import struct

MAX_INPUTS = 64

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x0101

EW_OP_ADD = 0
EW_OP_SUB = 1
EW_OP_MUL = 2
EW_OP_DIV = 3
# Pure copy/reshape mode: the NPU side of a CPU-fallback op runs the same 69
# register block as the element-wise ops, just configured to copy rather than
# compute. It is a regular member of the op-dispatch tables below.
EW_OP_COPY = 4

_EW_CFG = {
    EW_OP_ADD: 0x108202c0,
    EW_OP_SUB: 0x108402c0,
    EW_OP_MUL: 0x108003c4,
    EW_OP_DIV: 0x108303c0,
    EW_OP_COPY: 0x00000383,
}

_DPU_OUT_RES = {
    EW_OP_ADD: 0x00010001,
    EW_OP_SUB: 0x00010001,
    EW_OP_MUL: 0x00010001,
    EW_OP_DIV: 0x00000001,
    EW_OP_COPY: 0x00000001,
}

_RDMA_BN_MUL = {
    EW_OP_ADD: 0x00017849,
    EW_OP_SUB: 0x00017849,
    EW_OP_MUL: 0x00017849,
    EW_OP_DIV: 0x00017841,
    EW_OP_COPY: 0x00007801,
}

# Copy mode overrides two more cfg registers vs the element-wise ops, and bakes
# the fixed [1,4] bool reshape geometry into the fields the element-wise path
# leaves zero and patches per-block at build time (copy geometry is constant).
_COPY_DPU_MODE = 0x00000000      # 0x4010 (element-wise: 0x48000002)
_COPY_RDMA_MODE = 0x00000001     # 0x5034 (element-wise: 0x40000008)
_COPY_SHAPE_FILL = {
    0x4024: 0x00000040, 0x4030: 0x00000001, 0x4034: 0x00000001,
    0x403c: 0x000f000f, 0x4058: 0x0000000f, 0x405c: 0x00010001,
    0x40c0: 0x00000040, 0x500c: 0x00000001, 0x5010: 0x00000001,
    0x5014: 0x0000000f,
}

_OP_NAMES = {"Add": EW_OP_ADD, "Sub": EW_OP_SUB, "Mul": EW_OP_MUL, "Div": EW_OP_DIV}

# CPU-fallback ops execute on a CPU kernel, with the NPU running only reshape/copy
# blocks (no DPU compute).  The NPU-side machinery (reshape/copy RC blocks,
# taskdesc, FlatBuffer node geometry) is COMPLETELY op-agnostic: And / Or models of
# the same arity are byte-identical except for the op-name strings (and the f7
# value below).
#
# IMPORTANT (verified on-device, RK3588, librknnrt.so 2.3.2): the runtime selects
# the CPU kernel by the node's OP-NAME STRING ("And" / "Or"), NOT by the f7 enum --
# an "Or" node computes OR for any f7, and an op whose name has no CPU kernel (e.g.
# "Xor") is rejected at load regardless of f7.  f7 is therefore cosmetic for
# dispatch; we still set it to the runtime's own op-type enum for fidelity.
#
# Provenance of the f7 values:
#   And = 85 (0x55): decoded byte-exact from toolkit references AND seen in the
#         runtime CPU-op dispatch switch (cmp w0,#0x55 at librknnrt.so:0x2aafa4).
#   Or  = 86 (0x56): runtime has an "Or" CPU kernel (string at 0x60e270); the
#         dispatch case 0x56 sits right after And on the same handler path.
#         Verified on-device: out = a|b, 0 mismatches vs OR (verify_bool).
#
# UNSUPPORTED on this runtime:
#   Xor: librknnrt.so 2.3.2 has NO "Xor" CPU kernel (no "Xor" string in the .so).
#        rknn_init() rejects it: "Unsupport CPU op: Xor in this librknnrt.so".
#        Confirmed by sweeping every f7 56..90 -- all rejected.  Running Xor would
#        require rknn_register_custom_ops (a custom op) or a newer runtime, so it is
#        intentionally NOT listed here.  (XOR can also be expressed without a native
#        kernel as (a OR b) AND NOT(a AND b), but that needs a multi-node graph.)
#
# Each value is (enum, verified) so tooling can tell ground-truth from candidates;
# cpu_op_id() returns the int regardless.
#
# UNARY ops (1 input, 1 output) -- listed in CPU_UNARY_OPS below -- use a distinct
# graph topology (1 input reshape -> op -> 1 output reshape, 2 RC copy blocks)
# rather than the binary 2-input chain.  Not is a NOT gate (out = ~a, bool):
#   Not: runtime HAS a "Not" CPU kernel (string at librknnrt.so:0x60e190).  f7 is
#        cosmetic for dispatch (runtime keys on the op name), confirmed for And/Or;
#        the value here is set for fidelity and verified on-device via verify_not.
CPU_OP_SPECS = {
    "And": (85, True),
    "Or":  (86, True),
    "Not": (78, True),
}
CPU_OP_ENUMS = {name: val for name, (val, _verified) in CPU_OP_SPECS.items()}

# Ops that take a single input (unary).  Everything else in CPU_OP_SPECS is binary.
CPU_UNARY_OPS = {"Not"}


def is_unary_cpu_op(name):
    return name in CPU_UNARY_OPS

# Single modular op classification used by both the FlatBuffer node builder and
# the regcmd generator. NPU ops dispatch to element-wise _canon compute blocks;
# CPU ops dispatch to reshape/copy-only blocks.
def is_cpu_op(name):
    return name in CPU_OP_ENUMS


def is_npu_op(name):
    return name in _OP_NAMES


def cpu_op_id(name):
    """Runtime CPU op-type enum (node field f7) for a CPU-fallback op."""
    return CPU_OP_ENUMS[name]


def cpu_op_verified(name):
    """True if this CPU op's f7 enum is confirmed ground-truth (vs a candidate)."""
    return CPU_OP_SPECS.get(name, (None, False))[1]


def ew_op_id(name):
    return _OP_NAMES.get(name, EW_OP_ADD)


def _ew_cfg(op):
    return _EW_CFG.get(op, _EW_CFG[EW_OP_ADD])


def _w(tgt, reg, val):
    return (tgt << 48) | ((val & 0xFFFFFFFF) << 16) | reg


def _canon(op=EW_OP_ADD):
    copy = (op == EW_OP_COPY)
    return [
    (0x1001, 0x4004, 0x0000000e, False),
    (0x2001, 0x5004, 0x0000000e, False),
    (0x1001, 0x400c, 0x000001e5, False),
    (0x1001, 0x4010, _COPY_DPU_MODE if copy else 0x48000002, False),
    (0x1001, 0x4014, 0x00000000, False),
    (0x1001, 0x4020, 0x00000000, True),
    (0x1001, 0x4024, 0x00000000, True),
    (0x1001, 0x4030, 0x00000000, True),
    (0x1001, 0x4034, 0x00000000, True),
    (0x1001, 0x4038, 0x00000000, False),
    (0x1001, 0x403c, 0x00000000, True),
    (0x1001, 0x4040, 0x00000053, False),
    (0x1001, 0x4044, 0x00000000, False),
    (0x1001, 0x4048, 0x00000000, False),
    (0x1001, 0x404c, 0x00000000, False),
    (0x1001, 0x4050, 0x00000002, False),
    (0x1001, 0x4054, 0x00000000, False),
    (0x1001, 0x4058, 0x00000000, True),
    (0x1001, 0x405c, 0x00000000, True),
    (0x1001, 0x4060, 0x00000053, False),
    (0x1001, 0x4064, 0x00000000, False),
    (0x1001, 0x4068, 0x00000000, False),
    (0x1001, 0x406c, 0x00000000, False),
    (0x1001, 0x4070, _ew_cfg(op), False),
    (0x1001, 0x4074, 0x00000000, False),
    (0x1001, 0x4078, 0x00000001, False),
    (0x1001, 0x407c, 0x00000000, False),
    (0x1001, 0x4080, 0x00000000, False),
    (0x1001, 0x4084, _DPU_OUT_RES.get(op, 0x00010001), False),
    (0x1001, 0x4088, 0x00000000, False),
    (0x1001, 0x4090, 0x00000000, False),
    (0x1001, 0x4094, 0x00000000, False),
    (0x1001, 0x4098, 0x00000000, False),
    (0x1001, 0x409c, 0x00000000, False),
    (0x1001, 0x40a0, 0x00000000, False),
    (0x1001, 0x40a4, 0x00000000, False),
    (0x1001, 0x40a8, 0x00000000, False),
    (0x1001, 0x40ac, 0x00000000, False),
    (0x1001, 0x40c0, 0x00000000, True),
    (0x1001, 0x40c4, 0x00000000, False),
    (0x1001, 0x4100, 0x00000000, False),
    (0x1001, 0x4104, 0x00000000, False),
    (0x1001, 0x4108, 0x00000000, False),
    (0x1001, 0x410c, 0x00000000, False),
    (0x1001, 0x4110, 0x00000000, False),
    (0x1001, 0x4114, 0x00000000, False),
    (0x1001, 0x4118, 0x00000000, False),
    (0x1001, 0x411c, 0x00000000, False),
    (0x1001, 0x4120, 0x00000000, False),
    (0x1001, 0x4124, 0x00000000, False),
    (0x1001, 0x4128, 0x00000000, False),
    (0x1001, 0x412c, 0x00000000, False),
    (0x2001, 0x500c, 0x00000000, True),
    (0x2001, 0x5010, 0x00000000, True),
    (0x2001, 0x5014, 0x00000000, True),
    (0x2001, 0x5018, 0x00000000, True),
    (0x2001, 0x501c, 0x00000000, False),
    (0x2001, 0x5020, 0x00000000, False),
    (0x2001, 0x5028, 0x00000000, False),
    (0x2001, 0x502c, 0x00000000, False),
    (0x2001, 0x5034, _COPY_RDMA_MODE if copy else 0x40000008, False),
    (0x2001, 0x5038, 0x00000000, True),
    (0x2001, 0x5040, 0x00000000, True),
    (0x2001, 0x5044, _RDMA_BN_MUL.get(op, 0x00017849), False),
    (0x2001, 0x5048, 0x00000000, False),
    (0x2001, 0x504c, 0x00000000, True),
    (0x2001, 0x5064, 0x00000000, False),
    (0x2001, 0x5068, 0x01010101, False),
    (0x2001, 0x506c, 0x00000000, True),
]

def reshape_copy_canon_words():
    """The 69-word NPU reshape/copy canon (CPU-op block) for the [1,4] bool shape.

    This is the same _canon register block as the element-wise ops, dispatched
    through EW_OP_COPY: copy cfg values plus the constant [1,4] reshape geometry
    baked into the otherwise-patched fields. Byte-exact vs the chained-And refs.
    """
    return _canon_words(EW_OP_COPY)


GAP69 = [
    0x0000000000000000, 0x0101000000000014,
    0x0041000000000000, 0x0081000000180008,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000,
]

GAP71 = [
    0x0041000000000000, 0x0081000000180008,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000,
]

PC14 = 0x0101000000240014


def _fixed_header(name=b"a_rs_i1\x00"):
    """The constant 30-u32 descriptor header shared by the NPU and CPU prefixes.

    Structure: a length-prefixed reshape tensor name, a few small constants, an
    interleaved 16-bit ascending offset stream, and a descending copy-descriptor
    size table. Identical for every n; only the 2-input NPU case swaps the name
    ("x_rs_i1" vs "a_rs_i1"). The CPU prefix uses it as-is; the NPU prefix wraps
    it in the even/odd u32-alignment phase.
    """
    out = [7]                                            # name length
    out += list(struct.unpack("<2I", name))              # tensor name
    out += [1, 4, 1, 4]                                  # bool [1,4] shape pair
    out.append((0x34 << 16) | 0x28)                      # packed (0x28, 0x34)
    # interleaved 16-bit offsets: 3 zeros, 4..36 step 4, 3 zeros, 40,44,48
    stream = [0, 0, 0] + list(range(4, 40, 4)) + [0, 0, 0] + [40, 44, 48]
    for i in range(0, len(stream), 2):
        out.append((stream[i + 1] << 16) | stream[i])
    # descending offset table: a 40 lead, then the values 104,96,...,4 (an
    # ascending run 4..60 step 8, then 72..104 step 12, reversed).
    asc = [4 + 8 * k for k in range(8)] + [72 + 12 * k for k in range(3)] + [104]
    out += [40] + asc[::-1]
    return out


def _u32_to_u64(words32):
    """Pack an even-length u32 list into u64 words (lo, hi) -> one u64 each."""
    out = []
    for i in range(0, len(words32), 2):
        out.append((words32[i + 1] << 32) | words32[i])
    return out


def _npu_prefix(n):
    """NPU element-wise descriptor prefix (u64 words). See _build_prefix."""
    n_adds = n - 1
    even = (n % 2 == 0)

    # The 30-u32 _fixed_header is shared with the CPU prefix; the NPU stream wraps
    # it in the even/odd u32-alignment phase (even-n carries a half-word offset,
    # so the header is bracketed by a leading and trailing zero u32).
    fixed = _fixed_header(b"x_rs_i1\x00" if n == 2 else b"a_rs_i1\x00")
    if even:
        n_lead = min((n - 2) // 2 + 1, 3)
        header = [0] * n_lead + _u32_to_u64([0] + fixed + [0])
    else:
        n_lead = 3 + (1 if n >= 7 else 0)
        header = [0] * n_lead + _u32_to_u64(fixed)

    words = list(header) + [0] * 7

    n_inter = (n - 1) // 2
    topmost_inter = n_inter + 1

    if even:
        words.append(((148 + 28 * (n - 2)) << 32) | (n + 4))
    else:
        words.append((n + 4) << 32)

    for k in range(topmost_inter, 1, -1):
        sz = 40 + 56 * k
        off = 68 + 56 * k
        if not even and k == topmost_inter:
            off -= 4
        words.append((sz << 32) | off)

    words += [0x000000600000007c, 0x0000002800000044]

    # chain1/canon_base shrink by 64 for every 4 inputs (matches the toolkit's
    # surface-tiling step). Verified byte-exact against the n=4,5,10..16 reference
    # bodies; n=2..9 keep their original (n>=6 -> 64) value since (n-2)//4 == 1 there.
    anomaly = 64 * ((n - 2) // 4)
    chain0 = 0x74 + 24 * (n - 2)
    chain1 = 0x15c + 280 * (n - 2) - anomaly
    canon_base = 0x1084 + 4120 * (n - 2) - anomaly

    words += [0x000600000000000c, 0x0000000600040012, chain0]
    words += [0x0006000000000000, 0x0000000600040012, chain1]
    for k in range(n + 1):
        addr = canon_base + k * 0x28
        marker = 0x0000000600040010 if k == n else 0x0000000600040012
        words += [0x0006000000000000, marker, addr]

    words += [0x0004000400000000, (240 * n_adds) << 32 | 4]

    for tile in range(6):
        for add in range(n_adds):
            counter = (n + 2 + 2 * add) << 32
            addr = (tile * n_adds + add) * 0x280
            words += [counter, 0x0000030000000018, 0x000000000001ffff,
                      0x0000000000000045, addr]

    n_zeros = 2 * ((n - 2) % 4) + 1
    words += [0] * n_zeros
    words.append((3840 * n_adds) << 32)

    return words


OFF = {n: (4 if n % 2 == 0 else 0) for n in range(2, 99)}
PREFIX = {n: _npu_prefix(n) for n in range(2, 99)}


def _build_trailing(n):
    ng = max(0, n - 2)
    if ng == 0:
        return [0] * 6
    w = [0] * 7
    w += [0x0000001000000000, 0x000000000000000a, 0x000000000000000a]
    if ng > 1:
        w += [0] * 5
    for k in range(1, ng):
        w += [0x0000002000000000, 0x0000000000000001, 0x000000000000000a,
              0x0000000000000001, 0x000000000000000a]
        if k < ng - 1:
            w += [0] * 3
    trail = max(0, 5 - 2 * ng)
    if ng == 3:
        trail = 1
    w += [0] * trail
    return w


def _canon_words(op=EW_OP_ADD):
    if op == EW_OP_COPY:
        # Copy mode bakes the constant [1,4] reshape geometry into the fields the
        # element-wise path patches per-block, so they are not zeroed here.
        return [_w(t, r, _COPY_SHAPE_FILL.get(r, 0) if p else v)
                for (t, r, v, p) in _canon(op)]
    return [_w(t, r, 0 if p else v) for (t, r, v, p) in _canon(op)]


def _normalize_ops(n_adds, ops):
    """Return a list of op-name strings, one per binary op in the chain.

    Accepts None (all Add), a single int EW id, a single op-name string, a list
    of int EW ids, or a list of op-name strings. This is the single modular entry
    used to decide NPU- vs CPU-dispatch per op.
    """
    _EW_ID_TO_NAME = {v: k for k, v in _OP_NAMES.items()}
    if ops is None:
        return ["Add"] * n_adds
    if isinstance(ops, int):
        return [_EW_ID_TO_NAME.get(ops, "Add")] * n_adds
    if isinstance(ops, str):
        return [ops] * n_adds
    out = []
    for o in ops:
        if isinstance(o, int):
            out.append(_EW_ID_TO_NAME.get(o, "Add"))
        else:
            out.append(o)
    if len(out) < n_adds:
        out += ["Add"] * (n_adds - len(out))
    return out[:n_adds]


def _build_cpu_copy_blocks(n_inputs, rc_word_off=0):
    """CPU-fallback reshape/copy RC stream as a uint32 byte string.

    n_inputs+1 reshape/copy `_canon(EW_OP_COPY)` blocks framed by PC chain
    addresses + GAP, preceded by the closed-form CPU descriptor prefix (aligned
    via rc_word_off) and followed by the CPU trailing descriptors.  Shared by the
    binary chain (n>=2) and the unary path (n=1, e.g. Not -> 2 blocks).
    """
    n_blocks = n_inputs + 1
    u = _build_prefix(n_inputs, EW_OP_COPY, rc_word_off)
    prefix_words = len(u)
    canon = _canon_u32(EW_OP_COPY)
    for blk in range(n_blocks):
        if blk < n_blocks - 1:
            u += canon + _pc_chain_u32((blk + 1) * 0x280) + _pc14_u32() + _gap71_u32()
        else:
            u += canon + _gap69_u32()
    u += _cpu_trailing_u32(n_inputs, prefix_words)
    return struct.pack(f"<{len(u)}I", *u)


def build_template(n_inputs, ops=None, rc_word_off=0):
    """Modular regcmd generator (single path for NPU and CPU ops).

    Dispatches on op type: a chain of NPU element-wise ops (Add/Sub/Mul/Div)
    produces 6 tiles of compute `_canon` blocks; a chain of CPU-fallback ops
    (And/...) produces n+1 reshape/copy `_canon` blocks (op `EW_OP_COPY`) with a
    CPU descriptor prefix. Both use the same `_canon`/PC/GAP framing; they differ
    only in the prefix, the block count/order, and the trailing schedule. Mixed
    NPU/CPU chains are not yet supported here (the FlatBuffer node builder
    already handles them modularly).

    rc_word_off is the RC section's u32 file offset (FlatBuffer-body length // 4);
    it is used only by the CPU path to align the reshape/copy canon to a 64-byte
    boundary, matching the toolkit's placement.
    """
    # Unary CPU op (1 input, e.g. Not): a single op node, with 1 input reshape and
    # 1 output reshape -> 2 reshape/copy RC blocks.  Uses the same closed-form CPU
    # prefix/canon/trailing math as the binary path, evaluated at n_inputs=1.
    if n_inputs == 1:
        unary = [o for o in (ops if isinstance(ops, (list, tuple)) else [ops]) if o]
        if not unary or not all(is_unary_cpu_op(o) for o in unary):
            raise NotImplementedError(
                f"n_inputs=1 is only supported for unary CPU ops {sorted(CPU_UNARY_OPS)}, "
                f"got ops={ops}")
        return _build_cpu_copy_blocks(1, rc_word_off)

    if not 2 <= n_inputs <= MAX_INPUTS:
        raise NotImplementedError(
            f"regcmd template generation supports 2..{MAX_INPUTS} inputs, got {n_inputs}")
    n_adds = n_inputs - 1
    op_names = _normalize_ops(n_adds, ops)

    cpu_flags = [is_cpu_op(o) for o in op_names]
    if any(cpu_flags) and not all(cpu_flags):
        raise NotImplementedError(
            "mixed NPU/CPU op chains within a single chain are not supported by "
            f"the RC generator (got ops={op_names}). For independent CPU+NPU "
            "branches (e.g. parallel And + Add) the schedule is "
            "[CPU copy blocks] + [NPU compute blocks]; see "
            "mixed_parallel_block_schedule() and rknn_mixed_gen.py.")

    if all(cpu_flags):
        # CPU-fallback chain: n+1 reshape/copy canon blocks (uint32 stream, with
        # the CPU descriptor prefix and an alignment lead computed from rc_word_off).
        return _build_cpu_copy_blocks(n_inputs, rc_word_off)

    # NPU element-wise chain: 6 tiles of compute canon blocks (uint64 stream).
    ew_ids = [ew_op_id(o) for o in op_names]
    canon_per = [_canon_words(op) for op in ew_ids]
    words = list(_build_prefix(n_inputs))
    gbi = 0
    for _tile in range(6):
        for j in range(n_adds):
            if j < n_adds - 1:
                base = _w(_PC, 0x0010, (gbi + 1) * 0x280)
                words += canon_per[j] + [base, PC14] + GAP71
            else:
                words += canon_per[j] + GAP69
            gbi += 1
    words += _build_trailing(n_inputs)
    return b"\x00" * OFF[n_inputs] + struct.pack(f"<{len(words)}Q", *words)


# ── Mixed parallel CPU+NPU schedule (decoded from vendor references) ──
#
# A *parallel* mixed model holds independent CPU-fallback branches (e.g.
# out1 = a AND b) and independent NPU element-wise branches (e.g. out2 = x + y).
# Decoding the toolkit's output (RKNN_CREATION.md section 6f) shows the RC block
# region is simply the CONCATENATION:
#
#     [ all CPU branches' copy/reshape blocks ]   (DPU_MODE=0, EW_CFG=0x383)
#     [ all NPU branches' 6-tile compute blocks ] (DPU_MODE=0x48000002, op cfg)
#
# Verified block counts:
#   And(2-in) + Add(2-in)   -> 3 copy  + 6  compute = 9   (sg.f10=[0,0,0,4,9])
#   And(3-in) + Add(2-in)   -> 4 copy  + 6  compute = 10
#   And(2-in) + Add(3-in)   -> 3 copy  + 12 compute = 15  (sg.f10=[0,0,0,5,12])
#
# The CPU copy block = _canon(EW_OP_COPY); the NPU compute blocks are exactly the
# per-tile blocks build_template() already emits for the Add chain.  Only the RC
# *prefix* (descriptor + DMA schedule, with runtime-validated chain addresses and
# size table) is not yet reproduced from scratch -- that is the same
# compiler-tiler port that gates large-N Add.  rknn_mixed_gen.py therefore builds
# mixed models via the hybrid (reference-body) path today.


def mixed_parallel_block_schedule(cpu_branches, npu_branches):
    """Return the mixed-parallel RC block schedule as a list of descriptors.

    cpu_branches: list of n_inputs (>=2) for each CPU (And) branch.
    npu_branches: list of (n_inputs, ops) for each NPU (Add/...) branch.

    Each returned entry is a dict describing one RC block:
      {"kind": "copy"|"compute", "branch": idx, "op": name, "len": 69|71}.
    This is the decoded, generatable block ordering; the prefix/relocation is
    separate (see module docstring).
    """
    sched = []
    for bi, n_in in enumerate(cpu_branches):
        # one CPU op chain of n_in inputs -> n_in + 1 reshape/copy blocks. In a
        # mixed stream every copy block is a 71-block (its PC preamble chains to
        # the next block); only the very last block of the whole stream is a
        # 69-block, handled in the final fix-up below.
        for _k in range(n_in + 1):
            sched.append({"kind": "copy", "branch": bi, "op": "And", "len": 71})
    for bi, (n_in, ops) in enumerate(npu_branches):
        n_adds = n_in - 1
        names = _normalize_ops(n_adds, ops)
        for _tile in range(6):
            for j in range(n_adds):
                # within a tile the last add is a 69-block, the rest are 71.
                sched.append({"kind": "compute", "branch": bi, "op": names[j],
                              "len": 69 if j == n_adds - 1 else 71})
    return sched


# ── CPU-fallback op regcmd helpers (used by build_template's CPU branch) ──
#
# A CPU-fallback op (And/...) runs no NPU compute: the NPU only reshapes/copies
# data and the CPU kernel does the logic. build_template emits this as a uint32
# stream of:
#
#   CPU descriptor prefix (u32)   header + copy-descriptor table + DMA cmd list
#   (n+1) x _canon(EW_OP_COPY) block (u32) + PC/GAP framing
#   CPU trailing (u32)            cosmetic task descriptors
#
# The canon block, and the PC chain-address + PC14 + GAP framing, are the SAME as
# the NPU element-wise path (just op=EW_OP_COPY). The CPU prefix is closed-form;
# the alignment lead derives from the RC file offset. Working on a uint32 basis
# (rather than uint64) handles the half-word alignment of even-n streams.

def _u64_list_to_u32(words64):
    out = []
    for w in words64:
        out.append(w & 0xFFFFFFFF)
        out.append((w >> 32) & 0xFFFFFFFF)
    return out


def _canon_u32(op=EW_OP_ADD):
    """The 69-register _canon block for `op` as a uint32 stream (CPU path)."""
    return _u64_list_to_u32(_canon_words(op))


def _gap71_u32():
    return _u64_list_to_u32(GAP71)


def _gap69_u32():
    return _u64_list_to_u32(GAP69)


def _pc14_u32():
    return _u64_list_to_u32([PC14])


def _pc_chain_u32(addr):
    # PC(0x0010, addr) as one u64 -> two u32
    return _u64_list_to_u32([_w(_PC, 0x0010, addr)])


# Offset-lead suffix sequence: an offset lead of length L (odd, 1..7) is the last
# L entries of this list, followed by 11 zeros.
_CPU_OFFSET_LEAD_SEQ = [0x2c, 0x24, 0x20, 0x18, 0x10, 0x08, 0x01]


def _cpu_nonlead_words(n):
    """u32 word count of the CPU prefix excluding the alignment lead."""
    header = 30 + 15 + 1 + (n + 4)         # fixed header + zeros + count + addr list
    copydesc = (n + 3) * 6
    preamble = 4
    dma = 3 * (n + 1) * 10 + ((2 * n) % 16) + 1
    return header + copydesc + preamble + dma


def _cpu_lead_u32(n, rc_word_off):
    """Alignment lead so the reshape/copy canon lands on a 64-byte boundary.

    rc_word_off is the RC section's u32 offset from the start of the file; the
    section base header (0x40 bytes) contributes nothing mod 16, so this is
    equivalently (FlatBuffer-body length // 4). The lead length is the pad
    (mod 16 u32 = 64 bytes) needed to align the canon, bumped by a full 16-word
    block when the raw pad is < 4 words. Leads of >= 12 words carry the
    descending offset suffix; shorter leads are all zeros.
    """
    residue = rc_word_off % 16
    pad = (-(residue + _cpu_nonlead_words(n))) % 16
    length = pad if pad >= 4 else pad + 16
    if length >= 12:
        return _CPU_OFFSET_LEAD_SEQ[-(length - 11):] + [0] * 11
    return [0] * length


def _cpu_prefix(n, rc_word_off):
    """Closed-form CPU descriptor prefix (u32 words) for a bool [1,4] op chain."""
    words = _cpu_lead_u32(n, rc_word_off)
    words += _fixed_header()
    words += [0] * 15

    words.append(n + 4)
    words.append(0x0c + 28 * (n + 3) - 4)
    for i in range(1, n + 4):
        words.append(0x0c + 28 * (n + 3 - i))

    anomaly = 64 * (n // 8)
    chain0 = 0x74 + 24 * (n - 2)
    chain1 = 0x1dc + 152 * (n - 2) - anomaly
    canon_base = 0x984 + 792 * (n - 2) - anomaly
    copy_addrs = [chain0, chain1]
    copy_addrs += [canon_base + 40 * k for k in range(n + 1)]
    for i, addr in enumerate(copy_addrs):
        marker = 0x00040010 if i == len(copy_addrs) - 1 else 0x00040012
        words += [0x00060000, marker, 0x00000006, addr, 0, 0]

    words += [0x00040004, 0x00000004, 0x168 + 120 * (n - 2), 0]
    for _group in range(3):
        for k in range(n + 1):
            count = n if k == 0 else n + 2 * k - 1
            words += [
                count, 0x00000018, 0x00000300, 0x0001ffff, 0,
                0x00000045, 0, 0x280 * k, 0, 0,
            ]

    words += [0] * ((2 * n) % 16)
    words.append(0x280 * (n + 1))
    return words


def _mixed_lead(rc_word_off):
    """Alignment lead (u32 words) for the mixed prefix, like _cpu_lead_u32.

    The mixed prefix body (header + tables + descriptors) must land so the first
    canon block is 16-u32 aligned.  For the available references the lead is 4
    zero words (B=9 @ residue 2) or a 3-word offset suffix + zeros (B=12 @
    residue 3); both reduce to the same alignment math the CPU path uses.
    """
    # The B=9 reference (rc_word_off%16==2) uses a 4-word zero lead; derive the
    # generic lead from the residue with the same suffix sequence as the CPU path.
    residue = rc_word_off % 16
    pad = (-(residue + 2)) % 16           # +2: the body starts 2 words into the bank
    length = pad if pad >= 4 else pad + 16
    if length >= 12:
        return _CPU_OFFSET_LEAD_SEQ[-(length - 11):] + [0] * 11
    return [0] * length


def mixed_prefix(cpu_blocks, n_npu, rc_word_off=None):
    """From-scratch mixed-parallel RC descriptor prefix (u32 words).

    Parameterized by the CPU copy-block count C (= sum of n_in+1 over CPU
    branches) and the NPU branch input count n_npu.  Decoded and VALIDATED
    byte-exact (non-cosmetic fields) against toolkit references for
    (C,n_npu) in {(3,2),(4,2),(5,2),(3,3),(4,3),(3,4)}.

    Closed forms (anomaly = 64*(n_npu//4), the NPU surface-tiling step):

        K        = 6*(n_npu-1)                 NPU compute blocks
        n_desc   = C + n_npu + 3               canon descriptors after chain0/1
        size cnt = C + n_npu + 4
        chain0     = 24*C + 24*n_npu + 68
        chain1     = 152*C + 280*n_npu - 212  - anomaly
        canon_base = 792*C + 4120*n_npu - 4012 - anomaly   (stride 0x28)
        top size   = 28*C + 28*n_npu + 92
        post size  = 120*C + 240*n_npu - 240
        DMA: 3 groups, each = ramp[0x280*i, i<C] + tails[0x280*(C+j)]; the K tails
             split across groups as [1,2,3]*(n_npu-1).  Counts are cosmetic.

    rc_word_off (RC section u32 file offset) selects the alignment lead; None uses
    the 4-word lead of the C=3,n_npu=2 reference.
    """
    C = cpu_blocks
    nn = n_npu
    K = 6 * (nn - 1)
    B = C + K
    anomaly = 64 * (nn // 4)
    sc = C + nn + 4
    n_desc = C + nn + 3

    words = list(_mixed_lead(rc_word_off)) if rc_word_off is not None else [0, 0, 0, 0]
    words += _fixed_header()                  # a_rs_i1 header (30 u32)
    words += [0] * 15

    # size table: count then `sc` descending sizes (step 0x1c, +0x18 at the top).
    top = 28 * C + 28 * nn + 92
    words.append(sc)
    words.append(top)
    words += [0xc + 0x1c * k for k in range(sc - 2, -1, -1)]

    # copy descriptors: chain0, chain1, then n_desc canon descriptors @ stride 0x28.
    chain0 = 24 * C + 24 * nn + 68
    chain1 = 152 * C + 280 * nn - 212 - anomaly
    canon_base = 792 * C + 4120 * nn - 4012 - anomaly
    copy_addrs = [chain0, chain1] + [canon_base + 0x28 * k for k in range(n_desc)]
    for i, addr in enumerate(copy_addrs):
        marker = 0x00040010 if i == len(copy_addrs) - 1 else 0x00040012
        words += [0x00060000, marker, 0x00000006, addr, 0, 0]

    # DMA section: post-size header, then 3 groups.
    post_size = 120 * C + 240 * nn - 240
    words += [0x00040004, 0x00000004, post_size, 0]
    ramp_addrs = [0x280 * i for i in range(C)]
    tail_addrs = [0x280 * (C + j) for j in range(K)]
    nadd = nn - 1
    grp_sizes = [1 * nadd, 2 * nadd, 3 * nadd]
    splits, off = [], 0
    for gs in grp_sizes:
        splits.append((off, off + gs)); off += gs
    for g, (lo, hi) in enumerate(splits):
        for i, a in enumerate(ramp_addrs):
            words += [C + 1 + i, 0x18, 0x300, 0x1ffff, 0, 0x45, 0, a, 0, 0]
        for j in range(lo, hi):
            words += [B + 2, 0x18, 0x300, 0x1ffff, 0, 0x45, 0, tail_addrs[j], 0, 0]

    words += [0] * 8
    words.append(0x280 * B)
    return words


def mixed_block_region(cpu_blocks, npu_ops):
    """The mixed RC block region (u32 words): C copy canons then NPU compute canons.

    cpu_blocks: number of CPU copy/reshape blocks (= sum of n_in+1 per CPU branch).
    npu_ops: list of EW op ids, one per NPU compute block (6 tiles x n_adds, in
             tile-major order).  Each block is a 160-word canon: 138-word canon +
             a PC chain link (to 0x280*(idx+1)) + PC14 + GAP, exactly as the
             standalone CPU/NPU paths emit.  Verified address-exact vs the 2+2 ref.
    """
    out = []
    total = cpu_blocks + len(npu_ops)
    for b in range(total):
        if b < cpu_blocks:
            canon = _canon_u32(EW_OP_COPY)
        else:
            canon = _canon_u32(npu_ops[b - cpu_blocks])
        out += canon
        if b < total - 1:
            out += _pc_chain_u32((b + 1) * 0x280) + _pc14_u32() + _gap71_u32()
        else:
            out += _gap69_u32()
    return out


def _build_prefix(n, op=EW_OP_ADD, rc_word_off=0):
    """Unified descriptor prefix for NPU compute and CPU copy chains.

    Both prefixes share the same _fixed_header and the same conceptual layout
    (header -> descriptor/size table -> copy descriptors -> DMA list -> tail);
    they differ only in encoding (NPU packs u64 with the even/odd alignment phase,
    CPU is a u32 stream with an alignment lead). op selects the body:
      - EW_OP_COPY  -> CPU prefix (u32), aligned via rc_word_off
      - anything else -> NPU element-wise prefix (u64), from the PREFIX cache
    """
    if op == EW_OP_COPY:
        return _cpu_prefix(n, rc_word_off)
    return PREFIX[n]


def _cpu_trailing(n, prefix_words):
    """CPU-op regcmd tail as u64 words (cosmetic task descriptors; never patched).

    Record family parallel to _build_trailing, for the [1,4] bool shape: the
    output reshape record carries dim 4 (count 1); intermediate records carry
    [1,1,1,4]. The content is a prefix of the repeated intermediate-record
    stream; the target length follows the toolkit's total RC section size, so
    it absorbs whatever the alignment lead added to the prefix.
    """
    if n == 2:
        return [0]

    total_u32 = 677 + 215 * (n - 2) - 16 * (n // 8)
    target_u64 = (total_u32 - prefix_words - 160 * (n + 1)) // 2

    w = [0] * 7 + [0x0000001000000000, 0x1, 0x4]
    w += [0] * 5
    record = [0x0000002000000000, 0x1, 0x1, 0x1, 0x4, 0, 0, 0]
    while len(w) < target_u64:
        w += record
    return w[:target_u64]


def _cpu_trailing_u32(n, prefix_words):
    return _u64_list_to_u32(_cpu_trailing(n, prefix_words))


def build_cpu_template(n_inputs, ops=None, rc_word_off=0):
    """Backward-compatible alias: build a CPU-fallback (all-`And`) regcmd chain.

    Kept for callers that explicitly want the CPU path; it simply forwards to the
    unified build_template with a chain of CPU ops.
    """
    if ops is None:
        ops = ["And"] * (n_inputs - 1)
    return build_template(n_inputs, ops, rc_word_off=rc_word_off)


def verify_cpu_rc(refs_dir="/tmp"):
    """Prove the CPU regcmd == the chained-op reference RC, byte-for-byte.

    Returns {n: bool}. Reference files are and_chain{n}_ref.rknn.
    """
    import os
    results = {}
    for n in range(2, MAX_INPUTS + 1):
        path = os.path.join(refs_dir, f"and_chain{n}_ref.rknn")
        if not os.path.exists(path):
            continue
        d = open(path, "rb").read()
        hdr = 0x40
        root = hdr + struct.unpack_from("<I", d, hdr)[0]

        def u32(o):
            return struct.unpack_from("<I", d, o)[0]

        def i32(o):
            return struct.unpack_from("<i", d, o)[0]

        def u16(o):
            return struct.unpack_from("<H", d, o)[0]

        def fld(p, f):
            vt = p - i32(p)
            vts = u16(vt)
            e = vt + 4 + f * 2
            return u16(e) if e + 2 <= vt + vts else 0

        rc_off = u32(root + fld(root, 20))
        ref_rc = d[rc_off:u32(root + fld(root, 21))]
        gen = build_template(n, ops=["And"] * (n - 1), rc_word_off=rc_off // 4)
        results[n] = gen == ref_rc
    return results



def all_templates(max_n=None):
    upper = max_n or MAX_INPUTS
    if upper > MAX_INPUTS:
        raise NotImplementedError(
            f"regcmd template generation supports up to {MAX_INPUTS} inputs, got {upper}")
    return {n: build_template(n) for n in range(2, upper + 1)}
