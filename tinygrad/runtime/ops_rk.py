import ctypes, functools, itertools, json, os, tempfile
import numpy as np

from tinygrad.device import Allocator, Compiled, Compiler
from tinygrad.helpers import cpu_profile, mv_address
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import _to_np_dtype, dtypes

def _addr(buf) -> int: return mv_address(memoryview(buf).cast("B"))

import ctypes, mmap, os, struct, sys
from fcntl import ioctl
import numpy as np

# ── DRM IOCTL ────────────────────────────────────────────────────────────────
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1
RKNPU_JOB_PINGPONG = 4

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("obj_addr", ctypes.c_uint64),
                ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("offset", ctypes.c_uint64)]

class rknpu_mem_sync(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64)]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

class rknpu_submit(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32),
                ("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32),
                ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
                ("task_obj_addr", ctypes.c_uint64),
                ("iommu_domain_id", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("task_base_addr", ctypes.c_uint64), ("hw_elapse_time", ctypes.c_int64),
                ("core_mask", ctypes.c_uint32), ("fence_fd", ctypes.c_int32),
                ("subcore_task", rknpu_subcore_task * 5)]

class rknpu_task(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32),
                ("enable_mask", ctypes.c_uint32), ("int_mask", ctypes.c_uint32),
                ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
                ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32),
                ("regcmd_addr", ctypes.c_uint64)]

class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

def _IOWR(ty, nr, sz): return (3 << 30) | (ord(ty) << 8) | nr | (sz << 16)
IOCTL_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
IOCTL_MEM_MAP    = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
IOCTL_SUBMIT     = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
IOCTL_MEM_SYNC   = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_sync))
IOCTL_ACTION     = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

# ── FlatBuffer Reader (minimal) ──────────────────────────────────────────────
class FB:
    def __init__(self, data):
        self.b = data
    def u16(self, o): return struct.unpack_from("<H", self.b, o)[0]
    def u32(self, o): return struct.unpack_from("<I", self.b, o)[0]
    def i32(self, o): return struct.unpack_from("<i", self.b, o)[0]
    def root(self): return self.u32(0)

    def _vt(self, pos):
        return pos - self.i32(pos)

    def field_abs(self, pos, field):
        vt = self._vt(pos)
        entry = vt + 4 + field * 2
        if entry + 2 > vt + self.u16(vt):
            return None
        off = self.u16(entry)
        return pos + off if off else None

    def scalar_u32(self, pos, field, default=None):
        a = self.field_abs(pos, field)
        return self.u32(a) if a is not None else default

    def string(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return None
        t = a + self.u32(a)
        n = self.u32(t)
        return self.b[t + 4:t + 4 + n].decode("ascii", "replace")

    def vec_u32(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return None
        v = a + self.u32(a)
        n = self.u32(v)
        return [self.u32(v + 4 + i * 4) for i in range(n)]

    def vec_tables(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return []
        v = a + self.u32(a)
        n = self.u32(v)
        return [v + 4 + i * 4 + self.u32(v + 4 + i * 4) for i in range(n)]

# ── Tensor / Node / Block types ─────────────────────────────────────────────
class Tensor:
    __slots__ = ("idx", "name", "native", "logical", "size", "offset", "n_elems", "_dma_addr", "_buf")
    def __init__(self, idx, name, native, logical, size, offset):
        self.idx = idx; self.name = name
        self.native = native or []; self.logical = logical or []
        self.size = size; self.offset = offset
        self.n_elems = 1
        for d in (logical or native or [1]):
            self.n_elems *= d
        self._dma_addr = None
        self._buf = None
    def __repr__(self):
        return f"Tensor({self.idx}, {self.name}, native={self.native}, logical={self.logical}, {self.size}@{self.offset})"

class Node:
    __slots__ = ("idx", "op", "name", "inputs", "outputs", "target", "cpu_kernel")
    def __init__(self, idx, op, name):
        self.idx = idx; self.op = op; self.name = name
        self.inputs = []; self.outputs = []
        self.target = None; self.cpu_kernel = None
    def __repr__(self):
        return f"Node({self.idx}, {self.op}, {self.name}, target={self.target})"

class CommandBlock:
    __slots__ = ("word_offset", "n_words", "words", "kind")
    def __init__(self, word_offset, n_words, words, kind):
        self.word_offset = word_offset; self.n_words = n_words
        self.words = words; self.kind = kind

# ── RKNN Container Parser ───────────────────────────────────────────────────
HEADER_SIZE = 0x40

def parse_rknn(data):
    data = bytes(data)
    if data[:4] != b"RKNN":
        raise ValueError("not an RKNN file")
    version = struct.unpack_from("<Q", data, 0x08)[0]
    body_size = struct.unpack_from("<Q", data, 0x10)[0]
    body = data[HEADER_SIZE:HEADER_SIZE + body_size]

    fb = FB(body)
    root = fb.root()
    subgraphs = fb.vec_tables(root, 2)
    if not subgraphs:
        raise ValueError("no subgraphs")
    sg = subgraphs[0]

    tensors = []
    for i, p in enumerate(fb.vec_tables(sg, 0)):
        tensors.append(Tensor(
            idx=i,
            name=fb.string(p, 5),
            native=fb.vec_u32(p, 3),
            logical=fb.vec_u32(p, 4),
            size=fb.scalar_u32(p, 12),
            offset=fb.scalar_u32(p, 13),
        ))

    nodes = []
    for i, p in enumerate(fb.vec_tables(sg, 1)):
        n = Node(i, fb.string(p, 1) or "", fb.string(p, 2) or "")
        n.target = fb.scalar_u32(p, 6, default=None)
        n.cpu_kernel = fb.scalar_u32(p, 7, default=None)
        n.inputs = fb.vec_u32(p, 4) or []
        n.outputs = fb.vec_u32(p, 5) or []
        nodes.append(n)

    n_words = len(body) // 8
    all_words = list(struct.unpack_from(f"<{n_words}Q", body, 0))

    NPU_TARGETS = {0x0101, 0x0201, 0x0801, 0x1001, 0x2001, 0x4001, 0x8001}
    blocks = []
    i = 0
    while i < n_words:
        if ((all_words[i] >> 48) & 0xFFFF) in NPU_TARGETS:
            j = i
            while j < n_words and ((all_words[j] >> 48) & 0xFFFF) in NPU_TARGETS:
                j += 1
            if j - i >= 20:
                kind = _classify_block(all_words[i:j])
                blocks.append(CommandBlock(i, j - i, all_words[i:j], kind))
            i = j
        else:
            i += 1

    return {"version": version, "body": body, "body_size": body_size,
            "tensors": tensors, "nodes": nodes, "blocks": blocks}

def _classify_block(words):
    reg_map = {}
    for w in words:
        target = (w >> 48) & 0xFFFF
        reg = w & 0xFFFF
        val = (w >> 16) & 0xFFFFFFFF
        reg_map[(target, reg)] = val
    fmt = reg_map.get((0x1001, 0x4010), 0)
    ew = reg_map.get((0x1001, 0x4070), 0)
    is_binary = (ew & 0x40) and not (ew & 0x01)
    if fmt == 0x48000002 and is_binary:
        return "EW_BINARY"
    return "COPY"

# ── NPU Device ───────────────────────────────────────────────────────────────
class NPUDevice:
    def __init__(self, dev="/dev/dri/card1"):
        self.fd = os.open(dev, os.O_RDWR)

    def mem_alloc(self, size, flags=0):
        mc = rknpu_mem_create(flags=flags, size=size)
        ctypes.memset(ctypes.addressof(mc), 0, ctypes.sizeof(mc))
        mc.flags = flags; mc.size = size
        ioctl(self.fd, IOCTL_MEM_CREATE, mc)
        mm = rknpu_mem_map(handle=mc.handle)
        ctypes.memset(ctypes.addressof(mm), 0, ctypes.sizeof(mm))
        mm.handle = mc.handle
        ioctl(self.fd, IOCTL_MEM_MAP, mm)
        buf = mmap.mmap(self.fd, mc.size, mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
        return buf, mc

    def mem_sync(self, handle, size, offset=0, flags=1):
        s = rknpu_mem_sync(handle=handle, flags=flags, offset=offset, size=size)
        ioctl(self.fd, IOCTL_MEM_SYNC, s)

    def reset(self):
        ioctl(self.fd, IOCTL_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

    def submit(self, task_obj_addr, task_start, task_number):
        self.reset()
        s = rknpu_submit(
            flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG, timeout=6000,
            task_start=task_start, task_number=task_number,
            task_counter=0, priority=0, task_obj_addr=task_obj_addr,
            core_mask=0, fence_fd=-1)
        s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
        s.subcore_task[1] = rknpu_subcore_task(task_start=0, task_number=1)
        s.subcore_task[2] = rknpu_subcore_task(task_start=0, task_number=1)
        return ioctl(self.fd, IOCTL_SUBMIT, s)

    def close(self):
        os.close(self.fd)

# ── NC1HWC2 Layout Conversion ───────────────────────────────────────────────
def _elem_size_from_native(native):
    """Determine bytes per element from NC1HWC2 shape [N, C1, H, W, C2]."""
    if len(native) == 5:
        return 16 // native[4]  # atom is 16 bytes, C2 elements per atom
    return 2

def _contiguous_to_nc1hwc2(src, native_shape, stride_atoms=None):
    """Pack flat data into NC1HWC2 layout with optional stride padding.
    native_shape = [N, C1, H, W, C2]. stride_atoms = atoms per C1 row (default W)."""
    if not native_shape or len(native_shape) < 5:
        n = len(src)
        out = np.zeros(n * max(src.dtype.itemsize, 2), dtype=np.uint8)
        out[:n * src.dtype.itemsize] = src.view(np.uint8)
        return out

    N, C1, H, W, C2 = native_shape
    row_atoms = stride_atoms if stride_atoms else W
    total_atoms = N * C1 * H * row_atoms
    elem_bytes = src.dtype.itemsize
    out = np.zeros(total_atoms * C2 * elem_bytes, dtype=np.uint8).view(src.dtype)

    flat = src.ravel()
    idx = 0
    for n in range(N):
        for c1 in range(C1):
            for h in range(H):
                for w in range(W):
                    atom = ((n * C1 + c1) * H + h) * row_atoms + w
                    take = min(C2, len(flat) - idx)
                    if take <= 0:
                        break
                    out[atom * C2:atom * C2 + take] = flat[idx:idx + take]
                    idx += take

    return out.view(np.uint8)

def _nc1hwc2_to_contiguous(packed, native_shape, dtype, stride_atoms=None):
    """Unpack NC1HWC2 layout with stride back to flat contiguous data."""
    if not native_shape or len(native_shape) < 5:
        n = len(packed) // np.dtype(dtype).itemsize
        return packed[:n * np.dtype(dtype).itemsize].view(dtype).copy()

    N, C1, H, W, C2 = native_shape
    row_atoms = stride_atoms if stride_atoms else W
    src = packed.view(dtype) if packed.dtype != dtype else packed.copy()

    result = []
    for n in range(N):
        for c1 in range(C1):
            for h in range(H):
                for w in range(W):
                    atom = ((n * C1 + c1) * H + h) * row_atoms + w
                    for c2 in range(C2):
                        si = atom * C2 + c2
                        if si < len(src):
                            result.append(src[si])

    return np.array(result, dtype=dtype)

# ── CPU Op Kernels ───────────────────────────────────────────────────────────
# Unified kernel table for both the RKNN CPU-fallback ops (And/Or/Not/...) and
# the uop-graph ONNX ops. All kernels take (inputs, attrs={}); attrs carries
# ONNX attributes (BitShift direction, Cast target, Concat axis, ...).
def _not_kernel(inputs, attrs={}):
    a = inputs[0]
    # bitwise_not on bool == logical not; the uint-view trick is float-only.
    if a.dtype.kind == 'f':
        return np.bitwise_not(a.view('u' + str(a.dtype.itemsize))).view(a.dtype)
    return np.bitwise_not(a)

_ONNX_CAST = {1: np.float32, 4: np.uint16, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 12: np.uint32, 13: np.uint64}

# Op types the toolkit-free NPU EW path could service today (fp16 only).
UOP_NPU_SUPPORTED = {"Add", "Sub", "Mul", "Div"}

CPU_KERNELS = {
    "Add":        lambda i, a={}: i[0] + i[1],
    "Sub":        lambda i, a={}: i[0] - i[1],
    "Mul":        lambda i, a={}: i[0] * i[1],
    "Div":        lambda i, a={}: i[0] / i[1],
    "Mod":        lambda i, a={}: i[0] % i[1],
    "Neg":        lambda i, a={}: -i[0],
    # bitwise ops; identical to logical ops on bool arrays
    "And":        lambda i, a={}: np.bitwise_and(i[0], i[1]),
    "Or":         lambda i, a={}: np.bitwise_or(i[0], i[1]),
    "Xor":        lambda i, a={}: np.bitwise_xor(i[0], i[1]),
    "Not":        _not_kernel,
    "BitwiseAnd": lambda i, a={}: np.bitwise_and(i[0], i[1]),
    "BitwiseOr":  lambda i, a={}: np.bitwise_or(i[0], i[1]),
    "BitwiseXor": lambda i, a={}: np.bitwise_xor(i[0], i[1]),
    "BitShift":   lambda i, a={}: (i[0] << i[1]) if a.get("direction") == "LEFT" else (i[0] >> i[1]),
    # comparisons / dtype / shape
    "Equal":      lambda i, a={}: i[0] == i[1],
    "Less":       lambda i, a={}: i[0] < i[1],
    "Cast":       lambda i, a={}: i[0].astype(_ONNX_CAST[a["to"]]),
    "Gather":     lambda i, a={}: np.take(i[0], i[1].astype(np.int64), axis=a.get("axis", 0)),
    "Concat":     lambda i, a={}: np.concatenate(i, axis=a.get("axis", 0)),
    "Identity":   lambda i, a={}: i[0],
    "Reshape":    lambda i, a={}: i[0].reshape(i[1].astype(np.int64)),
}

# RKNN graph scheduling treats these as structural (NPU COPY / IO), never as
# CPU compute ops, even though CPU_KERNELS has a Reshape kernel for uop graphs.
RKNN_GRAPH_CPU_OPS = set(CPU_KERNELS) - {"Reshape", "Identity", "Gather", "Concat", "Cast"}

# ── DMA Address Patching ────────────────────────────────────────────────────
REG_DST_BASE  = 0x4020  # DPU_DST_BASE_ADDR
REG_RDMA_SRC  = 0x5018  # DPU_RDMA_RDMA_SRC_BASE_ADDR
REG_RDMA_EW   = 0x5038  # DPU_RDMA_RDMA_EW_BASE_ADDR
DPU_TARGET    = 0x1001
RDMA_TARGET   = 0x2001

def _patch_dma_addr(words, reg, target, addr):
    for i, w in enumerate(words):
        t = (w >> 48) & 0xFFFF
        r = w & 0xFFFF
        if t == target and r == reg:
            val = (addr & 0xFFFFFFFF)
            words[i] = (target << 48) | (val << 16) | reg
            return True
    return False

# ── RKNN Runtime ─────────────────────────────────────────────────────────────
class RKNNRuntime:
    def __init__(self, rknn_path):
        with open(rknn_path, "rb") as f:
            self.raw = f.read()
        self.model = parse_rknn(self.raw)
        self.dev = NPUDevice()
        self._buffers = {}
        self._feature_buf = None
        self._feature_mc = None
        self._input_bufs = []
        self._output_bufs = []
        self._task_buf = None
        self._task_mc = None
        self._model_buf = None
        self._model_mc = None
        self._graph = None
        self._npu_submit_order = []
        self._cpu_ops = []
        self._input_tensors = []
        self._output_tensors = []
        self._alu_fallback = False
        self._alu_input_arrays = []
        self._alu_input_shapes = []
        self._alu_result = None
        self._uop = None
        self._npu_ew = None
        self._init()

    def _init(self):
        m = self.model
        tensors = m["tensors"]
        nodes = m["nodes"]
        blocks = m["blocks"]

        # ── uop-graph models (node-per-op export, trailer-flagged) ──
        # Same dispatch pattern as the COPY/EW graph paths below: detect the
        # model kind, set up its executor state, and return.
        body_size = struct.unpack_from("<Q", self.raw, 0x10)[0]
        off = HEADER_SIZE + body_size
        if off + 8 <= len(self.raw):
            tlen = struct.unpack_from("<Q", self.raw, off)[0]
            try:
                import json
                trailer = json.loads(self.raw[off + 8: off + 8 + tlen])
            except Exception:
                trailer = None
            if isinstance(trailer, dict) and trailer.get("uop_graph"):
                self._uop = trailer
                self._uop_inputs = [n.outputs[0] for n in nodes if n.op == "InputOperator"]
                self._uop_outputs = [n.inputs[0] for n in nodes if n.op == "OutputOperator"]
                self._uop_values = {int(ti): np.array(c["data"], dtype=np.dtype(c["dtype"]))
                                    for ti, c in trailer["consts"].items()}
                # Optional companion fp16 EW model: its baked register blocks
                # let NPU-supported fp16 ops execute on the NPU (hybrid).
                ew_path = trailer.get("npu_ew_model")
                if ew_path and os.path.exists(ew_path):
                    self._npu_ew = RKNNRuntime(ew_path)
                op_types = sorted({n.op for n in nodes if n.op not in ("InputOperator", "OutputOperator")})
                npu_ok = [o for o in op_types if o in UOP_NPU_SUPPORTED]
                cpu_only = [o for o in op_types if o not in UOP_NPU_SUPPORTED]
                print(f"UopGraph executor: {len(nodes)} nodes, {len(self._uop_values)} consts, "
                      f"npu_ew={'yes' if self._npu_ew else 'no'}")
                print(f"  NPU ops (fp16 EW): {npu_ok}")
                print(f"  CPU ops: {cpu_only}")
                return

        # Classify tensors
        for t in tensors:
            if t.name and t.name.startswith("InputOperator") or (t.logical and len(t.logical) >= 2 and t.name and t.name in [n.name.split(":")[-1] for n in nodes if n.op == "InputOperator"]):
                pass

        # Find input/output tensors by name
        input_node_names = [n.name for n in nodes if n.op == "InputOperator"]
        output_node_names = [n.name for n in nodes if n.op == "OutputOperator"]

        # Input tensors: named "x", "y" etc or linked from InputOperator nodes
        # Input tensors: external inputs named x/y or a/b or with InputOperator
        self._input_tensors = [t for t in tensors
            if t.name and t.name.lower() in ("x", "y", "a", "b", "input0", "input1")]
        self._output_tensors = [t for t in tensors
            if t.name and t.name.lower() in ("z", "c", "output0", "output")]

        # If no named matches, use position: first external tensors with logical shape
        if not self._input_tensors:
            self._input_tensors = [t for t in tensors if t.logical and len(t.logical) >= 2 and t.offset is None and "rs" not in (t.name or "")][:2]
        if not self._output_tensors:
            self._output_tensors = [t for t in tensors if t.logical and len(t.logical) >= 2 and t.offset is None and "rs" not in (t.name or "")][-1:]

        # Classify: NPU-op vs CPU-op
        cpu_ops = []
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator", "Reshape"):
                continue
            cpu_ops.append(n)

        # Decide execution path:
        # - If blocks are EW_BINARY with fp16/int8 precision → NPU can execute them
        # - If blocks are COPY or EW with int32/fp32 precision → CPU fallback
        has_ew_binary = any(blk.kind == "EW_BINARY" for blk in blocks)
        # Check precision from first block's DATA_FORMAT register
        npu_compatible = False
        proc_prec = None
        if has_ew_binary and blocks:
            fmt_val = None
            for w in blocks[0].words:
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                val = (w >> 16) & 0xFFFFFFFF
                if target == DPU_TARGET and reg == 0x4010:  # DATA_FORMAT
                    fmt_val = val
                    break
            if fmt_val is not None:
                proc_prec = fmt_val & 0xF
                npu_compatible = proc_prec in (2, 0)  # fp16=2, int8=0

        self._cpu_ops = cpu_ops

        all_copy = all(blk.kind == "COPY" for blk in blocks) if blocks else False
        has_cpu_ops = any(n.op in RKNN_GRAPH_CPU_OPS for n in nodes)
        if all_copy and has_cpu_ops:
            self._graph = True
            self._all_cpu = False
            self._init_graph()
            return

        has_copy = any(blk.kind == "COPY" for blk in blocks)
        has_ew = any(blk.kind == "EW_BINARY" for blk in blocks)
        if has_copy and has_ew and has_cpu_ops:
            self._graph = True
            self._all_cpu = False
            self._init_graph_hybrid()
            return

        # Element-wise ops (Add/Sub/Mul/Div) whose baked blocks are fp16/int8 run
        # on the NPU directly from the .rknn body — the register commands are
        # extracted from the file (no synthesis), inputs are NC1HWC2-reshaped, the
        # 3 DMA bases are patched, and the body's task blocks are submitted. This
        # holds for ANY size baked into the file: a larger .rknn simply carries
        # more blocks. fp16 EW therefore NEVER falls back to CPU.
        all_cpu = not npu_compatible
        self._all_cpu = all_cpu

        # For NPU path: determine how many tasks to submit
        # Vendor runtime submits len(blocks)//2 tasks (ping half only)
        if not all_cpu and len(blocks) >= 2:
            self._npu_task_count = len(blocks) // 2
        else:
            self._npu_task_count = 0

        # Allocate DMA buffers
        # 1. Model buffer (holds .rknn data including regcmd blocks)
        self._model_buf, self._model_mc = self.dev.mem_alloc(
            max(len(self.raw), 4096), RKNPU_MEM_KERNEL_MAPPING)

        # Copy model data into model buffer
        mv = memoryview(self._model_buf)
        mv[:len(self.raw)] = self.raw

        # 2. Feature/working buffer
        feature_tensors = [t for t in tensors if t.offset is not None and t.size and "rs" in (t.name or "")]
        if not feature_tensors:
            feature_tensors = [t for t in tensors if t.offset is not None and t.size]
        rs_tensors_nc1hwc2 = {t.name: t for t in tensors
                              if t.name and "rs" in t.name and t.native and len(t.native) == 5
                              and t.name not in (n.name for n in feature_tensors if n.name)}
        for name, t in rs_tensors_nc1hwc2.items():
            if t.offset is None:
                max_off = max((ft.offset + ft.size for ft in feature_tensors), default=0)
                aligned = (max_off + 63) & ~63
                t.offset = aligned
                feature_tensors.append(t)

        # Also assign offsets for input/output tensors that have offset=None
        for t in list(self._input_tensors) + list(self._output_tensors):
            if t.offset is None and t.size:
                max_off = max((ft.offset + ft.size for ft in feature_tensors), default=0)
                aligned = (max_off + 63) & ~63
                t.offset = aligned
                feature_tensors.append(t)

        feature_size = max((t.offset + t.size for t in feature_tensors), default=4096)
        feature_size = max(feature_size, 4096)

        # For NPU path: allocate separate DMA buffers per RS tensor (like vendor)
        self._rs_bufs = {}
        if not all_cpu:
            for t in feature_tensors:
                if t.native and len(t.native) == 5:
                    sz = max(t.size, 4096)
                    buf, mc = self.dev.mem_alloc(sz, RKNPU_MEM_KERNEL_MAPPING)
                    t.offset = 0
                    t._dma_addr = mc.dma_addr
                    t._buf = buf
                    self._rs_bufs[t.name] = (buf, mc, t)
            # Also allocate for any NC1HWC2 tensor that was added
            self._feature_buf, self._feature_mc = self.dev.mem_alloc(4096, RKNPU_MEM_KERNEL_MAPPING)
        else:
            self._feature_buf, self._feature_mc = self.dev.mem_alloc(feature_size, RKNPU_MEM_KERNEL_MAPPING)
            for t in feature_tensors:
                t._dma_addr = self._feature_mc.dma_addr

        # 3. Input buffers (one per input tensor)
        for t in self._input_tensors:
            sz = max(t.size or (np.prod(t.logical) * 4), 4096)
            buf, mc = self.dev.mem_alloc(sz)
            self._input_bufs.append((buf, mc, t))

        # 4. Output buffer
        for t in self._output_tensors:
            sz = max(t.size or (np.prod(t.logical) * 4), 4096)
            buf, mc = self.dev.mem_alloc(sz)
            self._output_bufs.append((buf, mc, t))

        # 5. Task array buffer
        n_tasks = len(blocks)
        task_size = max(n_tasks * 40, 4096)
        self._task_buf, self._task_mc = self.dev.mem_alloc(task_size, RKNPU_MEM_KERNEL_MAPPING)

        # Build task array from blocks
        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(self._task_buf)),
                            ctypes.POINTER(rknpu_task))
        body = m["body"]
        regcmd_offset = blocks[0].word_offset * 8 if blocks else 0

        for i, blk in enumerate(blocks):
            tasks[i].flags = 0
            tasks[i].op_idx = 4
            tasks[i].enable_mask = 0x18
            tasks[i].int_mask = 0x300
            tasks[i].int_clear = 0x1ffff
            tasks[i].int_status = 0
            tasks[i].regcfg_amount = blk.n_words
            tasks[i].regcfg_offset = 0
            tasks[i].regcmd_addr = self._model_mc.dma_addr + HEADER_SIZE + blk.word_offset * 8

        self._tasks = tasks
        self._n_tasks = n_tasks
        self._blocks = blocks
        self._feature_tensors = feature_tensors

        # Extract NPU stride from first block (for stride-aware NC1HWC2 packing)
        self._npu_stride = None
        if blocks:
            for w in blocks[0].words:
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                val = (w >> 16) & 0xFFFFFFFF
                if target == DPU_TARGET and reg == 0x4024:
                    self._npu_stride = val
                    break

        # Fix z-rs/c-rs offset if None (vendor runtime places output RS at offset 0)
        z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
        if z_rs and z_rs.offset is None:
            z_rs.offset = 0
            print(f"  Fixed {z_rs.name} offset to 0")

        print(f"RKNNRuntime init: {len(nodes)} nodes, {len(blocks)} blocks, "
              f"{len(cpu_ops)} CPU ops, all_cpu={all_cpu}")
        print(f"  inputs: {[t.name for t in self._input_tensors]}")
        print(f"  outputs: {[t.name for t in self._output_tensors]}")
        print(f"  feature_buf: {feature_size} bytes, model_buf: {len(self.raw)} bytes")

    def _get_feature_tensor(self, name):
        alt_map = {"x_rs": "a_rs", "a_rs": "x_rs",
                   "y_rs": "b_rs", "b_rs": "y_rs",
                   "z-rs": "c-rs", "c-rs": "z-rs"}
        candidates = [name, name.lower(), name.upper()]
        # Also add alternative names
        for n in list(candidates):
            if n in alt_map:
                candidates.extend([alt_map[n], alt_map[n].upper(), alt_map[n].lower()])
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        candidates = unique

        # Exact match first (case-insensitive)
        for c in candidates:
            for t in self._feature_tensors:
                if t.name and t.name.lower() == c.lower():
                    return t
        # Then NC1HWC2 prefix match
        for c in candidates:
            nc1hwc2 = [t for t in self._feature_tensors
                       if t.name and t.name.lower().startswith(c.lower()) and t.native and len(t.native) == 5]
            if nc1hwc2:
                return nc1hwc2[0]
        return None

    def _detect_model_dtype(self):
        for n in self.model["nodes"]:
            if n.op == "InputOperator":
                for ti in n.outputs:
                    t = self.model["tensors"][ti]
                    if t.native and len(t.native) == 5:
                        c2 = t.native[4]
                        if c2 == 8: return np.float16
                        elif c2 == 4: return np.int32
                        elif c2 == 2: return np.float16
                        elif c2 == 1: return np.int8
        return np.int32

    def _init_graph(self):
        m = self.model
        tensors = m["tensors"]
        nodes = m["nodes"]
        blocks = m["blocks"]

        self._input_tensors = []
        self._output_tensors = []
        for n in nodes:
            if n.op == "InputOperator":
                for ti in n.outputs:
                    self._input_tensors.append(tensors[ti])
            elif n.op == "OutputOperator":
                for ti in n.inputs:
                    self._output_tensors.append(tensors[ti])

        self._graph_dtype = self._detect_model_dtype()
        elem_size = self._graph_dtype().itemsize

        data_tensor_indices = set()
        for n in nodes:
            if n.op == "InputOperator":
                data_tensor_indices.update(n.outputs)
            elif n.op == "OutputOperator":
                data_tensor_indices.update(n.inputs)
            else:
                data_tensor_indices.update(n.outputs)

        self._tensor_bufs = {}
        for ti in sorted(data_tensor_indices):
            t = tensors[ti]
            sz = max(t.size or 0,
                     int(np.prod(t.logical)) * elem_size if t.logical else 0,
                     4096)
            buf, mc = self.dev.mem_alloc(sz, RKNPU_MEM_KERNEL_MAPPING)
            t._dma_addr = mc.dma_addr
            t._buf = buf
            self._tensor_bufs[ti] = (buf, mc)

        self._model_buf, self._model_mc = self.dev.mem_alloc(
            max(len(self.raw), 4096), RKNPU_MEM_KERNEL_MAPPING)
        mv = memoryview(self._model_buf)
        mv[:len(self.raw)] = self.raw

        n_tasks = len(blocks)
        task_size = max(n_tasks * 40, 4096)
        self._task_buf, self._task_mc = self.dev.mem_alloc(task_size, RKNPU_MEM_KERNEL_MAPPING)

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(self._task_buf)),
                            ctypes.POINTER(rknpu_task))
        for i, blk in enumerate(blocks):
            tasks[i].flags = 0
            tasks[i].op_idx = 4
            tasks[i].enable_mask = 0x18
            tasks[i].int_mask = 0x300
            tasks[i].int_clear = 0x1ffff
            tasks[i].int_status = 0
            tasks[i].regcfg_amount = blk.n_words
            tasks[i].regcfg_offset = 0
            tasks[i].regcmd_addr = self._model_mc.dma_addr + HEADER_SIZE + blk.word_offset * 8

        self._tasks = tasks
        self._n_tasks = n_tasks
        self._blocks = blocks

        block_to_node = {}
        bi = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op in RKNN_GRAPH_CPU_OPS:
                continue
            if bi < len(blocks):
                block_to_node[bi] = n
                bi += 1

        hdr_qwords = HEADER_SIZE // 8
        model_view = (ctypes.c_uint64 * (len(self.raw) // 8)).from_buffer(self._model_buf)

        for bi, blk in enumerate(blocks):
            node = block_to_node.get(bi)
            if node is None:
                continue
            data_inputs = [ti for ti in node.inputs if ti in data_tensor_indices]
            data_outputs = [ti for ti in node.outputs if ti in data_tensor_indices]
            if not data_inputs or not data_outputs:
                continue
            src_ti = data_inputs[0]
            dst_ti = data_outputs[0]
            src_addr = tensors[src_ti]._dma_addr
            dst_addr = tensors[dst_ti]._dma_addr
            for wi in range(blk.n_words):
                abs_idx = hdr_qwords + blk.word_offset + wi
                w = model_view[abs_idx]
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                old_val = (w >> 16) & 0xFFFFFFFF
                if target == RDMA_TARGET and reg == REG_RDMA_SRC:
                    model_view[abs_idx] = (target << 48) | (((src_addr + old_val) & 0xFFFFFFFF) << 16) | reg
                elif target == DPU_TARGET and reg == REG_DST_BASE:
                    model_view[abs_idx] = (target << 48) | (((dst_addr + old_val) & 0xFFFFFFFF) << 16) | reg

        self.dev.mem_sync(self._model_mc.handle, self._model_mc.size)

        schedule = []
        task_start = 0
        pending_tasks = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op in RKNN_GRAPH_CPU_OPS:
                if pending_tasks > 0:
                    schedule.append(("npu", task_start, pending_tasks))
                    task_start += pending_tasks
                    pending_tasks = 0
                schedule.append(("cpu", n))
            else:
                pending_tasks += 1
        if pending_tasks > 0:
            schedule.append(("npu", task_start, pending_tasks))

        self._graph_schedule = schedule
        print(f"Graph executor: {len(blocks)} COPY blocks, "
              f"{len([s for s in schedule if s[0]=='cpu'])} CPU ops, dtype={self._graph_dtype}")
        print(f"  inputs:  {[t.name for t in self._input_tensors]}")
        print(f"  outputs: {[t.name for t in self._output_tensors]}")
        print(f"  data_tensors: {sorted(data_tensor_indices)}")
        print(f"  schedule: {[(s[0], s[1].op if s[0]=='cpu' else f'tasks {s[1]}-{s[1]+s[2]-1}') for s in schedule]}")

    def _init_graph_hybrid(self):
        m = self.model
        tensors = m["tensors"]
        nodes = m["nodes"]
        blocks = m["blocks"]

        self._input_tensors = []
        self._output_tensors = []
        for n in nodes:
            if n.op == "InputOperator":
                for ti in n.outputs:
                    self._input_tensors.append(tensors[ti])
            elif n.op == "OutputOperator":
                for ti in n.inputs:
                    self._output_tensors.append(tensors[ti])

        self._graph_dtype = self._detect_model_dtype()

        cpu_op_names = set(RKNN_GRAPH_CPU_OPS)
        copy_block_indices = set()
        for bi, blk in enumerate(blocks):
            if blk.kind == "COPY":
                copy_block_indices.add(bi)
        ew_block_start = min(bi for bi in range(len(blocks)) if blocks[bi].kind == "EW_BINARY") if any(blocks[bi].kind == "EW_BINARY" for bi in range(len(blocks))) else len(blocks)

        copy_reshape_outputs = set()
        bi = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op == "Reshape" and bi in copy_block_indices:
                for ti in n.outputs:
                    copy_reshape_outputs.add(ti)
                bi += 1
            elif n.op == "Reshape":
                bi += 1

        cpu_nodes = []
        npu_ew_nodes = []
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator", "Reshape"):
                continue
            has_copy_input = any(ti in copy_reshape_outputs for ti in n.inputs)
            if has_copy_input:
                cpu_nodes.append(n)
            else:
                npu_ew_nodes.append(n)

        data_tensor_indices = set()
        for n in nodes:
            if n.op == "InputOperator":
                data_tensor_indices.update(n.outputs)
            elif n.op == "OutputOperator":
                data_tensor_indices.update(n.inputs)
            else:
                data_tensor_indices.update(n.outputs)

        self._tensor_bufs = {}
        for ti in sorted(data_tensor_indices):
            t = tensors[ti]
            sz = max(t.size or 0, 4096)
            buf, mc = self.dev.mem_alloc(sz, RKNPU_MEM_KERNEL_MAPPING)
            t._dma_addr = mc.dma_addr
            t._buf = buf
            self._tensor_bufs[ti] = (buf, mc)

        self._model_buf, self._model_mc = self.dev.mem_alloc(
            max(len(self.raw), 4096), RKNPU_MEM_KERNEL_MAPPING)
        mv = memoryview(self._model_buf)
        mv[:len(self.raw)] = self.raw

        n_tasks = len(blocks)
        task_size = max(n_tasks * 40, 4096)
        self._task_buf, self._task_mc = self.dev.mem_alloc(task_size, RKNPU_MEM_KERNEL_MAPPING)
        self._tasks = ctypes.cast(
            ctypes.addressof(ctypes.c_char.from_buffer(self._task_buf)),
            ctypes.POINTER(rknpu_task))
        for i, blk in enumerate(blocks):
            self._tasks[i].flags = 0
            self._tasks[i].op_idx = 4
            self._tasks[i].enable_mask = 0x18
            self._tasks[i].int_mask = 0x300
            self._tasks[i].int_clear = 0x1ffff
            self._tasks[i].int_status = 0
            self._tasks[i].regcfg_amount = blk.n_words
            self._tasks[i].regcfg_offset = 0
            self._tasks[i].regcmd_addr = self._model_mc.dma_addr + HEADER_SIZE + blk.word_offset * 8
        self._n_tasks = n_tasks
        self._blocks = blocks

        copy_nodes = []
        bi = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op in cpu_op_names:
                continue
            if n.op in ("Reshape",):
                if bi < len(blocks) and blocks[bi].kind == "COPY":
                    copy_nodes.append((bi, n))
                    bi += 1
                else:
                    bi += 1
            elif n.op in ("Add", "Sub", "Mul", "Div") and blocks[bi].kind == "EW_BINARY":
                bi += 1

        hdr_qwords = HEADER_SIZE // 8
        model_view = (ctypes.c_uint64 * (len(self.raw) // 8)).from_buffer(self._model_buf)
        for bi_copy, node in copy_nodes:
            blk = blocks[bi_copy]
            data_inputs = [ti for ti in node.inputs if ti in data_tensor_indices]
            data_outputs = [ti for ti in node.outputs if ti in data_tensor_indices]
            if not data_inputs or not data_outputs:
                continue
            src_ti = data_inputs[0]
            dst_ti = data_outputs[0]
            src_addr = tensors[src_ti]._dma_addr
            dst_addr = tensors[dst_ti]._dma_addr
            for wi in range(blk.n_words):
                abs_idx = hdr_qwords + blk.word_offset + wi
                w = model_view[abs_idx]
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                old_val = (w >> 16) & 0xFFFFFFFF
                if target == RDMA_TARGET and reg == REG_RDMA_SRC:
                    model_view[abs_idx] = (target << 48) | (((src_addr + old_val) & 0xFFFFFFFF) << 16) | reg
                elif target == DPU_TARGET and reg == REG_DST_BASE:
                    model_view[abs_idx] = (target << 48) | (((dst_addr + old_val) & 0xFFFFFFFF) << 16) | reg

        ew_node = npu_ew_nodes[0] if npu_ew_nodes else None
        if ew_node:
            ew_inputs = [ti for ti in ew_node.inputs if ti in data_tensor_indices]
            ew_outputs = [ti for ti in ew_node.outputs if ti in data_tensor_indices]
            x_rs_addr = tensors[ew_inputs[0]]._dma_addr if len(ew_inputs) > 0 else 0
            y_rs_addr = tensors[ew_inputs[1]]._dma_addr if len(ew_inputs) > 1 else 0
            z_rs_addr = tensors[ew_outputs[0]]._dma_addr if len(ew_outputs) > 0 else 0
            ew_start = len(copy_nodes)
            for blk in blocks[ew_start:]:
                if blk.kind != "EW_BINARY":
                    continue
                for wi in range(blk.n_words):
                    abs_idx = hdr_qwords + blk.word_offset + wi
                    w = model_view[abs_idx]
                    target = (w >> 48) & 0xFFFF
                    reg = w & 0xFFFF
                    old_val = (w >> 16) & 0xFFFFFFFF
                    if target == RDMA_TARGET and reg == REG_RDMA_SRC:
                        model_view[abs_idx] = (target << 48) | (((x_rs_addr + old_val) & 0xFFFFFFFF) << 16) | reg
                    elif target == RDMA_TARGET and reg == REG_RDMA_EW:
                        model_view[abs_idx] = (target << 48) | (((y_rs_addr + old_val) & 0xFFFFFFFF) << 16) | reg
                    elif target == DPU_TARGET and reg == REG_DST_BASE:
                        model_view[abs_idx] = (target << 48) | (((z_rs_addr + old_val) & 0xFFFFFFFF) << 16) | reg

        ew_block_count = len(blocks) - ew_block_start
        ew_ping = ew_block_count // 2 if ew_block_count >= 2 else ew_block_count
        self._hybrid_ew_start = ew_block_start
        self._hybrid_ew_ping = ew_ping

        copy_block_to_node = {}
        for bi_copy, node in copy_nodes:
            copy_block_to_node[bi_copy] = node

        cpu_node_set = {n.idx for n in cpu_nodes}
        ew_node_set = {n.idx for n in npu_ew_nodes}

        schedule = []
        copy_queue = []
        ew_emitted = False
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.idx in cpu_node_set:
                if copy_queue:
                    copy_queue.sort()
                    schedule.append(("npu_copy", copy_queue[0], len(copy_queue)))
                    copy_queue = []
                schedule.append(("cpu", n))
            elif n.idx in ew_node_set and not ew_emitted:
                if copy_queue:
                    copy_queue.sort()
                    schedule.append(("npu_copy", copy_queue[0], len(copy_queue)))
                    copy_queue = []
                schedule.append(("npu_ew", ew_block_start, 1))
                ew_emitted = True
            elif n.op == "Reshape":
                for bi, bn in copy_block_to_node.items():
                    if bn.idx == n.idx and bi not in copy_queue:
                        copy_queue.append(bi)
                        break
        if copy_queue:
            schedule.append(("npu_copy", copy_queue[0], len(copy_queue)))
        if not ew_emitted and ew_block_count > 0:
            schedule.append(("npu_ew", ew_block_start, 1))

        self._graph_schedule = schedule

        npu_input_rs = {}
        if ew_node:
            for ti in ew_node.inputs:
                if ti in data_tensor_indices:
                    rs_t = tensors[ti]
                    for n in nodes:
                        if n.op == "Reshape":
                            d_ins = [x for x in n.inputs if x in data_tensor_indices]
                            if d_ins and d_ins[0] != ti and n.outputs and n.outputs[0] == ti:
                                npu_input_rs[d_ins[0]] = ti
                    for n in nodes:
                        if n.op == "InputOperator" and ti not in [o for nn in nodes if nn.op == "InputOperator" for o in nn.outputs]:
                            for nn in nodes:
                                if nn.op == "Reshape":
                                    d_ins = [x for x in nn.inputs if x in data_tensor_indices]
                                    if d_ins and nn.outputs and nn.outputs[0] == ti:
                                        src_ti = d_ins[0]
                                        npu_input_rs[src_ti] = ti

        npu_input_rs_final = {}
        for src_ti, rs_ti in npu_input_rs.items():
            rs_t = tensors[rs_ti]
            if rs_t.native and len(rs_t.native) >= 5 and rs_t.native[4] == 8:
                npu_input_rs_final[src_ti] = rs_ti
        self._npu_input_rs = npu_input_rs_final

        self._rs_bufs = {}
        for src_ti, rs_ti in npu_input_rs_final.items():
            rs_t = tensors[rs_ti]
            self._rs_bufs[rs_t.name] = (rs_t._buf, None, rs_t)
        if ew_node:
            z_rs_ti = ew_outputs[0] if ew_outputs else None
            if z_rs_ti is not None:
                z_rs_t = tensors[z_rs_ti]
                self._rs_bufs[z_rs_t.name] = (z_rs_t._buf, None, z_rs_t)

        self._npu_stride = None
        for blk in blocks:
            if blk.kind == "EW_BINARY":
                for w in blk.words:
                    target = (w >> 48) & 0xFFFF
                    reg = w & 0xFFFF
                    val = (w >> 16) & 0xFFFFFFFF
                    if target == DPU_TARGET and reg == 0x4024:
                        self._npu_stride = val
                        break
                break

        print(f"Hybrid executor: {len(copy_nodes)} COPY + {ew_block_count} EW_BINARY blocks, "
              f"{len(cpu_nodes)} CPU ops")
        print(f"  inputs:  {[t.name for t in self._input_tensors]}")
        print(f"  outputs: {[t.name for t in self._output_tensors]}")
        print(f"  schedule: {[(s[0], s[1].op if s[0]=='cpu' else f'blk {s[1]}-{s[1]+s[2]-1}') for s in schedule]}")

    def _inputs_set_graph(self, input_arrays):
        arrays = [np.asarray(a) for a in input_arrays]
        user_dtype = arrays[0].dtype if arrays else np.float16
        if user_dtype in (np.int16, np.float32):
            self._alu_fallback = True
            self._alu_result = None
            self._alu_input_arrays = [a.ravel() for a in arrays]
            self._alu_input_shapes = [a.shape for a in arrays]
            self._graph_user_dtypes = [a.dtype for a in arrays]
            self._graph_user_dtype = user_dtype
            return

        for i, arr in enumerate(input_arrays):
            arr = np.asarray(arr)
            t = self._input_tensors[i]
            buf = t._buf
            if t.native and len(t.native) >= 5:
                packed = _contiguous_to_nc1hwc2(arr, t.native)
            else:
                packed = arr.view(np.uint8)
            n_bytes = packed.nbytes if isinstance(packed, np.ndarray) else len(packed)
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = packed.tobytes() if isinstance(packed, np.ndarray) else bytes(packed)
            mc = self._tensor_bufs[t.idx][1]
            self.dev.mem_sync(mc.handle, mc.size)

        if hasattr(self, '_npu_input_rs') and self._npu_input_rs:
            for src_ti, rs_ti in self._npu_input_rs.items():
                src_idx = None
                for i, it in enumerate(self._input_tensors):
                    if it.idx == src_ti:
                        src_idx = i
                        break
                if src_idx is None or src_idx >= len(input_arrays):
                    continue
                arr = np.asarray(input_arrays[src_idx])
                rs_t = self.model["tensors"][rs_ti]
                packed = _contiguous_to_nc1hwc2(arr, rs_t.native, self._stride_for(rs_t))
                n_bytes = packed.nbytes if isinstance(packed, np.ndarray) else len(packed)
                rs_buf = rs_t._buf
                ct = (ctypes.c_uint8 * n_bytes).from_buffer(rs_buf)
                ct[:n_bytes] = packed.tobytes() if isinstance(packed, np.ndarray) else bytes(packed)

        self._graph_user_dtypes = [a.dtype for a in input_arrays] if input_arrays else [self._graph_dtype]
        self._graph_user_dtype = self._graph_user_dtypes[0]

    def _run_graph(self):
        if self._alu_fallback:
            ops = [n for n in self.model["nodes"]
                   if n.op not in ("InputOperator", "OutputOperator", "Reshape")]
            result = self._alu_input_arrays[0]
            for op_node in ops:
                kernel = CPU_KERNELS.get(op_node.op)
                if kernel is None:
                    continue
                if len(self._alu_input_arrays) >= 2 and op_node.op not in ("Not", "Neg"):
                    result = kernel([result, self._alu_input_arrays[1]])
                else:
                    result = kernel([result])
            self._alu_result = result
            return

        tensors = self.model["tensors"]
        for item in self._graph_schedule:
            if item[0] == "npu" or item[0] == "npu_copy":
                _, start, count = item
                for ti in range(start, start + count):
                    blk = self._blocks[ti]
                    self._tasks[0].regcmd_addr = (self._model_mc.dma_addr
                                                   + HEADER_SIZE + blk.word_offset * 8)
                    self._tasks[0].regcfg_amount = blk.n_words
                    self.dev.submit(self._task_mc.obj_addr, 0, 1)
            elif item[0] == "npu_ew":
                ew_start = self._hybrid_ew_start
                ew_ping = self._hybrid_ew_ping
                self._tasks[0].regcmd_addr = (self._model_mc.dma_addr
                                               + HEADER_SIZE + self._blocks[ew_start].word_offset * 8)
                self._tasks[0].regcfg_amount = self._blocks[ew_start].n_words
                self.dev.submit(self._task_mc.obj_addr, 0, ew_ping)
            elif item[0] == "cpu":
                _, node = item
                self._execute_graph_cpu_op(node, tensors)

    def _execute_graph_cpu_op(self, node, tensors):
        kernel = CPU_KERNELS.get(node.op)
        if kernel is None:
            return
        dtype = self._graph_user_dtype if hasattr(self, '_graph_user_dtype') else self._graph_dtype
        data_inputs = [ti for ti in node.inputs if ti in self._tensor_bufs]
        if data_inputs:
            rs_ti = data_inputs[0]
            rs_t = tensors[rs_ti]
            if rs_t.native and len(rs_t.native) >= 5:
                c2 = rs_t.native[4]
                if c2 == 8: dtype = np.float16
                elif c2 == 4: dtype = np.int32
                elif c2 == 1: dtype = np.int8
        dt = np.dtype(dtype)
        elem_size = dt.itemsize

        data_inputs = [ti for ti in node.inputs if ti in self._tensor_bufs]
        data_outputs = [ti for ti in node.outputs if ti in self._tensor_bufs]

        input_data = []
        for ti in data_inputs:
            t = tensors[ti]
            buf = t._buf
            n_bytes = t.size or (int(np.prod(t.logical)) * elem_size if t.logical else 4096)
            n_elems = n_bytes // elem_size
            raw = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            input_data.append(np.frombuffer(raw, dtype=dt, count=n_elems).copy())

        result = kernel(input_data)

        for oi, ti in enumerate(data_outputs):
            t = tensors[ti]
            buf = t._buf
            out_data = result if len(data_outputs) == 1 else result[oi]
            n_bytes = len(out_data) * elem_size
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = out_data.view(np.uint8).tobytes()
            mc = self._tensor_bufs[ti][1]
            self.dev.mem_sync(mc.handle, mc.size)

    def _outputs_get_graph(self):
        if self._alu_fallback and self._alu_result is not None:
            shape = self._alu_input_shapes[0] if self._alu_input_shapes else None
            return [self._alu_result.reshape(shape) if shape else self._alu_result]
        results = []
        for oi, t in enumerate(self._output_tensors):
            buf = t._buf
            dt = None
            rs_t_found = None
            for n in self.model["nodes"]:
                if n.op == "Reshape":
                    d_outs = [x for x in n.outputs if x in self._tensor_bufs]
                    if d_outs and d_outs[0] == t.idx:
                        rs_cand = self.model["tensors"][n.inputs[0]] if n.inputs else None
                        if rs_cand and rs_cand.native and len(rs_cand.native) >= 5:
                            rs_t_found = rs_cand
                        break
            if rs_t_found:
                c2 = rs_t_found.native[4]
                if c2 == 8: dt = np.float16
                elif c2 == 4: dt = np.int32
                elif c2 == 1: dt = np.int8
            if dt is None and t.native and len(t.native) >= 5:
                c2 = t.native[4]
                if c2 == 8: dt = np.float16
                elif c2 == 4: dt = np.int32
                elif c2 == 1: dt = np.int8
            if dt is None:
                if hasattr(self, '_graph_user_dtypes') and oi < len(self._graph_user_dtypes):
                    dt = np.dtype(self._graph_user_dtypes[oi])
                else:
                    dt = np.dtype(self._graph_user_dtype if hasattr(self, '_graph_user_dtype') else self._graph_dtype)
            elem_size = dt.itemsize
            rs_t = None
            if hasattr(self, '_rs_bufs'):
                for rs_name, (rs_buf, _, rs_tensor) in self._rs_bufs.items():
                    for n in self.model["nodes"]:
                        if n.op == "Reshape":
                            data_outs = [x for x in n.outputs if x in self._tensor_bufs]
                            if data_outs and data_outs[0] == t.idx:
                                rs_t = self.model["tensors"][n.inputs[0]] if n.inputs else None
                                break
                    if rs_t:
                        break
            if rs_t and rs_t.native and len(rs_t.native) >= 5:
                sz = rs_t.size or 4096
                raw = (ctypes.c_uint8 * sz).from_buffer(rs_t._buf)
                packed = np.frombuffer(raw, dtype=np.uint8, count=sz).copy()
                data = _nc1hwc2_to_contiguous(packed, rs_t.native, dt, self._stride_for(rs_t))
                n_out = int(np.prod(t.logical)) if t.logical else len(data)
                results.append(data[:n_out].reshape(t.logical) if t.logical else data[:n_out])
            elif t.native and len(t.native) >= 5:
                sz = t.size or 4096
                raw = (ctypes.c_uint8 * sz).from_buffer(buf)
                packed = np.frombuffer(raw, dtype=np.uint8, count=sz).copy()
                data = _nc1hwc2_to_contiguous(packed, t.native, dt)
                results.append(data.reshape(t.logical) if t.logical else data)
            else:
                n_elems = int(np.prod(t.logical)) if t.logical else 1
                n_bytes = n_elems * elem_size
                raw = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
                data = np.frombuffer(raw, dtype=dt, count=n_elems).copy()
                results.append(data.reshape(t.logical) if t.logical else data)
        return results

    def inputs_set(self, input_arrays):
        """Write input data into the model (contiguous → NC1HWC2 + DMA patching)."""
        if self._uop is not None:
            for ti, arr in zip(self._uop_inputs, input_arrays):
                self._uop_values[ti] = np.asarray(arr).ravel()
            return
        if self._graph:
            return self._inputs_set_graph(input_arrays)

        arrays = [np.asarray(arr) for arr in input_arrays]
        user_dtype = arrays[0].dtype if arrays else np.float16
        self._alu_fallback = user_dtype in (np.int16, np.float32)
        self._alu_result = None

        if self._alu_fallback:
            self._alu_input_arrays = [a.ravel() for a in arrays]
            self._alu_input_shapes = [a.shape for a in arrays]
            return

        # Write raw input data to input buffers
        for i, arr in enumerate(input_arrays):
            if i >= len(self._input_bufs):
                break
            buf, mc, tensor = self._input_bufs[i]
            arr = np.asarray(arr)
            n_bytes = arr.nbytes
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = arr.view(np.uint8).tobytes()

        # Reshape inputs into NC1HWC2 and write to feature buffer
        # For each input tensor, find its NC1HWC2 version (x_rs, y_rs)
        for i, arr in enumerate(input_arrays):
            arr = np.asarray(arr)
            name = self._input_tensors[i].name if i < len(self._input_tensors) else f"input{i}"
            rs_names = [f"{name}_rs", f"{name}-rs"]
            alt_map = {"x": "a", "a": "x", "y": "b", "b": "y"}
            if name.lower() in alt_map:
                rs_names.append(f"{alt_map[name.lower()]}_rs")
                rs_names.append(f"{alt_map[name.lower()]}-rs")
            rs_tensor = None
            for rn in rs_names:
                rs_tensor = self._get_feature_tensor(rn)
                if rs_tensor:
                    break

            if rs_tensor is None or rs_tensor.native is None:
                # No RS tensor: write raw input data directly to feature buffer
                inp_t = self._input_tensors[i]
                if inp_t.offset is not None:
                    raw = arr.view(np.uint8).tobytes()
                    n_bytes = min(len(raw), inp_t.size or len(raw))
                    ct = (ctypes.c_uint8 * n_bytes).from_buffer(self._feature_buf, inp_t.offset)
                    ct[:n_bytes] = raw[:n_bytes]
                continue

            packed = _contiguous_to_nc1hwc2(arr, rs_tensor.native, self._stride_for(rs_tensor))
            buf = getattr(rs_tensor, '_buf', None) or self._feature_buf
            off = rs_tensor.offset
            n_bytes = min(len(packed), rs_tensor.size)
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf, off)
            ct[:n_bytes] = packed[:n_bytes].tobytes() if isinstance(packed, np.ndarray) else bytes(packed[:n_bytes])

        # Patch DMA addresses in regcmd blocks
        # Patch DST_BASE_ADDR (0x4020) → z-rs offset in feature buffer
        # Patch RDMA_SRC (0x5018) → x_rs offset in feature buffer
        # Patch RDMA_EW (0x5038) → y_rs offset in feature buffer
        x_rs = self._get_feature_tensor("x_rs")
        y_rs = self._get_feature_tensor("y_rs")
        z_rs = self._get_feature_tensor("z-rs")

        x_dma = getattr(x_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + x_rs.offset if x_rs else None)
        y_dma = getattr(y_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + y_rs.offset if y_rs else None)
        z_dma = getattr(z_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + z_rs.offset if z_rs else None)

        # Patch all blocks in the model buffer (modifying the mmap'd copy)
        # Block word offsets are relative to the body, model buffer has 64B header
        hdr_qwords = HEADER_SIZE // 8
        model_view = (ctypes.c_uint64 * (len(self.raw) // 8)).from_buffer(self._model_buf)
        for blk in self._blocks:
            for wi in range(blk.n_words):
                abs_idx = hdr_qwords + blk.word_offset + wi
                w = model_view[abs_idx]
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                old_val = (w >> 16) & 0xFFFFFFFF

                if target == RDMA_TARGET and reg == REG_RDMA_SRC and x_dma is not None:
                    model_view[abs_idx] = (RDMA_TARGET << 48) | (((x_dma + old_val) & 0xFFFFFFFF) << 16) | reg

                elif target == RDMA_TARGET and reg == REG_RDMA_EW and y_dma is not None:
                    model_view[abs_idx] = (RDMA_TARGET << 48) | (((y_dma + old_val) & 0xFFFFFFFF) << 16) | reg

                elif target == DPU_TARGET and reg == REG_DST_BASE and z_dma is not None:
                    model_view[abs_idx] = (DPU_TARGET << 48) | (((z_dma + old_val) & 0xFFFFFFFF) << 16) | reg

        for name, (buf, mc, t) in self._rs_bufs.items():
            self.dev.mem_sync(mc.handle, mc.size)
        self.dev.mem_sync(self._model_mc.handle, self._model_mc.size)

    def run(self):
        """Execute the model: NPU submit or CPU fallback."""
        if self._uop is not None:
            # uop graph: per-node dispatch. fp16 EW-supported ops go to the NPU
            # via the companion EW model's baked register blocks; the rest run
            # on CPU kernels.
            attrs_map = self._uop.get("node_attrs", {})
            self._uop_exec_log = []
            for ni, n in enumerate(self.model["nodes"]):
                if n.op in ("InputOperator", "OutputOperator"):
                    continue
                ins = [self._uop_values[ti] for ti in n.inputs]
                if (self._npu_ew is not None and n.op in UOP_NPU_SUPPORTED
                        and len(ins) == 2 and all(getattr(a, 'dtype', None) == np.float16 for a in ins)):
                    self._npu_ew.inputs_set([ins[0], ins[1]])
                    self._npu_ew.run()
                    out = self._npu_ew.outputs_get()[0][:ins[0].size]
                    self._uop_exec_log.append((n.op, "NPU"))
                else:
                    kernel = CPU_KERNELS.get(n.op)
                    if kernel is None:
                        raise NotImplementedError(f"uop graph op {n.op} (node {ni}) has no CPU kernel")
                    out = kernel(ins, attrs_map.get(str(ni), {}))
                    self._uop_exec_log.append((n.op, "CPU"))
                self._uop_values[n.outputs[0]] = out
            return
        if self._graph:
            return self._run_graph()
        if self._alu_fallback:
            ops = [n for n in self.model["nodes"]
                   if n.op not in ("InputOperator", "OutputOperator", "Reshape")]
            result = self._alu_input_arrays[0]
            for op_node in ops:
                if op_node.op in CPU_KERNELS and len(self._alu_input_arrays) >= 2:
                    result = CPU_KERNELS[op_node.op]([result, self._alu_input_arrays[1]])
                elif op_node.op in CPU_KERNELS and len(self._alu_input_arrays) == 1:
                    result = CPU_KERNELS[op_node.op]([result])
            self._alu_result = result
            return
        if self._all_cpu:
            for cpu_node in self._cpu_ops:
                self._execute_cpu_op(cpu_node)
        else:
            # NPU path: submit EW_BINARY blocks
            n_submit = self._npu_task_count or self._n_tasks
            self.dev.submit(self._task_mc.obj_addr, 0, n_submit)

    def _execute_cpu_op(self, node):
        op = node.op
        if op not in CPU_KERNELS:
            print(f"  CPU op '{op}' not implemented, skipping")
            return

        x_rs = self._get_feature_tensor("x_rs")
        y_rs = self._get_feature_tensor("y_rs")
        z_rs = self._get_feature_tensor("z-rs")

        if x_rs is not None and z_rs is not None:
            self._execute_cpu_op_rs(op, x_rs, y_rs, z_rs)
        else:
            self._execute_cpu_op_flat(op)

    def _execute_cpu_op_flat(self, op):
        x_t = self._input_tensors[0]
        y_t = self._input_tensors[1] if len(self._input_tensors) > 1 else None
        z_t = self._output_tensors[0]

        n_elems = np.prod(x_t.logical) if x_t.logical else x_t.n_elems
        dtype = self._detect_dtype(x_t, n_elems)

        x_data = self._read_flat(x_t, dtype, n_elems)

        if y_t:
            y_data = self._read_flat(y_t, dtype, n_elems)
            result = CPU_KERNELS[op]([x_data, y_data])
        else:
            result = CPU_KERNELS[op]([x_data])

        self._write_flat(z_t, result, dtype)

    def _is_nc1hwc2(self, tensor):
        if not tensor.native or len(tensor.native) != 5:
            return False
        N, C1, H, W, C2 = tensor.native
        nc1hwc2_total = N * C1 * H * W * C2
        logical_total = int(np.prod(tensor.logical)) if tensor.logical else 0
        return nc1hwc2_total > logical_total and C2 in (1, 2, 4, 8)

    def _detect_dtype(self, tensor, n_elems):
        if self._is_nc1hwc2(tensor):
            C2 = tensor.native[4]
            if C2 == 8: return np.float16
            elif C2 == 4: return np.int32
            elif C2 == 1: return np.int8
        elem_bytes = (tensor.size or 0) // max(n_elems, 1)
        if elem_bytes == 2: return np.float16
        elif elem_bytes == 1: return np.int8
        return np.int32

    def _read_flat(self, tensor, dtype, n_elems):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset or 0
        sz = tensor.size or (n_elems * dtype().itemsize)
        raw = (ctypes.c_uint8 * sz).from_buffer(buf, off)
        return np.frombuffer(raw, dtype=dtype, count=n_elems).copy()

    def _write_flat(self, tensor, data, dtype):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset or 0
        n_bytes = min(len(data) * dtype().itemsize, tensor.size or len(data) * dtype().itemsize)
        ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf, off)
        ct[:n_bytes] = data[:n_bytes // dtype().itemsize].view(np.uint8).tobytes()

    def _execute_cpu_op_rs(self, op, x_rs, y_rs, z_rs):
        c2 = x_rs.native[4] if len(x_rs.native) >= 5 else 4
        if c2 == 8:
            dtype = np.float16
        elif c2 == 4:
            dtype = np.int32
        elif c2 == 2:
            dtype = np.float16
        elif c2 == 1:
            dtype = np.int8
        else:
            dtype = np.int32

        x_data = self._read_nc1hwc2(x_rs, dtype)

        if y_rs is not None:
            y_data = self._read_nc1hwc2(y_rs, dtype)
            result = CPU_KERNELS[op]([x_data, y_data])
        else:
            result = CPU_KERNELS[op]([x_data])

        self._write_nc1hwc2(z_rs, result)

    def _stride_for(self, tensor):
        if self._npu_stride and tensor.native and len(tensor.native) == 5:
            W = tensor.native[3]
            stride_atoms = self._npu_stride // 16
            if stride_atoms >= W:
                return stride_atoms
        return None

    def _read_nc1hwc2(self, tensor, dtype):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset
        sz = tensor.size
        raw = (ctypes.c_uint8 * sz).from_buffer(buf, off)
        packed = np.frombuffer(raw, dtype=np.uint8, count=sz).copy()
        return _nc1hwc2_to_contiguous(packed, tensor.native, dtype, self._stride_for(tensor))

    def _write_nc1hwc2(self, tensor, data):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        packed = _contiguous_to_nc1hwc2(data, tensor.native, self._stride_for(tensor))
        off = tensor.offset
        n = min(len(packed), tensor.size)
        ct = (ctypes.c_uint8 * n).from_buffer(buf, off)
        ct[:n] = packed[:n].tobytes() if isinstance(packed, np.ndarray) else bytes(packed[:n])

    def _get_output_rs_offset(self):
        """Determine the z-rs/c-rs offset in the feature buffer.
        The vendor runtime places the output NC1HWC2 tensor at offset 0.
        The FlatBuffer may have it as None or we compute from the memory plan."""
        # Check all known output RS tensor name patterns
        for prefix in ("z-rs", "c-rs", "out-rs", "output_rs"):
            for t in self._feature_tensors:
                if t.name and t.name.startswith(prefix) and t.offset is not None:
                    return t.offset, t.size, t.native
        # The vendor always puts output RS at offset 0 in the feature buffer
        # Find the tensor by size: it's the first NC1HWC2 tensor with the output size
        z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
        if z_rs:
            # Find from the output tensor's size
            for t in self._output_tensors:
                expected_size = z_rs.size
                # Offset 0 is the default for the output RS tensor
            return 0, z_rs.size, z_rs.native
        return None, None, None

    def outputs_get(self):
        if self._uop is not None:
            return [self._uop_values[ti] for ti in self._uop_outputs]
        if self._graph:
            return self._outputs_get_graph()
        if self._alu_fallback and self._alu_result is not None:
            shape = self._alu_input_shapes[0] if self._alu_input_shapes else None
            return [self._alu_result.reshape(shape) if shape else self._alu_result]
        results = []
        for i, (buf, mc, tensor) in enumerate(self._output_bufs):
            z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
            if z_rs and z_rs.native and len(z_rs.native) >= 5:
                c2 = z_rs.native[4]
                if c2 == 8:
                    dtype = np.float16
                elif c2 == 4:
                    dtype = np.int32
                elif c2 == 2:
                    dtype = np.float16
                elif c2 == 1:
                    dtype = np.int8
                else:
                    dtype = np.int32
                data = self._read_nc1hwc2(z_rs, dtype)
                n_out = np.prod(tensor.logical) if tensor.logical else len(data)
                results.append(data[:n_out])
            else:
                out_t = self._output_tensors[0] if self._output_tensors else tensor
                n_out = np.prod(out_t.logical) if out_t.logical else (out_t.size // 4)
                inp_t = self._input_tensors[0] if self._input_tensors else out_t
                n_inp = np.prod(inp_t.logical) if inp_t.logical else n_out
                dtype = self._detect_dtype(inp_t, n_inp)
                data = self._read_flat(out_t, dtype, n_out)
                results.append(data)
        return results

    def run_scalar(self, x, scalar, op="Add"):
        """Run a unary op with a scalar constant: out = op(x, scalar).

        Works for any dtype (fp16 on ALU, int16/float32 on ALU).
        For fp16 binary ops other than Add, uses ALU because the NPU RC
        blocks are hardcoded to one EW opcode.
        """
        x = np.asarray(x)
        kernel = CPU_KERNELS.get(op)
        if kernel is None:
            raise ValueError(f"unknown op {op!r}")

        result = x.ravel()
        if op in ("Not", "Neg"):
            result = kernel([result])
        else:
            c = np.full(result.shape, scalar, dtype=x.dtype)
            if x.dtype.kind in ('i', 'u'):
                c = np.broadcast_to(np.array(scalar, dtype=x.dtype), result.shape).copy()
            result = kernel([result, c])
        shape = x.shape
        return result.reshape(shape) if shape else result

    def destroy(self):
        self.dev.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.destroy()


# ── RK backend ───────────────────────────────────────────────────────────────

class _Blob(bytes):
    def __new__(cls, data:bytes, meta:dict|None=None):
        ret = bytes.__new__(cls, data)
        ret.meta = meta
        return ret

class _Compiler(Compiler):
    def compile(self, src:bytes) -> bytes: return src if isinstance(src, _Blob) else bytes(src)


# ── RC Template Generator ────────────────────────────────────────────────
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

# ── RKNN FlatBuffer Builder ───────────────────────────────────────────────
"""Build RKNN FlatBuffer bodies for fp16 element-wise Add/Sub/Mul/Div.

build_body() is the toolkit-free production path for 2..64 inputs.  It builds the
FlatBuffer metadata, register-command stream, and task descriptor data from
decoded formulas/templates.  build_body_scratch() keeps the old reference-body
splice path for reverse-engineering comparisons only.
"""
import math
import struct
from pathlib import Path

import flatbuffers



SURF_W = 8176
MAX_CH = (1 << 13) - 1
MAX_C1_PER_TILE = (MAX_CH + 1) // 8
BLOCKS_PER_ADD = 6
MAX_TILES = BLOCKS_PER_ADD
HEADER_SIZE = 0x40
MAX_INPUTS = 64
SUPPORTED_INPUTS = range(2, MAX_INPUTS + 1)

# dtype -> {internal FlatBuffer TensorType enum stored in tensor field f0 (NOT the
# public rknn_tensor_type enum), bytes per element of the EXTERNAL I/O buffer,
# root dtype string used in f12/f13 dtype_in/out + attrs}.
#
# Coverage on the NPU element-wise (DPU) path was established empirically against
# the vendor librknnrt.so on rk3588 (RKNN_CREATION.md 6g):
#
#   dtype     f0   bytes  status (Add/Sub/Div verified, mismatches=0)
#   float16   10     2    native DPU fp16 compute            -- WORKS
#   float32   10     4    runtime converts fp32<->fp16        -- WORKS (fp16 precision)
#   int8       3     1    DPU int8 element-wise               -- WORKS (exact integer)
#   bool       3     1    same enum as int8; Add~OR, Mul!=AND -- WORKS (int8 arithmetic)
#
# Unsupported on THIS path/runtime (kept out of DTYPES so they raise clearly):
#   uint8   -> input normalizes but OUTPUT uint8 dtype is unsupported by the EW path
#   int16/int32/int64 -> recognized input dtypes, but their native NC1HWC2 pack uses
#       C2=4 (not 8) so the fp16 RC template's working-buffer geometry mis-computes;
#       a per-width RC template + memory plan would be needed (compiler-tiler work)
#   uint16/uint32/int4/bfloat16 -> not recognized by this librknnrt's dtype parser
#       (report UNKNOW(12) regardless of spelling)
DTYPES = {
    "float16": {"fb_type": 10, "elem_bytes": 2, "str": "float16"},
    "float32": {"fb_type": 10, "elem_bytes": 4, "str": "float32"},
    "int8": {"fb_type": 3, "elem_bytes": 1, "str": "int8"},
    "bool": {"fb_type": 3, "elem_bytes": 1, "str": "bool"},
}

# Dtypes the runtime knows but which do NOT work on the toolkit-free EW path; we
# name them so the error message can be specific rather than a generic KeyError.
_DTYPE_KNOWN_UNSUPPORTED = {
    "uint8": "output uint8 dtype is unsupported by the element-wise path",
    "int16": "needs C2=4 native pack + per-width RC template (compiler-tiler work)",
    "int32": "needs C2=4 native pack + per-width RC template (compiler-tiler work)",
    "int64": "needs C2=4 native pack + per-width RC template (compiler-tiler work)",
    "uint16": "not recognized by this librknnrt dtype parser",
    "uint32": "not recognized by this librknnrt dtype parser",
    "int4": "not recognized by this librknnrt dtype parser",
    "bfloat16": "not recognized by this librknnrt dtype parser",
}


def _resolve_dtype(dtype):
    if dtype in DTYPES:
        return DTYPES[dtype]
    if dtype in _DTYPE_KNOWN_UNSUPPORTED:
        raise NotImplementedError(
            f"dtype {dtype!r} is not supported on the toolkit-free NPU "
            f"element-wise path: {_DTYPE_KNOWN_UNSUPPORTED[dtype]}. "
            f"Supported: {sorted(DTYPES)}.")
    raise ValueError(f"unknown dtype {dtype!r}; supported: {sorted(DTYPES)}")


_DPU = 0x1001
_RDMA = 0x2001
_TARGETS = {0x0101, 0x0201, 0x0801, _DPU, _RDMA, 0x4001,
            0x8001, 0x10001, 0x20001, 0x40001}

_RC_TEMPLATES = all_templates(max_n=MAX_INPUTS)

_EMBEDDED = {}

MEMORY_PLANS = {
    2: {
        "x": (0, False), "y": (1, False), "z": (2, False),
        "z-rs": (0, True), "x_rs": (1, True), "y_rs": (2, True),
    },
    3: {
        "a": (0, False), "b": (1, False), "c": (2, False), "d": (3, False),
        "d-rs": (0, True), "a_rs": (1, True), "b_rs": (2, True),
        "c_rs": (3, True), "t-rs": (None, True),
    },
    4: {
        "a": (0, False), "b": (1, False), "c": (2, False), "d": (3, False),
        "e": (4, False),
        "e-rs": (0, True), "a_rs": (1, True), "b_rs": (2, True),
        "c_rs": (3, True), "d_rs": (4, True),
        "t1-rs": (None, True), "t2-rs": (5, True),
    },
}


def _memory_plan_for(n_inputs):
    if n_inputs in MEMORY_PLANS:
        return MEMORY_PLANS[n_inputs]
    ins, outp = _io(n_inputs)
    plan = {nm: (i, False) for i, nm in enumerate(ins)}
    plan[outp] = (n_inputs, False)
    plan[f"{outp}-rs"] = (0, True)
    for i, nm in enumerate(ins):
        plan[f"{nm}_rs"] = (i + 1, True)
    n_inter = n_inputs - 2
    for k in range(1, n_inter + 1):
        nm = "t-rs" if n_inter == 1 else f"t{k}-rs"
        plan[nm] = (None if k == 1 else n_inputs + k - 1, True)
    return plan


def _get_template_body(n_inputs):
    # Reference 10x10 bodies are stored as raw binary .body files (no base64),
    # the same convention as _template_8192x8192.body.
    if n_inputs not in _EMBEDDED:
        path = Path(__file__).resolve().parent / f"_body_add{n_inputs}_10x10.body"
        if not path.exists():
            raise ValueError(f"no embedded body for {n_inputs} inputs ({path.name})")
        _EMBEDDED[n_inputs] = path.read_bytes()
    return bytearray(_EMBEDDED[n_inputs])


_LARGE_TEMPLATE = None

def _get_large_template_body():
    global _LARGE_TEMPLATE
    if _LARGE_TEMPLATE is None:
        path = Path(__file__).resolve().parent / "_template_8192x8192.body"
        _LARGE_TEMPLATE = bytearray(path.read_bytes())
    return bytearray(_LARGE_TEMPLATE)


_LARGE_MEMORY_PLAN_2D = {
    "x": (0, False), "y": (1, False), "z": (2, False),
    "z-rs": (0, True), "x_rs": (1, True), "y_rs": (2, True),
}


def _plan_memory_2d(body, rows, cols, n_inputs):
    C1 = math.ceil(rows / 8)
    W = cols
    Npad = C1 * W
    al = lambda s, a=256: ((s + a - 1) // a) * a
    ext = al(rows * cols * 2)
    work = al(Npad * 16)
    plan = {}
    for nm, (slot, is_work) in _memory_plan_for(n_inputs).items():
        if slot is None:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = None
        elif is_work:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = slot * work
        else:
            f3, f4, f12 = [rows, cols], [rows, cols], rows * cols * 2
            f13 = slot * ext
        plan[nm] = (f3, f4, f12, f13)
    for tp in _fb_tensor_positions(body):
        nm = _fb_string(body, tp, 5)
        if nm in plan:
            f3, f4, f12, f13 = plan[nm]
            _fb_set_vec(body, tp, 3, f3)
            _fb_set_vec(body, tp, 4, f4)
            _fb_set_scalar(body, tp, 12, f12)
            if f13 is not None:
                _fb_set_scalar(body, tp, 13, f13)


def _patch_large_add_tiles(body, rows, cols, n_inputs):
    DPU_M = 0x1001
    RDMA_M = 0x2001
    MODE_ADD = 0x48000002
    spans, w = _regcmd_spans(body)
    C1 = math.ceil(rows / 8)
    W = cols
    st = W * 16
    ch_full = C1 * 8 - 1

    for idx, (start, count) in enumerate(spans):
        mode = None
        for k in range(count):
            x = w[start + k]
            tgt = (x >> 48) & 0xFFFF
            reg = x & 0xFFFF
            if tgt == DPU_M and reg == 0x4010:
                mode = (x >> 16) & 0xFFFFFFFF
                break
        if mode != MODE_ADD:
            continue
        for reg in (0x4030, 0x405c):
            _rc_set(w, start, count, DPU_M, reg, W - 1)
        _rc_set(w, start, count, RDMA_M, 0x500c, W - 1)
        _rc_set(w, start, count, DPU_M, 0x403c, (ch_full << 16) | ch_full)
        _rc_set(w, start, count, DPU_M, 0x4058, ch_full)
        _rc_set(w, start, count, RDMA_M, 0x5014, ch_full)
        _rc_set(w, start, count, DPU_M, 0x4034, 0)
        _rc_set(w, start, count, RDMA_M, 0x5010, 0)
        _rc_set(w, start, count, DPU_M, 0x4020, 0)
        _rc_set(w, start, count, RDMA_M, 0x5018, 0)
        _rc_set(w, start, count, RDMA_M, 0x5038, 0)
        _rc_set(w, start, count, DPU_M, 0x4024, st)
        _rc_set(w, start, count, DPU_M, 0x40c0, st)
        _rc_set(w, start, count, RDMA_M, 0x5040, st)
        _rc_set(w, start, count, RDMA_M, 0x504c, 0)
        _rc_set(w, start, count, RDMA_M, 0x506c, 0)

    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)


MAX_N_6TILE = MAX_TILES * MAX_C1_PER_TILE * SURF_W


def _fb_u16(data, off):
    return struct.unpack_from("<H", data, off)[0]

def _fb_u32(data, off):
    return struct.unpack_from("<I", data, off)[0]

def _fb_i32(data, off):
    return struct.unpack_from("<i", data, off)[0]

def _fb_field_offset(data, pos, field):
    vt = pos - _fb_i32(data, pos)
    vts = _fb_u16(data, vt)
    entry = vt + 4 + field * 2
    if entry + 2 > vt + vts:
        return 0
    return _fb_u16(data, entry)

def _fb_field_abs(data, pos, field):
    off = _fb_field_offset(data, pos, field)
    return pos + off if off else None

def _fb_string(data, pos, field):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return None
    tgt = ab + _fb_u32(data, ab)
    n = _fb_u32(data, tgt)
    return data[tgt + 4:tgt + 4 + n].decode("ascii", errors="replace")

def _fb_vec_target(data, pos, field):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return None
    return ab + _fb_u32(data, ab)

def _fb_vec_u32(data, off):
    n = _fb_u32(data, off)
    return [_fb_u32(data, off + 4 + k * 4) for k in range(n)]

def _fb_tensor_positions(data):
    root = _fb_u32(data, 0)
    sg_abs = _fb_field_abs(data, root, 2)
    if sg_abs is None:
        return []
    sg_vec = sg_abs + _fb_u32(data, sg_abs)
    n_sg = _fb_u32(data, sg_vec)
    if n_sg < 1:
        return []
    sg = (sg_vec + 4) + _fb_u32(data, sg_vec + 4)
    tvec_abs = _fb_field_abs(data, sg, 0)
    if tvec_abs is None:
        return []
    tvec = tvec_abs + _fb_u32(data, tvec_abs)
    n = _fb_u32(data, tvec)
    out = []
    for i in range(n):
        entry = tvec + 4 + i * 4
        out.append(entry + _fb_u32(data, entry))
    return out

def _fb_set_vec(data, pos, field, vals):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return False
    tgt = ab + _fb_u32(data, ab)
    n = _fb_u32(data, tgt)
    if n != len(vals):
        return False
    for k, v in enumerate(vals):
        struct.pack_into("<I", data, tgt + 4 + 4 * k, v)
    return True

def _fb_set_scalar(data, pos, field, val):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return
    struct.pack_into("<I", data, ab, val)


def _regcmd_spans(body):
    n = len(body) // 8
    w = list(struct.unpack_from(f"<{n}Q", body, 0))
    out = []
    i = 0
    while i < n:
        if ((w[i] >> 48) & 0xFFFF) in _TARGETS:
            j = i
            while j < n and ((w[j] >> 48) & 0xFFFF) in _TARGETS:
                j += 1
            if j - i >= 20:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out, w


def _plan_memory(body, N, n_inputs):
    C1, W = surface_split(N)
    Npad = C1 * W
    al = lambda s, a=256: ((s + a - 1) // a) * a
    ext, work = al(N * 2), al(Npad * 16)
    plan = {}
    for nm, (slot, is_work) in _memory_plan_for(n_inputs).items():
        if slot is None:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = None
        elif is_work:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = slot * work
        else:
            f3, f4, f12 = [1, N], [1, N], N * 2
            f13 = slot * ext
        plan[nm] = (f3, f4, f12, f13)
    for tp in _fb_tensor_positions(body):
        nm = _fb_string(body, tp, 5)
        if nm in plan:
            f3, f4, f12, f13 = plan[nm]
            _fb_set_vec(body, tp, 3, f3)
            _fb_set_vec(body, tp, 4, f4)
            _fb_set_scalar(body, tp, 12, f12)
            if f13 is not None:
                _fb_set_scalar(body, tp, 13, f13)


def _patch_tiles(body, N, n_inputs):
    C1, W = surface_split(N)
    tiles = tile_split(C1)
    n_tiles = len(tiles)
    spans, w = _regcmd_spans(body)
    n_adds = n_inputs - 1
    spans_per_add = len(spans) // n_adds
    st = W * 16
    for g in range(n_adds):
        group = spans[g * spans_per_add:(g + 1) * spans_per_add]
        for tile_idx, (i, c) in enumerate(group):
            tidx = min(tile_idx, n_tiles - 1)
            C1_tile, surf_offset = tiles[tidx]
            ch = C1_tile * 8 - 1
            base = surf_offset * st
            _rc_patch_block(w, i, c, W, ch, base, st)
    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)


def build_body(N, n_inputs, rows=None, cols=None, ops=None, dtype="float16"):
    return _build_body_scratch_flatbuffers(N, n_inputs, ops, dtype)


def surface_split(N):
    C1 = max(1, math.ceil(N / SURF_W))
    return C1, math.ceil(N / C1)


def tile_split(C1):
    if C1 <= MAX_C1_PER_TILE:
        return [(C1, 0)]
    n_tiles = min(MAX_TILES, math.ceil(C1 / MAX_C1_PER_TILE))
    c1_per = math.ceil(C1 / n_tiles)
    tiles, off, rem = [], 0, C1
    for _ in range(n_tiles):
        c1 = min(c1_per, rem)
        tiles.append((c1, off))
        off += c1; rem -= c1
    return tiles


def _io(n):
    if n == 2:
        return ["x", "y"], "z"
    names = _io_names(n + 1)
    return names[:n], names[n]


def _io_names(count):
    # First 26 names are a..z (keeps byte-exact parity with toolkit reference
    # bodies for n<=25). Beyond that, spreadsheet-style names aa, ab, ... so the
    # input/output count is effectively unbounded. None collide with the
    # intermediate "t<k>-rs" tensors (those always start with 't' + digits).
    L = "abcdefghijklmnopqrstuvwxyz"
    out = []
    i = 0
    while len(out) < count:
        if i < 26:
            out.append(L[i])
        else:
            j = i - 26
            out.append(L[j // 26] + L[j % 26])
        i += 1
    return out


def _mem(n):
    ins, outp = _io(n)
    p = {}
    for i, nm in enumerate(ins):
        p[nm] = (i, False)
    p[outp] = (n, False)
    p[f"{outp}-rs"] = (0, True)
    for i, nm in enumerate(ins):
        p[f"{nm}_rs"] = (i + 1, True)
    ni = n - 2
    for k in range(1, ni + 1):
        nm = "t-rs" if ni == 1 else f"t{k}-rs"
        p[nm] = (None if k == 1 else n + k - 1, True)
    return p


def _ev(b):
    b.StartVector(4, 0, 4)
    return b.EndVector()


def _vec(b, vs):
    b.StartVector(4, len(vs), 4)
    for v in reversed(vs):
        b.PrependUint32(v)
    return b.EndVector()


def _ovec(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def _vec_u8(b, data):
    b.StartVector(1, len(data), 1)
    for v in reversed(data):
        b.PrependByte(v)
    return b.EndVector()


def _vec_pairs(b, pairs):
    b.StartVector(8, len(pairs), 4)
    for a, c in reversed(pairs):
        b.PrependUint32(c)
        b.PrependUint32(a)
    return b.EndVector()


def _str(b, s):
    return b.CreateString(s)


def _align(s, a=256):
    return ((s + a - 1) // a) * a


def _u32_table2(b, a, c):
    b.StartObject(2)
    b.PrependUint32Slot(1, c, 0)
    b.PrependUint32Slot(0, a, 0)
    return b.EndObject()


def _u32_table3(b, a, c, d):
    b.StartObject(3)
    b.PrependUint32Slot(2, d, 0)
    b.PrependUint32Slot(1, c, 0)
    b.PrependUint32Slot(0, a, 0)
    return b.EndObject()


def _vec_table3(b, a, c, d):
    va, vc, vd = _vec(b, a), _vec(b, c), _vec(b, d)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(2, vd, 0)
    b.PrependUOffsetTRelativeSlot(1, vc, 0)
    b.PrependUOffsetTRelativeSlot(0, va, 0)
    return b.EndObject()


def _str_vec_table3(b, name, c, d):
    vn, vc, vd = _str(b, name), _vec_pairs(b, c), _vec(b, d)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(2, vd, 0)
    b.PrependUOffsetTRelativeSlot(1, vc, 0)
    b.PrependUOffsetTRelativeSlot(0, vn, 0)
    return b.EndObject()


def _str_scalar_table2(b, name, val):
    vn = _str(b, name)
    b.StartObject(2)
    b.PrependUint32Slot(1, val, 0)
    b.PrependUOffsetTRelativeSlot(0, vn, 0)
    return b.EndObject()


def _u64_table2(b, a, c):
    b.StartObject(2)
    b.PrependUint64Slot(1, c, 0)
    b.PrependUint64Slot(0, a, 0)
    return b.EndObject()


def _root_attrs(n_inputs, N, dtype="float16"):
    ins, outp = _io(n_inputs)
    side = math.isqrt(N)
    shape = [side, side] if side * side == N else [1, N]
    # The attrs "dtype" is the model input dtype (float32 host buffer for the fp16
    # path); the quant_tab dtype is the on-device tensor dtype. For bool both are
    # bool. These strings are cosmetic to the runtime (it reads root f12/f13), but
    # we keep them consistent for tooling/round-trip fidelity.
    in_dtype = "bool" if dtype == "bool" else "float32"
    out_dtype = "bool" if dtype == "bool" else "float16"
    attrs = {}
    quant = {}
    for i, nm in enumerate(ins):
        attrs[nm] = {
            "idx": i, "shape": shape, "layout": "nchw", "layout_ori": "nchw",
            "is_output": False, "range": [0, 1], "origin_dynamic": False,
            "dtype": in_dtype, "mean": [0] * 10, "std": [1] * 10,
            "rgb2bgr": False,
        }
        quant[nm] = {
            "dtype": in_dtype, "qmethod": "", "qtype": "", "min": [],
            "max": [], "scale": [], "zero_point": [], "name": nm,
            "shape": shape,
        }
    attrs[outp] = {
        "is_output": True, "idx": 0, "shape": shape, "dtype": in_dtype,
        "layout": "nchw",
    }
    quant[outp] = {
        "dtype": out_dtype, "qmethod": "", "qtype": "", "min": [],
        "max": [], "scale": [], "zero_point": [], "name": outp,
        "shape": shape,
    }
    return str({"attrs": attrs, "quant_tab": quant, "dynamic_shapes": {}})


def _generate_sg_f7_specs(n_inputs):
    ins, outp = _io(n_inputs)
    n_adds = n_inputs - 1
    n_inter = n_inputs - 2
    step = 80 * n_adds
    _IO_OFF = [0, 0, 192, 0, 64, 112]
    _IO_MASK = [1, 0, 1, 0, 1, 1]
    _INT_MASK = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    specs = []
    for k, nm in enumerate(ins):
        base = 55 if k == 0 else (61 if k == 1 else 141 + (k - 2) * 80)
        specs.append((f"{nm}_rs",
                      [(_IO_OFF[i], base + i * step) for i in range(6)],
                      list(_IO_MASK)))
    out_base = 5 + (n_inputs - 2) * 80
    specs.append((f"{outp}-rs",
                  [(_IO_OFF[i], out_base + i * step) for i in range(6)],
                  list(_IO_MASK)))
    for k in range(1, n_inter + 1):
        t_name = "t-rs" if n_inter == 1 else f"t{k}-rs"
        g1 = 5 + (k - 1) * 80
        g2 = 135 + (k - 1) * 80
        pairs = []
        for i in range(6):
            pairs.append((_IO_OFF[i], g1 + i * step))
            pairs.append((_IO_OFF[i], g2 + i * step))
        specs.append((t_name, pairs, list(_INT_MASK)))
    return specs


def _generate_exsec_f13(n):
    _CYCLE = [384, 768, 1152, None]
    ins, outp = _io(n)
    result = {}
    for i, nm in enumerate(ins):
        if i <= 1:
            result[nm] = 320 * n
        else:
            result[nm] = _CYCLE[(i - 2) % 4]
    result[f"{outp}-rs"] = 384 if n % 4 == 2 else None
    return result



def _rc_set(w, b, c, t, reg, val):
    for k in range(c):
        r = w[b + k]
        if ((r >> 48) & 0xFFFF) == t and (r & 0xFFFF) == reg:
            w[b + k] = (r & ~(0xFFFFFFFF << 16)) | ((val & 0xFFFFFFFF) << 16)
            return


def _rc_patch_block(w, i, c, W, ch, base, st, op="Add"):
    for reg in (0x4030, 0x405c):
        _rc_set(w, i, c, _DPU, reg, W - 1)
    _rc_set(w, i, c, _RDMA, 0x500c, W - 1)
    ch_val = ((W - 1) << 16 | ch) if op != "Add" else (ch << 16 | ch)
    _rc_set(w, i, c, _DPU, 0x403c, ch_val)
    _rc_set(w, i, c, _DPU, 0x4058, ch)
    _rc_set(w, i, c, _RDMA, 0x5014, ch)
    _rc_set(w, i, c, _DPU, 0x4034, 0)
    _rc_set(w, i, c, _RDMA, 0x5010, 0)
    _rc_set(w, i, c, _DPU, 0x4020, base)
    _rc_set(w, i, c, _RDMA, 0x5018, base)
    _rc_set(w, i, c, _RDMA, 0x5038, base)
    _rc_set(w, i, c, _DPU, 0x4024, st)
    _rc_set(w, i, c, _DPU, 0x40c0, st)
    _rc_set(w, i, c, _RDMA, 0x5040, st)
    _rc_set(w, i, c, _RDMA, 0x504c, 0)
    _rc_set(w, i, c, _RDMA, 0x506c, 0)


def _tensor_indices(n_inputs, ins, outp, n_adds):
    idx = {}
    t = 0
    idx["empty"] = t; t += 1
    for nm in ins:
        idx[f"{nm}_rsi1"] = t; t += 1
    idx[f"{outp}_rsi1"] = t; t += 1
    idx[ins[0]] = t; t += 1
    for nm in ins[1:]:
        idx[nm] = t; t += 1
    for i, nm in enumerate(ins):
        idx[f"{nm}_exsec"] = t; t += 1
        idx[f"{nm}_rs"] = t; t += 1
        if 1 <= i <= n_inputs - 2:
            inter = "t-rs" if n_inputs == 3 else f"t{i}-rs"
            idx[inter] = t; t += 1
    idx[f"{outp}_rs"] = t; t += 1
    idx[f"{outp}_rs_exsec"] = t; t += 1
    idx[outp] = t; t += 1
    idx["regcmd"] = t; t += 1
    idx["task"] = t; t += 1
    return idx


def _build_tensors(b, n_inputs, ins, outp, n_adds, plan, idx,
                   s2, s4, s5, N, Npad, ext, work, C1, W):
    toffs = [None] * (idx["task"] + 1)
    sec2 = [10, 10]
    sec4 = [1, 10, 1, 10]

    toffs[idx["task"]] = _cmd_tensor(b, "task", f2=10, f18=n_adds + 3 + 1)
    toffs[idx["regcmd"]] = _cmd_tensor(b, "regcmd", f2=9, f18=n_adds + 3)

    toffs[idx[outp]] = _ext_tensor(b, outp, s2, N * 2,
                                    plan[outp][0] * ext, f2=2)
    out_exsec_f13 = _generate_exsec_f13(n_inputs)[f"{outp}-rs"]
    toffs[idx[f"{outp}_rs_exsec"]] = _exsec_tensor(
        b, f"{outp}-rs_exSecondary", sec4, 1, out_exsec_f13, f1=2)
    toffs[idx[f"{outp}_rs"]] = _rs_tensor(
        b, f"{outp}-rs", s5, s4, Npad * 16,
        plan[f"{outp}-rs"][0] * work, has_f13=True,
        emit_zero_f13=n_inputs > 2)

    for name, (slot, _) in plan.items():
        if name.startswith("t") and name.endswith("-rs") and name in idx:
            offset = 0 if slot is None else slot * work
            toffs[idx[name]] = _rs_tensor(
                b, name, s5, s4, Npad * 16, offset, has_f13=slot is not None)

    for nm in reversed(ins):
        rs_name = f"{nm}_rs"
        toffs[idx[f"{nm}_rs"]] = _rs_tensor(
            b, rs_name, s5, s4, Npad * 16,
            plan[rs_name][0] * work, has_f13=True)
        toffs[idx[f"{nm}_exsec"]] = _exsec_tensor(
            b, f"{nm}_exSecondary", sec2, 1, _generate_exsec_f13(n_inputs)[nm], f1=None)

    for nm in reversed(ins[1:]):
        toffs[idx[nm]] = _ext_tensor(b, nm, s2, N * 2, plan[nm][0] * ext)

    toffs[idx[ins[0]]] = _ext_tensor(b, ins[0], s2, N * 2, f13=0)

    for i in range(n_inputs, -1, -1):
        all_names = ins + [outp]
        nm = all_names[i]
        name_str = f"{nm}_rs_i1" if nm in ins else f"{outp}-rs_i1"
        f12 = 16 if i == n_inputs else 32
        toffs[idx[f"{all_names[i]}_rsi1"]] = _rsi1_tensor(b, name_str, i + 1, f12)

    toffs[0] = _empty_tensor(b)
    return toffs


def _build_nodes(b, n_inputs, ins, outp, n_adds, idx, ops=None):
    noffs = []

    for nm in ins:
        noffs.append(_input_node(b, nm, idx[nm]))

    for nm in ins[:2]:
        noffs.append(_reshape_node(
            b, f"{nm}_rs", nm,
            [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
            [idx[f"{nm}_rs"]]))

    for k in range(n_adds):
        if k == 0:
            rs_a = f"{ins[0]}_rs"
        else:
            rs_a = "t-rs" if n_inputs == 3 else f"t{k}-rs"
        rs_b = f"{ins[k + 1]}_rs"
        if k == n_adds - 1:
            rs_out = f"{outp}_rs"
        else:
            rs_out = "t-rs" if n_inputs == 3 else f"t{k + 1}-rs"
        op_name = (ops[k] if ops and k < len(ops) else None) or "Add"
        noffs.append(_add_node(b, k + 1, idx[rs_a], idx[rs_b], idx[rs_out], op_name))

        next_input = k + 2
        if next_input < n_inputs:
            nm = ins[next_input]
            noffs.append(_reshape_node(
                b, f"{nm}_rs", nm,
                [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
                [idx[f"{nm}_rs"]]))

    noffs.append(_reshape_node(b, f"{outp}-rs", outp,
                               [idx[f"{outp}_rs"], idx[f"{outp}_rsi1"],
                                idx[f"{outp}_rs_exsec"]],
                               [idx[outp]]))
    noffs.append(_output_node(b, outp, idx[outp]))
    return noffs


def _build_subgraph(b, toffs, noffs, n_adds, idx, ins, outp, n_external=None):
    tvec = _ovec(b, toffs)
    nvec = _ovec(b, noffs)
    n_inputs = len(ins)
    n_ext = n_external if n_external is not None else n_inputs
    sg_f4 = _ovec(b, [
        _vec_table3(b, [0] * 10, [0x3f800000] * 10, list(range(10)))
        for _ in ins
    ])
    sg_f7_tables = [
        _str_vec_table3(b, name, vals, mask)
        for name, vals, mask in _generate_sg_f7_specs(n_inputs)
    ]
    sg_f7 = _ovec(b, sg_f7_tables)
    sg_f10 = _vec(b, [0, 0, 0, n_adds, n_adds * 3])
    sg_f12 = _ovec(b, [_str_scalar_table2(b, outp, n_adds)])
    sg_f2 = _vec(b, [idx[ins[i]] for i in range(n_ext)])
    sg_f3 = _vec(b, [idx[outp]])
    evs = [_ev(b) for _ in range(7)]

    b.StartObject(17)
    b.PrependUOffsetTRelativeSlot(16, evs[0], 0)
    b.PrependUOffsetTRelativeSlot(15, evs[1], 0)
    b.PrependUOffsetTRelativeSlot(14, evs[2], 0)
    b.PrependUOffsetTRelativeSlot(13, evs[3], 0)
    b.PrependUOffsetTRelativeSlot(12, sg_f12, 0)
    b.PrependUOffsetTRelativeSlot(10, sg_f10, 0)
    b.PrependUOffsetTRelativeSlot(9, evs[4], 0)
    b.PrependUOffsetTRelativeSlot(8, evs[5], 0)
    b.PrependUOffsetTRelativeSlot(7, sg_f7, 0)
    b.PrependUOffsetTRelativeSlot(6, evs[6], 0)
    b.PrependUOffsetTRelativeSlot(4, sg_f4, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_f3, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_f2, 0)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()

    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    return b.EndVector()

def _emit_root_table(b, sgvec, f12_dtype, f13_dtype, f19_a, f19_c, attrs_str):
    """Emit the 22-field RKNN root table + Finish, returning the FlatBuffer bytes.

    The field layout is byte-identical between the NPU and CPU build paths; the
    only path-specific inputs are the f12/f13 dtype-JSON maps, the f19 size table
    operands, and the ev3 attrs string. Everything else (target/toolkit/platform/
    framework strings, f14..f18, and the Prepend order) is shared.
    """
    s_target = _str(b, "RKNPU v2")
    s_toolkit = _str(b, "2.3.2(compiler version: 2.3.2 (@2025-04-03T08:26:16))")
    s_platform = _str(b, "rk3588")
    s_framework = _str(b, "ONNX")
    import json
    root_f12 = _str(b, json.dumps(f12_dtype, separators=(", ", ": ")))
    root_f13 = _str(b, json.dumps(f13_dtype, separators=(", ", ": ")))
    root_f14 = _vec_u8(b, b"0")
    root_f15 = _ev(b)
    root_f16 = _str(b, "static_shape")
    root_f17 = _ev(b)
    root_f18 = _ev(b)
    root_f19 = _u64_table2(b, f19_a, f19_c)
    root_ev3 = _str(b, attrs_str)
    root_ev11 = _str(b, "")
    b.StartObject(22)
    b.PrependUint32Slot(21, 1, 0)
    b.PrependUint32Slot(20, 1, 0)
    b.PrependUOffsetTRelativeSlot(19, root_f19, 0)
    b.PrependUOffsetTRelativeSlot(18, root_f18, 0)
    b.PrependUOffsetTRelativeSlot(17, root_f17, 0)
    b.PrependUOffsetTRelativeSlot(16, root_f16, 0)
    b.PrependUOffsetTRelativeSlot(15, root_f15, 0)
    b.PrependUOffsetTRelativeSlot(14, root_f14, 0)
    b.PrependUOffsetTRelativeSlot(13, root_f13, 0)
    b.PrependUOffsetTRelativeSlot(12, root_f12, 0)
    b.PrependUOffsetTRelativeSlot(11, root_ev11, 0)
    b.PrependUint8Slot(10, 2, 0)
    b.PrependUOffsetTRelativeSlot(9, s_framework, 0)
    b.PrependUOffsetTRelativeSlot(8, s_platform, 0)
    b.PrependUOffsetTRelativeSlot(7, s_toolkit, 0)
    b.PrependUint32Slot(6, 20302, 0)
    b.PrependUOffsetTRelativeSlot(3, root_ev3, 0)
    b.PrependUOffsetTRelativeSlot(2, sgvec, 0)
    b.PrependUOffsetTRelativeSlot(1, s_target, 0)
    b.PrependUint32Slot(0, 6, 0)
    root = b.EndObject()
    b.Finish(root, b"RKNN")
    return bytearray(b.Output())


def _build_root(b, sgvec, n_adds, C1, W, tiles, N, n_inputs, ops=None, n_rc_inputs=None,
                dtype="float16"):
    rc_n = n_rc_inputs if n_rc_inputs is not None else n_inputs
    ins, outp = _io(rc_n)
    dstr = _resolve_dtype(dtype)["str"]
    dtype_in = {nm: {"dtype": dstr, "layout": "UNDEFINED"} for nm in ins}
    dtype_out = {outp: {"dtype": dstr, "layout": "NCHW"}}
    fb = _emit_root_table(
        b, sgvec, dtype_in, dtype_out,
        192 + (rc_n - 2) * 64, 1472 + (rc_n - 2) * 320,
        _root_attrs(rc_n, N, dtype))
    ew_ops = None
    if ops:
        ew_ops = [ew_op_id(o) for o in ops]
    rc_n = n_rc_inputs if n_rc_inputs is not None else n_inputs
    rc_raw = build_template(rc_n, ew_ops)
    taskdesc = _taskdesc(rc_n)

    full = fb + rc_raw + taskdesc
    rc_len = len(rc_raw)
    fb_len = len(fb)
    root_off = struct.unpack_from("<I", full, 0)[0]
    if root_off == 60 and full[8:12] == b"\x00\x00\x00\x00":
        if rc_n == 2:
            del full[8:12]
            struct.pack_into("<I", full, 0, 56)
            fb_len -= 4
        else:
            full[8:8] = b"\x00\x00\x00\x00"
            struct.pack_into("<I", full, 0, 64)
            fb_len += 4
    elif root_off == 56 and rc_n in (3, 4):
        full[8:8] = b"\x00" * 8
        struct.pack_into("<I", full, 0, 64)
        fb_len += 8

    required_mod = 4 if rc_n % 2 == 0 else 0
    pad = (required_mod - fb_len % 8) % 8
    if pad:
        full[fb_len:fb_len] = b"\x00" * pad
        fb_len += pad

    _patch_regcmd(full, fb_len, n_adds, C1, W, tiles, ops)

    _patch_root_command_offsets(full, fb_len, rc_raw, rc_len, taskdesc, rc_n)
    return bytes(full[:fb_len]), bytes(full[fb_len:])


def _patch_root_command_offsets(full, fb_len, rc_raw, rc_len, taskdesc, rc_n):
    """Point root f20 (regcmd target) / f21 (task target) at the RC/task sections.

    Shared by the NPU and CPU build paths: both append `rc_raw + taskdesc` after a
    FlatBuffer body of length fb_len and then rewrite the two root command-offset
    fields to the absolute byte positions of the RC target word and the task tail.
    """
    rc_target = _rc_target_offset(rc_raw, rc_n)
    root_rt = struct.unpack_from("<I", full, 0)[0]
    for field_idx, offset in [(20, rc_target),
                              (21, rc_len + len(taskdesc) - 8)]:
        ab = _fb_field_abs(full, root_rt, field_idx)
        struct.pack_into("<I", full, ab, fb_len + offset - ab)


def _rc_target_offset(rc, n_inputs):
    target_val = n_inputs + 4
    n_u32 = len(rc) // 4
    zero_run = 0
    for i in range(n_u32):
        v = struct.unpack_from("<I", rc, i * 4)[0]
        if v == 0:
            zero_run += 1
        else:
            if zero_run >= 8 and v == target_val:
                return i * 4
            zero_run = 0
    raise ValueError(f"RC target not found for n_inputs={n_inputs}")


def _patch_root_f20_f21(fb, f20, f21):
    rt = struct.unpack_from("<I", fb, 0)[0]
    vt = rt - struct.unpack_from("<i", fb, rt)[0]
    vts = struct.unpack_from("<H", fb, vt)[0]
    nf = (vts - 4) // 2
    if nf < 22:
        return
    for field_idx, val in [(20, f20), (21, f21)]:
        entry = vt + 4 + field_idx * 2
        off = struct.unpack_from("<H", fb, entry)[0]
        if off:
            struct.pack_into("<I", fb, rt + off, val)


# Task command-tensor data (the embedded "task" tensor). A sequence of 64-byte
# (8-word) reshape descriptors: word0 hi-32 = the reshape-info tensor's f12 byte
# size (0x10 output / 0x20 input), following words' lo-32 = its dims, zero-padded;
# bracketed by leading/trailing zero words. Shape-independent (never patched), so
# it is generated rather than stored as a literal.
_TD_F12_OUT, _TD_F12_IN = 0x10, 0x20    # f12 of output / input reshape-info tensors
_TD_DIM = 0x0a                          # template dim (cosmetic, from add_10x10)
_TD_REC_WORDS = 8                       # 64 bytes per descriptor


def _td_rec(f12, dims):
    return [f12 << 32] + list(dims) + [0] * (_TD_REC_WORDS - 1 - len(dims))


def _taskdesc(n_inputs):
    if n_inputs < 2:
        raise ValueError(f"taskdesc not available for {n_inputs} inputs")
    rec_out = _td_rec(_TD_F12_OUT, [_TD_DIM, _TD_DIM])
    rec_in = _td_rec(_TD_F12_IN, [1, _TD_DIM, 1, _TD_DIM])
    if n_inputs == 2:
        words = [0] + rec_out + rec_in + rec_in + [0]
    else:
        leading_zeros = n_inputs // 3 + 1
        # Taskdesc lengths of 288 + 64*k bytes (leading_zeros 11, 19, 27, ...
        # with the 3 reshape records + bracket zeros) are rejected by the
        # runtime's ModelBuffer verifier; every neighbouring length loads. The
        # records are cosmetic/never-patched, so nudge the zero-pad off any bad
        # length. Affects n_inputs in {30,31,32}, {54,55,56}, ... ; n<=25 is
        # unaffected (leading_zeros < 11 there).
        if leading_zeros >= 11 and (leading_zeros - 11) % 8 == 0:
            leading_zeros += 1
        words = [0] * leading_zeros + rec_in * 3 + [0]
    return struct.pack(f"<{len(words)}Q", *words)


# CPU-fallback (bool [1,4]) taskdesc: same 8-word reshape-descriptor family as
# _taskdesc, but with the [1,4] bool dims. The output reshape carries dims [1,4]
# (f12=0x10), each input reshape carries [1,1,1,4] (f12=0x20). For n>=3 the
# taskdesc is a cyclic window into repeated input records; the window offset is
# keyed by the same compact lead schedule used by the CPU regcmd prefix.
_TD_CPU_DIM = 0x04


def _taskdesc_cpu(n_inputs, rc_word_off=0):
    if n_inputs < 1:
        raise ValueError(f"taskdesc not available for {n_inputs} inputs")
    rec_out = _td_rec(_TD_F12_OUT, [1, _TD_CPU_DIM])
    rec_in = _td_rec(_TD_F12_IN, [1, 1, 1, _TD_CPU_DIM])
    if n_inputs == 1:
        # Unary (Not): one output reshape record + one input reshape record.
        words = [0] * 6 + rec_out + rec_in + [0]
    elif n_inputs == 2:
        words = [0] * 6 + rec_out + rec_in + rec_in + [0]
    else:
        # The taskdesc is a cyclic window into repeated input records; the window
        # offset is set by the CPU regcmd alignment lead length (longer leads -
        # which carry the offset suffix - start the window earlier). rc_word_off
        # is the RC section's u32 file offset, the same input the lead derives from.
        lead_len = len(_cpu_lead_u32(n_inputs, rc_word_off))
        if lead_len >= 12:
            offset = (18 - lead_len) // 2
        else:
            offset = 9 - lead_len // 2
        total_words = 33 - offset
        stream = rec_in * ((offset + total_words + _TD_REC_WORDS - 1) // _TD_REC_WORDS)
        words = stream[offset:offset + total_words - 1] + [0]
    return struct.pack(f"<{len(words)}Q", *words)


def _patch_regcmd(full, rc_offset, n_adds, C1, W, tiles, ops=None):
    n = len(full) // 8
    spans, w = _regcmd_spans(bytes(full))
    n_tiles = len(tiles)
    spans_per_add = len(spans) // n_adds
    st = W * 16

    for g in range(n_adds):
        group = spans[g::n_adds]
        op_name = (ops[g] if ops and g < len(ops) else None) or "Add"
        op_id = ew_op_id(op_name)
        ew_cfg_val = _ew_cfg(op_id)
        out_res = _DPU_OUT_RES.get(op_id, 0x00010001)
        bn_mul = _RDMA_BN_MUL.get(op_id, 0x00017849)
        for tile_idx, (i, c) in enumerate(group):
            tidx = min(tile_idx, n_tiles - 1)
            C1_tile, surf_offset = tiles[tidx]
            ch = C1_tile * 8 - 1
            base = surf_offset * st
            _rc_patch_block(w, i, c, W, ch, base, st, op_name)
            _rc_set(w, i, c, _DPU, 0x4070, ew_cfg_val)
            _rc_set(w, i, c, _DPU, 0x4084, out_res)
            _rc_set(w, i, c, _RDMA, 0x5044, bn_mul)

    for idx, val in enumerate(w):
        struct.pack_into("<Q", full, idx * 8, val)


def _empty_tensor(b):
    nm = _str(b, "")
    evs = [_ev(b) for _ in range(11)]
    b.StartObject(18)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, evs[9], 0)
    b.PrependUOffsetTRelativeSlot(3, evs[10], 0)
    return b.EndObject()


def _rsi1_tensor(b, name, f18, f12):
    nm = _str(b, name)
    sh = _vec(b, [f12 // 8])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(19)
    b.PrependUint32Slot(18, f18, 0)
    b.PrependUint32Slot(12, f12, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, 5, 0)
    b.PrependUint8Slot(0, 7, 0)
    return b.EndObject()


def _ext_tensor(b, name, shape, size, f13=0, f2=1):
    nm = _str(b, name)
    sh = _vec(b, shape)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    b.PrependUint32Slot(13, f13, 0)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, f2, 0)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _exsec_tensor(b, name, shape, size, offset, f1=0):
    nm = _str(b, name)
    sh = _vec(b, shape)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    if offset is not None:
        b.PrependUint32Slot(13, offset, 0xFFFFFFFF)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, 3, 0)
    if f1 is not None:
        b.PrependUint8Slot(1, f1, 1)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _rs_tensor(b, name, s5, s4, size, offset, has_f13=True, emit_zero_f13=False):
    nm = _str(b, name)
    v5, v4 = _vec(b, s5), _vec(b, s4)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(20)
    b.PrependUint32Slot(19, 4, 0)
    if has_f13 and emit_zero_f13 and offset == 0:
        b.PrependUint32(0)
        b.Slot(13)
    elif has_f13 and offset != 0:
        b.PrependUint32Slot(13, offset, 0)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v5, 0)
    b.PrependUint8Slot(2, 3, 0)
    b.PrependUint8Slot(1, 64, 0)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _cmd_tensor(b, name, f2, f18):
    nm = _str(b, name)
    evs = [_ev(b) for _ in range(11)]
    b.StartObject(19)
    b.PrependUint32Slot(18, f18, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, evs[9], 0)
    b.PrependUOffsetTRelativeSlot(3, evs[10], 0)
    b.PrependUint8Slot(2, f2, 0)
    b.PrependUint8Slot(0, 13, 0)
    return b.EndObject()


def _input_node(b, name, tensor_idx):
    op_s = _str(b, "InputOperator")
    nm_s = _str(b, f"InputOperator:{name}")
    f5 = _vec(b, [tensor_idx])
    f9 = _ev(b)
    f4 = _ev(b)
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _reshape_node(b, rs_name, src_name, input_indices, output_indices):
    op_s = _str(b, "Reshape")
    nm_s = _str(b, f"Reshape:{rs_name}")
    f4 = _vec(b, input_indices)
    f5 = _vec(b, output_indices)
    f9 = _ev(b)
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _empty_attrs_table(b):
    """Empty CPU-op attributes table (node field f8). StartObject(0)/EndObject()."""
    b.StartObject(0)
    return b.EndObject()


def _op_node(b, op_num, in_a_idx, in_b_idx, out_idx, op="Add",
             npu_f10=None, npu_f12=None, in_idxs=None):
    """Binary (or unary) op node, modular over NPU element-wise and CPU ops.

    NPU ops (Add/Sub/Mul/Div) carry node field f3 = DPU op-type (2) and the
    NPU geometry vectors. CPU ops (And/Or/Not/...) instead carry field f7 = the
    runtime CPU op-type enum and field f8 = an empty attributes table, with the
    geometry vectors zeroed (the NPU only reshapes; the CPU kernel does compute).

    in_idxs overrides the input-index vector f4 (default [in_a_idx, in_b_idx]);
    pass a single-element list for unary CPU ops like Not.

    npu_f10/npu_f12 override the default NPU geometry vectors (used by the mixed
    model builder for non-standard tensor shapes like [1,4]).
    """
    cpu = is_cpu_op(op)
    op_s = _str(b, op)
    if cpu:
        nm_s = _str(b, f"{op}:{op}:{op.lower()}{op_num}")
    else:
        nm_s = _str(b, f"{op}:{op.lower()}{op_num}")
    f4 = _vec(b, in_idxs if in_idxs is not None else [in_a_idx, in_b_idx])
    f5 = _vec(b, [out_idx])
    if cpu:
        f8 = _empty_attrs_table(b)
        f9 = _ev(b)
        f10 = _vec(b, [0, 0, 0, 0, 0, 0])
        f11 = _vec(b, [0, 0, 0, 0, 0, 0])
        f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        f9 = _ev(b)
        f10 = _vec(b, npu_f10 if npu_f10 else [1, 1, 1, 1, 1, 1])
        f11 = _vec(b, [0, 0, 0, 0, 0, 0])
        f12 = _vec(b, npu_f12 if npu_f12 else [160, 0, 0, 80, 80, 0, 64, 48, 48])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    if cpu:
        b.PrependUOffsetTRelativeSlot(8, f8, 0)
        b.PrependUint8Slot(7, cpu_op_id(op), 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    if not cpu:
        b.PrependUint8Slot(3, 2, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


# Backwards-compatible alias.
def _add_node(b, add_num, in_a_idx, in_b_idx, out_idx, op="Add"):
    return _op_node(b, add_num, in_a_idx, in_b_idx, out_idx, op)


def _output_node(b, outp, tensor_idx):
    op_s = _str(b, "OutputOperator")
    nm_s = _str(b, f"OutputOperator:{outp}")
    full_name = f"OutputOperator:{outp}"
    name_bytes = full_name.encode("ascii")
    op_bytes = b"OutputOperator"
    f9_data = [0, 0, 0, 0, 0, 1, tensor_idx, len(name_bytes)]
    name_aligned = name_bytes[:len(name_bytes) - len(name_bytes) % 4] if len(name_bytes) % 4 else name_bytes
    f9_data += list(struct.unpack(f"<{len(name_aligned)//4}I", name_aligned))
    f9_data += [0, len(op_bytes)]
    op_space = max(0, 16 - len(f9_data))
    op_trunc = op_bytes[:op_space * 4]
    op_padded = op_trunc + b'\x00' * ((4 - len(op_trunc) % 4) % 4)
    if op_padded:
        f9_data += list(struct.unpack(f"<{len(op_padded)//4}I", op_padded))
    f9_data = (f9_data + [0] * 16)[:16]
    f4 = _vec(b, [tensor_idx])
    f5 = _ev(b)
    f9 = _vec(b, f9_data[:16])
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


# ── CPU-fallback op body builder (bool [1,4] shape) ──

_CPU_SLOT = 64


def _cpu_io(n):
    return _io_names(n), "out"


def _cpu_mem_offsets(n):
    """Workspace memory offsets for CPU ops, decoded from vendor references."""
    plan = {}
    ins, _ = _cpu_io(n)
    for i, nm in enumerate(ins):
        plan[nm] = (i * _CPU_SLOT, i > 0)
    plan["out"] = (2 * _CPU_SLOT, True)
    plan[f"{ins[0]}_exsec"] = (n * _CPU_SLOT, True)
    plan[f"{ins[0]}_rs"] = ((n + 1) * _CPU_SLOT, True)
    for i in range(1, n):
        nm = ins[i]
        plan[f"{nm}_exsec"] = (_CPU_SLOT, True) if i % 2 == 0 else (0, False)
        plan[f"{nm}_rs"] = (n * _CPU_SLOT if i <= 2 else (i - 1) * _CPU_SLOT, True)
    for k in range(1, n - 1):
        plan[f"t{k}-rs" if n > 3 else "t-rs"] = (0, False)
    if n % 2 == 0:
        plan["out-rs"] = (0, False)
        plan["out-rs_exsec"] = (_CPU_SLOT, True)
    else:
        plan["out-rs"] = (_CPU_SLOT, True)
        plan["out-rs_exsec"] = (0, False)
    return plan


def _cpu_ext_input(b, name, f13_val, has_f13):
    nm = _str(b, name)
    v3 = _vec(b, [1, 1, 1, 1, 4])
    v4 = _vec(b, [1, 4])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    if has_f13 and f13_val:
        b.PrependUint32Slot(13, f13_val, 0)
    b.PrependUint32Slot(12, 4, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v3, 0)
    b.PrependUint8Slot(2, 1, 0)
    b.PrependUint8Slot(0, 3, 0)
    return b.EndObject()


def _cpu_ext_output(b, name, f13=None):
    f13_val = 2 * _CPU_SLOT if f13 is None else f13
    nm = _str(b, name)
    v3 = _vec(b, [1, 4])
    v4 = _vec(b, [1, 4])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    b.PrependUint32Slot(13, f13_val, 0)
    b.PrependUint32Slot(12, _CPU_SLOT, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v3, 0)
    b.PrependUint8Slot(2, 2, 0)
    b.PrependUint8Slot(0, 3, 0)
    return b.EndObject()


def _cpu_exsec(b, name, is_output, f13_val, has_f13):
    nm = _str(b, name)
    if is_output:
        v3 = _vec(b, [1, 1, 1, 4, 16])
        v4 = _vec(b, [1, 1, 1, 4])
    else:
        v3 = _vec(b, [1, 1, 1, 1, 16])
        v4 = _vec(b, [1, 4])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(20 if is_output else 18)
    if is_output:
        b.PrependUint32Slot(19, 4, 0)
    if has_f13 and f13_val:
        b.PrependUint32Slot(13, f13_val, 0)
    b.PrependUint32Slot(12, _CPU_SLOT, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v3, 0)
    b.PrependUint8Slot(2, 3, 0)
    b.PrependUint8Slot(1, 64, 0)
    b.PrependUint8Slot(0, 3, 0)
    return b.EndObject()


def _cpu_rs(b, name, f13_val, has_f13):
    nm = _str(b, name)
    v3 = _vec(b, [1, 1, 1, 4])
    v4 = _vec(b, [1, 1, 1, 4])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(20)
    b.PrependUint32Slot(19, 4, 0)
    if has_f13 and f13_val:
        b.PrependUint32Slot(13, f13_val, 0)
    b.PrependUint32Slot(12, _CPU_SLOT, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v3, 0)
    b.PrependUint8Slot(2, 3, 0)
    b.PrependUint8Slot(0, 3, 0)
    return b.EndObject()


def _cpu_reshape_node(b, rs_name, src_name, input_indices, output_indices):
    op_s = _str(b, "Reshape")
    nm_s = _str(b, f"Reshape:{rs_name}")
    f4 = _vec(b, input_indices)
    f5 = _vec(b, output_indices)
    f9 = _ev(b)
    f10 = _vec(b, [1, 1, 0, 1, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [64, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUint8Slot(3, 2, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _cpu_output_node(b, outp, tensor_idx):
    op_s = _str(b, "OutputOperator")
    nm_s = _str(b, f"OutputOperator:{outp}")
    full_name = f"OutputOperator:{outp}"
    name_bytes = full_name.encode("ascii")
    op_bytes = b"OutputOperator"
    f9_data = [0, 0, 0, 0, 0, 1, tensor_idx, len(name_bytes)]
    name_padded = name_bytes + b'\x00' * ((4 - len(name_bytes) % 4) % 4)
    f9_data += list(struct.unpack(f"<{len(name_padded)//4}I", name_padded))
    f9_data.append(len(op_bytes))
    remaining = 16 - len(f9_data)
    if remaining > 0:
        op_padded = op_bytes + b'\x00' * ((4 - len(op_bytes) % 4) % 4)
        op_u32s = list(struct.unpack(f"<{len(op_padded)//4}I", op_padded))
        f9_data += op_u32s[:remaining]
    f9_data = (f9_data + [0] * 16)[:16]
    f4 = _vec(b, [tensor_idx])
    f5 = _ev(b)
    f9 = _vec(b, f9_data[:16])
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _build_cpu_nodes(b, n, ins, outp, n_adds, idx, ops=None):
    noffs = []
    for nm in ins:
        noffs.append(_input_node(b, nm, idx[nm]))
    for nm in ins[:2]:
        noffs.append(_cpu_reshape_node(
            b, f"{nm}_rs", nm,
            [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
            [idx[f"{nm}_rs"]]))
    for k in range(n_adds):
        if k == 0:
            rs_a = f"{ins[0]}_rs"
        else:
            rs_a = "t-rs" if n == 3 else f"t{k}-rs"
        rs_b = f"{ins[k + 1]}_rs"
        if k == n_adds - 1:
            rs_out = f"{outp}_rs"
        else:
            rs_out = "t-rs" if n == 3 else f"t{k + 1}-rs"
        op_name = (ops[k] if ops and k < len(ops) else None) or "And"
        noffs.append(_op_node(b, k + 1, idx[rs_a], idx[rs_b], idx[rs_out], op_name))
        next_input = k + 2
        if next_input < n:
            nm = ins[next_input]
            noffs.append(_cpu_reshape_node(
                b, f"{nm}_rs", nm,
                [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
                [idx[f"{nm}_rs"]]))
    noffs.append(_cpu_reshape_node(
        b, f"{outp}-rs", outp,
        [idx[f"{outp}_rs"], idx[f"{outp}_rsi1"], idx[f"{outp}_rs_exsec"]],
        [idx[outp]]))
    noffs.append(_cpu_output_node(b, outp, idx[outp]))
    return noffs


def _build_cpu_tensors(b, n, ins, outp, n_adds, idx, mem):
    toffs = [None] * (idx["task"] + 1)
    toffs[idx["task"]] = _cmd_tensor(b, "task", f2=10, f18=n + 3)
    toffs[idx["regcmd"]] = _cmd_tensor(b, "regcmd", f2=9, f18=n + 2)
    toffs[idx[outp]] = _cpu_ext_output(b, outp)
    omem = mem[f"{outp}-rs_exsec"]
    toffs[idx[f"{outp}_rs_exsec"]] = _cpu_exsec(
        b, f"{outp}-rs_exSecondary", True, omem[0], omem[1])
    omem = mem[f"{outp}-rs"]
    toffs[idx[f"{outp}_rs"]] = _cpu_rs(b, f"{outp}-rs", omem[0], omem[1])
    for name in list(mem.keys()):
        if name.startswith("t") and name.endswith("-rs") and name != f"{outp}-rs" and name in idx:
            m = mem[name]
            toffs[idx[name]] = _cpu_rs(b, name, m[0], m[1])
    for nm in reversed(ins):
        rs_name = f"{nm}_rs"
        m = mem[rs_name]
        toffs[idx[rs_name]] = _cpu_rs(b, rs_name, m[0], m[1])
        m = mem[f"{nm}_exsec"]
        toffs[idx[f"{nm}_exsec"]] = _cpu_exsec(
            b, f"{nm}_exSecondary", False, m[0], m[1])
    for nm in reversed(ins[1:]):
        m = mem[nm]
        toffs[idx[nm]] = _cpu_ext_input(b, nm, m[0], m[1])
    m = mem[ins[0]]
    toffs[idx[ins[0]]] = _cpu_ext_input(b, ins[0], m[0], m[1])
    for i in range(n, -1, -1):
        all_names = ins + [outp]
        nm = all_names[i]
        name_str = f"{nm}_rs_i1" if nm in ins else f"{outp}-rs_i1"
        f12 = 16 if i == n else 32
        toffs[idx[f"{all_names[i]}_rsi1"]] = _rsi1_tensor(b, name_str, i + 1, f12)
    toffs[0] = _empty_tensor(b)
    return toffs


def _cpu_root_attrs(n):
    ins, outp = _cpu_io(n)
    shape = [1, 4]
    attrs, quant = {}, {}
    for i, nm in enumerate(ins):
        attrs[nm] = {"idx": i, "shape": shape, "layout": "nchw", "layout_ori": "nchw",
                      "is_output": False, "range": [0, 1], "origin_dynamic": False,
                      "dtype": "bool", "mean": [0] * 4, "std": [1] * 4, "rgb2bgr": False}
        quant[nm] = {"dtype": "bool", "qmethod": "", "qtype": "", "min": [],
                      "max": [], "scale": [], "zero_point": [], "name": nm, "shape": shape}
    attrs[outp] = {"is_output": True, "idx": 0, "shape": shape,
                    "dtype": "bool", "layout": "nchw"}
    quant[outp] = {"dtype": "bool", "qmethod": "", "qtype": "", "min": [],
                    "max": [], "scale": [], "zero_point": [], "name": outp, "shape": shape}
    return str({"attrs": attrs, "quant_tab": quant, "dynamic_shapes": {}})


def _cpu_sg_f7_entry(b, name, offset):
    vn = _str(b, name)
    v1 = _vec_pairs(b, [(0, offset)])
    v2 = _vec(b, [1])
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(2, v2, 0)
    b.PrependUOffsetTRelativeSlot(1, v1, 0)
    b.PrependUOffsetTRelativeSlot(0, vn, 0)
    return b.EndObject()


def _build_cpu_subgraph(b, toffs, noffs, n, ins, outp, idx):
    tvec = _ovec(b, toffs)
    nvec = _ovec(b, noffs)
    sg_f10 = _vec(b, [0, 0, 0, n + 1, 2 * (n + 1)])
    sg_f12 = _ovec(b, [_str_scalar_table2(b, outp, n - 1)])
    sg_f2 = _vec(b, [idx[ins[i]] for i in range(n)])
    sg_f3 = _vec(b, [idx[outp]])
    evs = [_ev(b) for _ in range(7)]
    sg_f4 = _ovec(b, [
        _vec_table3(b, [0] * 10, [0x3f800000] * 10, list(range(10)))
        for _ in range(n)
    ])
    f7_entries = []
    for i, nm in enumerate(ins):
        f7_entries.append(_cpu_sg_f7_entry(b, nm, 55 + 80 * i))
        f7_entries.append(_cpu_sg_f7_entry(b, f"{nm}_rs", 5 + 80 * i))
    f7_entries.append(_cpu_sg_f7_entry(b, outp, 5 + 80 * n))
    f7_entries.append(_cpu_sg_f7_entry(b, f"{outp}-rs", 55 + 80 * n))
    sg_f7 = _ovec(b, f7_entries)
    b.StartObject(17)
    b.PrependUOffsetTRelativeSlot(16, evs[0], 0)
    b.PrependUOffsetTRelativeSlot(15, evs[1], 0)
    b.PrependUOffsetTRelativeSlot(14, evs[2], 0)
    b.PrependUOffsetTRelativeSlot(13, evs[3], 0)
    b.PrependUOffsetTRelativeSlot(12, sg_f12, 0)
    b.PrependUOffsetTRelativeSlot(10, sg_f10, 0)
    b.PrependUOffsetTRelativeSlot(9, evs[4], 0)
    b.PrependUOffsetTRelativeSlot(8, evs[5], 0)
    b.PrependUOffsetTRelativeSlot(7, sg_f7, 0)
    b.PrependUOffsetTRelativeSlot(6, evs[6], 0)
    b.PrependUOffsetTRelativeSlot(4, sg_f4, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_f3, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_f2, 0)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    return b.EndVector()


def _build_cpu_body(n_inputs, ops=None):
    n = n_inputs
    n_adds = n - 1
    if ops is None:
        ops = ["And"] * n_adds
    ins, outp = _cpu_io(n)
    idx = _tensor_indices(n, ins, outp, n_adds)
    mem = _cpu_mem_offsets(n)
    b = flatbuffers.Builder(65536)
    noffs = _build_cpu_nodes(b, n, ins, outp, n_adds, idx, ops)
    toffs = _build_cpu_tensors(b, n, ins, outp, n_adds, idx, mem)
    sg_vec = _build_cpu_subgraph(b, toffs, noffs, n, ins, outp, idx)

    dtype_in = {nm: {"dtype": "bool", "layout": "UNDEFINED"} for nm in ins}
    dtype_out = {outp: {"dtype": "bool", "layout": "NCHW"}}
    fb = _emit_root_table(
        b, sg_vec, dtype_in, dtype_out,
        192 + (n - 2) * 64, 256 + (n - 2) * 64,
        _cpu_root_attrs(n))
    fb_len = len(fb)

    rc_word_off = (HEADER_SIZE + fb_len) // 4
    lead = _cpu_lead_u32(n, rc_word_off)
    prefix_words = len(lead) + _cpu_nonlead_words(n)
    canon_total = 160 * (n + 1)
    total_u32 = 677 + 215 * (n - 2) - 16 * (n // 8)
    trailing_needed = total_u32 - canon_total - prefix_words
    if trailing_needed % 2 != 0:
        fb = fb + b"\x00" * 4
        fb_len += 4

    rc_word_off = (HEADER_SIZE + fb_len) // 4
    rc_raw = build_template(n, ops=ops, rc_word_off=rc_word_off)
    taskdesc = _taskdesc_cpu(n, rc_word_off=rc_word_off)
    full = fb + rc_raw + taskdesc
    rc_len = len(rc_raw)
    fb_len = len(fb)

    _patch_root_command_offsets(full, fb_len, rc_raw, rc_len, taskdesc, n)
    return bytes(full[:fb_len]), bytes(full[fb_len:])


# ── Unary CPU-fallback op body builder (e.g. Not: out = ~a, bool [1,4]) ──
#
# Topology (1 input, 1 output): input(a) -> reshape(a_rs) -> Not(a_rs -> out_rs)
# -> reshape(out-rs) -> output(out).  Reuses the same per-tensor reshape/exsec
# tables and the same closed-form CPU RC prefix/canon/trailing as the binary path
# evaluated at n=1 (2 reshape/copy RC blocks; sg.f10 = [0,0,0,2,4]).

def _build_cpu_unary_nodes(b, ins, outp, idx, op):
    a = ins[0]
    noffs = [_input_node(b, a, idx[a])]
    noffs.append(_cpu_reshape_node(
        b, f"{a}_rs", a,
        [idx[a], idx[f"{a}_rsi1"], idx[f"{a}_exsec"]],
        [idx[f"{a}_rs"]]))
    # Unary op node: single input (the reshaped a), single output (out_rs).
    noffs.append(_op_node(b, 1, 0, 0, idx[f"{outp}_rs"], op=op,
                          in_idxs=[idx[f"{a}_rs"]]))
    noffs.append(_cpu_reshape_node(
        b, f"{outp}-rs", outp,
        [idx[f"{outp}_rs"], idx[f"{outp}_rsi1"], idx[f"{outp}_rs_exsec"]],
        [idx[outp]]))
    noffs.append(_cpu_output_node(b, outp, idx[outp]))
    return noffs


def _build_cpu_unary_body(op="Not"):
    """Build a from-scratch unary CPU-fallback model (bool [1,4]).

    Mirrors _build_cpu_body but for a single-input op.  The n-dependent binary
    size formulas are evaluated at n=1; the RC and taskdesc come from the shared
    n=1 generators (build_template / _taskdesc_cpu).
    """
    n = 1
    ins, outp = _cpu_io(n)
    idx = _tensor_indices(n, ins, outp, 0)
    mem = _cpu_mem_offsets(n)
    b = flatbuffers.Builder(65536)
    noffs = _build_cpu_unary_nodes(b, ins, outp, idx, op)
    toffs = _build_cpu_tensors(b, n, ins, outp, 0, idx, mem)
    sg_vec = _build_cpu_subgraph(b, toffs, noffs, n, ins, outp, idx)

    dtype_in = {nm: {"dtype": "bool", "layout": "UNDEFINED"} for nm in ins}
    dtype_out = {outp: {"dtype": "bool", "layout": "NCHW"}}
    fb = _emit_root_table(
        b, sg_vec, dtype_in, dtype_out,
        192 + (n - 2) * 64, 256 + (n - 2) * 64,
        _cpu_root_attrs(n))
    fb_len = len(fb)

    # Parity fix-up (same idea as _build_cpu_body): keep the CPU RC trailing even
    # so the total RC u32 count matches the canon+prefix layout.
    rc_word_off = (HEADER_SIZE + fb_len) // 4
    rc_raw = build_template(n, ops=[op], rc_word_off=rc_word_off)
    if (len(rc_raw) // 4) % 2 != 0:
        fb = fb + b"\x00" * 4
        fb_len += 4
        rc_word_off = (HEADER_SIZE + fb_len) // 4
        rc_raw = build_template(n, ops=[op], rc_word_off=rc_word_off)

    taskdesc = _taskdesc_cpu(n, rc_word_off=rc_word_off)
    full = fb + rc_raw + taskdesc
    rc_len = len(rc_raw)
    fb_len = len(fb)
    _patch_root_command_offsets(full, fb_len, rc_raw, rc_len, taskdesc, n)
    return bytes(full[:fb_len]), bytes(full[fb_len:])


# ── Fused XOR via And/Or/Not CPU ops (n-input parity, single .rknn) ──
#
# XOR has no native CPU kernel on this runtime, but is the identity
#   XOR(a,b) = (a OR b) AND NOT(a AND b)
# expressed as a CPU DAG.  N-input XOR (parity a0^a1^...^a{n-1}) chains one such
# 4-op stage per extra input, left-associated:  acc = XOR(acc, a_k).
#
# Crucial decode: CPU compute nodes touch the NPU only via Reshape nodes;
# intermediates between compute nodes flow CPU-side WITHOUT reshapes.  So the only
# reshapes are the n input reshapes + the output reshape -> the RC stream,
# taskdesc, sg.f10 and sg.f7 are byte-identical to the n-input And model.  Only the
# FlatBuffer compute-node set + the per-stage intermediate tensors are new.
#
# Each stage k computes  acc_k = (acc_{k-1} OR x_k) AND NOT(acc_{k-1} AND x_k)
# using 3 intermediates (s{k}_and, s{k}_or, s{k}_not) + the stage result s{k}.
# Every intermediate gets a DISTINCT workspace offset (f13) so the CPU kernels
# never alias (aliasing collapses the result, as seen when all shared offset 0).

def _build_cpu_xor_body(n_inputs=2):
    n = n_inputs
    if n < 2:
        raise ValueError(f"XOR needs >=2 inputs, got {n}")
    ins, outp = _cpu_io(n)
    mem = _cpu_mem_offsets(n)            # base layout (inputs, exsec, output reshape)
    n_stages = n - 1

    # The n-input And reshape plan REUSES offsets for inputs consumed immediately in
    # a linear chain (e.g. b_rs and c_rs share a slot).  Chained XOR keeps every
    # input reshape live until its stage, so each needs a DISTINCT offset.  Assign
    # input reshapes to distinct slots; intermediates get slots above them.
    rs_off = {}
    for i, nm in enumerate(ins):
        rs_off[nm] = (i + 1) * _CPU_SLOT          # 64,128,192,... distinct per input

    # Per-stage intermediate tensor names (3 per stage) + per-stage result name.
    # The final stage writes directly into the output-reshape tensor (out_rs).
    inter_names = []
    stage_out = {}
    for k in range(n_stages):
        inter_names += [f"s{k}_and", f"s{k}_or", f"s{k}_not"]
        if k < n_stages - 1:
            sres = f"s{k}_res"
            inter_names.append(sres)
            stage_out[k] = sres
        else:
            stage_out[k] = f"{outp}_rs"

    # Distinct, non-colliding workspace offsets for every intermediate, placed above
    # all the (distinct) input-reshape slots.
    base = (len(ins) + 2) * _CPU_SLOT
    inter_off = {nm: base + i * _CPU_SLOT for i, nm in enumerate(inter_names)}

    # ---- tensor index map (regcmd/task LAST for command-offset patching) ----
    idx = {}
    t = 0
    idx["empty"] = t; t += 1
    for nm in ins:
        idx[f"{nm}_rsi1"] = t; t += 1
    idx[f"{outp}_rsi1"] = t; t += 1
    for nm in ins:
        idx[nm] = t; t += 1
    for nm in ins:
        idx[f"{nm}_exsec"] = t; t += 1
        idx[f"{nm}_rs"] = t; t += 1
    for nm in inter_names:
        idx[nm] = t; t += 1
    idx[f"{outp}_rs"] = t; t += 1
    idx[f"{outp}_rs_exsec"] = t; t += 1
    idx[outp] = t; t += 1
    idx["regcmd"] = t; t += 1
    idx["task"] = t; t += 1

    b = flatbuffers.Builder(65536)

    # ---- nodes ----
    noffs = []
    for nm in ins:
        noffs.append(_input_node(b, nm, idx[nm]))
    for nm in ins:
        noffs.append(_cpu_reshape_node(
            b, f"{nm}_rs", nm,
            [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]], [idx[f"{nm}_rs"]]))
    # XOR stages: acc starts as a_rs; fold in each subsequent input reshape.
    acc = idx[f"{ins[0]}_rs"]
    for k in range(n_stages):
        x = idx[f"{ins[k + 1]}_rs"]
        a_t, o_t, n_t = f"s{k}_and", f"s{k}_or", f"s{k}_not"
        noffs.append(_op_node(b, 1, acc, x, idx[a_t], op="And"))
        noffs.append(_op_node(b, 1, acc, x, idx[o_t], op="Or"))
        noffs.append(_op_node(b, 1, 0, 0, idx[n_t], op="Not", in_idxs=[idx[a_t]]))
        noffs.append(_op_node(b, 2, idx[o_t], idx[n_t], idx[stage_out[k]], op="And"))
        acc = idx[stage_out[k]]
    noffs.append(_cpu_reshape_node(
        b, f"{outp}-rs", outp,
        [idx[f"{outp}_rs"], idx[f"{outp}_rsi1"], idx[f"{outp}_rs_exsec"]], [idx[outp]]))
    noffs.append(_cpu_output_node(b, outp, idx[outp]))

    # ---- tensors ----
    toffs = [None] * (idx["task"] + 1)
    toffs[idx["task"]] = _cmd_tensor(b, "task", f2=10, f18=n + 3)
    toffs[idx["regcmd"]] = _cmd_tensor(b, "regcmd", f2=9, f18=n + 2)
    toffs[idx[outp]] = _cpu_ext_output(b, outp)
    om = mem[f"{outp}-rs_exsec"]
    toffs[idx[f"{outp}_rs_exsec"]] = _cpu_exsec(b, f"{outp}-rs_exSecondary", True, om[0], om[1])
    om = mem[f"{outp}-rs"]
    toffs[idx[f"{outp}_rs"]] = _cpu_rs(b, f"{outp}-rs", om[0], om[1])
    for nm in inter_names:
        toffs[idx[nm]] = _cpu_rs(b, nm, inter_off[nm], True)
    for nm in reversed(ins):
        toffs[idx[f"{nm}_rs"]] = _cpu_rs(b, f"{nm}_rs", rs_off[nm], True)
        m = mem[f"{nm}_exsec"]
        toffs[idx[f"{nm}_exsec"]] = _cpu_exsec(b, f"{nm}_exSecondary", False, m[0], m[1])
    for nm in reversed(ins[1:]):
        toffs[idx[nm]] = _cpu_ext_input(b, nm, *mem[nm])
    toffs[idx[ins[0]]] = _cpu_ext_input(b, ins[0], *mem[ins[0]])
    all_names = ins + [outp]
    for i in range(n, -1, -1):
        nm = all_names[i]
        name_str = f"{nm}_rs_i1" if nm in ins else f"{outp}-rs_i1"
        f12 = 16 if i == n else 32
        toffs[idx[f"{nm}_rsi1"]] = _rsi1_tensor(b, name_str, i + 1, f12)
    toffs[0] = _empty_tensor(b)

    sg_vec = _build_cpu_subgraph(b, toffs, noffs, n, ins, outp, idx)

    dtype_in = {nm: {"dtype": "bool", "layout": "UNDEFINED"} for nm in ins}
    dtype_out = {outp: {"dtype": "bool", "layout": "NCHW"}}
    fb = _emit_root_table(b, sg_vec, dtype_in, dtype_out,
                          192 + (n - 2) * 64, 256 + (n - 2) * 64, _cpu_root_attrs(n))
    fb_len = len(fb)

    # RC + taskdesc are identical to the n-input And model (n+1 reshape/copy blocks).
    rc_word_off = (HEADER_SIZE + fb_len) // 4
    rc_ops = ["And"] * (n - 1)
    rc_raw = build_template(n, ops=rc_ops, rc_word_off=rc_word_off)
    if (len(rc_raw) // 4) % 2 != 0:
        fb = fb + b"\x00" * 4
        fb_len += 4
        rc_word_off = (HEADER_SIZE + fb_len) // 4
        rc_raw = build_template(n, ops=rc_ops, rc_word_off=rc_word_off)
    taskdesc = _taskdesc_cpu(n, rc_word_off=rc_word_off)
    full = fb + rc_raw + taskdesc
    _patch_root_command_offsets(full, fb_len, rc_raw, len(rc_raw), taskdesc, n)
    return bytes(full[:fb_len]), bytes(full[fb_len:])


def _mixed_root_attrs():
    shape = [1, 4]
    attrs, quant = {}, {}
    for i, nm in enumerate(["a", "b"]):
        attrs[nm] = {"idx": i, "shape": shape, "layout": "nchw", "layout_ori": "nchw",
                     "is_output": False, "range": [0, 1], "origin_dynamic": False,
                     "dtype": "bool", "mean": [0] * 4, "std": [1] * 4, "rgb2bgr": False}
        quant[nm] = {"dtype": "bool", "qmethod": "", "qtype": "", "min": [],
                     "max": [], "scale": [], "zero_point": [], "name": nm, "shape": shape}
    for i, nm in enumerate(["x", "y"]):
        attrs[nm] = {"idx": i + 2, "shape": shape, "layout": "nchw", "layout_ori": "nchw",
                     "is_output": False, "range": [0, 1], "origin_dynamic": False,
                     "dtype": "float32", "mean": [0] * 4, "std": [1] * 4, "rgb2bgr": False}
        quant[nm] = {"dtype": "float16", "qmethod": "", "qtype": "", "min": [],
                     "max": [], "scale": [], "zero_point": [], "name": nm, "shape": shape}
    attrs["out1"] = {"is_output": True, "idx": 0, "shape": shape,
                     "dtype": "bool", "layout": "nchw"}
    quant["out1"] = {"dtype": "bool", "qmethod": "", "qtype": "", "min": [],
                     "max": [], "scale": [], "zero_point": [], "name": "out1", "shape": shape}
    attrs["out2"] = {"is_output": True, "idx": 1, "shape": shape,
                     "dtype": "float32", "layout": "nchw"}
    quant["out2"] = {"dtype": "float16", "qmethod": "", "qtype": "", "min": [],
                     "max": [], "scale": [], "zero_point": [], "name": "out2", "shape": shape}
    return str({"attrs": attrs, "quant_tab": quant, "dynamic_shapes": {}})


_MIXED_NPU_IO_OFF = [0, 0, 32, 0, 32, 48]
_MIXED_NPU_IO_MASK = [1, 0, 1, 0, 1, 1]
_MIXED_CPU_SHIFT = 240
_MIXED_NPU_F10 = [1, 1, 1, 1, 1, 1]
_MIXED_NPU_F12 = [32, 0, 0, 16, 16, 0, 16, 8, 8]


def _build_mixed_and_add_body(cpu_op="And", npu_op="Add"):
    n_t = 27
    n_n = 14
    b = flatbuffers.Builder(65536)

    noffs = [None] * n_n
    noffs[13] = _cpu_output_node(b, "out2", 24)
    noffs[12] = _reshape_node(b, "out2-rs", "out2", [22, 6, 23], [24])
    noffs[11] = _op_node(b, 1, 19, 21, 22, op=npu_op,
                          npu_f10=_MIXED_NPU_F10, npu_f12=_MIXED_NPU_F12)
    noffs[10] = _reshape_node(b, "y_rs", "y", [10, 5, 20], [21])
    noffs[9] = _reshape_node(b, "x_rs", "x", [9, 4, 18], [19])
    noffs[8] = _cpu_output_node(b, "out1", 17)
    noffs[7] = _cpu_reshape_node(b, "out1-rs", "out1", [15, 3, 16], [17])
    noffs[6] = _op_node(b, 1, 12, 14, 15, op=cpu_op)
    noffs[5] = _cpu_reshape_node(b, "b_rs", "b", [8, 2, 13], [14])
    noffs[4] = _cpu_reshape_node(b, "a_rs", "a", [7, 1, 11], [12])
    noffs[3] = _input_node(b, "y", 10)
    noffs[2] = _input_node(b, "x", 9)
    noffs[1] = _input_node(b, "b", 8)
    noffs[0] = _input_node(b, "a", 7)

    toffs = [None] * n_t
    toffs[26] = _cmd_tensor(b, "task", f2=10, f18=8)
    toffs[25] = _cmd_tensor(b, "regcmd", f2=9, f18=7)
    toffs[24] = _ext_tensor(b, "out2", [1, 4], 64, f13=128, f2=2)
    toffs[23] = _exsec_tensor(b, "out2-rs_exSecondary", [1, 1, 1, 4], 1, 64, f1=2)
    toffs[22] = _rs_tensor(b, "out2-rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 0,
                            has_f13=False)
    toffs[21] = _rs_tensor(b, "y_rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 128,
                            has_f13=True)
    toffs[20] = _exsec_tensor(b, "y_exSecondary", [1, 4], 1, None, f1=None)
    toffs[19] = _rs_tensor(b, "x_rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 64,
                            has_f13=True)
    toffs[18] = _exsec_tensor(b, "x_exSecondary", [1, 4], 1, None, f1=None)
    toffs[17] = _cpu_ext_output(b, "out1", f13=256)
    toffs[16] = _cpu_exsec(b, "out1-rs_exSecondary", True, 64, True)
    toffs[15] = _cpu_rs(b, "out1-rs", None, False)
    toffs[14] = _cpu_rs(b, "b_rs", 256, True)
    toffs[13] = _cpu_exsec(b, "b_exSecondary", False, None, False)
    toffs[12] = _cpu_rs(b, "a_rs", 320, True)
    toffs[11] = _cpu_exsec(b, "a_exSecondary", False, 256, True)
    toffs[10] = _ext_tensor(b, "y", [1, 4], 8, f13=192, f2=1)
    toffs[9] = _ext_tensor(b, "x", [1, 4], 8, f13=128, f2=1)
    toffs[8] = _cpu_ext_input(b, "b", 64, True)
    toffs[7] = _cpu_ext_input(b, "a", None, False)
    toffs[6] = _rsi1_tensor(b, "out2-rs_i1", 7, 16)
    toffs[5] = _rsi1_tensor(b, "y_rs_i1", 6, 32)
    toffs[4] = _rsi1_tensor(b, "x_rs_i1", 5, 32)
    toffs[3] = _rsi1_tensor(b, "out1-rs_i1", 4, 16)
    toffs[2] = _rsi1_tensor(b, "b_rs_i1", 3, 32)
    toffs[1] = _rsi1_tensor(b, "a_rs_i1", 2, 32)
    toffs[0] = _empty_tensor(b)

    tvec = _ovec(b, toffs)
    nvec = _ovec(b, noffs)
    sg_f4 = _ovec(b, [
        _vec_table3(b, [0] * 10, [0x3f800000] * 10, list(range(10)))
        for _ in range(4)
    ])

    cpu_f7 = [
        _cpu_sg_f7_entry(b, "a", 55),
        _cpu_sg_f7_entry(b, "a_rs", 5),
        _cpu_sg_f7_entry(b, "b", 135),
        _cpu_sg_f7_entry(b, "b_rs", 85),
        _cpu_sg_f7_entry(b, "out1", 165),
        _cpu_sg_f7_entry(b, "out1-rs", 215),
    ]
    npu_bases = {"out2-rs": 5, "x_rs": 55, "y_rs": 61}
    npu_f7 = []
    for nm, base in npu_bases.items():
        pairs = [(_MIXED_NPU_IO_OFF[i], base + i * 80 + _MIXED_CPU_SHIFT)
                 for i in range(6)]
        npu_f7.append(_str_vec_table3(b, nm, pairs, list(_MIXED_NPU_IO_MASK)))
    sg_f7 = _ovec(b, cpu_f7 + npu_f7)
    sg_f10 = _vec(b, [0, 0, 0, 4, 9])
    sg_f12 = _ovec(b, [_str_scalar_table2(b, "out1", 3),
                        _str_scalar_table2(b, "out2", 4)])
    sg_f2 = _vec(b, [7, 8, 9, 10])
    sg_f3 = _vec(b, [17, 24])
    evs = [_ev(b) for _ in range(7)]

    b.StartObject(17)
    b.PrependUOffsetTRelativeSlot(16, evs[0], 0)
    b.PrependUOffsetTRelativeSlot(15, evs[1], 0)
    b.PrependUOffsetTRelativeSlot(14, evs[2], 0)
    b.PrependUOffsetTRelativeSlot(13, evs[3], 0)
    b.PrependUOffsetTRelativeSlot(12, sg_f12, 0)
    b.PrependUOffsetTRelativeSlot(10, sg_f10, 0)
    b.PrependUOffsetTRelativeSlot(9, evs[4], 0)
    b.PrependUOffsetTRelativeSlot(8, evs[5], 0)
    b.PrependUOffsetTRelativeSlot(7, sg_f7, 0)
    b.PrependUOffsetTRelativeSlot(6, evs[6], 0)
    b.PrependUOffsetTRelativeSlot(4, sg_f4, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_f3, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_f2, 0)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    sg_vec = b.EndVector()

    dtype_in = {"a": {"dtype": "bool", "layout": "UNDEFINED"},
                "b": {"dtype": "bool", "layout": "UNDEFINED"},
                "x": {"dtype": "float16", "layout": "UNDEFINED"},
                "y": {"dtype": "float16", "layout": "UNDEFINED"}}
    dtype_out = {"out1": {"dtype": "bool", "layout": "NCHW"},
                 "out2": {"dtype": "float16", "layout": "NCHW"}}
    fb = _emit_root_table(b, sg_vec, dtype_in, dtype_out, 384, 384,
                          _mixed_root_attrs())
    return bytes(fb)


def _build_mixed_xor_and_add_body(npu_op="Add"):
    """FlatBuffer body for parallel fused-Xor CPU branch + one NPU EW branch.

    CPU branch computes out1 = (a OR b) AND NOT(a AND b) using the same three
    CPU reshape/copy blocks as the 2-input And/Or branch: a_rs, b_rs, out1-rs.
    Thus the mixed RC schedule stays 3 CPU copy blocks + 6 NPU compute blocks;
    only the FlatBuffer node/tensor graph grows by 3 CPU compute nodes and three
    intermediate tensors.
    """
    n_t = 30
    n_n = 17
    b = flatbuffers.Builder(65536)

    noffs = [None] * n_n
    # CPU inputs and reshapes.
    noffs[0] = _input_node(b, "a", 7)
    noffs[1] = _input_node(b, "b", 8)
    noffs[2] = _input_node(b, "x", 9)
    noffs[3] = _input_node(b, "y", 10)
    noffs[4] = _cpu_reshape_node(b, "a_rs", "a", [7, 1, 11], [12])
    noffs[5] = _cpu_reshape_node(b, "b_rs", "b", [8, 2, 13], [14])
    # Fused XOR CPU DAG.
    noffs[6] = _op_node(b, 1, 12, 14, 15, op="And")
    noffs[7] = _op_node(b, 1, 12, 14, 16, op="Or")
    noffs[8] = _op_node(b, 1, 0, 0, 17, op="Not", in_idxs=[15])
    noffs[9] = _op_node(b, 2, 16, 17, 18, op="And")
    noffs[10] = _cpu_reshape_node(b, "out1-rs", "out1", [18, 3, 19], [20])
    noffs[11] = _cpu_output_node(b, "out1", 20)
    # NPU branch.
    noffs[12] = _reshape_node(b, "x_rs", "x", [9, 4, 21], [22])
    noffs[13] = _reshape_node(b, "y_rs", "y", [10, 5, 23], [24])
    noffs[14] = _op_node(b, 1, 22, 24, 25, op=npu_op,
                          npu_f10=_MIXED_NPU_F10, npu_f12=_MIXED_NPU_F12)
    noffs[15] = _reshape_node(b, "out2-rs", "out2", [25, 6, 26], [27])
    noffs[16] = _cpu_output_node(b, "out2", 27)

    toffs = [None] * n_t
    toffs[29] = _cmd_tensor(b, "task", f2=10, f18=8)
    toffs[28] = _cmd_tensor(b, "regcmd", f2=9, f18=7)
    toffs[27] = _ext_tensor(b, "out2", [1, 4], 64, f13=128, f2=2)
    toffs[26] = _exsec_tensor(b, "out2-rs_exSecondary", [1, 1, 1, 4], 1, 64, f1=2)
    toffs[25] = _rs_tensor(b, "out2-rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 0,
                            has_f13=False)
    toffs[24] = _rs_tensor(b, "y_rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 128,
                            has_f13=True)
    toffs[23] = _exsec_tensor(b, "y_exSecondary", [1, 4], 1, None, f1=None)
    toffs[22] = _rs_tensor(b, "x_rs", [1, 1, 1, 4, 8], [1, 1, 1, 4], 64, 64,
                            has_f13=True)
    toffs[21] = _exsec_tensor(b, "x_exSecondary", [1, 4], 1, None, f1=None)
    toffs[20] = _cpu_ext_output(b, "out1", f13=256)
    toffs[19] = _cpu_exsec(b, "out1-rs_exSecondary", True, 64, True)
    toffs[18] = _cpu_rs(b, "out1-rs", None, False)
    # XOR intermediates: distinct workspace offsets above a_rs/b_rs (320/256).
    toffs[17] = _cpu_rs(b, "t_not", 512, True)
    toffs[16] = _cpu_rs(b, "t_or", 448, True)
    toffs[15] = _cpu_rs(b, "t_and", 384, True)
    toffs[14] = _cpu_rs(b, "b_rs", 256, True)
    toffs[13] = _cpu_exsec(b, "b_exSecondary", False, None, False)
    toffs[12] = _cpu_rs(b, "a_rs", 320, True)
    toffs[11] = _cpu_exsec(b, "a_exSecondary", False, 256, True)
    toffs[10] = _ext_tensor(b, "y", [1, 4], 8, f13=192, f2=1)
    toffs[9] = _ext_tensor(b, "x", [1, 4], 8, f13=128, f2=1)
    toffs[8] = _cpu_ext_input(b, "b", 64, True)
    toffs[7] = _cpu_ext_input(b, "a", None, False)
    toffs[6] = _rsi1_tensor(b, "out2-rs_i1", 7, 16)
    toffs[5] = _rsi1_tensor(b, "y_rs_i1", 6, 32)
    toffs[4] = _rsi1_tensor(b, "x_rs_i1", 5, 32)
    toffs[3] = _rsi1_tensor(b, "out1-rs_i1", 4, 16)
    toffs[2] = _rsi1_tensor(b, "b_rs_i1", 3, 32)
    toffs[1] = _rsi1_tensor(b, "a_rs_i1", 2, 32)
    toffs[0] = _empty_tensor(b)

    tvec = _ovec(b, toffs)
    nvec = _ovec(b, noffs)
    sg_f4 = _ovec(b, [
        _vec_table3(b, [0] * 10, [0x3f800000] * 10, list(range(10)))
        for _ in range(4)
    ])
    cpu_f7 = [
        _cpu_sg_f7_entry(b, "a", 55),
        _cpu_sg_f7_entry(b, "a_rs", 5),
        _cpu_sg_f7_entry(b, "b", 135),
        _cpu_sg_f7_entry(b, "b_rs", 85),
        _cpu_sg_f7_entry(b, "out1", 165),
        _cpu_sg_f7_entry(b, "out1-rs", 215),
    ]
    npu_bases = {"out2-rs": 5, "x_rs": 55, "y_rs": 61}
    npu_f7 = []
    for nm, base in npu_bases.items():
        pairs = [(_MIXED_NPU_IO_OFF[i], base + i * 80 + _MIXED_CPU_SHIFT)
                 for i in range(6)]
        npu_f7.append(_str_vec_table3(b, nm, pairs, list(_MIXED_NPU_IO_MASK)))
    sg_f7 = _ovec(b, cpu_f7 + npu_f7)
    sg_f10 = _vec(b, [0, 0, 0, 4, 9])
    sg_f12 = _ovec(b, [_str_scalar_table2(b, "out1", 3),
                        _str_scalar_table2(b, "out2", 4)])
    sg_f2 = _vec(b, [7, 8, 9, 10])
    sg_f3 = _vec(b, [20, 27])
    evs = [_ev(b) for _ in range(7)]
    b.StartObject(17)
    b.PrependUOffsetTRelativeSlot(16, evs[0], 0)
    b.PrependUOffsetTRelativeSlot(15, evs[1], 0)
    b.PrependUOffsetTRelativeSlot(14, evs[2], 0)
    b.PrependUOffsetTRelativeSlot(13, evs[3], 0)
    b.PrependUOffsetTRelativeSlot(12, sg_f12, 0)
    b.PrependUOffsetTRelativeSlot(10, sg_f10, 0)
    b.PrependUOffsetTRelativeSlot(9, evs[4], 0)
    b.PrependUOffsetTRelativeSlot(8, evs[5], 0)
    b.PrependUOffsetTRelativeSlot(7, sg_f7, 0)
    b.PrependUOffsetTRelativeSlot(6, evs[6], 0)
    b.PrependUOffsetTRelativeSlot(4, sg_f4, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_f3, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_f2, 0)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    sg_vec = b.EndVector()

    dtype_in = {"a": {"dtype": "bool", "layout": "UNDEFINED"},
                "b": {"dtype": "bool", "layout": "UNDEFINED"},
                "x": {"dtype": "float16", "layout": "UNDEFINED"},
                "y": {"dtype": "float16", "layout": "UNDEFINED"}}
    dtype_out = {"out1": {"dtype": "bool", "layout": "NCHW"},
                 "out2": {"dtype": "float16", "layout": "NCHW"}}
    fb = _emit_root_table(b, sg_vec, dtype_in, dtype_out, 384, 384,
                          _mixed_root_attrs())
    return bytes(fb)


def build_mixed_and_add(cpu_op="And", npu_op="Add", ref_rc_path=None):
    fb_bytes = (_build_mixed_xor_and_add_body(npu_op) if cpu_op == "Xor"
                else _build_mixed_and_add_body(cpu_op, npu_op))
    ref_name = "_ref_parallel_and_add.rknn"
    ref_path = ref_rc_path or str(Path(__file__).resolve().parent / ref_name)
    ref_data = Path(ref_path).read_bytes()
    body_size = struct.unpack_from("<Q", ref_data, 0x10)[0]

    root_file = HEADER_SIZE + struct.unpack_from("<I", ref_data, HEADER_SIZE)[0]

    def i32f(o): return struct.unpack_from("<i", ref_data, o)[0]
    def u16f(o): return struct.unpack_from("<H", ref_data, o)[0]
    def u32f(o): return struct.unpack_from("<I", ref_data, o)[0]

    vt_file = root_file - i32f(root_file)
    rc_off = u32f(root_file + u16f(vt_file + 4 + 20 * 2))
    task_off = u32f(root_file + u16f(vt_file + 4 + 21 * 2))
    ref_rc = _patch_mixed_npu_op(ref_data[rc_off:task_off], npu_op)
    ref_taskdesc = ref_data[task_off:HEADER_SIZE + body_size]

    fb_len = len(fb_bytes)
    if fb_len % 4:
        pad = 4 - fb_len % 4
        fb_bytes += b"\x00" * pad
        fb_len += pad

    full = bytearray(fb_bytes + ref_rc + ref_taskdesc)
    _patch_root_command_offsets(full, fb_len, ref_rc, len(ref_rc),
                                ref_taskdesc, 5)
    return bytes(full[:fb_len]), bytes(full[fb_len:])


def _patch_mixed_npu_op(rc, npu_op):
    """Patch the NPU compute blocks of a mixed RC stream to compute `npu_op`.

    The reference RC was built for Add; the per-op DPU/RDMA registers (EW_CFG
    0x4070, DPU_OUT_RES 0x4084, RDMA_BN_MUL 0x5044) must be rewritten to match
    Sub/Mul/Div.  Compute blocks are identified by 0x4070 != the copy value 0x383
    (copy blocks keep 0x383).  Returns a new bytes object.
    """
    op_id = ew_op_id(npu_op)
    cfg = _EW_CFG[op_id]
    out_res = _DPU_OUT_RES.get(op_id, 0x00010001)
    bn_mul = _RDMA_BN_MUL.get(op_id, 0x00017849)
    reg_to_val = {0x4070: cfg, 0x4084: out_res, 0x5044: bn_mul}
    COPY_CFG = _EW_CFG[EW_OP_COPY]   # 0x383 (copy block)
    KNOWN_CFGS = set(_EW_CFG.values())
    buf = bytearray(rc)
    n = len(buf)

    # Pass 1: find compute-block extents.  Mixed RC streams are u32-aligned, so the
    # 64-bit register words may start at byte phase 0 or 4 depending on the prefix
    # alignment.  Each block spans from one DPU 0x4070 write to the next; copy
    # blocks carry 0x383 and compute blocks carry an EW op cfg.
    phase_marks = []
    for phase in (0, 4):
        marks = []   # (byte_index_of_0x4070_write, is_compute)
        for i in range(phase, n - 8 + 1, 8):
            w = struct.unpack_from("<Q", buf, i)[0]
            tgt = (w >> 48) & 0xFFFF
            reg = w & 0xFFFF
            val = (w >> 16) & 0xFFFFFFFF
            if tgt == _DPU and reg == 0x4070 and val in KNOWN_CFGS:
                marks.append((i, val != COPY_CFG))
        phase_marks.append(marks)
    marks = max(phase_marks, key=len)

    if npu_op != "Add" and not any(is_c for _, is_c in marks):
        raise ValueError("no mixed NPU compute blocks found to patch")

    bounds = []
    for k, (start, is_c) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else n
        if is_c:
            bounds.append((start, end))

    # Pass 2: within each compute block, rewrite the per-op registers.
    for (start, end) in bounds:
        for i in range(start, end - 8 + 1, 8):
            w = struct.unpack_from("<Q", buf, i)[0]
            reg = w & 0xFFFF
            if reg in reg_to_val:
                # Preserve the command target (top 16 bits) and register (low 16
                # bits); replace only the 32-bit payload in bits 16..47.
                struct.pack_into("<Q", buf, i,
                                 (w & 0xFFFF00000000FFFF) | (reg_to_val[reg] << 16))
    return bytes(buf)


def _make_trailer(rows, cols, n_inputs, body=None):
    import json
    ins, outp = _io(n_inputs)
    if body is not None and len(body) > 100:
        try:
            ba = bytearray(body)
            positions = _fb_tensor_positions(ba)
            if len(positions) > 1:
                first_name = _fb_string(ba, positions[1], 5)
                if first_name and first_name.startswith("a_rs"):
                    ins, outp = _cpu_io(n_inputs)
        except Exception:
            pass
    norm_tensor = []
    for i, nm in enumerate(ins + [outp]):
        norm_tensor.append({
            "dim_num": 2, "dtype": {"qnt_method": "", "qnt_type": "", "vx_type": ""},
            "size": [rows, cols], "tensor_id": i, "url": nm
        })
    connection = []
    for i in range(n_inputs):
        connection.append({"left": "input", "left_tensor_id": i, "node_id": 0,
                           "right_tensor": {"tensor_id": i, "type": "norm_tensor"}})
    connection.append({"left": "output", "left_tensor_id": 0, "node_id": 0,
                       "right_tensor": {"tensor_id": n_inputs, "type": "norm_tensor"}})
    graph = []
    for i in range(n_inputs):
        graph.append({"left": "input", "left_tensor_id": i,
                       "right": "norm_tensor", "right_tensor_id": i})
    graph.append({"left": "output", "left_tensor_id": 0,
                   "right": "norm_tensor", "right_tensor_id": n_inputs})
    js = {
        "connection": connection, "const_tensor": [], "graph": graph,
        "input_num": n_inputs, "name": "rknn model", "network_platform": "ONNX",
        "node_num": 1,
        "nodes": [{"input_num": n_inputs, "lid": "npu_network_bin_graph", "name": "nnbg",
                    "nn": {"nbg": {"type": "RKNN_OP_NNBG"}}, "op": "RKNN_OP_NNBG",
                    "output_num": 1, "uid": 0}],
        "norm_tensor": norm_tensor, "norm_tensor_num": n_inputs + 1,
        "ori_network_platform": "ONNX", "output_num": 1,
        "target_platform": ["rk3588"], "version": "2.3.2", "virtual_tensor": [],
    }
    nj = json.dumps(js, separators=(",", ":")).encode()
    return struct.pack("<Q", len(nj)) + nj


def assemble_rknn(body, rows, cols, n_inputs):
    trailer = _make_trailer(rows, cols, n_inputs, body)
    h = bytearray(HEADER_SIZE)
    h[0:4] = b"RKNN"
    struct.pack_into("<Q", h, 0x08, 6)
    struct.pack_into("<Q", h, 0x10, len(body))
    return bytes(h) + body + trailer


def _root_command_offsets(body):
    root = _fb_u32(body, 0)
    out = []
    for field in (20, 21):
        ab = _fb_field_abs(body, root, field)
        if ab is None:
            raise ValueError(f"template body is missing root field {field}")
        val = _fb_u32(body, ab)
        if val < HEADER_SIZE:
            raise ValueError(f"root field {field} has invalid absolute offset {val}")
        out.append(val - HEADER_SIZE)
    return tuple(out)


def build_body_scratch(N, n_inputs):
    """Rebuild a spec-conformant body from decoded components.

    The RKNN runtime's schema verifier rejects the generic FlatBuffers builder
    layout below even though it is structurally valid FlatBuffers. The accepted
    layout keeps the toolkit-produced FlatBuffer skeleton, regenerates the
    command/template bytes from readable specs, then patches shapes/memory/regcmd.
    """
    if n_inputs not in _RC_TEMPLATES:
        raise ValueError(f"scratch regcmd template not available for {n_inputs} inputs")

    template = _get_template_body(n_inputs)
    fb_end, _ = _root_command_offsets(template)
    rc_raw = _RC_TEMPLATES[n_inputs]
    body = bytearray(template[:fb_end] + rc_raw + _taskdesc(n_inputs))
    _patch_root_f20_f21(body, HEADER_SIZE + fb_end, HEADER_SIZE + fb_end + len(rc_raw))
    _plan_memory(body, N, n_inputs)
    _patch_tiles(body, N, n_inputs)
    return bytes(body)


def _build_body_scratch_flatbuffers(N, n_inputs, ops=None, dtype="float16"):
    # Fused XOR (n inputs, parity): chained (a OR b) AND NOT(a AND b) CPU DAG.
    _op_list = ops if isinstance(ops, (list, tuple)) else [ops]
    if n_inputs >= 2 and _op_list and _op_list[0] == "Xor":
        fb_part, rc_part = _build_cpu_xor_body(n_inputs)
        return fb_part + rc_part

    # Unary CPU op (1 input, e.g. Not): distinct single-input topology.
    if n_inputs == 1:
        unary = [o for o in (ops if isinstance(ops, (list, tuple)) else [ops]) if o]
        if unary and all(is_unary_cpu_op(o) for o in unary):
            fb_part, rc_part = _build_cpu_unary_body(unary[0])
            return fb_part + rc_part
        raise NotImplementedError(
            f"n_inputs=1 is only supported for unary CPU ops; got ops={ops}")

    n_adds_check = max(1, n_inputs - 1)
    op_names = _normalize_ops(n_adds_check, ops)
    if op_names and all(is_cpu_op(o) for o in op_names):
        fb_part, rc_part = _build_cpu_body(n_inputs, op_names)
        return fb_part + rc_part
    min_ops = max(1, n_inputs - 1)
    n_ops = len(ops) if ops and len(ops) >= min_ops else min_ops
    is_multiop = n_ops > n_inputs - 1
    n_external = n_inputs
    n_virtual = n_ops + 1 if is_multiop else n_inputs
    if n_virtual not in _RC_TEMPLATES:
        raise NotImplementedError(
            f"pure FlatBuffers generation not available for {n_inputs} inputs "
            f"with {n_ops} ops (internal n={n_virtual}); "
            f"available: {sorted(_RC_TEMPLATES)}"
        )
    C1, W = surface_split(N)
    Npad = C1 * W
    n_adds = n_virtual - 1
    tiles = tile_split(C1)
    ins, outp = _io(n_virtual)
    ext = _align(N * 2)
    work = _align(Npad * 16)
    plan = _mem(n_virtual)

    idx = _tensor_indices(n_virtual, ins, outp, n_adds)

    if is_multiop:
        for k in range(n_external, n_virtual):
            tgt = 1 + (k - 1) % (n_external - 1)
            plan[ins[k]] = plan[ins[tgt]]
            plan[f"{ins[k]}_rs"] = plan[f"{ins[tgt]}_rs"]

    s5 = [1, C1, 1, W, 8]
    s4 = [1, C1, 1, W]
    s2 = [1, N]

    idx_nodes = dict(idx)
    if is_multiop:
        for k in range(n_external, n_virtual):
            tgt = 1 + (k - 1) % (n_external - 1)
            for suffix in ["", "_rs", "_exsec", "_rsi1"]:
                src_key = f"{ins[tgt]}{suffix}"
                dst_key = f"{ins[k]}{suffix}"
                if src_key in idx_nodes:
                    idx_nodes[dst_key] = idx_nodes[src_key]

    b = flatbuffers.Builder(65536)

    noffs = _build_nodes(b, n_virtual, ins, outp, n_adds, idx_nodes, ops)
    toffs = _build_tensors(b, n_virtual, ins, outp, n_adds, plan, idx,
                           s2, s4, s5, N, Npad, ext, work, C1, W)
    n_sg2 = n_external if is_multiop else n_virtual
    sg = _build_subgraph(b, toffs, noffs, n_adds, idx, ins, outp, n_sg2)
    fb_bytes, rc_bytes = _build_root(b, sg, n_adds, C1, W, tiles, N, n_inputs, ops,
                                     n_rc_inputs=n_virtual, dtype=dtype)
    body = bytearray(fb_bytes + rc_bytes)
    _patch_tensor_dtype(body, dtype)
    return bytes(body)


def _patch_tensor_dtype(body, dtype):
    """Set every data tensor's f0 dtype enum to the requested dtype.

    Only the FlatBuffer-prefix tensor tables carry f0; the regcmd/task tensors
    keep their command kind (13). Data tensors are the ones whose f0 currently
    holds the float16 enum (10). For dtypes whose internal tensor class IS fp16
    (float16, float32 — the runtime converts on I/O) this is a no-op.
    """
    fb_type = _resolve_dtype(dtype)["fb_type"]
    fp16_type = DTYPES["float16"]["fb_type"]
    if fb_type == fp16_type:
        return
    for tp in _fb_tensor_positions(body):
        ab = _fb_field_abs(body, tp, 0)
        if ab is not None and body[ab] == fp16_type:
            body[ab] = fb_type

# ── RKNN Graph Builder ──────────────────────────────────────────────────
"""Build a node-per-op "uop_graph" .rknn from a small op list (toolkit-free).

This is the CPU-graph counterpart to `_rknn_flatbuf.build_body`: instead of baking
an NPU register-command stream for fp16 element-wise ops, it emits a FlatBuffer node
table (one RKNN node per op) plus a JSON trailer flagged `uop_graph`. The custom
runtime (`rknn_runtime.RKNNRuntime`) walks that graph node-by-node, running each op on
its CPU kernels (or on the NPU via a companion fp16 EW model). This is the path used
for dtypes the NPU element-wise tiler can't service today (int32, etc.), which the
runtime executes on the CPU inside the rknn runtime.

Mirrors `/data/rk3588/rknn-creation/onnx_to_rknn.py`'s flatbuffer writer, but is fed
from an explicit (op, ins, outs) spec rather than an ONNX graph -- no onnx dependency.
"""
import json
import struct

import flatbuffers

HEADER_SIZE = 0x40


def _str(b, s): return b.CreateString(s)


def _vec(b, vs):
  b.StartVector(4, len(vs), 4)
  for v in reversed(vs): b.PrependUint32(v)
  return b.EndVector()


def _ovec(b, offsets):
  b.StartVector(4, len(offsets), 4)
  for o in reversed(offsets): b.PrependUOffsetTRelative(o)
  return b.EndVector()


def _tensor(b, name, shape):
  nm = _str(b, name); f3 = _vec(b, shape); f4 = _vec(b, shape)
  b.StartObject(18)
  b.PrependUOffsetTRelativeSlot(5, nm, 0)
  b.PrependUOffsetTRelativeSlot(4, f4, 0)
  b.PrependUOffsetTRelativeSlot(3, f3, 0)
  return b.EndObject()


def _node(b, op, name, ins, outs):
  op_s = _str(b, op); nm_s = _str(b, name); f4 = _vec(b, ins); f5 = _vec(b, outs)
  b.StartObject(13)
  b.PrependUOffsetTRelativeSlot(5, f5, 0)
  b.PrependUOffsetTRelativeSlot(4, f4, 0)
  b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
  b.PrependUOffsetTRelativeSlot(1, op_s, 0)
  return b.EndObject()


def build_graph_rknn(tensor_names, tensor_shapes, node_specs, inputs, outputs, output_shapes,
                     consts=None, node_attrs=None, npu_ew_model=None, dtype_tag=None) -> bytes:
  """Assemble a uop_graph .rknn.

  tensor_names: ordered list of value names (index == tensor id used by node_specs)
  tensor_shapes: {name: [dims]}
  node_specs:   list of (op_type, node_name, [in_ids], [out_ids])
  inputs/outputs: ordered value names bound at runtime / read back
  output_shapes: {name: [dims]}
  consts:       {tensor_id(str): {"dtype": np-dtype-str, "data": [...]}}
  node_attrs:   {node_index(str): {attr: val}}
  """
  b = flatbuffers.Builder(1 << 20)
  toffs = [_tensor(b, nm, tensor_shapes[nm]) for nm in tensor_names]
  noffs = [_node(b, *spec) for spec in node_specs]
  tvec, nvec = _ovec(b, toffs), _ovec(b, noffs)
  b.StartObject(16)
  b.PrependUOffsetTRelativeSlot(1, nvec, 0)
  b.PrependUOffsetTRelativeSlot(0, tvec, 0)
  sg = b.EndObject()
  sgs = _ovec(b, [sg])
  b.StartObject(22)
  b.PrependUOffsetTRelativeSlot(2, sgs, 0)
  root = b.EndObject()
  b.Finish(root)
  body = bytes(b.Output())

  trailer = {
    "uop_graph": 1,
    "npu_ew_model": npu_ew_model,
    "consts": consts or {},
    "node_attrs": node_attrs or {},
    "inputs": list(inputs),
    "outputs": list(outputs),
    "output_shapes": output_shapes,
    # I/O dtype signature: keeps otherwise-identical graphs that differ only in dtype
    # (e.g. float32 vs fp16 matmul of the same shape) byte-distinct, so the compile cache
    # does not return the wrong-dtype program for them.
    "dtype_tag": dtype_tag,
  }
  tj = json.dumps(trailer).encode()
  hdr = bytearray(HEADER_SIZE)
  hdr[0:4] = b"RKNN"
  struct.pack_into("<Q", hdr, 0x08, 6)
  struct.pack_into("<Q", hdr, 0x10, len(body))
  return bytes(hdr) + body + struct.pack("<Q", len(tj)) + tj


def elementwise_graph_rknn(op_name: str, N: int, n_inputs: int = 2) -> bytes:
  """A single element-wise op over N elements, n_inputs operands, run on the CPU.

  Graph: InputOperator x n_inputs -> <op chain> -> OutputOperator. For >2 inputs the op
  is applied left-associatively (in0 OP in1 OP in2 ...), matching the EW chain semantics.
  """
  shape = [1, N]
  names = [f"in{i}" for i in range(n_inputs)]
  shapes = {nm: shape for nm in names}
  specs = [("InputOperator", f"InputOperator:in{i}", [], [i]) for i in range(n_inputs)]
  # chain of binary ops producing intermediates n_inputs, n_inputs+1, ...
  acc = 0
  next_id = n_inputs
  for i in range(1, n_inputs):
    name = f"out{i}" if i < n_inputs - 1 else "out"
    shapes[name] = shape
    names.append(name)
    specs.append((op_name, f"{op_name}_{i}", [acc, i], [next_id]))
    acc = next_id
    next_id += 1
  out_name = names[-1]
  specs.append(("OutputOperator", f"OutputOperator:{out_name}", [acc], []))   # bind the graph output
  return build_graph_rknn(names, shapes, specs, [f"in{i}" for i in range(n_inputs)],
                          [out_name], {out_name: shape})

# ── UOp Synthesizer ─────────────────────────────────────────────────────
"""Synthesize a runnable .rknn from tinygrad UOps (toolkit-free).

This is the tinygrad-side port of rknn-decode's `helpers.rknn_synth`: it reads the
fully-lowered element-wise UOp graph that reaches `RKRenderer.render`, recovers the
element count N and the op, and emits the .rknn (FlatBuffer body + NPU
register-command stream + container) from scratch via the vendored builders
(`_rknn_flatbuf` + `_rc_template_gen`, which depend only on `flatbuffers`).

Scope: an element-wise op `z = a OP b [OP c ...]` (OP in Add/Sub/Mul/Div) over N elements,
in either the scalar full-unroll form or tinygrad's vectorized UPCAST form. Two-input fp16/
int8 ops bake NPU register commands; everything else (more inputs, int32, ...) emits an
rknn CPU-op graph. Non-element-wise kernels (matmul, rand, bitops) are rejected at render.
"""





_NPU_EW_FB_DTYPE = {"float16": "float16", "float32": "float32", "int8": "int8", "bool": "bool"}


def _index_param(u:UOp):
  if u.op is Ops.CAST: u = u.src[0]
  if u.op is not Ops.INDEX or u.src[0].op is not Ops.PARAM:
    raise ValueError("expected (CAST of) INDEX(PARAM, ...)")
  return u.src[0]


def _is_neg(u:UOp):
  if u.op is Ops.NEG: return u.src[0]
  if u.op is Ops.MUL and len(u.src) == 2:
    for i in (0, 1):
      if u.src[i].op is Ops.CONST and u.src[i].arg == -1: return u.src[1 - i]
  return None


def _operand_param(u:UOp):
  # peel dtype CASTs at every level: a leaf can be (CAST*)(GEP of)?(LOAD of)?(CAST*) INDEX(PARAM),
  # e.g. CAST(GEP(LOAD(CAST(INDEX)))) when float32 operands are cast to fp16 inside a vectorized store.
  u = _peel_cast(u)
  if u.op is Ops.GEP: u = _peel_cast(u.src[0])
  if u.op is Ops.LOAD: u = u.src[0]
  return _index_param(u)


def _resolve_op(alu:UOp):
  if alu.op is Ops.ADD and len(alu.src) == 2:
    for i in (0, 1):
      if (other := _is_neg(alu.src[i])) is not None: return "Sub", (alu.src[1 - i], other)
    return "Add", (alu.src[0], alu.src[1])
  if alu.op is Ops.MUL and len(alu.src) == 2:
    for i in (0, 1):
      if alu.src[i].op is Ops.RECIPROCAL: return "Div", (alu.src[1 - i], alu.src[i].src[0])
    return "Mul", (alu.src[0], alu.src[1])
  if alu.op is Ops.FDIV and len(alu.src) == 2: return "Div", (alu.src[0], alu.src[1])
  if alu.op is Ops.SUB and len(alu.src) == 2: return "Sub", (alu.src[0], alu.src[1])
  raise ValueError(f"unsupported element-wise op {alu.op}")


def _np_dtype_name(dt) -> str:
  return str(_to_np_dtype(dt.scalar()).__name__)


def _flatten_ew(alu:UOp, op_name:str):
  # Flatten associative Add/Mul chains (a OP b OP c ...) into ordered leaf operands so the
  # rknn graph path can carry >2 inputs. Sub/Div stay binary (left-associative).
  raw = {"Add": Ops.ADD, "Mul": Ops.MUL}.get(op_name)
  if raw is None: return list(_resolve_op(alu)[1])
  leaves:list[UOp] = []
  def walk(u:UOp):
    if u.op is raw and len(u.src) == 2 and _is_neg(u) is None and not any(s.op is Ops.RECIPROCAL for s in u.src):
      for s in u.src: walk(s)
    else: leaves.append(u)
  walk(alu)
  return leaves


def analyze_elementwise(uops:list[UOp]):
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores: raise ValueError("no STORE in uops")
  out_params, in_params, ops = set(), [], set()
  for st in stores:
    out = _index_param(st.src[0])
    out_params.add(out.arg)
    val = st.src[1]
    # peel an output-dtype CAST (e.g. float32 operands -> fp16 store) and the STACK of vectorized lanes
    alu = _peel_cast(val.src[0] if val.op is Ops.STACK else val)
    op_name, _ = _resolve_op(alu)
    ops.add(op_name)
    for operand in _flatten_ew(alu, op_name):
      p = _operand_param(operand).arg
      if p not in in_params: in_params.append(p)
  if len(ops) != 1: raise ValueError(f"all stores must use the same op, got {ops}")
  if len(out_params) != 1: raise ValueError(f"expected one output PARAM, got {out_params}")
  if len(in_params) < 2: raise ValueError(f"expected at least two input PARAMs, got {in_params}")
  out_param = next(u for u in uops if u.op is Ops.PARAM and u.arg == next(iter(out_params)))
  N = out_param.dtype.size
  if not isinstance(N, int) or N < 1: raise ValueError(f"could not recover element count (ptr size={N})")
  return N, next(iter(ops)), _np_dtype_name(out_param.dtype), len(in_params)


def uops_to_rknn(uops:list[UOp]) -> bytes:
  N, op_name, dtype, n_inputs = analyze_elementwise(uops)
  if n_inputs == 2 and dtype in _NPU_EW_FB_DTYPE:
    body = build_body(N, 2, ops=[op_name], dtype=_NPU_EW_FB_DTYPE[dtype])
    return bytes(assemble_rknn(body, 1, N, 2))
  return elementwise_graph_rknn(op_name, N, n_inputs)


def _unroll_uops(uops:list[UOp]) -> list[UOp]:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if not ranges: return uops
  for r in ranges:
    if r.src[0].op is not Ops.CONST: raise ValueError(f"cannot unroll non-constant range bound {r.src[0]}")
  sink = next(u for u in uops if u.op is Ops.SINK)
  stores = [u for u in uops if u.op is Ops.STORE]
  per_range = [[(r, UOp.const(dtypes.int, i)) for i in range(int(r.src[0].arg))] for r in ranges]
  new_stores = [st.substitute(dict(combo)).simplify() for combo in itertools.product(*per_range) for st in stores]
  return list(UOp.sink(*new_stores, arg=sink.arg).toposort())


def _extract_reduce_groups(unrolled):
  from collections import defaultdict
  groups = defaultdict(list)
  for u in unrolled:
    if u.op is not Ops.STORE: continue
    idx_node = u.src[0]
    red_node = u.src[1]
    if red_node.op is not Ops.REDUCE: return None
    mul_node = red_node.src[0]
    if mul_node.op is not Ops.MUL: return None
    try:
      out_pos = idx_node.src[1].arg
      a_idx_node = mul_node.src[0]
      b_idx_node = mul_node.src[1]
      if a_idx_node.op is not Ops.INDEX or b_idx_node.op is not Ops.INDEX: return None
      a_pos = a_idx_node.src[1].arg
      b_pos = b_idx_node.src[1].arg
      a_param_id = a_idx_node.src[0].arg
      b_param_id = b_idx_node.src[0].arg
      groups[out_pos].append((a_param_id, a_pos, b_param_id, b_pos))
    except (AttributeError, IndexError):
      return None
  return groups


def _peel_cast(u:UOp):
  while u.op is Ops.CAST: u = u.src[0]
  return u


def _peel_end(u:UOp):
  # an AFTER's ordering dep may be the producing STORE directly or wrapped in loop-END markers.
  while u.op is Ops.END: u = u.src[0]
  return u


def _dtype_tag(uops) -> str:
  """I/O dtype signature (param dtypes ordered by arg) used to keep dtype-distinct graphs byte-distinct."""
  return ",".join(_np_dtype_name(u.dtype) for u in sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u:u.arg))


def _unrolled_leaf_ref(u:UOp):
  """Resolve a value-tree leaf in a fully-unrolled graph to (param_id, flat_index).

  Leaves are (CAST*)(GEP of)? (LOAD of)? (CAST*) INDEX(PARAM, CONST); the constant
  index is the flat element offset, the GEP lane (if any) adds to it (vectorized load).
  """
  u = _peel_cast(u)
  off = 0
  if u.op is Ops.GEP:
    if len(u.arg) != 1: raise ValueError("multi-element GEP leaf")
    off = int(u.arg[0]); u = _peel_cast(u.src[0])
  if u.op is Ops.LOAD: u = _peel_cast(u.src[0])
  if u.op is not Ops.INDEX or u.src[0].op is not Ops.PARAM:
    raise ValueError(f"leaf is not INDEX(PARAM), got {u.op.name}")
  if u.src[1].op is not Ops.CONST: raise ValueError("non-constant index in unrolled leaf")
  return u.src[0].arg, int(u.src[1].arg) + off


def _unrolled_terms(u:UOp):
  """Flatten a CAST/ADD-wrapped tree of MUL(a,b) products into (a_param,a_idx,b_param,b_idx)."""
  u = _peel_cast(u)
  if u.op is Ops.ADD: return _unrolled_terms(u.src[0]) + _unrolled_terms(u.src[1])
  if u.op is Ops.MUL:
    a, b = _unrolled_leaf_ref(u.src[0]), _unrolled_leaf_ref(u.src[1])
    return [(a[0], a[1], b[0], b[1])]
  raise ValueError(f"value node is not an add-of-mul reduce, got {u.op.name}")


def _extract_unrolled_matmul(unrolled):
  """Extract gather-reduce groups from an already-lowered matmul (REDUCE expanded to ADD/MUL)."""
  from collections import defaultdict
  groups = defaultdict(list)
  for u in unrolled:
    if u.op is not Ops.STORE: continue
    idx_node = _peel_cast(u.src[0])
    if idx_node.op is not Ops.INDEX or idx_node.src[1].op is not Ops.CONST: return None
    try:
      groups[int(idx_node.src[1].arg)] = _unrolled_terms(u.src[1])
    except (ValueError, AttributeError, IndexError):
      return None
  ks = {len(v) for v in groups.values()}
  return groups if (groups and len(ks) == 1) else None


def _symbolic_reduce_groups(unrolled):
  """Symbolically interpret a fully-unrolled multiply-accumulate program into gather groups.

  Handles both the add-tree reduce (matmul, small conv) and the DEFINE_REG accumulator
  loop (conv with many outputs): each value is interpreted as a sum-of-products list of
  (a_param, a_idx, b_param, b_idx); accumulator registers hold partial sums and output
  PARAM stores capture the final term list per output element. Returns groups or None.
  """
  groups = {}

  def ev(node):
    # interpret a value node as a sum-of-products term list, following the register
    # dataflow: a LOAD of a register reads its producing STORE's value (linked via AFTER).
    node = _peel_cast(node)
    if node.op is Ops.CONST:
      if node.arg != 0: raise ValueError("non-zero const in reduce value")
      return []                                              # additive identity (acc init / uninit reg)
    if node.op is Ops.ADD: return ev(node.src[0]) + ev(node.src[1])
    if node.op is Ops.MUL:
      a, b = _unrolled_leaf_ref(node.src[0]), _unrolled_leaf_ref(node.src[1])
      return [(a[0], a[1], b[0], b[1])]
    if node.op is Ops.LOAD:
      idxn = _peel_cast(node.src[0])
      if idxn.op is not Ops.INDEX: raise ValueError("bad load index")
      base = idxn.src[0]
      if base.op is Ops.AFTER:
        producer = next((s for s in (_peel_end(d) for d in base.src[1:]) if s is not None and s.op is Ops.STORE), None)
        if producer is None: raise ValueError("AFTER without producer store")
        return ev(producer.src[1])                           # value written by the producing store
      if base.op is Ops.DEFINE_REG: return []                # uninitialized register reads as 0
      raise ValueError("param load in additive position")
    raise ValueError(f"cannot interpret {node.op.name} in reduce value")

  try:
    for u in unrolled:
      if u.op is not Ops.STORE: continue
      tgt = _peel_cast(u.src[0])
      if tgt.op is not Ops.INDEX or tgt.src[1].op is not Ops.CONST: continue
      base = tgt.src[0]
      while base.op is Ops.AFTER: base = base.src[0]
      if base.op is Ops.PARAM and base.arg == 0:             # final output element store
        if u.src[1].op is Ops.STACK: return None             # vectorized output unsupported here
        groups[int(tgt.src[1].arg)] = ev(u.src[1])
  except (ValueError, AttributeError, IndexError, RecursionError):
    return None
  ks = {len(v) for v in groups.values()}
  return groups if (groups and len(ks) == 1 and 0 not in ks) else None


def _reduce_graph_rknn(groups, out_param, params_info, dtype_tag=None):
  n_out = len(groups)
  sample = next(iter(groups.values()))
  K = len(sample)
  a_param_id = sample[0][0]
  b_param_id = sample[0][2]
  a_size = params_info[a_param_id]
  b_size = params_info[b_param_id]
  # reject patterns we can't express as a plain gather (e.g. masked/padded accesses with
  # out-of-range indices, or inconsistent operand params) so render fails cleanly here.
  for terms in groups.values():
    for (ap, ai, bp, bi) in terms:
      if ap != a_param_id or bp != b_param_id or not (0 <= ai < a_size and 0 <= bi < b_size):
        raise ValueError("reduce pattern not expressible as a plain gather (masked/padded or mixed params)")

  shape = [n_out]
  tid = 0
  tensor_names = []
  tensor_shapes = {}
  node_specs = []
  consts = {}

  def add_tensor(name, sh):
    nonlocal tid
    tensor_names.append(name)
    tensor_shapes[name] = sh
    cur = tid
    tid += 1
    return cur

  a_id = add_tensor("A", [a_size])
  b_id = add_tensor("B", [b_size])
  out_id = add_tensor("out", shape)
  node_specs += [("InputOperator", "InputOperator:A", [], [a_id]),
                 ("InputOperator", "InputOperator:B", [], [b_id])]

  prod_ids = []
  for k in range(K):
    a_indices = [groups[o][k][1] for o in range(n_out)]
    b_indices = [groups[o][k][3] for o in range(n_out)]
    a_idx_id = add_tensor(f"a_idx{k}", shape)
    b_idx_id = add_tensor(f"b_idx{k}", shape)
    ag_id = add_tensor(f"a_g{k}", shape)
    bg_id = add_tensor(f"b_g{k}", shape)
    consts[str(a_idx_id)] = {"dtype": "int64", "data": a_indices}
    consts[str(b_idx_id)] = {"dtype": "int64", "data": b_indices}
    node_specs.append(("Gather", f"a_gather_{k}", [a_id, a_idx_id], [ag_id]))
    node_specs.append(("Gather", f"b_gather_{k}", [b_id, b_idx_id], [bg_id]))
    p_id = add_tensor(f"p{k}", shape)
    prod_ids.append(p_id)
    node_specs.append(("Mul", f"mul_{k}", [ag_id, bg_id], [p_id]))

  acc_id = prod_ids[0]
  for k in range(1, K):
    sum_id = add_tensor(f"s{k}", shape) if k < K - 1 else out_id
    node_specs.append(("Add", f"add_{k}", [acc_id, prod_ids[k]], [sum_id]))
    acc_id = sum_id
  if K == 1:
    node_specs.append(("Identity", "copy", [prod_ids[0]], [out_id]))
  node_specs.append(("OutputOperator", "OutputOperator:out", [out_id], []))

  return build_graph_rknn(tensor_names, tensor_shapes, node_specs, ["A", "B"],
                          ["out"], {"out": shape}, consts=consts, dtype_tag=dtype_tag)


def _eval_index(u, env):
  """Evaluate an integer index expression over a range environment {range_uop: value}."""
  if u.op is Ops.CONST: return int(u.arg)
  if u.op is Ops.RANGE: return env[u]
  if u.op is Ops.ADD: return _eval_index(u.src[0], env) + _eval_index(u.src[1], env)
  if u.op is Ops.MUL: return _eval_index(u.src[0], env) * _eval_index(u.src[1], env)
  if u.op is Ops.IDIV: return _eval_index(u.src[0], env) // _eval_index(u.src[1], env)
  if u.op is Ops.MOD: return _eval_index(u.src[0], env) % _eval_index(u.src[1], env)
  if u.op is Ops.CAST: return _eval_index(u.src[0], env)
  raise ValueError(f"cannot evaluate index op {u.op.name}")


def _operand_ref(u):
  """Resolve a multiplicand to (param_id, index_expr_uop, lane_offset)."""
  u = _peel_cast(u)
  lane = 0
  if u.op is Ops.GEP:
    if len(u.arg) != 1: raise ValueError("multi-element GEP")
    lane = int(u.arg[0]); u = _peel_cast(u.src[0])
  if u.op is Ops.LOAD: u = _peel_cast(u.src[0])
  if u.op is not Ops.INDEX or u.src[0].op is not Ops.PARAM: raise ValueError("operand not INDEX(PARAM)")
  return u.src[0].arg, u.src[1], lane


def _extract_range_groups(uops):
  """Iterate the range-based loop nest, accumulating out[idx] += a[..]*b[..] into gather groups.

  Works directly on the original (non-unrolled) graph, so it captures reduction loops
  (matmul, conv with a DEFINE_REG accumulator) without collapsing the sum. Returns
  (groups, params_info) or None.
  """
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if not ranges: return None
  # the reduction products: float MULs whose two operands both load from input PARAMs (arg != 0)
  products = []
  for u in uops:
    if u.op is not Ops.MUL or u.dtype.count != 1 or u.dtype.scalar() not in (dtypes.float16, dtypes.float32): continue
    try:
      a, b = _operand_ref(u.src[0]), _operand_ref(u.src[1])
    except ValueError:
      continue
    if a[0] != 0 and b[0] != 0: products.append((a, b))
  out_stores = []
  for u in uops:
    if u.op is not Ops.STORE: continue
    tgt = _peel_cast(u.src[0])
    if tgt.op is not Ops.INDEX: continue
    base = tgt.src[0]
    while base.op is Ops.AFTER: base = base.src[0]
    if base.op is Ops.PARAM and base.arg == 0: out_stores.append(tgt.src[1])
  if not products or len(out_stores) != 1: return None
  out_idx_expr = out_stores[0]
  params = {u.arg: u.dtype.size for u in uops if u.op is Ops.PARAM}
  from collections import defaultdict
  groups = defaultdict(list)
  try:
    for vals in itertools.product(*[range(int(r.src[0].arg)) for r in ranges]):
      env = dict(zip(ranges, vals))
      out_pos = _eval_index(out_idx_expr, env)
      for (ap, ax, al), (bp, bx, bl) in products:
        ai, bi = _eval_index(ax, env) + al, _eval_index(bx, env) + bl
        # masked (padded) accesses produce out-of-range indices we can't model as a plain gather
        if not (0 <= out_pos < params[0] and 0 <= ai < params[ap] and 0 <= bi < params[bp]): return None
        groups[out_pos].append((ap, ai, bp, bi))
  except (ValueError, KeyError):
    return None
  ks = {len(v) for v in groups.values()}
  if not groups or len(ks) != 1 or 0 in ks: return None
  return dict(groups), params


def uops_to_reduce_rknn(uops:list[UOp]) -> bytes:
  # range-based loop nests (matmul / conv with a reduction accumulator) are read directly off
  # the original graph; fully-unrolled add-tree reduces are read off the unrolled graph.
  rg = _extract_range_groups(uops)
  if rg is not None:
    groups, params = rg
    return _reduce_graph_rknn(groups, 0, params, dtype_tag=_dtype_tag(uops))
  unrolled = _unroll_uops(uops)
  groups = _extract_reduce_groups(unrolled)
  if groups is None:
    groups = _extract_unrolled_matmul(unrolled)
  if groups is None:
    groups = _symbolic_reduce_groups(unrolled)
  if groups is None:
    raise ValueError("unrolled uops are not a pure reduce(mul(a,b)) pattern")
  params = {}
  for u in unrolled:
    if u.op is Ops.PARAM:
      params[u.arg] = u.dtype.size
  return _reduce_graph_rknn(groups, 0, params, dtype_tag=_dtype_tag(unrolled))


def uops_to_linear_rknn(uops:list[UOp]) -> bytes:
  """Handle already-linearized matmul uops: explicit MUL+ADD tree with RANGE/LOAD/GEP."""
  params = {u.arg: u for u in uops if u.op is Ops.PARAM}
  if len(params) != 3: raise ValueError(f"expected 3 params, got {len(params)}")
  from collections import defaultdict
  groups = defaultdict(list)

  def eval_int(u, env):
    if u.op is Ops.CONST: return int(u.arg)
    if u.op is Ops.RANGE: return env[u]
    if u.op is Ops.ADD: return eval_int(u.src[0], env) + eval_int(u.src[1], env)
    if u.op is Ops.MUL: return eval_int(u.src[0], env) * eval_int(u.src[1], env)
    raise ValueError(f"unsupported index op {u.op}")

  def load_ref(u, env):
    if u.op is Ops.GEP:
      if len(u.arg) != 1 or u.src[0].op is not Ops.LOAD: raise ValueError("unsupported GEP")
      idx = u.src[0].src[0]
      if idx.op is Ops.CAST: idx = idx.src[0]
      if idx.op is not Ops.INDEX: raise ValueError("GEP load is not from INDEX")
      return idx.src[0].arg, eval_int(idx.src[1], env) + int(u.arg[0])
    if u.op is Ops.LOAD:
      idx = u.src[0]
      if idx.op is Ops.CAST: idx = idx.src[0]
      if idx.op is not Ops.INDEX: raise ValueError("LOAD is not from INDEX")
      return idx.src[0].arg, eval_int(idx.src[1], env)
    if u.op is Ops.INDEX:
      return u.src[0].arg, eval_int(u.src[1], env)
    raise ValueError(f"not a load/index ref: {u.op}")

  def terms(u, env):
    if u.op is Ops.ADD: return terms(u.src[0], env) + terms(u.src[1], env)
    if u.op is Ops.MUL:
      a, b = load_ref(u.src[0], env), load_ref(u.src[1], env)
      return [(a[0], a[1], b[0], b[1])]
    raise ValueError(f"unsupported value op {u.op}")

  ranges = [u for u in uops if u.op is Ops.RANGE]
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1: raise ValueError(f"expected one STORE, got {len(stores)}")
  store = stores[0]
  extents = [int(r.src[0].arg) for r in ranges]
  for vals in itertools.product(*[range(e) for e in extents]):
    env = dict(zip(ranges, vals))
    out_idx = load_ref(store.src[0], env)[1]
    groups[out_idx] = terms(store.src[1], env)

  if not groups: raise ValueError("could not extract gather pattern from linearized uops")
  params_info = {arg: dt.dtype.size for arg, dt in params.items()}
  return _reduce_graph_rknn(groups, 0, params_info, dtype_tag=_dtype_tag(uops))


class RKRenderer(Renderer):
    compiler = _Compiler()
    has_local = False
    device = "RK"
    def render(self, uops:list[UOp]) -> bytes:
        out_params = [u for u in uops if u.op is Ops.PARAM and u.arg == 0]
        # per-input param dtypes (ordered by PARAM arg) so the runtime reads each input buffer
        # at its own width; inputs may differ from the output dtype (e.g. float32 in, fp16 matmul out).
        in_dtypes = [_np_dtype_name(p.dtype) for p in sorted((u for u in uops if u.op is Ops.PARAM and u.arg != 0), key=lambda u:u.arg)]
        meta = {"kind":"rknn", "in_dtypes":in_dtypes}
        if out_params: meta["dtype"] = _np_dtype_name(out_params[0].dtype)
        try:
            N, op_name, dtype, _ = analyze_elementwise(uops)
            return _Blob(uops_to_rknn(uops), {"kind":"rknn", "N":N, "op":op_name, "dtype":dtype, "in_dtypes":in_dtypes})
        except Exception: pass
        try:
            return _Blob(uops_to_reduce_rknn(uops), meta)
        except Exception: pass
        try:
            return _Blob(uops_to_linear_rknn(uops), meta)
        except Exception: pass
        raise RuntimeError(f"RKRenderer: cannot render {len(uops)}-uop kernel "
                           f"(ops: {sorted(set(u.op.name for u in uops))})")

_WORKDIR = "/dev/shm" if os.path.isdir("/dev/shm") else None

def _run_once(blob:bytes, inputs:list, want_n:int):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".rknn", dir=_WORKDIR)
    with os.fdopen(fd, "wb") as f: f.write(bytes(blob))
    try:
        r = RKNNRuntime(path)
        try:
            r.inputs_set(inputs)
            r.run()
            return np.asarray(r.outputs_get()[0]).ravel()[:want_n].copy()
        finally: r.destroy()
    finally: os.unlink(path)

class RKProgram:
    def __init__(self, device:str, name:str, lib:bytes, *args, **kwargs):
        self.device, self.name, self.lib = device, name, lib
        self.meta = getattr(lib, "meta", None) or {"kind":"rknn"}
    def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait=False, **kw):
        with cpu_profile(self.name, self.device):
            self._run_rknn(bufs)
        return 1e-3 if wait else None

    def _run_rknn(self, bufs):
        dtype = np.dtype(self.meta.get("dtype", "float16"))
        in_dtypes = self.meta.get("in_dtypes") or []
        out_buf, in_bufs = bufs[0], bufs[1:]
        ins = [np.frombuffer(bytes(b), dtype=np.dtype(in_dtypes[i]) if i < len(in_dtypes) else dtype).copy()
               for i, b in enumerate(in_bufs)]
        n_out = len(out_buf) // dtype.itemsize
        got = _run_once(self.lib, ins, n_out).astype(dtype)
        res = np.ascontiguousarray(got)
        ctypes.memmove(_addr(out_buf), _addr(res), min(len(out_buf), res.nbytes))

class RKAllocator(Allocator['RKDevice']):
    def _alloc(self, size, options): return bytearray(size)
    def _copyin(self, dest, src:memoryview): ctypes.memmove(_addr(dest), mv_address(src), src.nbytes)
    def _copyout(self, dest:memoryview, src): ctypes.memmove(mv_address(dest), _addr(src), dest.nbytes)
    def _offset(self, buf, size:int, offset:int): return memoryview(buf)[offset:offset+size]

class RKDevice(Compiled):
    def __init__(self, device:str): super().__init__(device, RKAllocator(self), [RKRenderer], functools.partial(RKProgram, device))
