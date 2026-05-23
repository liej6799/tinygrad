# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING, cast
import pickle, base64, itertools, time, struct, sys, functools, array, ctypes, mmap, os, math, numpy as np
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate, float_to_fp16, float_to_bf16, float_to_fp8, fp8_to_float
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, EMULATE, mv_address, to_mv, DEBUG
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair, BufferSpec
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_python import storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar, _load, load, _store, generic_wmma_helper

def _rk_env(name:str, default:int) -> int:
  try: return int(os.getenv(name, str(default)))
  except ValueError: return default

class RockchipProgram:
  def __init__(self, dev:'RockchipDevice', name:str, lib:bytes):
    self.uops: list[tuple[Ops, DType, list[int], Any]] = pickle.loads(lib)
    self.device = dev
    self.q = []
    self.hardware_ops = {Ops.WMMA:0, Ops.TRUNC:0, Ops.CUSTOM:0, Ops.MUL:0, Ops.NEG:0, Ops.MAX:0, Ops.EXP2:0, Ops.CMPLT:0, Ops.CMPEQ:0, Ops.ADD:2, Ops.FDIV:3, Ops.SUB:4}
    self.cmd_buf_size = 16384
    self.exp2_inv_scale = 1.0
    self.lut_size = 513
    self.fused_matmul_meta = self._parse_fused_matmul_name(name)
    self.fused_matmul_hits = 0
    self.fused_matmul_fallbacks = 0

  def _parse_fused_matmul_name(self, name:str) -> dict[str, int]|None:
    if not name.startswith("rkmm_v1_"): return None
    toks = name.split("_")
    if len(toks) != 23: return None
    try:
      vals = [int(x) for x in toks[2:]]
    except ValueError:
      return None
    keys = ["m", "n", "k", "batch", "a_bs", "b_bs", "c_bs", "a_ms", "a_ks", "b_ks", "b_ns", "c_ms", "c_ns",
            "a_slot", "b_slot", "c_slot", "ta", "tb", "a_dt", "b_dt", "c_dt"]
    if any(v < 0 for v in vals): return None
    return dict(zip(keys, vals))

  def _dtype_from_code(self, code:int):
    if code == 0: return np.float16
    if code == 1: return np.float32
    raise RuntimeError(f"dtype_code_{code}")

  def _run_wmma_matmul(self, a_matrix:np.ndarray, b_matrix:np.ndarray) -> np.ndarray:
    m, k = a_matrix.shape
    if b_matrix.shape[0] != k: raise RuntimeError("k_mismatch")
    n = int(b_matrix.shape[1])
    wmma_meta = self._wmma_params(int(m), int(n), int(k))
    in_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["m"], dtype=np.float16)
    wt_pack = np.zeros(wmma_meta["align_out"] * wmma_meta["align_in"], dtype=np.float16)
    if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) == (64, 64, 64):
      for mm in range(1, 65):
        for kk in range(1, 65):
          plane = (kk - 1) // 8
          offset = (kk - 1) % 8
          in_pack[plane * 64 * 8 + (mm - 1) * 8 + offset] = a_matrix[mm - 1, kk - 1]
      for nn in range(1, 65):
        for kk in range(1, 65):
          kpg, cpg = (nn - 1) // 16, (kk - 1) // 32
          wt_idx = ((cpg * 32) * 16) + (kpg * 16 * wmma_meta["align_in"]) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32)
          wt_pack[wt_idx] = b_matrix[kk - 1, nn - 1]
    else:
      in_pack = in_pack.reshape(wmma_meta["m"], wmma_meta["align_in"])
      wt_pack = wt_pack.reshape(wmma_meta["align_out"], wmma_meta["align_in"])
      in_pack[:, :wmma_meta["k"]] = a_matrix
      wt_pack[:wmma_meta["n"], :wmma_meta["k"]] = b_matrix.T
      in_pack = in_pack.reshape(-1)
      wt_pack = wt_pack.reshape(-1)
    src = memoryview(bytearray(in_pack.tobytes()))
    src2 = memoryview(bytearray(wt_pack.tobytes()))
    self.task_buf = self.device._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
    self.cmd_buf = self.device._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
    self.input_buf = self.device._gpu_alloc(src.nbytes, 0, name="input")
    self.weight_buf = self.device._gpu_alloc(src2.nbytes, 0, name="weight")
    out_stride = wmma_meta["align_out"] * dtypes.float32.itemsize
    out_nbytes = max(0x100, (wmma_meta["m"]-1)*out_stride + wmma_meta["n"]*dtypes.float32.itemsize)
    self.output_buf = self.device._gpu_alloc(out_nbytes, 0, name="output")
    try:
      ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
      ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
      self.device._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.device._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.q, self.lut_enable = [], False
      self.boilerplate(op=Ops.WMMA, size=int(m*k), arg=None,
        feature_addr=self.input_buf.meta.dma_addr, weight_addr=self.weight_buf.meta.dma_addr, dst_addr=self.output_buf.meta.dma_addr,
        wmma_meta=wmma_meta)
      self.submit(Ops.WMMA)
      self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      dst = memoryview(bytearray(self.output_buf.size))
      ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
      raw = np.frombuffer(dst.tobytes(), dtype=np.float32)
      out = np.empty((m, n), dtype=np.float32)
      if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) in {(64, 64, 64), (256, 256, 256)}:
        c2 = 4
        for col in range(n):
          plane, offset = col // c2, col % c2
          plane_base = plane * m * c2
          for row in range(m): out[row, col] = raw[plane_base + row * c2 + offset]
      else:
        stride = wmma_meta["align_out"]
        for row in range(m): out[row, :] = raw[row * stride:row * stride + n]
      return out
    finally:
      self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])

  def _run_fused_matmul(self, bufs:tuple[Any, ...]) -> None:
    if self.fused_matmul_meta is None: raise RuntimeError("missing_meta")
    m = self.fused_matmul_meta["m"]
    n = self.fused_matmul_meta["n"]
    k = self.fused_matmul_meta["k"]
    batch = self.fused_matmul_meta["batch"]
    a_slot, b_slot, c_slot = self.fused_matmul_meta["a_slot"], self.fused_matmul_meta["b_slot"], self.fused_matmul_meta["c_slot"]
    if max(a_slot, b_slot, c_slot) >= len(bufs): raise RuntimeError("slot_oob")
    a_arr = np.frombuffer(bufs[a_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["a_dt"]))
    b_arr = np.frombuffer(bufs[b_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["b_dt"]))
    c_arr = np.frombuffer(bufs[c_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["c_dt"]))
    if self.fused_matmul_meta["ta"] == 0 and (self.fused_matmul_meta["a_ms"], self.fused_matmul_meta["a_ks"]) != (k, 1):
      raise RuntimeError("lhs_layout")
    if self.fused_matmul_meta["ta"] == 1 and (self.fused_matmul_meta["a_ms"], self.fused_matmul_meta["a_ks"]) != (1, m):
      raise RuntimeError("lhs_layout")
    if self.fused_matmul_meta["tb"] == 0 and (self.fused_matmul_meta["b_ks"], self.fused_matmul_meta["b_ns"]) != (n, 1):
      raise RuntimeError("rhs_layout")
    if self.fused_matmul_meta["tb"] == 1 and (self.fused_matmul_meta["b_ks"], self.fused_matmul_meta["b_ns"]) != (1, k):
      raise RuntimeError("rhs_layout")
    if (self.fused_matmul_meta["c_ms"], self.fused_matmul_meta["c_ns"]) != (n, 1): raise RuntimeError("out_layout")
    if batch > 1 and self.fused_matmul_meta["a_bs"] != m*k: raise RuntimeError("lhs_batch_stride")
    if batch > 1 and self.fused_matmul_meta["b_bs"] != k*n: raise RuntimeError("rhs_batch_stride")
    if batch > 1 and self.fused_matmul_meta["c_bs"] != m*n: raise RuntimeError("out_batch_stride")
    if len(a_arr) < (batch-1)*self.fused_matmul_meta["a_bs"] + m*k: raise RuntimeError("lhs_buffer_too_small")
    if len(b_arr) < (batch-1)*self.fused_matmul_meta["b_bs"] + k*n: raise RuntimeError("rhs_buffer_too_small")
    if len(c_arr) < (batch-1)*self.fused_matmul_meta["c_bs"] + m*n: raise RuntimeError("out_buffer_too_small")

    for bidx in range(batch):
      a_base = bidx * self.fused_matmul_meta["a_bs"]
      b_base = bidx * self.fused_matmul_meta["b_bs"]
      c_base = bidx * self.fused_matmul_meta["c_bs"]
      a_block = a_arr[a_base:a_base + m*k]
      b_block = b_arr[b_base:b_base + k*n]
      if len(a_block) != m*k or len(b_block) != k*n: raise RuntimeError("batch_slice_oob")
      if self.fused_matmul_meta["ta"] == 0:
        a_matrix = a_block.reshape(m, k).astype(np.float16, copy=False)
      else:
        a_matrix = a_block.reshape(k, m).T.astype(np.float16, copy=False)
      if self.fused_matmul_meta["tb"] == 0:
        b_matrix = b_block.reshape(k, n).astype(np.float16, copy=False)
      else:
        b_matrix = b_block.reshape(n, k).T.astype(np.float16, copy=False)
      out_matrix = self._run_wmma_matmul(a_matrix, b_matrix)
      # Validate one dot-product entry to avoid silently returning bad fused outputs.
      ref00 = float(np.dot(a_matrix[0, :].astype(np.float32), b_matrix[:, 0].astype(np.float32)))
      got00 = float(out_matrix[0, 0])
      tol = max(1e-2, abs(ref00) * 5e-3)
      if not np.isfinite(got00) or abs(got00 - ref00) > tol: raise RuntimeError("npu_verify_mismatch")
      c_block = c_arr[c_base:c_base + m*n]
      if len(c_block) != m*n: raise RuntimeError("out_batch_slice_oob")
      if self.fused_matmul_meta["c_dt"] == 0: c_block[:] = out_matrix.astype(np.float16).reshape(-1)
      else: c_block[:] = out_matrix.reshape(-1)

  def check_lut_enable(self, op, arg):
    return op in (Ops.EXP2, Ops.TRUNC) or (op is Ops.CUSTOM and arg == "silu")
  def reg(self, val, shift, mask):
    return ((val) << shift) & mask
  def _align_up(self, val:int, align:int) -> int:
    if align <= 0: return val
    return ((val + align - 1) // align) * align
  def _wmma_params(self, m:int, n:int, k:int) -> dict[str, int]:
    m = max(1, m)
    n = max(1, n)
    k = max(1, k)
    align_in = max(32, self._align_up(k, 32))
    align_out = max(32, self._align_up(n, 32))
    data_in_width, data_in_height = 1, m
    dataout_width, dataout_height = 1, m
    out_width_stride = 1
    is_kn_64 = k == 64 and n == 64
    is_kn_256 = k == 256 and n == 256
    is_kn_512 = k == 512 and n == 512
    is_kn_lg_512 = k > 512 and n > 512
    is_matmul_64 = m == 64 and k == 64 and n == 64
    is_matmul_256 = m == 256 and k == 256 and n == 256
    feature_grains = data_in_height + 1
    if k > 7872:
      feature_grains = 2
    elif 128 < k <= 192:
      feature_grains = data_in_height
    elif k > 192 and k != 256:
      denom = align_in * dtypes.float16.itemsize
      grains = (2 * 32768 + denom - 1) // denom
      grains = (grains + 1) & ~1
      feature_grains = max(80, grains)
    weight_bytes_per_kernel = align_in * dtypes.float16.itemsize
    fd_bytes = data_in_width * data_in_height * align_in * dtypes.float16.itemsize
    data_bank = max(1, min(11, (fd_bytes + 32768 - 1) // 32768))
    line_stride = data_in_width * 4
    if 32 < k < 512 and k not in (64, 256):
      line_stride = min(13, (k + 31) // 32) * 4
    surf_groups = data_in_height // 4
    surf_stride = (line_stride * (surf_groups - 1) + int(surf_groups == 0)) * int(align_in >= 64)
    if (32 < k < 64) or (64 < k <= 128) or (128 < k < 256) or (256 < k < 512):
      surf_stride = 0
    dst_surf_stride = 64 if is_matmul_64 else (256 if is_matmul_256 else out_width_stride)
    notch_blocks = min(13, align_out // 32)
    notch_val = 8 * notch_blocks - 1
    if is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or k > 7872:
      notch_val = 0
    return {
      "m":m, "n":n, "k":k, "align_in":align_in, "align_out":align_out,
      "data_in_width":data_in_width, "data_in_height":data_in_height,
      "dataout_width":dataout_width, "dataout_height":dataout_height,
      "feature_grains":feature_grains, "weight_bytes_per_kernel":weight_bytes_per_kernel,
      "data_bank":data_bank, "line_stride":line_stride, "surf_stride":surf_stride,
      "dst_surf_stride":dst_surf_stride, "notch_val":notch_val,
    }
  def emit_raw(self, target, reg, value):
    # Pack the values into a 64-bit integer as per hardware spec
    target = target + 0x1
    packed_value = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF)
    self.q.append(packed_value)
  def fill_lut(self, lut):
    for table_id, base in ((0, 0), (1, self.lut_size)):
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
          self.reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK) |
          self.reg(table_id, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK) |
          self.reg(0, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK))
      for i in range(self.lut_size):
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
          self.reg(lut[base + i], rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))

  def boilerplate(self, op, size, arg, feature_addr=0, weight_addr=0, dst_addr=0, wmma_meta:dict[str, int]|None=None):
    if self.lut_enable:
      lut = [0] * self.lut_size * 2
      index_shift = 5
      index_scale = 0.0
      if op is Ops.EXP2:
        x_min, x_max = -2.0, 2.0
        step = (x_max - x_min) / (len(lut) - 1)
        index_scale = (1 << index_shift) / step
        
        max_val = max(math.exp2(x_min), math.exp2(x_max))
        self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for i in range(len(lut)):
          x = x_min + i * step
          y = math.exp2(x) * self.inv_scale
          q = int(math.floor((y + 1.0) * 2**14 + 0.5))
          lut[i] = np.clip(q, 0, 32767)
      elif op is Ops.CUSTOM and arg == "silu":
        x_min, x_max = 0, 5.8
        step = (x_max - x_min) / (self.lut_size - 1)
        index_scale = (1 << index_shift) / step

        max_val = max(x_min / (1.0 + math.exp(-x_min)), x_max / (1.0 + math.exp(-x_max)))
        self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for i in range(self.lut_size * 2):
          x = (i - self.lut_size + (i < self.lut_size)) * step
          y = x / (1.0 + math.exp(-x)) * self.inv_scale 
          q = int(math.floor(y * (2**15 - 1) + 0.5)) if y >= 0.0 else int(math.ceil(y * (2**15 - 1) - 0.5))
          lut[i] = np.clip(q, -32768, 32767)
      elif op is Ops.TRUNC:
        max_val = 1 << 14
        for table_id in range(2):
          base = table_id * self.lut_size
          for i in range(self.lut_size):
            lut[base + i] = 0 if (i % 2 == 0) else max_val
      bn_mul_operand = int(np.float16(index_scale).view(np.int16)) if index_scale!=0 else 0x3C00

      self.fill_lut(lut)
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_CFG,
          self.reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
          self.reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
          self.reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
      index_select = 14 if op is Ops.TRUNC else 5
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_INFO,
          self.reg(index_select, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
          self.reg(index_select, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
      if op is Ops.TRUNC:
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
            self.reg(0x00000000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_END,
            self.reg(0x44000000, rk.DPU_LUT_LE_END_LUT_LE_END__SHIFT, rk.DPU_LUT_LE_END_LUT_LE_END__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_START,
            self.reg(0x44000000, rk.DPU_LUT_LO_START_LUT_LO_START__SHIFT, rk.DPU_LUT_LO_START_LUT_LO_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
            self.reg(0x44800000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
      else:
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
            self.reg(0xffffc000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
            self.reg(0x00004000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
          self.reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT,
                  rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
          self.reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT,
                  rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
        self.reg(bn_mul_operand, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
      
    elif op is Ops.CUSTOM and arg == "cmplt_diff2bool":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      # DPU_BS perform ALU first then MUL
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
        self.reg(0x33800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x4000, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(4, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT, rk.DPU_BN_CFG_BN_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
        self.reg(0x7C00, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
        self.reg(0x3F800000, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK))
    elif op is Ops.CUSTOM and arg == "cmpeq_diff_zero_to_nan_to_32800":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(2, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x7C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
        self.reg(1, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    elif op is Ops.CUSTOM and arg == "cmpeq_32800_to_bool":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
        self.reg(0x47001F00, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x3C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      # REG_DPU_OUT_CVT_SHIFT need manual reset to 0
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
        self.reg(0, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    
    if op is Ops.WMMA:
      p = wmma_meta if wmma_meta is not None else self._wmma_params(2, 2, 2)
      self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

      is_kn_64 = p["k"] == 64 and p["n"] == 64
      is_kn_256 = p["k"] == 256 and p["n"] == 256
      is_kn_512 = p["k"] == 512 and p["n"] == 512
      is_kn_lg_512 = p["k"] > 512 and p["n"] > 512
      is_m_1_kn_768 = p["m"] == 1 and p["k"] == 768 and p["n"] == 768
      is_m_1_k768_n2048 = p["m"] == 1 and p["k"] == 768 and p["n"] == 2048
      is_m_1_kn_2048 = p["m"] == 1 and p["k"] == 2048 and p["n"] == 2048
      conv_con1 = self.reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) | \
                  self.reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK)
      if not (is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or is_m_1_kn_768 or is_m_1_k768_n2048 or is_m_1_kn_2048):
        conv_con1 |= self.reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK)
      self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1)
      self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON2,
        self.reg(p["feature_grains"], rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON3,
        self.reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
        self.reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE0,
        self.reg(p["data_in_width"], rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
        self.reg(p["data_in_height"], rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE1,
        self.reg(p["align_in"]-1, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
        self.reg(p["align_in"], rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE2,
        self.reg(p["dataout_width"], rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE3,
        self.reg(p["dataout_width"]*p["dataout_height"], rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0,
        self.reg(p["weight_bytes_per_kernel"]*p["align_out"], rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__SHIFT, rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
        self.reg(p["weight_bytes_per_kernel"], rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT,
                 rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
        self.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
        self.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
        self.reg(p["align_out"], rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON0,
        self.reg(12-p["data_bank"], rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
        self.reg(p["data_bank"], rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON1,
        self.reg((p["data_in_width"]*p["align_in"]+31)//32, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT,
                 rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON0,
        self.reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
        self.reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK) |
        self.reg(1, rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT, rk.CNA_CVT_CON0_CVT_BYPASS__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON1,
        self.reg(1, rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT, rk.CNA_CVT_CON1_CVT_SCALE0__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON2,
        self.reg(1, rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT, rk.CNA_CVT_CON2_CVT_SCALE1__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON3,
        self.reg(1, rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT, rk.CNA_CVT_CON3_CVT_SCALE2__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON4,
        self.reg(1, rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT, rk.CNA_CVT_CON4_CVT_SCALE3__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR,
        self.reg(feature_addr, rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__SHIFT,
                  rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK))

      self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON0,
        self.reg(15, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK) |
        self.reg(15, rk.CNA_DMA_CON0_DATA_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_DATA_BURST_LEN__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON1,
        self.reg(p["line_stride"], rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON2,
        self.reg(p["surf_stride"], rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
        self.reg(p["data_in_width"], rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
        self.reg(p["data_in_height"], rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
        self.reg(p["align_in"], rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
      self.emit_raw(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
        self.reg(weight_addr, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT,
                  rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))

      self.emit_raw(rk.CORE, rk.REG_CORE_MISC_CFG,
        self.reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK) |
        self.reg(1, rk.CORE_MISC_CFG_QD_EN__SHIFT, rk.CORE_MISC_CFG_QD_EN__MASK))
      self.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
        self.reg(p["dataout_height"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
        self.reg(p["dataout_width"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
      self.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
        self.reg(p["align_out"]-1, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
        self.reg(dst_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                  rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
        self.reg(p["dst_surf_stride"], rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(p["dataout_width"]-1, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
        self.reg(p["dataout_height"]-1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,
        self.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK) |
        self.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
        self.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
        self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
        self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
        self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
        self.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
        self.reg(p["align_out"]-1, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
        self.reg(p["dataout_height"]-1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
        self.reg(p["dataout_width"]-1, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
        self.reg(p["dst_surf_stride"]*4, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
      return

    burst_len = 15
    output_mode  = 2
    flying_mode = 1
    channel = 7
    dataout_height = 0
    dataout_width = math.ceil(size / ((dataout_height+1) * (channel+1))) - 1

    precision_float16 = 2

    ew_cvt_type = 0
    ew_data_mode = 1
    ew_data_size = 2
    ew_relu_bypass = arg != "relu"
    ew_alu_algo = self.hardware_ops.get(op, 0)
    ew_op_src = 1
    erdma_data_size_16bit=2
    if self.lut_enable: 
      ew_data_mode = 0; ew_data_size = 0; ew_op_src = 0; 
    
    self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
        self.reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(op == Ops.MUL, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
        self.reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(op in [Ops.MUL, Ops.FDIV] or self.lut_enable, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(self.lut_enable == False, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
        self.reg(self.lut_enable == True, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        self.reg(arg in ["cmplt_diff2bool", "cmpeq_diff_zero_to_nan_to_32800", "cmpeq_32800_to_bool"], rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK) 
      )
    # 0 or 1 both passed test_div, do not emit OUT_CVT_SCALE for other ops
    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      self.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK)) if op == Ops.FDIV else None

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        self.reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
        self.reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))

  def submit(self, uop):
    # TODO fix special if, maybe MUL output defaulted as fp32 amd need FP16TOFP32
    if uop not in (Ops.FDIV, Ops.WMMA):
      # EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1));
      self.q.append(0x2001000178495044), 
    # self.q.append(0x0081000000180008), # EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12))
    self.q.append(0x00810000000d0008 if uop is Ops.WMMA else 0x0081000000180008)
    tasks = ctypes.cast(self.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
    assert len(self.q) <= self.cmd_buf_size
    regcmd = ctypes.cast(self.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * self.cmd_buf_size)).contents
    for i in range(len(self.q)):
      regcmd[i] = self.q[i]

    tasks[0].flags  = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(self.q)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = self.cmd_buf.meta.dma_addr

    # TODO: update parameter name as driver updated
    submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=self.task_buf.meta.obj_addr,   # Placeholder, would be actual address in real code
            regcfg_obj_addr=0,
            task_base_addr=0,
            user_data=0,
            core_mask=1,
            fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
            )
    )
    if uop is Ops.WMMA and DEBUG >= 7:
      os.system("cd ~/npu/ops_reg/ && python dump.py 2 | grep EMIT | sed 's/\x1B\\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/tinygrad_emit ")
      os.system("cd ~/npu/ops_reg/ && python dump.py 3 > /tmp/tinygrad_input")
      os.system("cd ~/npu/ops_reg/ && python dump.py 4 > /tmp/tinygrad_weight")
    res = rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl,__payload=submit_res)
    # os.system("cd ~/npu/ops_reg/ && python dump.py 5")
    if DEBUG >= 7: print(res)
    if uop is Ops.WMMA and DEBUG >= 7:
      os.system("cd ~/npu/ops_reg/ && python dump.py 5 > /tmp/tinygrad_output")

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    self.device.reset_npu()
    st = time.perf_counter()
    if self.fused_matmul_meta is not None:
      try:
        self._run_fused_matmul(bufs)
        self.fused_matmul_hits += 1
        return time.perf_counter() - st
      except Exception as e:
        reason = str(e)
        self.fused_matmul_fallbacks += 1

    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == Ops.END}
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    has_control_flow = any(op in (Ops.RANGE, Ops.IF, Ops.ENDIF) for op,_,_,_ in self.uops)
    vectorize_global = False
    global_iters = itertools.product(*[range(x) for x in global_size[::-1]])
    # why has_control_flow
    if not has_control_flow and all(x == 1 for x in local_size):
      total_elems = math.prod(global_size)
      # why 16384
      if 1 < total_elems <= 16384:
        warp = list(itertools.product(*[range(x) for x in global_size[::-1]]))
        warp_size = len(warp)
        global_iters = [tuple(0 for _ in global_size)]
        vectorize_global = True

    for idxs in global_iters:
      values: dict[int, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        if DEBUG >= 7:
          print()
          print(i, uop, arg, src_values)
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        if DEBUG >= 7: print(i, uop, dtype, arg, src_values, src_dtypes)
        if uop is Ops.END:
          i = srcs[1]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        if uop is Ops.STORE:
          for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
            for (m,o,g),v in zip(src_values[0], val):
              if g: _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if uop is Ops.AFTER: values[i] = src_values[0]
        elif uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType), dtype
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if uop is Ops.DEFINE_REG:
            # REGs are per thread
            values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.DEFINE_GLOBAL else pbufs.pop(0)
            values[i] = [buf.cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          values[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': values[i] = values[i] = [x[2-int(arg[-1])] for x in warp] if vectorize_global else [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': values[i] = values[i] = [0] * warp_size if vectorize_global else [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: values[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
              if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          values[i] = src_values[0]
        elif uop is Ops.RANGE:
          if i not in values: values[i] = [0] * warp_size
          else:
            for j in range(len(values[i])):
              values[i][j] += 1
          if values[i][0] == src_values[0][0]:
            del values[i]
            i = loop_ends[i] + 1
            continue
        elif uop is Ops.VECTORIZE: values[i] = src_values
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()),
                               *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
          values[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
          values[i] = [from_storage_scalar(x, dtype.scalar()) for x in values[i]]
        elif uop is Ops.CAST:
          values[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in src_values[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            values[i] = [load([src_values[i][j] if i != 0 and src_dtypes[i].count > 1 else src_values[i] \
                               for i in range(len(src_values))], j, dtype.scalar()) for j in range(dtype.count)]
          else:
            values[i] = load(src_values, 0, dtype)
        elif uop is Ops.GEP:
          v = src_values[0][get_single_element(arg)]
          values[i] = v if isinstance(v, (list, tuple)) else [v]
        # elif uop is Ops.WMMA:
        elif uop is Ops.WMMA and False:
          first_src_dtype = self.uops[srcs[0]][1]
          assert isinstance(first_src_dtype, DType) # mypy
          dims, dtype_in, device, threads = arg[1], first_src_dtype.scalar(), arg[4], arg[5]
          wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
          # TODO: refactor these to a shared TensorCoreLayout
          # if device == "ROCKCHIP":
          #   if threads == 1 and dims == (2,2,1):
          #     def a_elem(x, _k, row, goff): return x[row][goff]
          #     def b_elem(x, col, _k, goff): return x[col][goff]
          #     def c_map(_lane, elem): return (elem%2, elem//2)
          #     values[i] = wmma_helper(1, 1, 2, 4, 4, a_elem, b_elem, c_map)
          #     i += 1
          #     continue
          if device == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            values[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif device == "AMD" and threads == 64:
            def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            values[i] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
          elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            values[i] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif device == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            values[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if dims == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              values[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,32):
              def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
              values[i] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {arg}")
          elif device == "INTEL":
            # A (16 elements on 8 threads)
            def a_elem(x, k, row, goff): return x[k%2+row*2][goff+k//2]
            # B (16 elements on 8 threads)
            def b_elem(x, col, k, goff): return x[k][goff+col]
            # C, D (8 elements on 8 threads)
            def c_map(lane, elem): return (lane, elem)
            values[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CPU":
            def elem(x, col, row, _): return x[col+row][0] # k is always 0
            def c_map(lane, elem): return (elem%16, elem//16)
            values[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
          else: raise NotImplementedError(f"unimplemented tensor core {arg}")
        elif uop in [Ops.CUSTOM, Ops.WMMA] or uop in GroupOp.ALU:
          if uop is not Ops.WMMA: assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {uop}"
          assert all_same([dtype] + src_dtypes) or uop in {*GroupOp.Comparison, Ops.WHERE, Ops.WMMA}, f"dtype mismatch on {uop}"
          # Ops.CMPLT seperated out for dtype 
          if uop in [Ops.CMPLT, Ops.WMMA] or (uop in self.hardware_ops and dtype.scalar() in [dtypes.float16]):
            self.device.reset_npu()
            self.q = []
            self.lut_enable = self.check_lut_enable(uop, arg)
            if len(src_values)==1: 
              if uop is Ops.NEG:
                src_values.append([-1]*len(src_values[0]))
                uop = Ops.MUL
              else:
                src_values.append(src_values[0])
            if uop is Ops.WMMA:
              in_full = np.asarray(src_values[0], dtype=np.float16).reshape(-1)
              wt_full = np.asarray(src_values[1], dtype=np.float16).reshape(-1)
              c_full = np.asarray(src_values[2], dtype=np.float32).reshape(-1)
              wmma_dims = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else (2, 2, 1)
              wmma_m, wmma_n, wmma_k = int(wmma_dims[0]), int(wmma_dims[1]), int(wmma_dims[2])
              if wmma_m <= 0 or wmma_n <= 0 or wmma_k <= 0:
                wmma_m, wmma_n, wmma_k = 2, 2, 1
              wmma_meta = self._wmma_params(wmma_m, wmma_n, wmma_k)
              in_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["m"], dtype=np.float16)
              wt_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["align_out"], dtype=np.float16)
              if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) == (2, 2, 1) and len(in_full) == 4 and len(wt_full) == 4:
                in_pack[0], in_pack[1], in_pack[32], in_pack[33] = in_full[0], in_full[1], in_full[2], in_full[3]
                wt_pack[0], wt_pack[1], wt_pack[32], wt_pack[33] = wt_full[0], wt_full[2], wt_full[1], wt_full[3]
              else:
                in_need = wmma_meta["m"] * wmma_meta["k"]
                wt_need = wmma_meta["k"] * wmma_meta["n"]
                in_matrix = np.zeros((wmma_meta["m"], wmma_meta["k"]), dtype=np.float16)
                wt_matrix = np.zeros((wmma_meta["k"], wmma_meta["n"]), dtype=np.float16)
                in_take = min(in_need, len(in_full))
                wt_take = min(wt_need, len(wt_full))
                if in_take: in_matrix.reshape(-1)[:in_take] = in_full[:in_take]
                if wt_take: wt_matrix.reshape(-1)[:wt_take] = wt_full[:wt_take]
                for r in range(wmma_meta["m"]):
                  dst_off = r * wmma_meta["align_in"]
                  in_pack[dst_off:dst_off+wmma_meta["k"]] = in_matrix[r, :]
                for n in range(wmma_meta["n"]):
                  dst_off = n * wmma_meta["align_in"]
                  wt_pack[dst_off:dst_off+wmma_meta["k"]] = wt_matrix[:, n]
              src = memoryview(bytearray(in_pack.tobytes()))
              src2 = memoryview(bytearray(wt_pack.tobytes()))
            else:
              src = memoryview(bytearray(np.asarray(src_values[0], dtype=np.float16).tobytes()))
              src2 = memoryview(bytearray(np.asarray(src_values[1], dtype=np.float16).tobytes()))
            self.task_buf = self.device._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
            self.cmd_buf = self.device._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
            self.input_buf = self.device._gpu_alloc(src.nbytes, 0, name="input")
            self.weight_buf = self.device._gpu_alloc(src2.nbytes, 0, name="weight")
            if uop is Ops.WMMA:
              output_stride = wmma_meta["align_out"] * dtypes.float32.itemsize
              output_nbytes = max(0x100, (wmma_meta["m"]-1)*output_stride + wmma_meta["n"]*dtypes.float32.itemsize)
            else:
              output_nbytes = src.nbytes
            self.output_buf = self.device._gpu_alloc(output_nbytes, 0, name="output")
            try:
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              self.device._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
              self.device._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
              if uop is Ops.WMMA:
                self.boilerplate(op=uop, size=len(src_values[0]), arg=arg,
                  feature_addr=self.input_buf.meta.dma_addr,
                  weight_addr=self.weight_buf.meta.dma_addr,
                  dst_addr=self.output_buf.meta.dma_addr,
                  wmma_meta=wmma_meta)
              else:
                self.boilerplate(op=uop, size=len(src_values[0]), arg=arg)
                self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
                  self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                            rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
                self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
                  self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT,
                            rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
                self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
                  self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT,
                            rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))

              self.submit(uop)
              self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)

              dst = memoryview(bytearray(self.output_buf.size))
              ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
              if DEBUG >= 7: print(dst.tobytes().hex())
              if uop is Ops.WMMA:
                raw = np.frombuffer(dst.tobytes(), dtype=np.float32)
                stride_f32 = wmma_meta["align_out"]
                result = [float(raw[r*stride_f32 + c]) for r in range(wmma_meta["m"]) for c in range(wmma_meta["n"])]
                if len(c_full) == len(result): result = [x+y for x,y in zip(result, c_full.tolist())]
              else:
                result = struct.unpack(f'<{self.output_buf.size//2}e', dst.tobytes())
              if self.lut_enable:
                raw = np.rint(np.array(result, dtype=np.float32))
                # q14 decode
                if uop is Ops.EXP2:
                  result = ((raw.astype(np.uint16) / 2**14) - 1) / self.inv_scale
                elif arg == "silu":
                  result = raw.astype(np.int16) / (2**15 - 1) / self.inv_scale
              values[i] = list(result)
              if DEBUG >= 7:
                print('src', src_values[0])
                print('src2', src_values[1])
                print('result', values[i])
                try: print('expected', [exec_alu(uop, dtype, p) for p in zip(*src_values)])
                except: pass
            finally:
              self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])
          else:
            allow_fallback = uop in (Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR)
            if allow_fallback:
              print('ALLOWED FALLBACK TO CPU', uop, dtype)
              values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
            else:
              print('<!> EXIT OPERATION NOT SUPPORTED', uop, dtype, src_values)
        assert i in values, (uop, dtype, srcs, arg)
        i += 1
    return time.perf_counter() - st

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  tensor_cores = tc.rockchip
  code_for_op = {k:v for k,v in python_alu.items() if k not in [Ops.MULACC, Ops.RECIPROCAL, Ops.CMPNE]} | {Ops.FDIV: 0}
  # hacks, turned unsupported dtype to half and lut function to Ops.CUSTOM
  def _rk_trunc_fix(x):
    if x.tag == "rk_trunc": return None
    xh = x.src[0].cast(dtypes.half)
    zero = UOp.const(dtypes.half, 0)
    neg = xh.alu(Ops.CMPLT, zero)
    shifted = xh.alu(Ops.SUB, UOp.const(dtypes.half, 0.49951171875))
    absx = UOp(Ops.WHERE, dtypes.half, src=(shifted.alu(Ops.CMPLT, zero), shifted.alu(Ops.NEG), shifted))
    mag = absx.alu(Ops.TRUNC).rtag("rk_trunc")
    signed = UOp(Ops.WHERE, dtypes.half, src=(neg, mag.alu(Ops.NEG).alu(Ops.ADD, UOp.const(dtypes.half, 1)), mag))
    return signed.cast(x.dtype)
  pre_matcher = PatternMatcher([
    (UPat.const(dtypes.floats, 0).alu(Ops.CMPLT, UPat.var("x", dtypes.floats)).where(UPat.var("x", dtypes.floats), UPat.const(dtypes.floats, 0)),
     lambda x: UOp(Ops.CUSTOM, dtypes.half, src=(x.cast(dtypes.half),), arg="relu")),
  ])
  extra_matcher = PatternMatcher([
    (UPat(Ops.WMMA, dtype=dtypes.float.vec(4), name="x"),
     lambda x: UOp(Ops.WMMA, dtypes.half.vec(4), x.src,
                   (x.arg[0], x.arg[1], x.arg[2], dtypes.half, *x.arg[4:])).cast(dtypes.float.vec(4))),
    (UPat(Ops.MUL, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MUL, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.ADD, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.ADD, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.MAX, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MAX, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.ADD, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
    (UPat(Ops.MAX, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.MAX, x.src[1].cast(dtypes.half))),
    (UPat(Ops.NEG, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.NEG)),
    (UPat(Ops.EXP2, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.EXP2)),
    (UPat(Ops.TRUNC, dtypes.floats, name="x"),
     _rk_trunc_fix),
    (UPat.var("x", dtypes.floats).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    (UPat.var("x", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    (UPat(Ops.CMPLT, name="x"),
     lambda x: UOp(Ops.CUSTOM, dtypes.float16, src=(x.src[1].cast(dtypes.float16).alu(Ops.SUB, x.src[0].cast(dtypes.float16)),),
                   arg="cmplt_diff2bool").cast(dtypes.bool)),
    (UPat(Ops.CMPEQ, name="x"),
     lambda x: UOp(Ops.CUSTOM, dtypes.float16, arg="cmpeq_32800_to_bool", src=(
       UOp(Ops.CUSTOM, dtypes.float16, arg="cmpeq_diff_zero_to_nan_to_32800", src=(
         x.src[1].cast(dtypes.float16).alu(Ops.SUB, x.src[0].cast(dtypes.float16)),),
       ),
     )).cast(dtypes.bool)),
    # CMPNE(x) = 1 - CMPEQ(x)
    (UPat(Ops.CMPNE, name="x"),
      lambda x: UOp.const(dtypes.float16, 1).alu(
        Ops.SUB,
        x.src[0].cast(dtypes.float16).alu(Ops.CMPEQ, x.src[1].cast(dtypes.float16)).cast(dtypes.float16)
      ).cast(dtypes.bool)),
    # ax + b(1-x) 
    (UPat(Ops.WHERE, name="w", src=(UPat.var("c", dtypes.bool), UPat.var("a", dtypes.floats), UPat.var("b", dtypes.floats))),
     lambda w,c,a,b: a.cast(dtypes.float16).alu(Ops.MUL, c.cast(dtypes.float16)).alu(Ops.ADD,
       b.cast(dtypes.float16).alu(Ops.MUL, UOp.const(dtypes.float16, 1).alu(Ops.SUB, c.cast(dtypes.float16)))).cast(w.dtype)),
    (UPat(Ops.WHERE, name="w", src=(UPat.var("c", dtypes.bool), UPat.var("a", dtypes.ints), UPat.var("b", dtypes.ints))),
     lambda w,c,a,b: a.cast(dtypes.float16).alu(Ops.MUL, c.cast(dtypes.float16)).alu(Ops.ADD,
       b.cast(dtypes.float16).alu(Ops.MUL, UOp.const(dtypes.float16, 1).alu(Ops.SUB, c.cast(dtypes.float16)))).cast(w.dtype)),
  ])
  def render(self, uops:list[UOp]) -> str:
    # the value of SPECIAL comes from local/global_size, not form its source
    uop_to_idx = {u:i for i,u in enumerate(uops)}
    lops = [(u.op, u.dtype, ([] if u.op is Ops.SPECIAL else [uop_to_idx[v] for v in u.src]), u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class RockchipRegisterAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, src_addr, dest_addr, src_size):
    ctypes.memmove(dest_addr, src_addr, src_size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self._do_copy(src.va_addr, mv_address(dest), src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    return to_mv(ctypes.cast(int, src.va_addr), src.size)

class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)

    compilers = CompilerSet([CompilerPair(RockchipRenderer, RockchipCompiler)])
    super().__init__(device, RockchipAllocator(self), compilers, functools.partial(RockchipProgram, self))
  def create_flink_name(self, handle: int, name:str, virt_address:int|None=None, obj_addr:int|None=None, dma_address:int|None=None) -> int:
    flink_req = rk.struct_drm_gem_flink(handle=handle, name=0)
    result = rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=flink_req)
    # print(f"SUCCESS: Created flink name {flink_req.name} for handle {handle} {name} {hex(dma_address)}")
    return flink_req.name
  def _gpu_alloc(self, size:int, flags, name:str) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)
    mem_create.flink_name = self.create_flink_name(mem_create.handle, name, virt_address=va_addr, obj_addr=mem_create.obj_addr, dma_address=mem_create.dma_addr)

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)
  def _gpu_sync(self, buf:HCQBuffer, flags:int) -> None:
    if not getenv("ROCKCHIP_MEM_SYNC", 0): return
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd_ctl, __payload=rk.struct_rknpu_mem_sync(
      flags=flags, reserved=0, obj_addr=buf.meta.obj_addr, offset=0, size=buf.size))
  def _gpu_free(self, buf:HCQBuffer) -> None:
    FileIOInterface.munmap(buf.va_addr, buf.size)
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, __payload=rk.struct_rknpu_mem_destroy(
      handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr))
  def _gpu_free_multiple(self, buf_list) -> None:
    for buf in buf_list: self._gpu_free(buf)
  def reset_npu(self):
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, __payload=rk.struct_rknpu_action(flags=rk.RKNPU_ACT_RESET, value=0))
