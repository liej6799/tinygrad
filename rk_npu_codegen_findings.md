# RK NPU Codegen Findings: Unrolling Performance & Better Designs

Scope: the `ops_rk` backend (`tinygrad/runtime/ops_rk.py`) unroll/codegen path, profiled on
`test_add`, `simple_matmul`, `simple_conv` at various sizes, plus an architecture review against the
RK3588 NPU hardware model documented in `/data/rkt` and `/data/rk3588`.

Companion doc: `rockchip_backend_consideration.md` covers the *runtime* (register/submit) side. This
doc covers the *codegen* side: how a looped kernel is turned into the loop-free op stream the NPU needs,
why it is slow today, and the better designs.

---

## 1. TL;DR

1. **The slowness is not "rendering thousands of uops".** Rendering the chunk stream
   (`uops_to_chunks` + `chunks_to_c`) is ~0.1 s. The cost is the **generic codegen pipeline running on
   the giant unrolled graph**: once `pre_matcher` unrolls (e.g. add 64×64 → ~37k uops), ~8 graph-rewrite
   passes + the linearizer + estimates each re-traverse the whole explosion.

2. **One real O(n²) bug, fixed.** `Estimates.from_uops` rebuilt a growing set per LOAD/STORE
   (`dont_count = dont_count.union(...)`). Changed to in-place `.update()` (3 lines,
   `tinygrad/renderer/__init__.py`). Restores linear scaling: **add 128×128 66 s → 16 s (4.2×)**,
   matmul 96³ 14.5 s → 5.0 s (2.9×). All `test_uops_stats` estimate tests still pass.

3. **Per-element unrolling is a synthesizer simplification, NOT a hardware requirement.** The RK3588
   NPU's elementwise engine processes a **vector tile up to 8176 elements in one op**
   (`DPU_DATA_CUBE_WIDTH = W-1`, `/data/rk3588/.../RKNN_CREATION.md:114,210`). The current path emits
   **one width-1 tile per scalar uop** (`/data/rk3588/.../MAPPING.md:15,22,25`): a 16384-element add
   becomes 16384 tiles instead of **2** width-8176 tiles. So unrolling to N scalar uops is wasteful on
   *both* axes — compile time and hardware time.

4. **Better design:** keep ops *structured* (loop-free but not scalarized) as **affine chunk
   descriptors** `(op, dtype, ranges, affine index maps)` — generalizing the existing GEMM extraction to
   every op — and map each chunk to its hardware engine (elementwise→DPU EW tile, matmul/conv→CNA/CORE,
   pool→PPU, transcendental→activation LUT). Full scalar unrolling, if ever needed, should be the **last**
   step (in `render`), not at `pre_matcher`.

---

## 2. Measurements

Backend `DEV=RK RK_UNROLL=1`, clang `-O0`. "Codegen-only" excludes execution.

### 2.1 Total realize, before/after the `from_uops` fix

| case        | size  | before  | after   | speedup |
|-------------|-------|---------|---------|---------|
| add (s×s)   | 64    | 5491 ms | 3901 ms | 1.4×    |
| add         | 96    | 20139 ms| 9059 ms | 2.2×    |
| add         | 128   | 66266 ms| 15809 ms| 4.2×    |
| matmul (s³) | 64    | 3327 ms | 2395 ms | 1.4×    |
| matmul      | 96    | 14530 ms| 5041 ms | 2.9×    |

The speedup grows with size because the fix removes a quadratic term (scaling went from ~n² to ~n).

### 2.2 Where the time goes (add 64×64, after fix; graph ≈ 37k uops)

```
linearize/render 975ms  pre_matcher 341  decomp dtypes 262  final rewrite 196
decompositions   147    transcendental 144  move gates 134   add control flow 97
```

`transcendental` and `move gates` cost ~140 ms each while doing **nothing** for an add — they just
traverse 37k nodes. Every pass after `pre_matcher` pays O(unrolled-node-count).

### 2.3 The decisive number — codegen on the loop vs the unroll

| size   | RK_UNROLL=1 (unrolled) | RK_UNROLL=0 (compact loop) |
|--------|------------------------|-----------------------------|
| add 64 | 2905 ms                | **50.5 ms**                 |
| add 96 | 6638 ms                | **53.1 ms**                 |
| add 128| —                      | **52.2 ms**                 |

The compact-loop pipeline is **flat ~50 ms** regardless of size; the unrolled pipeline is O(elements).
This is the whole story: the generic passes are cheap on a loop body and expensive on the explosion.

### 2.4 conv

`simple_conv` currently **crashes** with `RK_UNROLL=1` (before and after my change): the chunk renderer
can't express a register-accumulator `STORE` (`STORE(INDEX(DEFINE_REG…), 0.0)` at `ops_rk.py:124`).
Pre-existing; orthogonal to performance.

---

## 3. Hardware execution model (why unrolling exists)

From `/data/rkt` (direct register driver) and `/data/rk3588` (ONNX/RKNN path). The RK3588 NPU is
**NVDLA-derived**.

- **Fixed-function dataflow pipeline**, not a programmable ISA: `CNA → CORE → DPU → PPU` with `RDMA` and
  a `PC` (program-control) block (`/data/rkt/README.md:27-66`). Engines are selected per task by an
  `enable_mask` (CONV `0xd`, GEMM/elementwise `0x18`, POOL `0x60`).
- **The "program" is a straight-line 64-bit register-command stream** with **PC-chained tasks** — no
  branches, no loops, no data-dependent control (`RUNTIME_SPEC.md`; `examples/gemm.py:409-429`).
- **Path to the device:** `tinygrad uops → fully_unroll → ONNX → rknn-toolkit2 → .rknn → librknnrt → NPU`
  (`uop_to_onnx.py:1-9`, `rknn-decode/MAPPING.md:6-7`). ONNX is a static DAG; the toolchain explicitly
  requires **loop-free, no RANGE/END** input and rejects accumulators
  (`rknn_synth.py:62-63`, `librknnrt/rknn_runtime.py:391,510-511`).

So *some* form of "no loops" is mandatory. The question is the **granularity**: per-element scalars, or
structured per-tile vector ops.

### 3.1 Supported ops (the engines we can target)

| hardware engine | ops |
|---|---|
| DPU EW          | ADD, SUB, MUL, MAX, NEG, FDIV (`elementwise.py:201-208`) — **vector tile, width ≤ 8176** |
| CNA + CORE (MAC)| conv; **matmul = 1×1 conv** (`README.md:158-233`); fp16 512 MAC/cyc, int8 1024 MAC/cyc |
| CACC            | reduction/accumulation for conv/matmul |
| PPU             | AVG / MAX pool |
| DPU activation  | LUT-based sigmoid/tanh/GELU/SiLU (approx, ~1e-2) (`act_lut_readme.md`) |
| CPU fallback    | And/Or/Xor/Not, Cast, Mod, compares (runtime splits the graph) |

### 3.2 Hard limits (these define the right chunk size)

| limit | value | source |
|---|---|---|
| elementwise single tile | **N ≤ 8176** | `RKNN_CREATION.md:210,222-232` |
| channel field | 13-bit → C1 ≤ 1024 | `RKNN_CREATION.md:389-396` |
| toolkit-free body | **≤ 6 regcmd blocks / 3 tasks** | `RKNN_CREATION.md:165,231-232` |
| ALU chain (toolkit-free) | 1..6 ops / 2..7 inputs (full path up to 64) | `rknn_synth.py:168` |
| dtypes | fp16 best; int8 ok (Mul broken); fp32/int16/int32 unsupported on EW | `RKNN_CREATION.md:1206-1264` |

**Key consequence:** a 16384-element elementwise op is naturally **2 tiles** (8176+8208→split), not
16384. The unroll-to-scalars approach produces 16384 width-1 tiles and then bumps into the 6-block/3-task
cap almost immediately — i.e. it is both slow to build and quickly hits a hardware wall.

---

## 4. The design space

Three designs, increasing quality. The mistake today is using design 1 *and* doing it early.

### Design 1 — Full scalar unroll, then group by op (current)
Expand every RANGE to per-element uops, bucket by (level, op-category, dtype). Output = the chunk stream.
- Pros: simple; matches the naive "one ONNX node per element" path.
- Cons: O(N) graph; the whole generic pipeline reprocesses it (Section 2); maps to width-1 tiles, so it
  fights the 8176/6-block hardware limits immediately.

### Design 2 — Late unroll (same output, fast compile)
Run the generic pipeline on the **compact loop**, expand to the scalar stream **only in `render()`**.
- Identical output to design 1, but the ~8 passes + linearizer + estimates stay on the loop body (~50 ms,
  Section 2.3) instead of the explosion.
- **Validated for elementwise:** moving the unroll into `render` took add 64×64 from 3901 ms → **1591 ms**.
- **Not yet working for matmul:** the GEMM is extracted at `pre_matcher` (correct), but the remaining
  M,N loops, when unrolled one-range-at-a-time *after* linearization, make `loop_unrolling` pathological
  (the post-linearize `AFTER/END` + C-load-back structure). Needs an all-ranges-at-once unroller (below).
- Verdict: good incremental win for the common (elementwise/epilogue) case; keeps today's exact format.

### Design 3 — Structured / affine chunks (recommended)
Do **not** scalarize. Represent each loop-carried op as a descriptor:

```
chunk = (op, dtype, ranges=[(var, trip_count), ...], src_maps=[affine(var)->offset], dst_map)
```

This is exactly what `extract_matmul` already does for GEMM (it pulls `(M,K,N, affine A/B/C maps)`,
`ops_rk.py:18-51`) — generalized to **every** op:
- `for i: out[i] = a[i] + b[i]`  →  one `ADD` chunk over range N with affine in/out maps.
- elementwise/epilogue chunk  →  one **DPU EW tile** of width = trip-count (split into ≤8176 pieces).
- contraction chunk           →  **CNA/CORE** (matmul=1×1 conv) or a MULACC chain.
- pooling/transcendental      →  PPU / activation LUT.

Why this is the right model for *this* hardware:
- The hardware op **is** a vector tile (`DATA_CUBE_WIDTH`), so a chunk descriptor maps 1:1 to a regcmd
  tile. A 16384-add = 2 tiles, not 16384.
- Elementwise vs "other" ops are separated **by construction** — one chunk per (op, level) — which is
  precisely the "separate elementwise from other ops by chunk so the runtime knows what to do" goal.
- Compile time becomes ~constant in problem size (no giant graph). Tiling to the 8176/6-block limits is a
  property of the descriptor (split a range), not a graph rewrite.
- The genuinely-scalar fallback (an op with no engine) is a **mechanical expansion** of one descriptor —
  a tight loop over affine maps — not a graph-rewrite over millions of nodes.

Design 3 subsumes design 2: design 2 is design 3 with every chunk's tile width forced to 1.

---

## 5. Recommendations

Concrete, in priority order.

1. **[DONE] Keep the `from_uops` O(n²)→O(n) fix** (`tinygrad/renderer/__init__.py`). General, tested,
   2–4× on every unrolled kernel. Independent of any redesign.

2. **Adopt the structured-chunk representation (design 3).** Generalize `extract_matmul` into a single
   "structure extractor" that pulls each op out as `(op, dtype, ranges, affine maps)` from the **compact
   loop** (cheap, ~constant). Emit:
   - elementwise/epilogue chunk → EW tile descriptor, split to ≤ 8176 / ≤ 6 blocks;
   - contraction chunk → CNA/CORE (matmul as 1×1 conv) descriptor;
   - keep the descriptor list as the program; the runtime/ONNX layer expands each descriptor to a tile
     (or, where the toolkit truly needs scalars, expands mechanically).
   This removes the giant-graph pipeline cost entirely and matches the hardware's native tile model.

3. **If you must keep the current scalar-stream format short-term, do design 2** (move `loop_unrolling`
   from `pre_matcher` into `render`). It is a drop-in speedup for elementwise. To also cover matmul,
   replace the one-range-at-a-time `loop_unrolling` (graph-rewrite fixed point) with a **single-pass
   all-ranges unroller**: enumerate the cartesian product of range values, substitute all range vars to
   consts, fold indices once with `symbolic`. This avoids the nested-loop duplication blow-up that hangs
   matmul today.

4. **Stop running no-op passes on the unrolled graph.** `transcendental` and `move gates` are pure
   overhead for RK's native-op stream (~280 ms/kernel at 64×64). With design 2/3 they run on the small
   loop body and the issue disappears; if design 1 is retained, gate these passes off for the RK target.

5. **Fix conv** by routing its reduction to the MAC engine (conv is the NPU's native op) or pre-unrolling
   it to a MULACC chain, instead of leaving a register accumulator the chunk renderer can't express.

---

## 6. Answer to "is there a way to support this hardware, better than unrolling to many uops?"

**Yes.** Unrolling to a flat scalar uop stream is the wrong granularity for this NPU. The hardware is a
fixed-function dataflow engine whose elementwise unit already executes a **width-N vector tile** and whose
MAC array already executes conv/matmul natively. The right design keeps each op **structured and
loop-free** as an affine chunk descriptor and maps it to the matching engine (EW tile / CNA-CORE / PPU /
LUT), splitting only to the hardware tile limits (≤ 8176 elements, ≤ 6 blocks). That:

- never builds the giant graph, so codegen stays ~constant in size (the 50 ms vs seconds gap in §2.3);
- emits ~`N/8176` tiles instead of `N` width-1 tiles, matching the hardware;
- cleanly separates elementwise chunks from other-op chunks for the runtime, which was the goal.

Full per-element materialization remains available as the last-resort expansion of a single descriptor,
but it is no longer the thing the whole compiler pipeline has to chew through.

---

## 7. Code changes in this investigation

- `tinygrad/renderer/__init__.py` — `Estimates.from_uops`: `dont_count = dont_count.union(...)` →
  `dont_count.update(...)` (3 lines). Removes the O(n²) set-rebuild. Verified: `test/null/test_uops_stats.py`
  (25 tests) and `test/backend/test_add_cmp.py` pass; add/matmul numerically correct on RK.

(Designs 2 and 3 are proposals; the elementwise design-2 prototype was validated for speedup but is not
committed because the matmul path needs the single-pass unroller from recommendation 3.)

---

## 8. Fusing the matmul epilogue onto the NPU (DPU bias / EW), and where it stops

This section is the *runtime* counterpart to the above: the GEMM always runs on the NPU MAC (vendor
`gemm.py:run_gemm`; the old `RK_NPU_ALU` flag is removed — the NPU path is unconditional), so what can the
`a@b + <epilogue>` add be folded into so it does not fall back to a clang loop? Measured on
`extra/gemm/simple_matmul.py` (`ADDC` const, `ADDV` per-column bias, `ADDT` [M,N] tensor).

> **Float-only on the NPU, NPU-first with a clang CPU fallback.** `extract_matmul` and `extract_elementwise` require
> the op be `dtypes.is_float` — the NPU is fp16. A non-float matmul/elementwise (e.g. int) stays a `compact_loop_to_c`
> C loop. The GEMM runs on the NPU MAC (`run_gemm`) and the elementwise epilogue on the NPU DPU-EW (`chain_add`,
> `cached_graph(n,tasks).run(leaf_data)` — the descriptor's `tasks`/`leaves` map 1:1 onto `AluGraph`). **Each NPU
> submit is wrapped in try/except: on any failure (`OSError`/ENXIO, incl. a missing device or the §8.3 limit) it
> recomputes that op on the CPU via clang** — `mm_lines` for the GEMM (fp32, the same affine A/B/C index maps),
> `ew_lines` for the EW. So the §8.3 gemm→chain_add session-state limit is now *handled*, not avoided: a post-GEMM EW
> that needs >1 task ENXIOs the 2nd task → caught → clang fallback (verified: a 256×256 `a@b + D` runs the GEMM on the
> NPU, the n=65536 EW ENXIOs and falls back to clang, result correct). A single post-GEMM EW task (M·N ≤ 64000)
> succeeds on the NPU (verified `a@b + D` at 8×8, fp16 `max_err 0.0027`). The per-GEMM clang fallback (`mm_lines`) is
> re-added (it was previously deleted when the NPU GEMM was unconditional).

### 8.1 Decisive driver fact: `enable_mask` is dead in PC mode
The rknpu driver's PC-mode commit (`rknpu_job.c`) reads only `regcmd_addr`, `regcfg_amount`, `int_mask`, and
the task count from each task descriptor. **`task.enable_mask` and `task.int_clear` are never read.** Engines
are selected purely by the PC `OPERATION_ENABLE` (reg 0x0008) word emitted in each task's PC-chain tail:
`0xd`=`(6<<1)|1` for CONV/GEMM, `0x18` for a standalone DPU-EW (both opaque "RESERVED, not in TRM"). This
means tweaking EW/RDMA **register values** is *recoverable*-risk (a bad config → DPU timeout → ENXIO, cleared
by `reset_npu` / `simple_add.py`), **not** the reboot-risk that wrong `npu_submit`/`task_count` args carry.

### 8.2 What fuses (shipped, any size)
| epilogue | mechanism | result |
|---|---|---|
| `a@b + c` (scalar) | DPU **BS (X1) ALU**: un-bypass BS, `BS_ALU_ALGO=SUM(2)`, operand in `BS_ALU_CFG=0x4044` | **exact**, 1 submit, N≤1024 |
| `a@b + b[n]` (per-col bias) | **matmul augmentation** `[x\|1]@[W;b]` — pure matmul, no DPU unit, no RDMA | **exact**, 1 submit, any N |

- **The BS ALU operand is FP32**, not fp16 (measured: fp16 `0x4700` adds 0 because it's a denorm in fp32;
  fp32 `0x40e00000` adds exactly 7.0). No vendor example uses the BS unit — all bypass it — so this was found
  empirically. `gemm.py:make_gemm_regs(bias=)` injects it; `run_gemm(bias=)` takes a scalar (→BS) or a vector
  (→augmentation). `ops_rk.py` auto-detects each epilogue (`extract_bias_vector`, render `bias` dict).
- Augmentation for the bias *vector* is preferred over a BS **memory** operand because BS-from-memory needs the
  SDP-RDMA (§8.3) — augmentation avoids it entirely (`[x|1]@[W;b]`, K→K+1, negligible).

### 8.3 What does NOT fuse: the full [M,N] residual `a@b + D`
This needs the DPU **EW (X2)** unit reading `D` from the SDP-**ERDMA** (`RDMA_EW_BASE_ADDR=0x5038`). It is
**provably fusable for a single EW tile**: emit fp16 GEMM output (16-interleaved), lay `D` out in the *same*
physical layout, and PC-chain a chain_add-style DPU-EW task (`OPERATION_ENABLE=0x18`, both operands from
memory, `EW_CFG=0x108202c0` — the critical bit is **`EW_OP_SRC=1`** = operand-from-memory) after the GEMM in
ONE submit. Verified exact: N=8 `max_err 2e-4`, N=128 `max_err 2e-3`.

**But it stops at one EW submit.** The EW flat-adds the physical fp16 buffer (`n_phys = 2·m·align_out`, the ×2
is interleave padding), so N≥~176 needs **multiple** EW tasks (chain_add cap `_MAX_ELEMENTS_PER_TASK=64000`).
The precise limit, isolated by submitting tiles with per-submit return codes:
**a gemm context permits exactly ONE subsequent DPU-EW TASK** (not one submit) — a *second* EW task ENXIOs no
matter how it is packaged. Evidence (N=256, 3 EW tiles needed):
- separate single-task ioctls → `gemm rc=0`, `EW#1 rc=0`, `EW#2 → ENXIO`;
- bundling all tiles into **one multi-task EW submit** (`task_count=3`) → `gemm rc=0`, EW submit `→ ENXIO`
  (rejected at submit — so "2 ALU ops in one ioctl" does **not** dodge it);
- gemm + all EW PC-chained in one submit → ENXIO.
- 2nd EW with **fresh task+regcmd+d buffers + `npu_reset`** (a "brand-new transaction") → still ENXIO.

So it is not buffer-level either: new memory, a reset, and a freshly-built EW call do **not** clear it — the
poison survives everything except a new **fd**. It is **not** the submission structure, the tile count-per-submit,
the tile size, or the address: reversing
the order so the high-offset 3072-elem tile goes first still gives `gemm ok, EW#1 ok, EW#2 ENXIO`; a single
≤64000-elem EW after a gemm always works; chain_add alone runs 16+ EW tiles; a **fresh process always recovers**.
So it is a driver/firmware **session-state** limit ("one DPU-EW task per gemm context"), confirmed identical
across two fds, shared fd, PC-chained single submit, one multi-task EW submit, and separate per-tile ioctls
(± pingpong). The flying-matmul + ERDMA single-task fusion (NVDLA's native residual path) also ENXIOs (its
`OPERATION_ENABLE` is undocumented and unused by any example). Not userspace-fixable short of per-EW process/fd
isolation; would need a driver/firmware change. (A single EW task caps at the DATA_CUBE_WIDTH limit ~65408
elems, so a residual fuses only when `n_phys ≤ 65408`: N≤128 with the fp16 interleaved buffer, ~N≤255 if the
GEMM emits fp32-contiguous output.)

**Consequence:** `a@b + [M,N] tensor` stays a clang epilogue (correct, fp32) for N≥256; small residuals are
fusable but were not wired (N≤128-only, marginal, and clang is already correct at all sizes).

### 8.4 Code changes in this investigation
- `extra/gemm/simple_matmul.py` — `ADDC`/`ADDT`/`ADDV` epilogue options for testing.
- `/data/rkt/examples/gemm.py` — `make_gemm_regs(bias=)` / `run_gemm(bias=)`: scalar→DPU-BS (fp32 operand),
  vector→augmentation. Backward-compatible (sweep still passes).
- `tinygrad/runtime/ops_rk.py` — `extract_bias_vector` + render/runtime wiring: auto-fuse `C+const` (DPU BS)
  and `C+bias[n]` (augmentation) into the NPU GEMM; the `[M,N]` tensor add falls back to clang.
- `extra/rockchip/npu-top.py`, `extra/rockchip/npu_ioctl_decode.py` — NPU load monitor + rknpu ioctl decoder
  (strace mislabels rknpu DRM ioctls via DRM_COMMAND_BASE collisions; `0x18` SUBMIT count proves NPU use).

---

## 9. `compact_loop_to_c`: direct-from-uop C codegen, now the DEFAULT (§4-6 Design 3)

§1-6 argued the slowness is `loop_unrolling` exploding a looped kernel to per-element scalars, after which the
generic pipeline re-traverses the explosion. **`compact_loop_to_c(uops)`** (`ops_rk.py`) is the structured
replacement: it renders the COMPACT looped sink straight to one C function with nested `for` loops by walking the
uops in **linearized order** (`RANGE`→`for{`, `END`→`}`, `STORE`→assignment, `DEFINE_REG`→a C local), directly
from the uops — no unroll, no per-element scalars, no uop interpreter. It reuses the `CT`/`C_BIN`/`C_FN` op→C
tables (MAX/WHERE/NEG/RECIPROCAL/CAST/BITCAST/FLOORDIV/FLOORMOD/MULACC/transcendentals) **and** handles
reductions via the register-accumulator form (`acc=0; for(reduce){ acc=acc+...; } out=acc;`).

**Now the only path.** `rk_sink` always keeps the loop; render emits the cloop as the last submit of a
`("chunks",...)` program, so an extracted **GEMM still runs first** (matmul + a non-extractable epilogue, e.g.
`(a@b).relu()`, is gemm→cloop). `full_unroll_reduces=False` keeps non-matmul reductions as register-accumulator
loops for cloop. `loop_unrolling`, the `RK_CLOOP` flag, `uops_to_chunks`, and the scalar-chunk machinery in
`chunks_to_c` (the per-node scratch, `rhs_of`/`expr`) are **deleted**; `test/backend/test_add_cmp.py` now asserts
the cloop C output instead of chunk names. The chunk path survives only for the fully-extracted case (GEMM +
fusable epilogue → `mm`/`ew` submits, no leftover compute).

**Measured (default cloop):**
- correct (`max_err 0.0` vs numpy): elementwise max/relu/where/exp; reductions full-sum / axis-sum; **conv2d**
  (which §2.4 reported *crashing* under `RK_UNROLL=1` — the register-accumulator STORE; cloop renders it as a C
  local); matmul+relu `(a@b).relu()` gemm→cloop;
- **works where the unroll path CRASHES**: 512×512 `maximum` under `loop_unrolling` →
  `RuntimeError: infinite loop in graph_rewrite (stack too big)`; under cloop → one C loop, `realize≈150 ms`;
  same for matmul+relu at N=256 (`305 ms`);
- compile time ~constant in problem size; the NPU GEMM + bias/EW fusion (chunk path) is unchanged (verified
  `fused DPU bias` / `fused augment` and exact NPU result); RK tests pass (`test_add`, `test_add3_rockchip`
  default; `test_add_cmp` with `RK_CLOOP=0`).

Unrenderable ops (if any: `compact_loop_to_c` returns `None`) still fall back to the uop interpreter, which is
correct. The `RK_UNROLL=0` mode (no extraction at all → everything to the interpreter) also remains as a debug
escape hatch.
