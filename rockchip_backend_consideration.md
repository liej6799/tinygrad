# Rockchip Backend Consideration

## Objective

Decide how to re-architect `tinygrad/runtime/ops_rockchip.py` so it looks more like the rest of tinygrad's runtimes, ideally under 500 lines, while keeping a realistic path to useful RK3588 NPU execution.

The specific question is whether Rockchip should move from runtime register emission to a compiled backend where almost all register command streams are built ahead of time, stored as a binary artifact, and patched at runtime only for input, output, scratch, and constant addresses. The AMD backend is the main comparison point because it is also a low-level driver backend that writes hardware command packets directly.

## Current Runtime Model In Tinygrad

Most `ops_*.py` files have the same high-level contract:

1. `Device` owns hardware discovery, global state, queues, and allocators.
2. `Allocator` turns tinygrad `Buffer` objects into device memory and implements copies.
3. `Compiler` turns rendered source into a byte blob.
4. `Program` loads that byte blob once, stores parsed metadata, and launches it many times with new buffers, scalar values, and grid sizes.

The small runtimes are the cleanest examples:

- CUDA loads a cubin/PTX module once, caches the function handle, encodes buffer and scalar arguments, and calls `cuLaunchKernel`.
- Metal compiles or loads `MTLB`, creates a compute pipeline, binds buffers and values, and dispatches threadgroups.
- OpenCL follows the same shape around `clCreateProgramWithBinary`, `clCreateKernel`, and `clEnqueueNDRangeKernel`.
- Python is unusual because its "compiled binary" is a pickled uop list, but it still keeps the same API: renderer creates a serialized program, compiler decodes it, runtime executes it.

The HCQ backends add another layer:

- `HCQProgram.__call__` creates an argument state, builds a hardware queue, waits on the timeline signal, emits memory barriers, calls `queue.exec(...)`, signals completion, submits, and optionally waits.
- Backend-specific queues implement only the command packet details: `exec`, `wait`, `signal`, `copy`, `timestamp`, and memory barriers.
- AMD, NV, QCOM, CPU, and RDMA all benefit from this split because command packet emission is separated from program parsing and memory allocation.

This matters for Rockchip because the current file does not follow that split. It has a `Compiled` runtime surface, but internally it behaves like a research script with several different execution modes in one class.

## Runtime Survey

The runtimes fall into a few families. Rockchip should try to join one of these families instead of being a one-off file.

### Native API Runtimes

These runtimes delegate most command submission details to a vendor API:

- `ops_cuda.py`: `CUDAProgram` loads a module with `cuModuleLoadData`, looks up a `CUfunction`, caches an argument struct, patches buffer and scalar values into that struct, and launches with `cuLaunchKernel`. The allocator is ordinary CUDA device or host allocation plus async HtoD and blocking DtoH copies. This is the smallest example of "compiled blob plus thin launch".
- `ops_hip.py`: same idea through HIP APIs. It is structurally close to CUDA: load module, get function, encode args, launch.
- `ops_cl.py`: compiles OpenCL C to a binary, creates a program and kernel, sets kernel args, and enqueues an NDRange. It is also "vendor owns queue details".
- `ops_metal.py`: `MetalCompiler` produces `MTLB`, `MetalProgram` creates a library/function/pipeline, and launch binds buffers and values into a command encoder. The device tracks in-flight command buffers and profiles from Metal timestamps.
- `ops_webgpu.py`: similar API-level model for WebGPU. The runtime manages bind groups, buffers, command encoders, and dispatch, while shader generation stays in the renderer/compiler side.

The common lesson is that the runtime is short because the compiled program is a real object. Launch code only binds dynamic state.

### HCQ Packet Runtimes

These runtimes own the packet format and submit queues directly:

- `ops_amd.py`: parses an AMDGPU ELF, loads it into GPU memory, extracts the kernel descriptor, and emits PM4 packets per launch. It has compute, AQL, and SDMA queues, plus KFD/PCI/USB interfaces. AMD is complex, but the boundaries are clean: program parsing, queue packet emission, allocation, profiling, and hardware discovery are separate concepts.
- `ops_nv.py`: parses CUDA/NV code objects, constructs QMD descriptors, emits NVC command methods, submits through GPFIFO, and has compute, copy, and video queues. Like AMD, dynamic launch state is patched into descriptors and command queues, not hardcoded into the compiled shader.
- `ops_qcom.py`: compiles or parses Adreno binaries/NIR, builds KGSL command buffers, emits CP packets and register writes, and creates argument descriptors for buffers, images, constants, samplers, and UAVs. It is a useful comparison for Rockchip because it writes register packets directly but still keeps the main launch path as `QCOMComputeQueue.exec`.
- `ops_cpu.py`: uses the same HCQ abstractions even though the "hardware queue" is a Python worker queue. A compiled CPU function is loaded into executable memory, then `CPUComputeQueue` schedules calls and signals.
- `ops_rdma.py`: only implements copy queue behavior. It shows that a runtime can expose a narrow hardware capability and still fit the HCQ model.

The common lesson is that direct packet emitters can still be idiomatic if packet emission lives in queue methods and program objects carry compiled metadata.

### Storage And Debug Runtimes

These are not accelerator launch backends, but they show tinygrad's device abstraction:

- `ops_disk.py`: maps files or shared memory into `DiskBuffer` objects and implements copy and sharded `io_uring` reads for disk-to-device paths. It has no real compiler/runtime because storage devices do not execute kernels.
- `ops_tinyfs.py`: similar storage-facing backend for tinygrad file service use cases.
- `ops_npy.py`: simple allocator-backed `.npy` storage path.
- `ops_null.py`: compile-only/testing backend. It accepts programs and buffers but does not execute real hardware work.
- `ops_python.py`: a software execution backend. It serializes uops with base64/pickle and interprets them. This is acceptable for Python because it is explicitly the Python runtime; it is a bad model for Rockchip long-term because it hides hardware misses inside a hardware device.
- `ops_dsp.py`: RPC/offload-oriented backend. It compiles C for a DSP support path and launches through a device-specific invocation layer, again keeping compile and launch concepts separate.

The common lesson is that unusual devices can be supported, but their responsibility should be explicit. A Rockchip hardware backend should not also be a generic Python interpreter.

## Current Rockchip Shape

`ops_rockchip.py` is around 1285 lines. It currently contains:

- Renderer graph rewrites for half-only arithmetic, relu, silu, comparisons, where, integer casts, and Rockchip WMMA patterns.
- Pattern detection for native elementwise and 1x1 convolution programs.
- Python uop interpreter fallback for unsupported or unfused programs.
- Tensor packing and unpacking for elementwise, conv1x1, and WMMA matmul.
- Register command construction for DPU, DPU_RDMA, CNA, and CORE blocks.
- DRM RKNPU memory allocation, mmap, flink, sync, submit, and reset calls.
- Temporary buffer allocation per launch for task, command, input, weight, and output buffers.
- Hardware validation and fallback counters for fused matmul.

This makes it difficult to see the actual runtime boundary. The file is not just a runtime. It is also a partial compiler, a binary layout packer, a register emitter, a command submitter, and a software fallback path.

The current memory model is also not a true device-buffer model. The public allocator returns CPU `memoryview(bytearray(...))`; launches copy data into short-lived RKNPU GEM buffers, run the NPU, then copy results back into CPU buffers. There is a `RockchipRegisterAllocator` that allocates `HCQBuffer`, but `RockchipDevice` does not use it. So tinygrad sees Rockchip as CPU-backed storage plus accelerator calls, not as a normal resident device.

## How AMD's Register Backend Works

AMD is not "precompiled register blobs". It is a compiled shader backend plus runtime command packets.

The AMD flow is:

1. Renderer/compiler produce a real AMD GPU code object, usually via HIP/LLVM.
2. `AMDProgram` parses the ELF, applies relocations, extracts the HSA kernel descriptor from `.rodata`, allocates GPU memory for the image, copies it to the GPU, and stores launch metadata such as SGPR requirements, LDS size, scratch size, kernel arg size, and program address.
3. `AMDComputeQueue.exec` emits PM4 packets for a particular launch. It writes registers like `COMPUTE_PGM_LO`, `COMPUTE_PGM_RSRC1/2/3`, `COMPUTE_TMPRING_SIZE`, `COMPUTE_USER_DATA_0`, `COMPUTE_RESOURCE_LIMITS`, `COMPUTE_START_X`, then emits `DISPATCH_DIRECT`.
4. Buffers and scalar values are not compiled into the PM4 stream. They are bound through a kernargs buffer, whose address is written into user data SGPRs.
5. Timeline synchronization, timestamps, profiling, copy engines, and memory barriers are handled through HCQ queues and signals.

AMD gives several useful hints:

- Precompile the expensive, semantic part: the shader binary and fixed metadata.
- Keep the launch packet dynamic where it naturally depends on buffers, scalar values, grid size, scratch, profiling, and timeline state.
- Use typed register wrappers where possible. AMD's `AMDReg` objects can encode fields by name, so command emission is not just magic integers.
- Separate hardware packet emission from compiler policy. `AMDProgram` parses the code object; `AMDComputeQueue` emits launch packets; `AMDAllocator` owns memory.
- A low-level backend can still fit tinygrad style if the runtime has a stable program/queue ABI.

AMD does not suggest that Rockchip should put every launch register into an immutable binary file. It suggests a hybrid: compile fixed hardware state into a launch template and patch the few dynamic fields at runtime.

## Runtime Register Emission vs Precompiled Register Templates

### Runtime Register Emission

Pros:

- Maximum flexibility while the hardware model is still being discovered.
- Easy to add shape-specific workarounds because Python can branch directly on dimensions and op kinds.
- Good for bring-up, debugging, and reverse engineering.
- No separate artifact format, relocation format, or validator is required.

Cons:

- The runtime becomes the compiler, which makes `ops_rockchip.py` large.
- Register programming details are repeated across elementwise, conv, and WMMA paths.
- Launch cost includes register stream construction, tensor packing, and temporary buffer setup every time.
- It is easy to accidentally mix hardware policy, layout transforms, and tinygrad runtime plumbing.
- Hard to cache, diff, inspect, and replay command streams.

### Fully Precompiled Register Binary

Pros:

- Very small runtime if all command streams are generated elsewhere.
- Launch can become "load blob, patch addresses, submit".
- Binary blobs are easy to replay and compare against hardware traces.
- Helps enforce that compilation is separate from execution.

Cons:

- Too rigid for Rockchip right now. Many register fields depend on shape, dtype, op, layout packing, output stride, LUT contents, and special-case hardware rules.
- Input/output addresses are not the only dynamic fields. Surface stride, data cube width/channel, CBUF banking, feature grains, weight size, LUT setup, task metadata, and sometimes operation mode depend on compile-time shapes, but shapes are only known at tinygrad program compile time.
- If the binary file is generated out-of-tree, tinygrad loses readability and reviewability.
- If every shape/op needs a separate blob, coverage becomes a cache-management problem.
- Debugging becomes harder unless the blob format has a disassembler and metadata.

### Recommended Hybrid

Rockchip should become a compiled register-template backend, not a pure runtime emitter and not a pile of opaque static binaries.

The renderer/compiler should produce a structured binary package per tinygrad program:

- Header: magic, version, target NPU generation, op family, task count, required scratch sizes, and validation metadata.
- Register templates: arrays of 64-bit RKNPU register commands.
- Patch table: offsets inside the register arrays that must be filled with runtime addresses or scalar launch values.
- Buffer roles: which tinygrad argument is input, weight, output, scratch, LUT, packed temporary, or constant.
- Layout contract: required input/output packing, dtype, shape, stride, alignment, and unpacking rule.
- Optional debug metadata: human-readable register names, source uop hash, shape, and a text dump for diffing.

Runtime launch then becomes:

1. Resolve tinygrad buffers and scalars.
2. Allocate or reuse temporary packed buffers only if the compiled layout requires them.
3. Pack host/device data according to the layout contract.
4. Copy the register template into a command buffer.
5. Apply patch table entries for DMA addresses, object addresses, task addresses, and scalar fields.
6. Submit the task through the RKNPU DRM ioctl.
7. Sync and unpack if the compiled layout requires it.

This keeps the runtime small while preserving shape-specialized hardware setup.

## Proposed File Split

The target is for `tinygrad/runtime/ops_rockchip.py` to be under 500 lines. That does not mean all Rockchip code must fit in one file. The clean split is:

- `tinygrad/runtime/ops_rockchip.py`
  - Device open and finalize.
  - Allocator.
  - Program loading and launch.
  - RKNPU submit path.
  - Minimal sync/reset helpers.

- `tinygrad/runtime/support/rockchip.py`
  - RKNPU buffer wrappers.
  - Register command helpers.
  - Patch table data classes.
  - Template packing/unpacking.
  - Debug dump/disassembler helpers.

- `tinygrad/renderer/rockchip.py` or a support compiler module
  - Uop pattern recognition.
  - Shape and layout decisions.
  - Register-template generation.
  - Serialization of the compiled package.

- `test/test_rockchip.py`
  - Keep behavior tests.
  - Add compiler package and patch-table tests that run without hardware.

If tinygrad maintainers prefer fewer files, `support/rockchip.py` can hold both template structures and register builder code, while `ops_rockchip.py` stays runtime-only.

## Proposed Compiled Package

A minimal binary package can be pickled at first, matching the current Python backend style and avoiding premature format work. Once stable, replace it with a compact binary struct.

Initial Python representation:

```python
@dataclass(frozen=True)
class RKPatch:
  section: str          # "regcmd", "task", "data"
  offset: int           # byte or element offset
  kind: str             # "dma32", "obj64", "u32", "regfield"
  arg_index: int | None
  role: str             # "input", "weight", "output", "scratch", "cmd", "task"
  shift: int = 0
  mask: int = 0xffffffff
  addend: int = 0

@dataclass(frozen=True)
class RKTemplate:
  version: int
  family: str           # "elementwise", "conv1x1", "wmma"
  regcmd: tuple[int, ...]
  tasks: tuple[RKTaskTemplate, ...]
  patches: tuple[RKPatch, ...]
  layouts: tuple[RKLayout, ...]
  temps: tuple[RKTempBuffer, ...]
```

The important part is the patch table. It lets the compiler own register construction while the runtime owns only address resolution.

## Register Builder Direction

The current `emit_raw(target, reg, value)` should move out of the program class and become a tiny helper:

```python
def rkcmd(target:int, reg:int, value:int) -> int:
  return (((target + 1) & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)
```

For fields, mirror the AMD/QCOM style:

```python
def field(value:int, shift:int, mask:int) -> int:
  return (value << shift) & mask
```

Then create small builders per hardware family:

- `build_ew_template(op, size, dtype, lut_kind)`
- `build_conv1x1_template(in_channels, out_channels, spatial)`
- `build_wmma_template(m, n, k, out_dtype)`

Each builder returns `RKTemplate` with symbolic patches instead of concrete addresses.

Avoid a large generic abstraction. Rockchip only needs three or four families initially, and tinygrad style favors a few direct functions over a class hierarchy.

## Runtime Direction

`RockchipProgram.__init__` should only decode the compiled package and cache metadata:

- `self.template`
- `self.layouts`
- `self.tmp_sizes`
- maybe `self.regcmd_size`

`RockchipProgram.__call__` should:

- reset only if required by stability, preferably behind `ROCKCHIP_RESET_EACH_LAUNCH`.
- prepare buffers according to `self.layouts`.
- allocate task and command buffers.
- copy `template.regcmd`.
- apply patches.
- submit.
- copy/unpack outputs.

This removes the Python uop interpreter from the runtime path. Unsupported programs should not silently execute by interpreting uops on CPU inside `ops_rockchip.py`; they should either fail compilation or use a normal tinygrad fallback outside this backend. Silent CPU fallback inside a device runtime makes correctness and performance hard to reason about.

## Memory Model Plan

There are two viable stages.

Stage 1, minimal risk:

- Keep public Rockchip buffers as CPU `memoryview`.
- Use short-lived RKNPU buffers for packed inputs, weights, outputs, task buffers, and command buffers.
- Move all packing and register construction out of runtime.
- This is closest to current behavior and easiest to test.

Stage 2, real device residency:

- Switch `RockchipDevice` to use an allocator that returns RKNPU `HCQBuffer` objects.
- Implement `_copyin`, `_copyout`, `_offset`, and `_as_buffer`.
- Reuse persistent buffers instead of allocating temporary GEM objects per launch.
- Make template patches point directly at resident tinygrad buffers where layout allows.
- Keep pack/unpack only for layouts that the NPU cannot consume directly.

Stage 1 is the right first milestone. Stage 2 is more important for performance, but it changes tinygrad-visible memory behavior and should be done after the compiler/runtime boundary is clean.

## Should Rockchip Use HCQ?

Eventually yes, but not as the first refactor.

Rockchip has a hardware queue concept through RKNPU tasks and DRM submit, but it does not currently expose a timeline signal model like AMD/NV/QCOM. Forcing it into `HCQCompiled` immediately would mix two hard changes:

- template compilation, and
- asynchronous queue/signal semantics.

The first rearchitecture should keep `Compiled` and a direct blocking submit. After that, evaluate an HCQ version if the driver supports fences or if timeline behavior can be modeled cleanly through submit fences and sync ioctls.

An HCQ Rockchip backend would have:

- `RockchipSignal` based on a mapped fence/signal buffer or sync file if supported.
- `RockchipComputeQueue` that writes task descriptors and register command buffers.
- `RockchipAllocator` derived from `HCQAllocatorBase`.

Until fences and nonblocking submit are understood, a simple blocking runtime is better.

## Detailed Migration Plan

### Phase 0: Freeze Current Behavior

Goal: make the current backend measurable before restructuring.

Tasks:

- Run the current Rockchip tests on hardware with representative envs:
  - `ROCKCHIP=1 FORWARD_ONLY=1 python3 test/test_rockchip.py TestOps.test_tiny_add`
  - `ROCKCHIP=1 FORWARD_ONLY=1 python3 test/test_rockchip.py TestOps.test_add TestOps.test_sub TestOps.test_scalar_mul TestOps.test_where`
  - matmul and conv tests that currently pass on the RK3588 board.
- Save failing tests explicitly. Do not design the refactor around tests that never passed.
- Add one hardware-independent unit test for register command packing.
- Add one hardware-independent unit test that compiles a tiny elementwise add and inspects the package shape.

Exit criteria:

- Known pass/fail matrix exists.
- Register pack helper is tested without hardware.
- The current backend behavior is not being changed yet.

### Phase 1: Extract Register Helpers Without Behavior Change

Goal: reduce risk by moving pure functions first.

Tasks:

- Move `reg`, `emit_raw` logic into support helpers.
- Move `_align_up`, `_wmma_params`, `_conv_params`, LUT table generation, and conv/WMMA packing helpers out of `RockchipProgram`.
- Keep call sites behavior-identical.
- Use `git diff` after each small move and avoid whitespace-only churn.

Exit criteria:

- `ops_rockchip.py` still passes the same tests.
- No runtime behavior intentionally changed.
- Extracted helpers have small direct tests where possible.

### Phase 2: Introduce `RKTemplate`

Goal: make compiled output represent register commands plus patches.

Tasks:

- Add dataclasses or named tuples for template, patch, temp buffer, and layout.
- Implement serialization in the current `RockchipCompiler`.
- Change elementwise fast path first:
  - renderer recognizes the same flat elementwise add/sub/mul/max patterns.
  - compiler emits an `RKTemplate` with placeholder addresses.
  - runtime patches `DST_BASE_ADDR`, `RDMA_SRC_BASE_ADDR`, and `RDMA_EW_BASE_ADDR`.
- Keep old path behind a temporary env, for example `ROCKCHIP_OLD_RUNTIME_EMIT=1`, only during migration.

Exit criteria:

- Elementwise native path no longer calls a runtime `boilerplate`.
- Elementwise add/sub/mul/max pass at the same tolerance.
- The compiled package can be dumped and read by a small debug tool.

### Phase 3: Move LUT Elementwise To Templates

Goal: cover exp2, trunc, silu, comparisons, and relu rewrites.

Tasks:

- Compile LUT data and LUT register writes into the template.
- Treat LUT table values as compile-time constants based on op kind.
- Add patch entries only for addresses and runtime dimensions.
- Keep comparison custom ops as named template families until they are better understood.

Exit criteria:

- Existing exp2, silu, relu, cmplt, cmpeq, cmpne, where tests match current behavior.
- No Python uop interpretation is needed for these ops.

### Phase 4: Move WMMA/Matmul To Templates

Goal: compile the RK3588 matrix path as a shape-specialized program.

Tasks:

- Move `_wmma_params` into the compiler side.
- Compile CNA/CORE/DPU register setup into `RKTemplate`.
- Patch feature, weight, and output addresses.
- Keep input and weight packing in a layout helper first.
- Preserve the current runtime verification for one dot-product only behind `ROCKCHIP_VERIFY_WMMA=1`; do not keep it as mandatory runtime behavior forever.

Exit criteria:

- Fused matmul metadata becomes a compiled template instead of being parsed from a name string.
- Program names no longer need to encode every matmul dimension.
- Current passing matmul cases continue to pass.

### Phase 5: Move Conv1x1 To Templates

Goal: compile conv register setup and keep shape policy out of runtime.

Tasks:

- Move `_conv1x1_meta` shape recognition into the renderer/compiler module.
- Compile `_conv_params` result into the template.
- Patch input, weight, and output addresses.
- Keep pack/unpack helpers as explicit layout transforms.

Exit criteria:

- Conv1x1 runtime path is the same launch path as elementwise and WMMA.
- Runtime only sees "template plus layouts", not conv-specific register policy.

### Phase 6: Remove Python Uop Interpreter From Runtime

Goal: make Rockchip a hardware backend, not a mixed software interpreter.

Tasks:

- Delete or quarantine the generic uop interpreter path.
- Unsupported uops should fail compilation with an actionable error.
- If fallback is desired, do it at scheduler/device selection level, not hidden inside `RockchipProgram.__call__`.

Exit criteria:

- `RockchipProgram.__call__` has no loop over uops.
- Every supported program family has a template path.
- Unsupported programs are explicit.

### Phase 7: Shrink `ops_rockchip.py`

Goal: get the runtime file under 500 lines.

Expected contents:

- Imports and constants.
- `RockchipProgram`.
- `RockchipCompiler`.
- `RockchipRenderer` shell or import from renderer module.
- `RockchipAllocator`.
- `RockchipDevice`.
- Minimal submit, alloc, sync, and reset helpers.

Most register policy, shape policy, and layout code should live outside the runtime file.

Exit criteria:

- `wc -l tinygrad/runtime/ops_rockchip.py` is under 500.
- No behavior-only code is hidden in giant data blobs without a debug dump.

### Phase 8: Optional Device-Resident Buffers

Goal: make Rockchip performance closer to a real backend.

Tasks:

- Switch allocator to RKNPU GEM buffers.
- Keep CPU staging only for copyin/copyout.
- Reuse command/task buffers with LRU or per-program cached allocations.
- Add `_offset` support for buffer views.
- Add sync behavior only where required by the DRM driver and cache mode.

Exit criteria:

- Elementwise and matmul no longer copy every tinygrad buffer through temporary bytearrays unless layout packing requires it.
- Copy behavior is covered by direct copyin/copyout tests.

## Risk Register

- Hardware state reset: current code resets NPU every launch. Removing it may expose stale state bugs. Keep reset configurable until templates prove complete.
- Address patching: RKNPU register fields may be 32-bit DMA addresses, object addresses, or shifted fields. The patch table must distinguish these.
- Cache coherency: `ROCKCHIP_MEM_SYNC` is optional today. Device-resident buffers will require a stricter policy.
- Shape coverage: pure binary templates can explode by shape. Compile templates per tinygrad program, not as a fixed library of blobs.
- Opaque blobs: binary templates are acceptable only with a text dump/disassembler.
- Silent fallback: hidden CPU fallback makes tests pass while hardware support is absent. Remove it from the runtime path.
- Line count target: under 500 lines is reasonable only if support/compiler code moves elsewhere. It is not reasonable if every register table must remain in `ops_rockchip.py`.

## Recommendation

Do not convert Rockchip to a fixed library of precompiled binary files where only input and output addresses are patched. That loses too much flexibility and does not match what AMD is doing.

Do convert Rockchip to compiled register templates. The compiler should own all shape-specific register decisions, layout decisions, and LUT contents. Runtime should patch addresses, allocate/copy required buffers, submit tasks, and sync. This matches tinygrad's runtime style much better and gives a credible path to `ops_rockchip.py` under 500 lines without hiding the backend in unreadable blobs.

The AMD backend's main lesson is the boundary: compiled program metadata is stable, launch packets are dynamic, and the runtime is a queue/allocator/program loader. Rockchip should copy that boundary, not AMD's exact PM4 machinery.
