# Rockchip Backend Consideration

## Objective

Decide how to re-architect `tinygrad/runtime/ops_rockchip.py` so it looks more like the rest of tinygrad's runtimes, using a few-hundred-line
runtime as a comparison target rather than a hard-coded limit, while keeping a realistic path to useful RK3588 NPU execution.

The specific question is whether Rockchip should move from runtime register emission to a compiled backend where almost all register command streams are built ahead of time, stored as a binary artifact, and patched at runtime only for input, output, scratch, and constant addresses. The AMD backend is the main comparison point because it is also a low-level driver backend that writes hardware command packets directly.

Important clarification: 500 lines is not a correctness rule. It is a smell test based on other runtimes: `ops_qcom.py` is currently 412 lines and
`ops_dsp.py` is 309 lines, while `ops_amd.py` is much larger because it contains several interface paths, profiling support, queues, allocation, and
program loading. This document should be as long as needed to make the refactor decision-complete. The final implementation should make
`ops_rockchip.py` small by moving compiler policy, register templates, layout transforms, and debug tooling into support modules.

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

### Size Comparisons

Line count is useful only as evidence about responsibility boundaries:

- `tinygrad/runtime/ops_qcom.py`: 412 lines. It directly emits Adreno command packets and register writes, but it still fits in a few hundred lines
  because it uses HCQ, keeps shader compilation in renderers/compiler support, and makes `QCOMComputeQueue.exec` the packet-emission boundary.
- `tinygrad/runtime/ops_dsp.py`: 309 lines. It is not a register backend; it is a compact compiled/offload backend where C generation, allocation,
  invocation, and mock support are all visible but narrowly scoped.
- `tinygrad/runtime/ops_amd.py`: 1110 lines. This is not evidence that Rockchip should stay monolithic. AMD is large because one runtime file covers
  KFD, direct PCI AM, USB AM, PM4 compute queues, AQL queues, SDMA copy queues, profiling, SQTT, allocation, and ELF code-object loading.
- `tinygrad/runtime/support/amd.py`: 110 lines. The typed register wrapper is small because generated register data is outside the runtime.
- `tinygrad/runtime/support/am/amdev.py` plus `tinygrad/runtime/autogen/am/*`: thousands of lines. This is direct hardware bring-up and generated
  ASIC structure/register material, not the normal launch path size target.
- `tinygrad/runtime/autogen/amd/*`: generated instruction/operand metadata for AMD codegen/disassembly. It is neither runtime policy nor launch code.

So the useful comparison is not "Rockchip must be less than 500 lines." The useful comparison is "QCOM can be a direct packet/register emitter in
412 lines because shader/compiler policy is not mixed into the runtime." Rockchip should aim for that shape. If the final runtime is 550 lines because
DRM ioctls need explicit structs or error handling, that is acceptable. If it is 900+ lines because layout transforms, op recognition, LUTs, and
register builders stayed in `ops_rockchip.py`, the refactor missed the point.

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

## What Must Leave `ops_rockchip.py`

To look like the compact tinygrad runtimes, the runtime file must stop being the owner of hardware policy. The current file has several large
responsibilities that should move out:

- Shape policy: `_wmma_params`, `_conv_params`, `_elementwise_meta`, `_conv1x1_meta`, and fused-matmul name parsing.
- Data layout policy: `_pack_conv_input`, `_pack_conv_weights`, `_unpack_conv_output`, WMMA input/weight packing, and output swizzle unpacking.
- Register policy: `_emit_conv_regs`, `boilerplate`, LUT register programming, `fill_lut`, and DPU/CNA/CORE field composition.
- Compiler policy: renderer rewrites for half-only arithmetic, custom comparison tricks, silu, trunc, relu, and `WHERE` lowering.
- Software execution policy: the generic Python uop interpreter in `RockchipProgram.__call__`.
- Debug trace shell-outs: `os.system("cd ~/npu/ops_reg/...")` dumps should move behind support debug helpers or an explicit dev-only tool.

The runtime should keep only the work that is genuinely launch-time:

- open `/dev/dri/card1`;
- allocate, mmap, sync, and free RKNPU GEM buffers;
- load a compiled template package;
- bind tinygrad buffers and scalar values to template patches;
- submit one or more RKNPU tasks;
- optionally block, sync, copy out, and profile.

That boundary is the practical meaning of "match the style of `ops_*.py`". A few-hundred-line runtime is only realistic if the compiler and register
builder live elsewhere.

## How The AMD Backends Work

There are four AMD-related layers in this tree, and they should not be collapsed into one lesson.

### `ops_amd.py`: compiled shader plus launch packets

`tinygrad/runtime/ops_amd.py` is the actual tinygrad AMD runtime. It is not "precompiled register blobs". It is a compiled shader backend plus
runtime command packets.

The launch flow is:

1. Renderers/compilers produce a real AMD GPU code object, usually through HIP/LLVM paths.
2. `AMDProgram` parses the ELF, applies relocations, extracts the HSA kernel descriptor from `.rodata`, allocates GPU memory for the image, copies it
   to the GPU, and stores launch metadata such as SGPR requirements, LDS size, scratch size, kernarg size, and program address.
3. `AMDComputeQueue.exec` emits PM4 packets for a particular launch. It writes registers like `COMPUTE_PGM_LO`, `COMPUTE_PGM_RSRC1/2/3`,
   `COMPUTE_TMPRING_SIZE`, `COMPUTE_USER_DATA_0`, `COMPUTE_RESOURCE_LIMITS`, `COMPUTE_START_X`, then emits `DISPATCH_DIRECT`.
4. Buffers and scalar values are not compiled into the PM4 stream. They are bound through a kernargs buffer, whose address is written into user data SGPRs.
5. Timeline synchronization, timestamps, profiling, copy engines, and memory barriers are handled through HCQ queues and signals.

This is the closest analogy for Rockchip's normal runtime boundary: compiled program metadata is stable, launch packets are dynamic, and queue methods
own hardware packet emission.

### AMD register/assembly compiler backend

There is also an AMD register/assembly backend in other branches. This is different from both `ops_amd.py` and the generated register metadata. The
clearest current branch in this checkout is `upstream/rdna3_vibes`:

- `tinygrad/renderer/rdna_new.py`: an RDNA3 renderer that walks tinygrad uops and emits AMD assembly DSL instructions directly.
- `tinygrad/renderer/rdna_regalloc.py`: a liveness-based VGPR/SGPR allocator. It tracks last uses, aliases, register ranges, vectorized allocations,
  scratch VGPRs, register-space buffers, loop-local buffers, and register reuse.
- `tinygrad/renderer/rdna_regalloc_ilp.py`: an alternate register allocator using an ILP strategy when `RDNA_ILP_REGALLOC` is enabled.
- `tinygrad/renderer/rdna_uops.py`: RDNA-specific uop lowering for unsupported or awkward operations, such as integer division/modulo, small dtype
  casts, half/bfloat16 lowering, and vector ALU devectorization.
- `extra/assembly/amd/*`: the AMD assembly DSL, instruction wrappers, emulator, pcode/ucode support, tests, and generated instruction families used by
  the renderer.

An older and simpler branch-local form exists on `upstream/amd_uop` in `extra/backends/rdna.py`. It converts uops into textual RDNA assembly with
explicit VGPR/SGPR assignment, emits `.amdgpu_metadata`, emits an HSA kernel descriptor, and relies on normal AMD program loading afterward.

This is a real register backend in the compiler sense:

- It chooses target instructions directly instead of asking LLVM/HIP to lower generic source.
- It owns VGPR/SGPR placement, liveness, scratch registers, and register-space temporary buffers.
- It performs target-specific uop rewrites before instruction selection.
- It still produces a kernel/program artifact. It is not a runtime path that rebuilds PM4 packets or hardware launch registers for every call.

For Rockchip, this changes the comparison. AMD has both:

- a register/assembly compiler path where target-specific instruction/register decisions happen before runtime; and
- a runtime path where `AMDProgram` loads the compiled artifact and `AMDComputeQueue.exec` emits launch packets.

That split is exactly the missing boundary in the current Rockchip file. If Rockchip needs a "register backend", it should mean a compiler-side
RKNPU register-template builder, not a runtime-side `Program.__call__` that continues to recognize ops, allocate layouts, and emit registers every
launch.

### AM: direct PCI/USB hardware ownership

The "AM" path is not a separate shader model. It is tinygrad's direct AMD hardware ownership path under `tinygrad/runtime/support/am/`, used by
`PCIIface` and `USBIface` in `ops_amd.py` instead of the kernel KFD interface. `AMDev` maps PCI BARs, discovers IP versions, loads firmware, builds
page tables, initializes PSP/SMU/GFX/SDMA blocks, creates rings, handles interrupts, and recovers from device hangs.

Important details for the Rockchip decision:

- AM contains register writes, firmware loading, page-table management, queue setup, and reset/recovery. That is bring-up and device management code,
  not per-kernel compiler policy.
- `PCIIface.create_queue` still returns an `AMDQueueDesc` consumed by the same `AMDComputeQueue`/`AMDCopyQueue` abstractions. AM changes how the queue
  is created and how memory is mapped; it does not make every kernel launch a giant Python register-builder function.
- AM is allowed to be large because it replaces a kernel driver path for AMD. Rockchip already has a DRM RKNPU driver, so copying AM's full direct
  hardware ownership model would add complexity without answering the current runtime/compiler split.

The useful AM lesson is narrower: low-level register access and device setup belong in support modules with typed helpers and explicit ownership, not
inside `Program.__call__`.

### AMD register metadata

There are also generated AMD register/instruction layers:

- `tinygrad/runtime/support/amd.py` defines `AMDReg` and `AMDIP`. These provide named register addresses, field masks, `encode`, `decode`, and
  version/IP lookup.
- `tinygrad/runtime/autogen/am/*` contains generated kernel/PM4/SDMA/SMU structs, constants, offsets, and firmware metadata used by AM and PM4 queue
  emission.
- `tinygrad/runtime/autogen/amd/*` contains generated AMD instruction/operand metadata for codegen/disassembly families such as CDNA and RDNA.

These files show why AMD command emission is still readable: the runtime can say `self.wreg(self.gc.regCOMPUTE_PGM_RSRC1, value)` or encode named
fields instead of scattering unexplained integers. The generated material is large, but it is data/support code. It does not imply that the launch
runtime should grow without bound.

### AMD lessons for Rockchip

AMD gives several useful hints:

- Precompile the expensive, semantic part: the shader or, for Rockchip, the shape-specialized NPU register template and layout metadata.
- Keep register allocation, target-specific lowering, and instruction/register selection on the compiler side, like the RDNA register/assembly
  branches do for VGPR/SGPR-backed AMD kernels.
- Keep launch state dynamic where it naturally depends on buffers, scalar values, grid size, scratch, profiling, timeline state, and task addresses.
- Use typed register wrappers where possible, so command emission is not just magic integers.
- Separate hardware packet emission from compiler policy. `AMDProgram` parses the code object; `AMDComputeQueue` emits launch packets; `AMDAllocator`
  owns memory; AM support owns direct hardware setup.
- A low-level backend can still fit tinygrad style if the runtime has a stable program/queue ABI.

AMD does not suggest that Rockchip should put every launch register into an immutable binary file. It suggests a hybrid: compile fixed hardware state
into a launch template and patch the few dynamic fields at runtime.

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

### Compiled Register Templates

One viable middle option is compiled register templates: not a pure runtime emitter and not a pile of opaque static binaries.

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

This would keep the runtime small while preserving shape-specialized hardware setup.

## Proposed File Split

The target is for `tinygrad/runtime/ops_rockchip.py` to be in the same few-hundred-line class as QCOM/DSP if possible. That does not mean all
Rockchip code must fit in one file. The clean split is:

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

## Target Runtime Line Budget

The refactored `ops_rockchip.py` should be budgeted before implementation. The budget is a design pressure, not a hard pass/fail gate, so new
register features do not drift back into the runtime file.

- Imports, constants, and tiny helpers: 35 lines.
- `RockchipProgram`: 120 lines.
  - decode template package;
  - allocate per-launch temporary buffers;
  - apply patches;
  - submit and copy/unpack outputs through support helpers.
- `RockchipCompiler`: 20 lines.
  - base64/pickle or final binary decode;
  - debug disassemble hook if cheap.
- `RockchipRenderer` shim: 40 lines.
  - ideally just imports or delegates to a renderer/support module;
  - no large pattern matcher in the runtime file.
- `RockchipAllocator`: 45 lines.
  - Stage 1 CPU-backed memoryview allocator, or Stage 2 RKNPU `HCQBuffer` allocator;
  - no register or layout logic.
- `RockchipDevice`: 140 lines.
  - file descriptor open;
  - GEM create/map/flink/destroy;
  - memory sync;
  - reset;
  - finalization.
- Submit helpers: 60 lines.
  - create `struct_rknpu_task`;
  - create `struct_rknpu_submit`;
  - call `DRM_IOCTL_RKNPU_SUBMIT`;
  - no op-specific register generation.

This rough budget lands near the QCOM/DSP scale. The review check should be comparative:

```sh
wc -l tinygrad/runtime/ops_rockchip.py tinygrad/runtime/ops_qcom.py tinygrad/runtime/ops_dsp.py tinygrad/runtime/ops_amd.py
```

If the runtime is modestly above 500 lines because DRM setup, validation, or errors are explicit, that can still be fine. If it is closer to the
current 1285 lines or grows because op-specific register generation, packers, LUTs, and debug tools stayed in the runtime, the implementation is not
done. Move code to support/compiler modules instead of arguing about the exact number.

## Current Code Movement Map

This is the concrete ownership map for the existing `ops_rockchip.py` functions.

- Keep in `ops_rockchip.py`:
  - `RockchipProgram.__init__`, rewritten to decode an `RKTemplatePackage`;
  - `RockchipProgram.__call__`, rewritten as a short bind/patch/submit path;
  - `RockchipCompiler`, only as package decode/cache glue;
  - `RockchipAllocator`, either CPU-backed Stage 1 or RKNPU-backed Stage 2;
  - `RockchipDevice.create_flink_name`, `_gpu_alloc`, `_gpu_sync`, `_gpu_free`, `_gpu_free_multiple`, `reset_npu`.

- Move to `tinygrad/runtime/support/rockchip.py`:
  - `reg`, `emit_raw`, `fill_lut`, and all register command packing;
  - RKNPU task-template construction;
  - patch application;
  - command dump/disassemble helpers;
  - reusable GEM buffer wrappers if `HCQBuffer` is not enough;
  - `submit` and `submit_conv` internals, generalized into one `submit_template`.

- Move to a Rockchip compiler/renderer support module:
  - `RockchipRenderer.pre_matcher`;
  - `RockchipRenderer.extra_matcher`;
  - `_elementwise_meta`;
  - `_conv1x1_meta`;
  - fused matmul metadata currently encoded in the program name;
  - conversion from tinygrad uops to `RKTemplatePackage`.

- Move to a layout support module:
  - `_pack_conv_input`, `_pack_conv_weights`, `_unpack_conv_output`;
  - WMMA input/weight packing and output swizzle decoding;
  - dtype conversions needed by half-only hardware paths;
  - shape-derived temporary buffer size calculations.

- Delete from `ops_rockchip.py` after templates cover the path:
  - generic uop interpreter loop in `RockchipProgram.__call__`;
  - runtime calls to `boilerplate`;
  - runtime construction of op-specific DPU/CNA/CORE registers;
  - hidden CPU fallback for unsupported hardware programs.

The final runtime file may import many helpers, but it should not contain the logic those helpers implement.

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

## Compiled Package Contract

The package must be explicit enough that `RockchipProgram.__call__` never asks "what op is this?" It should only ask "which buffers and scalar
values patch this template?"

Required package fields:

- `magic`: constant such as `b"RKTP"` so runtime can reject old pickled uop lists.
- `version`: integer, bumped whenever serialization or patch semantics change.
- `target`: string such as `"rk3588-rknpu2"`; runtime rejects unknown targets unless a compatibility flag is set.
- `families`: tuple of task families, usually one entry at first: `"elementwise"`, `"lut"`, `"wmma"`, or `"conv1x1"`.
- `regcmd`: immutable tuple of 64-bit register commands with placeholder values already encoded.
- `tasks`: task descriptors with `op_idx`, `enable_mask`, `int_mask`, `int_clear`, `core_mask`, and submit flags.
- `patches`: patch table describing every runtime-written field.
- `layouts`: buffer layout transforms needed before submit or after completion.
- `temps`: temporary buffers required by this program, with size expressions resolved at compile time where possible.
- `debug`: optional source uop hash, shape summary, register-name dump, and original tinygrad program name.

Patch kinds should be deliberately small:

- `dma32`: patch a 32-bit DMA address into a register command value field.
- `dma32_add`: patch a 32-bit DMA address plus compile-time addend, for cases like weight data after `REGCMD_RESERVED`.
- `obj64`: patch an RKNPU object address into a task or submit structure.
- `u32`: patch a plain scalar value.
- `regfield`: patch `((value + addend) << shift) & mask` into the value bits of an existing register command.
- `task_regcmd_addr`: patch the command-buffer DMA address into the task descriptor.

Avoid generic arbitrary Python callbacks in the package. If patching needs code execution, the template ABI is not precise enough.

The initial serializer can be pickle because current Rockchip and Python renderers already use pickle-style blobs. The package should still have a
magic/version wrapper so old blobs fail cleanly:

```python
payload = pickle.dumps(RKTemplatePackage(...))
lib = b"RKTP" + bytes([RK_TEMPLATE_VERSION]) + payload
```

After the refactor is stable, replace pickle with a packed binary format only if startup cost, cache stability, or reviewability requires it.

### Template ABI Invariants

The package should define invariants that are checked before launch. Without these checks, a bad template can turn into silent memory corruption or
an unrecoverable NPU hang.

- Register command arrays must have a byte size that fits the allocated command buffer and a command count that fits `struct_rknpu_task.regcfg_amount`.
- Every register patch must point at an existing command and must modify only the 32-bit value field, unless its kind explicitly targets task or data storage.
- Address patches must document whether the field expects `dma_addr`, `obj_addr`, flink name, or a CPU virtual address. The current code uses both
  `dma_addr` for register fields and `obj_addr` for submit/task buffers, and confusing the two is a launch-breaking bug.
- `task_obj_addr`, `regcmd_addr`, `regcfg_offset`, `op_idx`, `enable_mask`, `core_mask`, interrupt masks, submit flags, and timeout should be template data,
  not hidden constants in `submit_template`.
- The package should include `max_working_set_bytes` or per-temp byte sizes so runtime allocation failures happen before partially initialized submits.
- Patch application must reject overflow after addend and shift, not just before them.
- Template validation should be hardware-free and deterministic so CI can exercise it without opening `/dev/dri/card*`.

This means the first version of `RKTemplatePackage` should be treated as an ABI, even if it is serialized with pickle. The ABI can be small, but it
must be stricter than "a tuple of Python objects".

## Template Families

Elementwise template:

- compile DPU and DPU_RDMA register commands for one binary op over packed half values;
- patch output, lhs, and rhs DMA addresses;
- keep op-specific ALU mode, data cube width/channel, and output conversion in compile-time registers.

LUT/custom template:

- compile LUT data and all LUT setup registers;
- treat relu, exp2, silu, trunc, cmplt, cmpeq, cmpne, and where lowerings as named template variants;
- patch only input/output addresses and any scalar dimensions that are truly runtime symbolic.

WMMA template:

- compile CNA, CORE, and DPU register setup from `(m, n, k, dtype, layout)`;
- patch feature, weight, and output addresses;
- move current output swizzle handling into a layout descriptor;
- replace name-encoded `rkmm_v1_...` metadata with structured package fields.

Conv1x1 template:

- compile `_conv_params` result into register commands;
- patch packed input, packed weight, and packed output DMA addresses;
- keep input/weight/output packers in layout support, not runtime.

Generic uop template:

- do not implement in v1. If a program does not match a hardware family, compilation should fail with an explicit unsupported message.
- Future support should add more hardware template families, not reintroduce a Python interpreter into `ops_rockchip.py`.

Multi-task template:

- support in the package format from v1, even if the first compiler emits only one task;
- each task descriptor should carry `op_idx`, `enable_mask`, `int_mask`, `int_clear`, `regcfg_offset`, `regcfg_amount`, and subcore task routing;
- this prevents the runtime ABI from assuming the current elementwise/conv one-task shape and makes later fused pipelines possible without another
  package redesign.

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

### Runtime Launch Flow

The new runtime launch should be deterministic and short:

1. Optionally reset the NPU if `ROCKCHIP_RESET_EACH_LAUNCH=1` or until stability proves reset is unnecessary.
2. Create a launch context with the template package, tinygrad buffers, scalar values, and env-controlled debug flags.
3. Ask layout helpers to prepare input buffers.
   - Stage 1: pack from CPU `memoryview` into temporary RKNPU GEM buffers.
   - Stage 2: use resident RKNPU buffers directly when layout-compatible.
4. Allocate command and task buffers.
5. Copy template `regcmd` into the command buffer.
6. Apply patches in one helper call, using resolved DMA/object addresses and scalar values.
7. Materialize `struct_rknpu_task` descriptors from task templates.
8. Submit through `DRM_IOCTL_RKNPU_SUBMIT`.
9. Sync output buffers if required.
10. Ask layout helpers to unpack outputs.
11. Free or cache temporary buffers.
12. Return elapsed time only when `wait=True`, matching other tinygrad program APIs.

No step above requires knowing whether the program is add, conv, WMMA, or silu. That knowledge belongs to the template.

### Intended Runtime Skeleton

The runtime file should roughly look like this after refactor:

```python
class RockchipProgram:
  def __init__(self, dev, name, lib, **kwargs):
    self.dev, self.name, self.pkg = dev, name, decode_template(lib)
    validate_template(self.pkg, dev.target)

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kw):
    with rockchip_launch(self.dev, self.pkg, bufs, vals, wait) as launch:
      launch.prepare_layouts()
      launch.copy_regcmd()
      launch.apply_patches()
      self.dev.submit_template(launch)
      launch.finish_outputs()
      return launch.elapsed if wait else None
```

The real code can be flatter than this if it is shorter, but the ownership should remain: `ops_rockchip.py` orchestrates; helpers decide layout,
registers, and patch semantics.

### Runtime Failure Modes

The runtime should fail early and explicitly:

- bad magic/version: `RuntimeError("unsupported Rockchip template version ...")`;
- unsupported target: `RuntimeError("compiled for rk3588..., running on ...")`;
- missing buffer role: `RuntimeError("template requires role input0 but only ... buffers were passed")`;
- patch overflow: `RuntimeError("value ... does not fit patch ...")`;
- submit ioctl failure: preserve the underlying `OSError` and include program name and task family;
- output validation failure: only if an explicit verification env is enabled.

These errors are more useful than falling into the Python interpreter or returning silently corrupted output.

### Device Discovery

The current runtime hardcodes `/dev/dri/card1`. The refactor should not bake that into the new architecture.

- Keep an env override such as `ROCKCHIP_DRM=/dev/dri/card1` for bring-up and board-specific setups.
- Add best-effort discovery by probing DRM nodes for the RKNPU driver name or by attempting a harmless RKNPU ioctl and selecting the matching node.
- Store the discovered path and driver/version metadata on `RockchipDevice` and include it in template validation/debug dumps.
- If discovery fails, raise a clear error listing probed nodes and the override env.

This is not central to register templates, but it matters for making the backend usable across RK3588 images where the NPU card index is not stable.

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

### Buffer Lifetime Policy

Stage 1 should keep the current external behavior, even if it is not optimal:

- tinygrad buffers stay CPU-backed;
- each hardware launch allocates task, command, packed input, packed weight, and packed output RKNPU buffers;
- helper code owns pack/unpack copies;
- allocator behavior visible to the rest of tinygrad remains unchanged.

Stage 1 can still cache internal temporary buffers after correctness is stable. Cache only by exact size and role, and clear the cache on device
finalization. Do not let temporary caching change the public allocator contract.

Stage 2 should change one thing at a time:

- first make public allocations return RKNPU buffers while keeping pack/unpack temp paths;
- then skip temp buffers for layout-compatible elementwise inputs;
- then skip temp buffers for WMMA/conv only when template layouts can describe the resident format;
- finally add `_transfer` only if peer devices or disk copy paths need it.

This order prevents a memory-model refactor from being confused with the compiler/template refactor.

### Buffer Views And Offsets

The template ABI must account for tinygrad buffer views before Stage 2 begins. Other backends can launch kernels with non-zero buffer offsets; a
Rockchip template that only patches the base allocation address will read or write the wrong tensor slice.

- Stage 1 CPU-backed buffers can keep copying from the exact `memoryview` passed by tinygrad, which naturally includes the view offset.
- Stage 2 resident buffers need `_offset` support and patch resolution must use `base_dma + view_offset`, with alignment validation per template role.
- Layout descriptors should state whether a role accepts arbitrary byte offsets, element-aligned offsets, or only zero-offset packed temporaries.
- Tests should include at least one sliced/viewed elementwise input once resident buffers are introduced.

This should be designed before changing the public allocator, because retrofitting offset semantics into register patches is error-prone.

### Cache Coherency Policy

`ROCKCHIP_MEM_SYNC` is currently optional and allocations are forced non-cacheable. The refactor should make the policy explicit instead of inheriting
the current env-only behavior accidentally.

- Stage 1 can keep non-cacheable GEM buffers and optional sync for parity with current behavior.
- Any cacheable or write-combine allocation must have required sync directions recorded by the layout/temp descriptor.
- Command and task buffers need a `TO_DEVICE` sync if the driver/platform does not make CPU writes visible automatically.
- Output buffers need `FROM_DEVICE` before CPU unpacking unless non-cacheable mappings are proven coherent on the target board.
- Debug validation should be able to force syncs on every buffer to distinguish coherency bugs from register-template bugs.

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

Additional safety check:

- In a temporary debug mode, build the old runtime register stream and the new compiled template for the same elementwise program, normalize patched
  addresses, and diff the command stream before submit. This catches template emission regressions without relying only on numeric output.

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

Goal: get the runtime file into the QCOM/DSP style: a small orchestration layer, ideally a few hundred lines, with no compiler/layout/register policy.

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

- `wc -l tinygrad/runtime/ops_rockchip.py tinygrad/runtime/ops_qcom.py tinygrad/runtime/ops_dsp.py tinygrad/runtime/ops_amd.py` shows Rockchip is
  in the small-runtime range, or any extra size is justified by DRM/RKNPU plumbing rather than op policy.
- `rg -n "def boilerplate|def _wmma_params|def _conv_params|for idxs in itertools.product" tinygrad/runtime/ops_rockchip.py` returns nothing.
- `rg -n "emit_raw|fill_lut|_pack_conv|_unpack_conv" tinygrad/runtime/ops_rockchip.py` returns nothing, except imports if needed.
- The only op-family branching in `RockchipProgram.__call__` is through template/layout helper dispatch, not hardcoded register emission.
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

### Phase 9: Optional Async/Fence Support

Goal: only after blocking templates are stable, investigate whether Rockchip can expose nonblocking submit and profiling in a tinygrad-compatible way.

Tasks:

- Experiment with `RKNPU_JOB_NONBLOCK`, `RKNPU_JOB_FENCE_IN`, and `RKNPU_JOB_FENCE_OUT` in isolation.
- Determine whether fence file descriptors can model tinygrad timeline waits without busy polling or global synchronization.
- Add profiling timestamps only if the driver exposes trustworthy completion timing; otherwise keep wall-clock `wait=True` timing.

Exit criteria:

- Either document why HCQ/asynchronous support is not currently viable, or introduce a small queue layer without changing the template package ABI.

## Acceptance Matrix

The refactor is complete only when each current capability has a new owner and a test signal.

- Add/sub/mul/max elementwise:
  - owner: elementwise template builder;
  - runtime responsibility: patch three DMA addresses and submit;
  - tests: tiny add, add/sub/scalar mul, max/min where currently supported.
- LUT/custom elementwise:
  - owner: LUT template builder;
  - runtime responsibility: patch addresses only;
  - tests: exp2, silu, relu, trunc, cmplt, cmpeq, cmpne, where.
- WMMA/fused matmul:
  - owner: WMMA template builder and layout packer;
  - runtime responsibility: allocate/patch feature, weight, output buffers;
  - tests: current fused matmul test cases and optional `ROCKCHIP_VERIFY_WMMA=1`.
- Conv1x1:
  - owner: conv template builder and layout packer;
  - runtime responsibility: allocate/patch packed input, weight, output buffers;
  - tests: known passing conv1x1 cases, including `in_channels == 1` expansion path.
- Unsupported uops:
  - owner: compiler error path;
  - runtime responsibility: none;
  - tests: compile unsupported program and assert a clear unsupported error.
- Runtime size:
  - owner: test/check script;
  - runtime responsibility: remain small compared with QCOM/DSP and far below the current monolith;
  - tests: line-count comparison plus `rg` checks that op-specific helpers are gone from `ops_rockchip.py`.

## Suggested Hardware-Free Tests

Hardware-free tests are important because most contributors will not have RK3588 access.

- `test_rockchip_template_pack`: verify `rkcmd(target, reg, value)` packs exactly the same 64-bit command as old `emit_raw`.
- `test_rockchip_patch_dma32`: build a one-command template with a placeholder address and verify patching updates only value bits.
- `test_rockchip_patch_regfield`: verify shift/mask/addend behavior and overflow rejection.
- `test_rockchip_template_roundtrip`: serialize and deserialize an `RKTemplatePackage` and assert equality.
- `test_rockchip_elementwise_compile_shape`: compile a tiny half add and assert the package family, roles, and patch kinds.
- `test_rockchip_reject_old_pickle`: pass an old pickled uop list without magic and assert a clear version error.
- `test_rockchip_template_validate_task_bounds`: reject task descriptors whose register offset/count exceed the command array.
- `test_rockchip_template_validate_address_kinds`: verify `dma_addr` and `obj_addr` patch kinds cannot be applied to the wrong section.
- `test_rockchip_template_diff_old_elementwise`: under a helper-only path, compare normalized old and new elementwise register streams.
- `test_rockchip_layout_offset_reject`: verify a layout that requires zero-offset packed input rejects a non-zero resident-buffer offset.
- `test_rockchip_runtime_line_count`: compare `ops_rockchip.py` against QCOM/DSP/AMD line counts and fail only on a clearly monolithic regression or
  op-policy symbols returning to the runtime.

These tests should not open `/dev/dri/card1`. Put hardware-open tests behind existing Rockchip env behavior.

## Suggested Hardware Tests

Hardware tests should preserve the current pass/fail reality instead of inventing new coverage first.

- Smoke:
  - `ROCKCHIP=1 FORWARD_ONLY=1 python3 test/test_rockchip.py TestOps.test_tiny_add`
  - `ROCKCHIP=1 FORWARD_ONLY=1 python3 test/test_rockchip.py TestOps.test_tiny_mul`
- Elementwise:
  - add, sub, scalar mul, where, relu, exp2, silu, comparison tests that currently pass.
- Matrix:
  - fused matmul cases with and without `ROCKCHIP_VERIFY_WMMA=1`.
- Conv:
  - current conv1x1 cases, with `ROCKCHIP_NATIVE_CONV=1`.
- Debug:
  - run one program with template dump enabled and verify the dump can be disassembled without submitting again.
- Discovery:
  - run with default DRM discovery and with `ROCKCHIP_DRM=/dev/dri/card1` to ensure both paths select the same NPU on the RK3588 board.
- Coherency:
  - run one elementwise case with forced syncs and one with the default policy, then compare outputs to isolate sync-related failures.

## Risk Register

- Hardware state reset: current code resets NPU every launch. Removing it may expose stale state bugs. Keep reset configurable until templates prove complete.
- Address patching: RKNPU register fields may be 32-bit DMA addresses, object addresses, or shifted fields. The patch table must distinguish these.
- Cache coherency: `ROCKCHIP_MEM_SYNC` is optional today. Device-resident buffers will require a stricter policy.
- Shape coverage: pure binary templates can explode by shape. Compile templates per tinygrad program, not as a fixed library of blobs.
- Opaque blobs: binary templates are acceptable only with a text dump/disassembler.
- Silent fallback: hidden CPU fallback makes tests pass while hardware support is absent. Remove it from the runtime path.
- Line count target: a few-hundred-line runtime is reasonable only if support/compiler code moves elsewhere. The exact number is less important than
  ensuring register tables, layout transforms, op recognition, and debug tooling are not in `ops_rockchip.py`.
- Device node instability: `/dev/dri/card1` is board-image dependent. Hardcoding it will make the backend look broken on otherwise compatible RK3588 systems.
- Buffer views: resident-buffer support is unsafe unless patching accounts for tinygrad offsets and alignment constraints.
- Template ABI drift: pickle makes it easy to change fields accidentally. Magic/version checks and validation tests are required from the first template version.
- Multi-task assumptions: a one-task-only runtime may pass v1 tests but force another refactor for fused DPU/CNA/CORE pipelines.

## Final Plan Review

The full plan points to a boundary problem, not just a line-count problem.

What the comparisons show:

- QCOM proves a direct packet/register emitter can be a few hundred lines when shader compilation, binary parsing, allocation, and command queue
  emission have clear boundaries.
- DSP proves unusual devices can stay compact when the runtime is only an offload/invocation layer and does not also own compiler policy.
- `ops_amd.py` proves a low-level backend can emit hardware launch packets dynamically while still treating the compiled program as a durable object.
- AM proves direct hardware ownership and register writes belong in support/device-management layers, not in every program launch.
- The AMD register/assembly branches prove register allocation and target-specific instruction decisions can be compiler-side code. They do not
  justify keeping semantic lowering and register allocation in the runtime.

What the current Rockchip code shows:

- `ops_rockchip.py` is currently a runtime, compiler, layout packer, register builder, debug tool, and software fallback in one file.
- The current public buffer model is CPU-backed, so the first refactor should not also force a resident-device allocator.
- Current RKNPU work is shape/op-family-specific. A fixed library of prebuilt blobs would either be too rigid or become an unreviewable cache of
  shape variants.
- Runtime register emission is useful for bring-up, but keeping it as the architecture guarantees the file remains large and hard to test.

What should be staged:

- Phase 0 and Phase 1 are still necessary. Freeze current behavior and extract pure helpers before changing the launch model.
- Phase 2 should introduce the template ABI with elementwise first, because it gives the smallest proof that compile-time register construction plus
  runtime patching works.
- LUT, WMMA, and conv should move only after the elementwise template path can be compared against the old normalized register stream.
- Device-resident buffers and HCQ/fence support should remain later phases. They are important but orthogonal to the compiler/runtime boundary.

What should not be done:

- Do not rewrite Rockchip as a CUDA/HIP-style shader backend unless a real RK3588 NPU shader ISA/code-object path appears. The current known interface
  is RKNPU task/register submission, not general shader loading.
- Do not keep `Program.__call__` as the op recognizer and register emitter. That is the current failure mode.
- Do not use opaque static register blobs with only input/output address patching. Too many fields are shape/layout/op dependent, and reviewability
  would get worse.
- Do not hide unsupported programs behind a Python interpreter inside the Rockchip runtime. Compilation should fail explicitly or fallback should be
  handled outside this backend.

## Recommendation

Do not convert Rockchip to a fixed library of precompiled binary files where only input and output addresses are patched. That loses too much
flexibility and does not match AMD, QCOM, or the AMD register/assembly branches.

Do convert Rockchip to compiled register templates. The compiler should own all shape-specific register decisions, layout decisions, and LUT contents.
Runtime should patch addresses, allocate/copy required buffers, submit tasks, and sync. This matches tinygrad's runtime style much better and gives a
credible path to a QCOM/DSP-sized `ops_rockchip.py` without hiding the backend in unreadable blobs.

So the best description is: Rockchip should be rewritten as a compiled register-template backend.

- "Compiled" means tinygrad program compilation decides op family, shapes, layouts, LUT contents, task descriptors, register commands, validation
  metadata, and patch tables.
- "Register-template" means the compiled artifact is still RKNPU register/task data, not a shader ISA binary.
- "Runtime" means `ops_rockchip.py` only orchestrates allocation, layout preparation, patch application, DRM submit, sync, and output unpacking.

The AMD backend's main lesson is the boundary: compiler-side register/instruction decisions happen before runtime; compiled program metadata is
stable; launch packets are dynamic; and the runtime is a queue/allocator/program loader. Rockchip should copy that boundary, not AMD's exact PM4
machinery.
