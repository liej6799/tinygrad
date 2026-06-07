# RKNN Runtime Guide

This guide covers `tinygrad/runtime/ops_rk.py`, the experimental RKNN API runtime. It is a thin tinygrad wrapper around a
prebuilt `.rknn` model loaded through Rockchip's `librknnrt.so`.

## What This Runtime Does

`DEV=RK` opens `RKDevice`, whose allocator stores tensors in CPU `bytearray` buffers. When tinygrad realizes a kernel on
this device, `RKRenderer` prints the UOps it received. For a float elementwise add or multiply with two or more inputs,
it emits an ONNX chain, converts it to RKNN with Toolkit2, and returns the generated `.rknn` bytes. If add conversion
fails, it retries with a custom `cstAdd` CPU op. A square (`M==N==K`) float matmul is recognized from its UOps — a
multiply-accumulate has both `ADD` and `MUL` on the float dtype, unlike the elementwise paths — and emitted as an ONNX
`MatMul` node. Other kernels fall back to reading the file pointed to by `RKNN_MODEL`.

At launch, `RKProgram`:

1. loads `librknnrt.so` from `RKNNRT_PATH`;
2. initializes a new RKNN context from the model bytes;
3. selects all RK3588 NPU cores with `RKNN_NPU_CORE_0_1_2`;
4. queries the model input and output metadata;
5. binds tinygrad input buffers with `rknn_inputs_set`;
6. runs the model with `rknn_run`;
7. fetches the single output with `rknn_outputs_get`;
8. copies the RKNN output back into the tinygrad output buffer;
9. releases the output and destroys the RKNN context.

This is useful for testing tinygrad's device/runtime plumbing against a fixed RKNN model. It is not yet a general tinygrad
compiler for Rockchip NPU ops.

## Requirements

- Rockchip RKNN runtime library available on the machine running tinygrad.
- A precompiled `.rknn` model compatible with that runtime and device.
- Python environment for this checkout:

```sh
source ~/tinygrad/.venv/bin/activate
```

The runtime defaults are:

```sh
RKNNRT_PATH=/data/rk3588/rknn-header/librknnrt.so
RKNN_MODEL=/data/test/add_10x10.rknn
RKNN_TARGET=rk3588
RKNN_FLOAT_DTYPE=float16
```

Override them when your files live somewhere else:

```sh
DEV=RK RKNNRT_PATH=/path/to/librknnrt.so RKNN_MODEL=/path/to/model.rknn python3 your_script.py
```

`RK` is not part of tinygrad's automatic device selection list, so select it explicitly with `DEV=RK` or `device="RK"`.

## Minimal Smoke Test

Use a tinygrad expression whose realized kernel has the same buffer contract as the RKNN model. For the default
`add_10x10.rknn`, that generally means one output plus the exact number, shape, and dtype of model inputs.

Use real input data, not `Tensor.ones`: a constant expression like `ones + ones + ones` is folded to a single constant
store before it reaches the renderer, leaving no input params, so it never exercises the conversion path.

```python
import numpy as np
from tinygrad import Tensor

a = Tensor(np.random.rand(10, 10).astype(np.float32), device="RK")
b = Tensor(np.random.rand(10, 10).astype(np.float32), device="RK")
c = Tensor(np.random.rand(10, 10).astype(np.float32), device="RK")
z = (a + b + c).realize()
print(z.numpy())
```

For a square matmul, the renderer emits an ONNX `MatMul` instead:

```python
import numpy as np
from tinygrad import Tensor

a = Tensor(np.random.rand(12, 12).astype(np.float32), device="RK")
b = Tensor(np.random.rand(12, 12).astype(np.float32), device="RK")
z = a.matmul(b).realize()
print(z.numpy())
```

Run it with:

```sh
source ~/tinygrad/.venv/bin/activate
DEV=RK RKNNRT_PATH=/path/to/librknnrt.so RKNN_MODEL=/path/to/add_10x10.rknn python3 smoke_rk.py
```

The printed UOps come from `RKRenderer.render`. For this add pattern, those UOps are matched and converted through ONNX
to RKNN inside the renderer.

To force the custom CPU op path for add:

```sh
RKNN_FORCE_CUSTOM_CPUOP=1 DEV=RK python3 smoke_rk.py
```

## Buffer And Dtype Contract

`RKProgram.__call__` expects tinygrad to pass buffers in this order:

```text
output, input0, input1, ...
```

The RKNN model must report exactly one output. The number of tinygrad input buffers must match the model's input count.

Only float buffers are accepted:

- `float16` when buffer bytes equal `n_elems * 2`;
- `float32` when buffer bytes equal `n_elems * 4`.

For outputs, the runtime accepts either the model output dtype or `float32`. When the tinygrad output buffer is `float32`,
it asks RKNN to return float data by setting `want_float`.

## Current Limitations

- Tinygrad UOp to RKNN compilation is limited to float elementwise add/mul with two or more inputs and square
  (`M==N==K`) float matmul. Non-square matmul and every other kernel fall back to `RKNN_MODEL`.
- Custom CPU op fallback is currently implemented only for add as `cstAdd`.
- Generated float models use RKNN Toolkit2's `float16` lowering by default, so large float32 values can lose precision
  before a custom CPU callback sees them.
- No model caching across launches. Each call initializes and destroys an RKNN context.
- No device-resident tinygrad memory. The allocator is CPU-backed and copies through RKNN API buffers.
- Single-output models only.
- Float16 and float32 buffers only.
- No quantized input/output path beyond whatever RKNN handles internally for the fixed model.
- No dynamic shape management beyond the metadata returned for the current model.
- No profiling from RKNN performance queries.

For direct RKNPU register work, see `rockchip_backend_consideration.md` and `tinygrad/runtime/ops_rockchip.py`. That path
is separate from this RKNN API wrapper.

## Common Failures

`OSError: cannot open shared object file`

Check `RKNNRT_PATH` and the dynamic loader dependencies of `librknnrt.so`.

`FileNotFoundError` for the model path

Check `RKNN_MODEL`. `RKRenderer` reads the model during render, before `RKProgram` runs.

`static RKNN model expects output plus ... inputs`

The tinygrad operation produced a different input count than the RKNN model expects. Use an expression matching the fixed
model's input signature.

`RKNN input ... wants float16/float32 buffer`

The tinygrad buffer byte size does not match the model tensor element count as fp16 or fp32. Check shape and dtype.

`rknn_* failed: <code>`

The RKNN runtime returned an error. Confirm that the model, runtime library, kernel driver, and target platform match.

## Extending The Runtime

The next practical improvements are:

- keep one RKNN context per `RKProgram` instead of recreating it every launch;
- query and print tensor names, dims, format, dtype, and quantization at high debug levels;
- support preallocated output buffers through RKNN memory APIs;
- add optional RKNN perf query support;
- decide whether this remains a fixed-model runner or becomes a real compiler/import path.

Keep `ops_rk.py` small. If model conversion, graph import, or tensor layout policy grows, put that work in support modules
instead of mixing it into launch-time runtime code.
