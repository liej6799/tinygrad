This is the guide for RKNN Implementation in Tinygrad

The goal of the project is to allow perform Tinygrad Operations (UOPS) -> Convert to RKNN Binaries file (Containing NPU / CPU Command
Data Type, Graph Structure, Custom OPS). -> Run with RKNN-RUntime.

Here are the previous efforts made.
1) /data/rk3588/rknn-creation - Recreate RKNN File with rknn_flatbuf and rc_template
2) /data/rk3588/rknn-decode - Convert Tinygrad UOPS to RKNN Graph, including reducing OPS to allow support.
3) /data/rk3588/librknnrt - Runtime
4) /data/rkt - Clean implementation with documented NPU Graph

These are the test we would like to made on tinygrad running on NPU or Hybrid (CPU and NPU)
1. test_add
2. simple_conv
3. simple_matmul
Try with different data type fp16, int16, int32, float. Make sure if the NPU can support it will handle it. 

What to note:
1. Inside the RK Renderer make sure only output one RKNN binary file, try to pack all the uop there.
2. Inside the RK Runtime try to decode the RKNN, if its unsupported due to data type, or unsupported OPS, allow CPU fallback.
3. Dont try to hardcode specific function to pass the test above, it should be generalized enough and should be able
to handle any UOPS provided. 

Tools to help investigate:
1. radare2
2. gdb

TIP: 
1. Try to create an example in onnx and export to RKNN and run on official library to see if the operatios is supported.
Can also use the tools provided to check the memory structure to make sure out attempt earlier able to produce the same result.

