#ifndef RKNPU_UTILS_H
#define RKNPU_UTILS_H

#include <sys/ioctl.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <libdrm/drm.h>
#include "rknpu_ioctl.h"
#include "rknn_api.h"
#include "rknpu_register.h"


#ifdef __cplusplus
extern "C" {
#endif
struct reg_list {
  uint64_t *data;
  size_t size;
  size_t capacity;
};


static struct reg_list regs;

static void
reg_list_clear(struct reg_list *list)
{
  if (!list) return; // null pointer check
  
  list->size = 0;
  // Note: we keep the allocated memory for reuse
}

static void
reg_list_push(struct reg_list *list, uint64_t value)
{
  if (!list) return; // null pointer check
  
  if (list->size >= list->capacity) {
    size_t new_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
    uint64_t *new_data = NULL;
    
    if (list->data == NULL) {
      new_data = (uint64_t*)malloc(new_capacity * sizeof(uint64_t));
    } else {
      new_data = (uint64_t*)realloc(list->data, new_capacity * sizeof(uint64_t));
    }

    if (!new_data) return; // memory allocation failed
    list->data = new_data;
    list->capacity = new_capacity;
  }
    
  // Store the value - this was the missing line causing the segfault
  list->data[list->size++] = value;
}

static void
emit_raw(struct reg_list *arr, uint32_t target, uint32_t reg,
         uint64_t value)
{
  uint64_t packed_value = 0;
  packed_value = ((uint64_t)target) << 48;
  packed_value |= ((uint64_t)value) << 16;
  packed_value |= (uint64_t)reg;
  reg_list_push(arr, packed_value);
}


 static void
 emit(uint32_t reg, uint64_t value)
 {
    uint32_t target = rkt_get_target(reg) + 0x1;
    emit_raw(&regs, target, reg, value);
 }
 
 #define EMIT(offset, value) emit(offset, value);
 
 
struct MemHandles {
    void* input;
    void* weights;
    void* output;
    uint64_t input_dma, input_obj;
    uint64_t weights_dma, weights_obj;
    uint64_t output_dma, output_obj;
    uint64_t tasks_obj;
};

int get_type_size(rknn_tensor_type type){
    switch (type){
        case RKNN_TENSOR_INT8:
            return sizeof(int8_t);
        case RKNN_TENSOR_UINT8:
            return sizeof(uint8_t);
        case RKNN_TENSOR_INT16:
            return sizeof(int16_t);
        case RKNN_TENSOR_UINT16:
            return sizeof(uint16_t);
        case RKNN_TENSOR_INT32:
            return sizeof(int32_t);
        case RKNN_TENSOR_UINT32:
            return sizeof(uint32_t);
        case RKNN_TENSOR_INT64:
            return sizeof(int64_t);
        case RKNN_TENSOR_FLOAT16:
            return sizeof(__fp16);
        case RKNN_TENSOR_FLOAT32:
            return sizeof(float);
        case RKNN_TENSOR_BFLOAT16:
            return sizeof(ushort);
        default:
            printf("    get_type_size error: not support dtype %d\n", type);
            return 0;
    }
}
void create_channel(int channel)
{
    EMIT(REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(channel) |
    DPU_DATA_CUBE_CHANNEL_CHANNEL(channel));
    
    EMIT(REG_DPU_WDMA_SIZE_0,
    DPU_WDMA_SIZE_0_CHANNEL_WDMA(channel));

    EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(channel));
}

void create_size(int height, int width)
{
    EMIT(REG_DPU_WDMA_SIZE_1,
        DPU_WDMA_SIZE_1_HEIGHT_WDMA(height) | DPU_WDMA_SIZE_1_WIDTH_WDMA(width));
            
    EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(width));
        EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
            DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(height));
   
}

void create_stride(int stride)
{
    EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
        DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(stride));
    EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(stride))
}

void create_surf_notch(int notch)
{
    EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH,
        DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(notch));

    EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
        DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(notch)); 
}

int get_precision(int dtype)
{
    switch (dtype)
    {
        case RKNN_TENSOR_INT8:
            return 0;
        case RKNN_TENSOR_INT16:
            return 1;
        case RKNN_TENSOR_FLOAT16:
            return 2;
        case RKNN_TENSOR_BFLOAT16:
            return 3;
        case RKNN_TENSOR_INT32:
            return 4;
        case RKNN_TENSOR_FLOAT32:
            return 5;
        default:
            return 0;
    }
}

int get_edata_size(int dtype)
{
    switch (dtype)
    {
        case RKNN_TENSOR_INT8:
            return 1;
        case RKNN_TENSOR_INT16:
            return 2;
        case RKNN_TENSOR_FLOAT16:
            return 2;
        case RKNN_TENSOR_BFLOAT16:
            return 2;
        case RKNN_TENSOR_INT32:
            return 3;
        case RKNN_TENSOR_FLOAT32:
            return 3;
        default:
            return 0;
    }
}
void ops(char* op, int dtype)
{
    EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(get_precision(dtype)) |
    DPU_DATA_FORMAT_IN_PRECISION(get_precision(dtype)) |
    DPU_DATA_FORMAT_PROC_PRECISION(get_precision(dtype)));

    //0x2001000178495044
    uint32_t rdma_feat_mode_cfg = 0x0;
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(get_precision(dtype));
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15);
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(0);
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(get_precision(dtype));
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(0);
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1);
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(0);
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1);
    
    EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg);


    EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG,
        DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(get_edata_size(dtype))); // 16 bit

    EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(0) | DPU_BS_CFG_BS_ALU_SRC(0) |
    DPU_BS_CFG_BS_RELUX_EN(0) |
    DPU_BS_CFG_BS_RELU_BYPASS(1) |
    DPU_BS_CFG_BS_MUL_PRELU(0) |
    DPU_BS_CFG_BS_MUL_BYPASS(1) |
    DPU_BS_CFG_BS_ALU_BYPASS(1) | 
    DPU_BS_CFG_BS_BYPASS(1));
 
    EMIT(REG_DPU_BN_CFG,
        DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
        DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));


    if (op == "ADD")
    {
        EMIT(REG_DPU_EW_CFG,
                DPU_EW_CFG_EW_CVT_TYPE(0) |
                DPU_EW_CFG_EW_CVT_ROUND(0) |
                DPU_EW_CFG_EW_DATA_MODE(1) |
                DPU_EW_CFG_EDATA_SIZE(get_edata_size(dtype)) |
                DPU_EW_CFG_EW_EQUAL_EN(0) |
                DPU_EW_CFG_EW_BINARY_EN(0) |
                DPU_EW_CFG_EW_ALU_ALGO(2) |
                DPU_EW_CFG_EW_RELUX_EN(0) |
                DPU_EW_CFG_EW_RELU_BYPASS(1) |
                DPU_EW_CFG_EW_OP_CVT_BYPASS(0) |
                DPU_EW_CFG_EW_LUT_BYPASS(1) |
                DPU_EW_CFG_EW_OP_SRC(1) |
                DPU_EW_CFG_EW_MUL_PRELU(0) |
                DPU_EW_CFG_EW_OP_TYPE(0) |
                DPU_EW_CFG_EW_OP_BYPASS(0) |
                DPU_EW_CFG_EW_BYPASS(0))
    }    
    else if (op == "MUL")
    {
        EMIT(REG_DPU_EW_CFG,
            DPU_EW_CFG_EW_CVT_TYPE(0) |
            DPU_EW_CFG_EW_CVT_ROUND(0) |
            DPU_EW_CFG_EW_DATA_MODE(1) |
            DPU_EW_CFG_EDATA_SIZE(get_edata_size(dtype)) |
            DPU_EW_CFG_EW_EQUAL_EN(0) |
            DPU_EW_CFG_EW_BINARY_EN(0) |
            DPU_EW_CFG_EW_ALU_ALGO(0) |
            DPU_EW_CFG_EW_RELUX_EN(0) |
            DPU_EW_CFG_EW_RELU_BYPASS(1) |
            DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
            DPU_EW_CFG_EW_LUT_BYPASS(1) |
            DPU_EW_CFG_EW_OP_SRC(1) |
            DPU_EW_CFG_EW_MUL_PRELU(0) |
            DPU_EW_CFG_EW_OP_TYPE(1) |
            DPU_EW_CFG_EW_OP_BYPASS(0) |
            DPU_EW_CFG_EW_BYPASS(0)) 
    }    
}
void create_reg_list()
{
    reg_list_clear(&regs);
   
    EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
    DPU_S_POINTER_EXECUTER_PP_EN(1) |
    DPU_S_POINTER_POINTER_PP_EN(1));

    EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) |
    DPU_FEATURE_MODE_CFG_CONV_MODE(0) |
    DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) |
    DPU_FEATURE_MODE_CFG_FLYING_MODE(1));

    
    EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_OD_BYPASS(1));
    EMIT(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0));

    create_channel(7);
    create_size(0,9);
    create_stride(12);
    create_surf_notch(2);
    emit_raw(&regs, DPU | 0x1, 0x40c4, 0);

    EMIT(REG_DPU_LUT_ACCESS_CFG, 0);
    EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
    EMIT(REG_DPU_LUT_CFG, 0);
    EMIT(REG_DPU_LUT_INFO, 0);
    EMIT(REG_DPU_LUT_LE_START, 0);
    EMIT(REG_DPU_LUT_LE_END, 0);
    EMIT(REG_DPU_LUT_LO_START, 0);
    EMIT(REG_DPU_LUT_LO_END, 0);
    EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0);
    EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
    EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0);
    EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

    EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(0));

    EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
    EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);
      
    EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

    EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
    EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
       DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
          DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));
    EMIT(REG_DPU_BN_ALU_CFG,0);    
    EMIT(REG_DPU_BN_MUL_CFG,0);    
    EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0)

    EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
    EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
    EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
    EMIT(REG_DPU_OUT_CVT_OFFSET, 0);

    emit_raw(&regs, DPU | 0x1, REG_DPU_OUT_CVT_SCALE, 65537);

    EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(1-1));

    EMIT(REG_DPU_EW_OP_VALUE_0, 0);
    EMIT(REG_DPU_EW_OP_VALUE_1, 0);
    EMIT(REG_DPU_EW_OP_VALUE_2, 0);
    EMIT(REG_DPU_EW_OP_VALUE_3, 0);
    EMIT(REG_DPU_EW_OP_VALUE_4, 0);
    EMIT(REG_DPU_EW_OP_VALUE_5, 0);
    EMIT(REG_DPU_EW_OP_VALUE_6, 0);
    EMIT(REG_DPU_EW_OP_VALUE_7, 0);

}

void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags) {


    int ret;
    struct rknpu_mem_create mem_create = {
      .flags = flags | RKNPU_MEM_NON_CACHEABLE,
      .size = size,
    };
  
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if(ret < 0)  {
      printf("RKNPU_MEM_CREATE failed %d\n",ret);
      return NULL;
    }
  
    struct rknpu_mem_map mem_map = { .handle = mem_create.handle, .offset=0 };
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    if(ret < 0) {
      printf("RKNPU_MEM_MAP failed %d\n",ret);
      return NULL;
    }	
    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
  
    *dma_addr = mem_create.dma_addr;
    *obj = mem_create.obj_addr;
    return map;
  }
  
  void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr) {
  
    int ret;
    struct rknpu_mem_destroy destroy = {
      .handle = handle ,
      .obj_addr = obj_addr
    };
  
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
    if (ret <0) {
      printf("RKNPU_MEM_DESTROY failed %d\n",ret);
    }
}



struct MemHandles create_mem(int fd, int type_size)
{
    uint64_t input_dma, input_obj;
    void *input = mem_allocate(fd, type_size, &input_dma, &input_obj, 0);  

    uint64_t weights_dma, weights_obj;
    void *weights = mem_allocate(fd, type_size, &weights_dma, &weights_obj, 0);

    uint64_t output_dma, output_obj;
    void *output = mem_allocate(fd, type_size, &output_dma, &output_obj, 0);


    EMIT(REG_DPU_DST_BASE_ADDR, DPU_DST_BASE_ADDR_DST_BASE_ADDR(output_dma))
    EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(input_dma))
    EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR(weights_dma))

    struct MemHandles handles;
    handles.input = input;
    handles.weights = weights;
    handles.output = output;
    handles.input_dma = input_dma;
    handles.input_obj = input_obj;
    handles.weights_dma = weights_dma;
    handles.weights_obj = weights_obj;
    handles.output_dma = output_dma;
    handles.output_obj = output_obj;
    return handles;
}


int getDeviceFd()
{
    int fd = open("/dev/dri/card1", O_RDWR);
    if(fd<0) {
      printf("Failed to open /dev/dri/card1");
      exit(1);
    }
    return fd;  
}

int submitTask(int fd)
{
    uint64_t tasks_dma, tasks_obj;
    struct rknpu_task *tasks = (struct rknpu_task *)mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING);
    uint64_t regcmd_dma, regcmd_obj;
    uint64_t *regcmd = (uint64_t *)(mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0));
    
    reg_list_push(&regs, 0x1001000000000014);
    EMIT(REG_PC_REGISTER_AMOUNTS, 0);
    emit_raw(&regs, 0x81, REG_PC_OPERATION_ENABLE,
       PC_OPERATION_ENABLE_RESERVED_0(12) | PC_OPERATION_ENABLE_OP_EN(0));  
 
    uint64_t npu_regs_a[regs.size];
    memcpy(npu_regs_a, regs.data, regs.size * sizeof(uint64_t));  // Copy elements to array
    memcpy(regcmd,npu_regs_a,sizeof(npu_regs_a));

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300; // wait for DPU to finish
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = sizeof(npu_regs_a)/sizeof(uint64_t); //nInstrs - 1;
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_dma;
    
    //util_dynarray_append(regs, uint64_t, 0x0041000000000000);
    // 0x0081000000180008
    //
   
  struct rknpu_submit submit = {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = tasks_obj,
    .regcfg_obj_addr = 0,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = 1,
    .fence_fd = -1,
    .subcore_task = { // Only use core 1, nothing for core 2/3
     {
       .task_start = 0,
       .task_number = 1,
     }, { 1, 0}, {2, 0}
   },
  
  };

   return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
}


short add_op_int16(short a, short b)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_INT16;
    create_reg_list();
    ops("ADD", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    short *weights_fp16 = (short*)(handles.weights);
    short *feature_data_fp16 = (short*)(handles.input);
    short *output_data = (short*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (short)0.0;
    }
    printf("a: %f, b: %f\n", a, b);
    printf("output_data: %f\n", *output_data);

    return *output_data;   
}

short mul_op_int16(short a, short b)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_INT16;
    create_reg_list();
    ops("MUL", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    short *weights_fp16 = (short*)(handles.weights);
    short *feature_data_fp16 = (short*)(handles.input);
    short *output_data = (short*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (short)0.0;
    }
    printf("a: %f, b: %f\n", a, b);
    printf("output_data: %f\n", *output_data);

    return *output_data;   
}



__fp16 add_op_fp16(__fp16 a, __fp16 b)
{

    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;
    create_reg_list();
    ops("ADD", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    __fp16 *weights_fp16 = (__fp16*)(handles.weights);
    __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
    __fp16 *output_data = (__fp16*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (__fp16)0.0;
    }
    printf("a: %f, b: %f\n", a, b);
    printf("output_data: %f\n", *output_data);

    return *output_data;   
}


__fp16 mul_op_fp16(__fp16 a, __fp16 b)
{

    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;
    create_reg_list();
    ops("MUL", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    __fp16 *weights_fp16 = (__fp16*)(handles.weights);
    __fp16 *feature_data_fp16 = (__fp16*)(handles.input);
    __fp16 *output_data = (__fp16*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (__fp16)0.0;
    }
    printf("a: %f, b: %f\n", a, b);
    printf("output_data: %f\n", *output_data);

    return *output_data;   
}


signed char mul_op_int8(signed char a, signed char b)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_INT8;
    create_reg_list();
    ops("MUL", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    signed char *weights_fp16 = (signed char*)(handles.weights);
    signed char *feature_data_fp16 = (signed char*)(handles.input);
    signed char *output_data = (signed char*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (signed char)0.0;
    }
    printf("a: %d, b: %d\n", a, b);
    printf("output_data: %d\n", *output_data);

    return *output_data;   
}

int mul_op_int32(int a, int b)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_INT32;
    create_reg_list();
    ops("MUL", dtype);
    struct MemHandles handles = create_mem(fd, get_type_size(dtype));

    int *weights_fp16 = (int*)(handles.weights);
    int *feature_data_fp16 = (int*)(handles.input);
    int *output_data = (int*)(handles.output);
    
    memcpy(weights_fp16, &a, get_type_size(dtype));
    memcpy(feature_data_fp16, &b, get_type_size(dtype));

    int ret = submitTask(fd);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return (int)0.0;
    }
    printf("a: %d, b: %d\n", a, b);
    printf("output_data: %d\n", *output_data);

    return *output_data;   
}
// Type-generic macro for add_op (C11 feature)
#define add_op(a, b) _Generic((a), \
    short: add_op_int16, \
    __fp16: add_op_fp16, \
    signed char: add_op_int8 \
)(a, b)

// Type-generic macro for add_op (C11 feature)
#define mul_op(a, b) _Generic((a), \
    short: mul_op_int16, \
    __fp16: mul_op_fp16, \
    signed char: mul_op_int8, \
    int: mul_op_int32 \
)(a, b)


// Library initialization and cleanup
int rknpu_utils_init(void);


#ifdef __cplusplus
}
#endif

#endif // RKNPU_UTILS_H
