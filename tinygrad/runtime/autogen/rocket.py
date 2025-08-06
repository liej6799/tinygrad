# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')



# values for enumeration 'target'
target__enumvalues = {
    256: 'PC',
    512: 'CNA',
    2048: 'CORE',
    4096: 'DPU',
    8192: 'DPU_RDMA',
    16384: 'PPU',
    32768: 'PPU_RDMA',
    65536: 'DDMA',
    131072: 'SDMA',
    262144: 'GLOBAL',
}
PC = 256
CNA = 512
CORE = 2048
DPU = 4096
DPU_RDMA = 8192
PPU = 16384
PPU_RDMA = 32768
DDMA = 65536
SDMA = 131072
GLOBAL = 262144
target = ctypes.c_uint32 # enum
uint32_t = ctypes.c_uint32
try:
    PC_VERSION_VERSION = _libraries['FIXME_STUB'].PC_VERSION_VERSION
    PC_VERSION_VERSION.restype = uint32_t
    PC_VERSION_VERSION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_VERSION_NUM_VERSION_NUM = _libraries['FIXME_STUB'].PC_VERSION_NUM_VERSION_NUM
    PC_VERSION_NUM_VERSION_NUM.restype = uint32_t
    PC_VERSION_NUM_VERSION_NUM.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].PC_OPERATION_ENABLE_RESERVED_0
    PC_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    PC_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].PC_OPERATION_ENABLE_OP_EN
    PC_OPERATION_ENABLE_OP_EN.restype = uint32_t
    PC_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_BASE_ADDRESS_PC_SOURCE_ADDR = _libraries['FIXME_STUB'].PC_BASE_ADDRESS_PC_SOURCE_ADDR
    PC_BASE_ADDRESS_PC_SOURCE_ADDR.restype = uint32_t
    PC_BASE_ADDRESS_PC_SOURCE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_BASE_ADDRESS_RESERVED_0 = _libraries['FIXME_STUB'].PC_BASE_ADDRESS_RESERVED_0
    PC_BASE_ADDRESS_RESERVED_0.restype = uint32_t
    PC_BASE_ADDRESS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_BASE_ADDRESS_PC_SEL = _libraries['FIXME_STUB'].PC_BASE_ADDRESS_PC_SEL
    PC_BASE_ADDRESS_PC_SEL.restype = uint32_t
    PC_BASE_ADDRESS_PC_SEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_REGISTER_AMOUNTS_RESERVED_0 = _libraries['FIXME_STUB'].PC_REGISTER_AMOUNTS_RESERVED_0
    PC_REGISTER_AMOUNTS_RESERVED_0.restype = uint32_t
    PC_REGISTER_AMOUNTS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT = _libraries['FIXME_STUB'].PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT
    PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT.restype = uint32_t
    PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_INTERRUPT_MASK_RESERVED_0 = _libraries['FIXME_STUB'].PC_INTERRUPT_MASK_RESERVED_0
    PC_INTERRUPT_MASK_RESERVED_0.restype = uint32_t
    PC_INTERRUPT_MASK_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_INTERRUPT_CLEAR_RESERVED_0 = _libraries['FIXME_STUB'].PC_INTERRUPT_CLEAR_RESERVED_0
    PC_INTERRUPT_CLEAR_RESERVED_0.restype = uint32_t
    PC_INTERRUPT_CLEAR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_INTERRUPT_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].PC_INTERRUPT_STATUS_RESERVED_0
    PC_INTERRUPT_STATUS_RESERVED_0.restype = uint32_t
    PC_INTERRUPT_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_INTERRUPT_RAW_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].PC_INTERRUPT_RAW_STATUS_RESERVED_0
    PC_INTERRUPT_RAW_STATUS_RESERVED_0.restype = uint32_t
    PC_INTERRUPT_RAW_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_CON_RESERVED_0 = _libraries['FIXME_STUB'].PC_TASK_CON_RESERVED_0
    PC_TASK_CON_RESERVED_0.restype = uint32_t
    PC_TASK_CON_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_CON_TASK_COUNT_CLEAR = _libraries['FIXME_STUB'].PC_TASK_CON_TASK_COUNT_CLEAR
    PC_TASK_CON_TASK_COUNT_CLEAR.restype = uint32_t
    PC_TASK_CON_TASK_COUNT_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_CON_TASK_PP_EN = _libraries['FIXME_STUB'].PC_TASK_CON_TASK_PP_EN
    PC_TASK_CON_TASK_PP_EN.restype = uint32_t
    PC_TASK_CON_TASK_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_CON_TASK_NUMBER = _libraries['FIXME_STUB'].PC_TASK_CON_TASK_NUMBER
    PC_TASK_CON_TASK_NUMBER.restype = uint32_t
    PC_TASK_CON_TASK_NUMBER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR = _libraries['FIXME_STUB'].PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR
    PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR.restype = uint32_t
    PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_DMA_BASE_ADDR_RESERVED_0 = _libraries['FIXME_STUB'].PC_TASK_DMA_BASE_ADDR_RESERVED_0
    PC_TASK_DMA_BASE_ADDR_RESERVED_0.restype = uint32_t
    PC_TASK_DMA_BASE_ADDR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].PC_TASK_STATUS_RESERVED_0
    PC_TASK_STATUS_RESERVED_0.restype = uint32_t
    PC_TASK_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PC_TASK_STATUS_TASK_STATUS = _libraries['FIXME_STUB'].PC_TASK_STATUS_TASK_STATUS
    PC_TASK_STATUS_TASK_STATUS.restype = uint32_t
    PC_TASK_STATUS_TASK_STATUS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].CNA_S_STATUS_RESERVED_0
    CNA_S_STATUS_RESERVED_0.restype = uint32_t
    CNA_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].CNA_S_STATUS_STATUS_1
    CNA_S_STATUS_STATUS_1.restype = uint32_t
    CNA_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].CNA_S_STATUS_RESERVED_1
    CNA_S_STATUS_RESERVED_1.restype = uint32_t
    CNA_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].CNA_S_STATUS_STATUS_0
    CNA_S_STATUS_STATUS_0.restype = uint32_t
    CNA_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].CNA_S_POINTER_RESERVED_0
    CNA_S_POINTER_RESERVED_0.restype = uint32_t
    CNA_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].CNA_S_POINTER_EXECUTER
    CNA_S_POINTER_EXECUTER.restype = uint32_t
    CNA_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].CNA_S_POINTER_RESERVED_1
    CNA_S_POINTER_RESERVED_1.restype = uint32_t
    CNA_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].CNA_S_POINTER_EXECUTER_PP_CLEAR
    CNA_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    CNA_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].CNA_S_POINTER_POINTER_PP_CLEAR
    CNA_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    CNA_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].CNA_S_POINTER_POINTER_PP_MODE
    CNA_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    CNA_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].CNA_S_POINTER_EXECUTER_PP_EN
    CNA_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    CNA_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].CNA_S_POINTER_POINTER_PP_EN
    CNA_S_POINTER_POINTER_PP_EN.restype = uint32_t
    CNA_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_S_POINTER_POINTER = _libraries['FIXME_STUB'].CNA_S_POINTER_POINTER
    CNA_S_POINTER_POINTER.restype = uint32_t
    CNA_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].CNA_OPERATION_ENABLE_RESERVED_0
    CNA_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    CNA_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].CNA_OPERATION_ENABLE_OP_EN
    CNA_OPERATION_ENABLE_OP_EN.restype = uint32_t
    CNA_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CONV_CON1_RESERVED_0
    CNA_CONV_CON1_RESERVED_0.restype = uint32_t
    CNA_CONV_CON1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_NONALIGN_DMA = _libraries['FIXME_STUB'].CNA_CONV_CON1_NONALIGN_DMA
    CNA_CONV_CON1_NONALIGN_DMA.restype = uint32_t
    CNA_CONV_CON1_NONALIGN_DMA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_GROUP_LINE_OFF = _libraries['FIXME_STUB'].CNA_CONV_CON1_GROUP_LINE_OFF
    CNA_CONV_CON1_GROUP_LINE_OFF.restype = uint32_t
    CNA_CONV_CON1_GROUP_LINE_OFF.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_RESERVED_1 = _libraries['FIXME_STUB'].CNA_CONV_CON1_RESERVED_1
    CNA_CONV_CON1_RESERVED_1.restype = uint32_t
    CNA_CONV_CON1_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_DECONV = _libraries['FIXME_STUB'].CNA_CONV_CON1_DECONV
    CNA_CONV_CON1_DECONV.restype = uint32_t
    CNA_CONV_CON1_DECONV.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_ARGB_IN = _libraries['FIXME_STUB'].CNA_CONV_CON1_ARGB_IN
    CNA_CONV_CON1_ARGB_IN.restype = uint32_t
    CNA_CONV_CON1_ARGB_IN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_RESERVED_2 = _libraries['FIXME_STUB'].CNA_CONV_CON1_RESERVED_2
    CNA_CONV_CON1_RESERVED_2.restype = uint32_t
    CNA_CONV_CON1_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_PROC_PRECISION = _libraries['FIXME_STUB'].CNA_CONV_CON1_PROC_PRECISION
    CNA_CONV_CON1_PROC_PRECISION.restype = uint32_t
    CNA_CONV_CON1_PROC_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_IN_PRECISION = _libraries['FIXME_STUB'].CNA_CONV_CON1_IN_PRECISION
    CNA_CONV_CON1_IN_PRECISION.restype = uint32_t
    CNA_CONV_CON1_IN_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON1_CONV_MODE = _libraries['FIXME_STUB'].CNA_CONV_CON1_CONV_MODE
    CNA_CONV_CON1_CONV_MODE.restype = uint32_t
    CNA_CONV_CON1_CONV_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CONV_CON2_RESERVED_0
    CNA_CONV_CON2_RESERVED_0.restype = uint32_t
    CNA_CONV_CON2_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_KERNEL_GROUP = _libraries['FIXME_STUB'].CNA_CONV_CON2_KERNEL_GROUP
    CNA_CONV_CON2_KERNEL_GROUP.restype = uint32_t
    CNA_CONV_CON2_KERNEL_GROUP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_RESERVED_1 = _libraries['FIXME_STUB'].CNA_CONV_CON2_RESERVED_1
    CNA_CONV_CON2_RESERVED_1.restype = uint32_t
    CNA_CONV_CON2_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_FEATURE_GRAINS = _libraries['FIXME_STUB'].CNA_CONV_CON2_FEATURE_GRAINS
    CNA_CONV_CON2_FEATURE_GRAINS.restype = uint32_t
    CNA_CONV_CON2_FEATURE_GRAINS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_RESERVED_2 = _libraries['FIXME_STUB'].CNA_CONV_CON2_RESERVED_2
    CNA_CONV_CON2_RESERVED_2.restype = uint32_t
    CNA_CONV_CON2_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_CSC_WO_EN = _libraries['FIXME_STUB'].CNA_CONV_CON2_CSC_WO_EN
    CNA_CONV_CON2_CSC_WO_EN.restype = uint32_t
    CNA_CONV_CON2_CSC_WO_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_CSC_DO_EN = _libraries['FIXME_STUB'].CNA_CONV_CON2_CSC_DO_EN
    CNA_CONV_CON2_CSC_DO_EN.restype = uint32_t
    CNA_CONV_CON2_CSC_DO_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON2_CMD_FIFO_SRST = _libraries['FIXME_STUB'].CNA_CONV_CON2_CMD_FIFO_SRST
    CNA_CONV_CON2_CMD_FIFO_SRST.restype = uint32_t
    CNA_CONV_CON2_CMD_FIFO_SRST.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CONV_CON3_RESERVED_0
    CNA_CONV_CON3_RESERVED_0.restype = uint32_t
    CNA_CONV_CON3_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_NN_MODE = _libraries['FIXME_STUB'].CNA_CONV_CON3_NN_MODE
    CNA_CONV_CON3_NN_MODE.restype = uint32_t
    CNA_CONV_CON3_NN_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_RESERVED_1 = _libraries['FIXME_STUB'].CNA_CONV_CON3_RESERVED_1
    CNA_CONV_CON3_RESERVED_1.restype = uint32_t
    CNA_CONV_CON3_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_ATROUS_Y_DILATION = _libraries['FIXME_STUB'].CNA_CONV_CON3_ATROUS_Y_DILATION
    CNA_CONV_CON3_ATROUS_Y_DILATION.restype = uint32_t
    CNA_CONV_CON3_ATROUS_Y_DILATION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_ATROUS_X_DILATION = _libraries['FIXME_STUB'].CNA_CONV_CON3_ATROUS_X_DILATION
    CNA_CONV_CON3_ATROUS_X_DILATION.restype = uint32_t
    CNA_CONV_CON3_ATROUS_X_DILATION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_RESERVED_2 = _libraries['FIXME_STUB'].CNA_CONV_CON3_RESERVED_2
    CNA_CONV_CON3_RESERVED_2.restype = uint32_t
    CNA_CONV_CON3_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_DECONV_Y_STRIDE = _libraries['FIXME_STUB'].CNA_CONV_CON3_DECONV_Y_STRIDE
    CNA_CONV_CON3_DECONV_Y_STRIDE.restype = uint32_t
    CNA_CONV_CON3_DECONV_Y_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_DECONV_X_STRIDE = _libraries['FIXME_STUB'].CNA_CONV_CON3_DECONV_X_STRIDE
    CNA_CONV_CON3_DECONV_X_STRIDE.restype = uint32_t
    CNA_CONV_CON3_DECONV_X_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_RESERVED_3 = _libraries['FIXME_STUB'].CNA_CONV_CON3_RESERVED_3
    CNA_CONV_CON3_RESERVED_3.restype = uint32_t
    CNA_CONV_CON3_RESERVED_3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_CONV_Y_STRIDE = _libraries['FIXME_STUB'].CNA_CONV_CON3_CONV_Y_STRIDE
    CNA_CONV_CON3_CONV_Y_STRIDE.restype = uint32_t
    CNA_CONV_CON3_CONV_Y_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CONV_CON3_CONV_X_STRIDE = _libraries['FIXME_STUB'].CNA_CONV_CON3_CONV_X_STRIDE
    CNA_CONV_CON3_CONV_X_STRIDE.restype = uint32_t
    CNA_CONV_CON3_CONV_X_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DATA_SIZE0_RESERVED_0
    CNA_DATA_SIZE0_RESERVED_0.restype = uint32_t
    CNA_DATA_SIZE0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE0_DATAIN_WIDTH = _libraries['FIXME_STUB'].CNA_DATA_SIZE0_DATAIN_WIDTH
    CNA_DATA_SIZE0_DATAIN_WIDTH.restype = uint32_t
    CNA_DATA_SIZE0_DATAIN_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE0_RESERVED_1 = _libraries['FIXME_STUB'].CNA_DATA_SIZE0_RESERVED_1
    CNA_DATA_SIZE0_RESERVED_1.restype = uint32_t
    CNA_DATA_SIZE0_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE0_DATAIN_HEIGHT = _libraries['FIXME_STUB'].CNA_DATA_SIZE0_DATAIN_HEIGHT
    CNA_DATA_SIZE0_DATAIN_HEIGHT.restype = uint32_t
    CNA_DATA_SIZE0_DATAIN_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DATA_SIZE1_RESERVED_0
    CNA_DATA_SIZE1_RESERVED_0.restype = uint32_t
    CNA_DATA_SIZE1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL = _libraries['FIXME_STUB'].CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL
    CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL.restype = uint32_t
    CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE1_DATAIN_CHANNEL = _libraries['FIXME_STUB'].CNA_DATA_SIZE1_DATAIN_CHANNEL
    CNA_DATA_SIZE1_DATAIN_CHANNEL.restype = uint32_t
    CNA_DATA_SIZE1_DATAIN_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE2_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DATA_SIZE2_RESERVED_0
    CNA_DATA_SIZE2_RESERVED_0.restype = uint32_t
    CNA_DATA_SIZE2_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE2_DATAOUT_WIDTH = _libraries['FIXME_STUB'].CNA_DATA_SIZE2_DATAOUT_WIDTH
    CNA_DATA_SIZE2_DATAOUT_WIDTH.restype = uint32_t
    CNA_DATA_SIZE2_DATAOUT_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE3_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DATA_SIZE3_RESERVED_0
    CNA_DATA_SIZE3_RESERVED_0.restype = uint32_t
    CNA_DATA_SIZE3_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE3_SURF_MODE = _libraries['FIXME_STUB'].CNA_DATA_SIZE3_SURF_MODE
    CNA_DATA_SIZE3_SURF_MODE.restype = uint32_t
    CNA_DATA_SIZE3_SURF_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DATA_SIZE3_DATAOUT_ATOMICS = _libraries['FIXME_STUB'].CNA_DATA_SIZE3_DATAOUT_ATOMICS
    CNA_DATA_SIZE3_DATAOUT_ATOMICS.restype = uint32_t
    CNA_DATA_SIZE3_DATAOUT_ATOMICS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE0_WEIGHT_BYTES = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE0_WEIGHT_BYTES
    CNA_WEIGHT_SIZE0_WEIGHT_BYTES.restype = uint32_t
    CNA_WEIGHT_SIZE0_WEIGHT_BYTES.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE1_RESERVED_0
    CNA_WEIGHT_SIZE1_RESERVED_0.restype = uint32_t
    CNA_WEIGHT_SIZE1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL
    CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL.restype = uint32_t
    CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_RESERVED_0 = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_RESERVED_0
    CNA_WEIGHT_SIZE2_RESERVED_0.restype = uint32_t
    CNA_WEIGHT_SIZE2_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_WEIGHT_WIDTH = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_WEIGHT_WIDTH
    CNA_WEIGHT_SIZE2_WEIGHT_WIDTH.restype = uint32_t
    CNA_WEIGHT_SIZE2_WEIGHT_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_RESERVED_1 = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_RESERVED_1
    CNA_WEIGHT_SIZE2_RESERVED_1.restype = uint32_t
    CNA_WEIGHT_SIZE2_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT
    CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT.restype = uint32_t
    CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_RESERVED_2 = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_RESERVED_2
    CNA_WEIGHT_SIZE2_RESERVED_2.restype = uint32_t
    CNA_WEIGHT_SIZE2_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_WEIGHT_SIZE2_WEIGHT_KERNELS = _libraries['FIXME_STUB'].CNA_WEIGHT_SIZE2_WEIGHT_KERNELS
    CNA_WEIGHT_SIZE2_WEIGHT_KERNELS.restype = uint32_t
    CNA_WEIGHT_SIZE2_WEIGHT_KERNELS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CBUF_CON0_RESERVED_0
    CNA_CBUF_CON0_RESERVED_0.restype = uint32_t
    CNA_CBUF_CON0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_WEIGHT_REUSE = _libraries['FIXME_STUB'].CNA_CBUF_CON0_WEIGHT_REUSE
    CNA_CBUF_CON0_WEIGHT_REUSE.restype = uint32_t
    CNA_CBUF_CON0_WEIGHT_REUSE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_DATA_REUSE = _libraries['FIXME_STUB'].CNA_CBUF_CON0_DATA_REUSE
    CNA_CBUF_CON0_DATA_REUSE.restype = uint32_t
    CNA_CBUF_CON0_DATA_REUSE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_RESERVED_1 = _libraries['FIXME_STUB'].CNA_CBUF_CON0_RESERVED_1
    CNA_CBUF_CON0_RESERVED_1.restype = uint32_t
    CNA_CBUF_CON0_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_FC_DATA_BANK = _libraries['FIXME_STUB'].CNA_CBUF_CON0_FC_DATA_BANK
    CNA_CBUF_CON0_FC_DATA_BANK.restype = uint32_t
    CNA_CBUF_CON0_FC_DATA_BANK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_WEIGHT_BANK = _libraries['FIXME_STUB'].CNA_CBUF_CON0_WEIGHT_BANK
    CNA_CBUF_CON0_WEIGHT_BANK.restype = uint32_t
    CNA_CBUF_CON0_WEIGHT_BANK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON0_DATA_BANK = _libraries['FIXME_STUB'].CNA_CBUF_CON0_DATA_BANK
    CNA_CBUF_CON0_DATA_BANK.restype = uint32_t
    CNA_CBUF_CON0_DATA_BANK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CBUF_CON1_RESERVED_0
    CNA_CBUF_CON1_RESERVED_0.restype = uint32_t
    CNA_CBUF_CON1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CBUF_CON1_DATA_ENTRIES = _libraries['FIXME_STUB'].CNA_CBUF_CON1_DATA_ENTRIES
    CNA_CBUF_CON1_DATA_ENTRIES.restype = uint32_t
    CNA_CBUF_CON1_DATA_ENTRIES.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CVT_CON0_RESERVED_0
    CNA_CVT_CON0_RESERVED_0.restype = uint32_t
    CNA_CVT_CON0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_TRUNCATE_3 = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_TRUNCATE_3
    CNA_CVT_CON0_CVT_TRUNCATE_3.restype = uint32_t
    CNA_CVT_CON0_CVT_TRUNCATE_3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_TRUNCATE_2 = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_TRUNCATE_2
    CNA_CVT_CON0_CVT_TRUNCATE_2.restype = uint32_t
    CNA_CVT_CON0_CVT_TRUNCATE_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_TRUNCATE_1 = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_TRUNCATE_1
    CNA_CVT_CON0_CVT_TRUNCATE_1.restype = uint32_t
    CNA_CVT_CON0_CVT_TRUNCATE_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_TRUNCATE_0 = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_TRUNCATE_0
    CNA_CVT_CON0_CVT_TRUNCATE_0.restype = uint32_t
    CNA_CVT_CON0_CVT_TRUNCATE_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_DATA_SIGN = _libraries['FIXME_STUB'].CNA_CVT_CON0_DATA_SIGN
    CNA_CVT_CON0_DATA_SIGN.restype = uint32_t
    CNA_CVT_CON0_DATA_SIGN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_ROUND_TYPE = _libraries['FIXME_STUB'].CNA_CVT_CON0_ROUND_TYPE
    CNA_CVT_CON0_ROUND_TYPE.restype = uint32_t
    CNA_CVT_CON0_ROUND_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_TYPE = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_TYPE
    CNA_CVT_CON0_CVT_TYPE.restype = uint32_t
    CNA_CVT_CON0_CVT_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON0_CVT_BYPASS = _libraries['FIXME_STUB'].CNA_CVT_CON0_CVT_BYPASS
    CNA_CVT_CON0_CVT_BYPASS.restype = uint32_t
    CNA_CVT_CON0_CVT_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON1_CVT_SCALE0 = _libraries['FIXME_STUB'].CNA_CVT_CON1_CVT_SCALE0
    CNA_CVT_CON1_CVT_SCALE0.restype = uint32_t
    CNA_CVT_CON1_CVT_SCALE0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON1_CVT_OFFSET0 = _libraries['FIXME_STUB'].CNA_CVT_CON1_CVT_OFFSET0
    CNA_CVT_CON1_CVT_OFFSET0.restype = uint32_t
    CNA_CVT_CON1_CVT_OFFSET0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON2_CVT_SCALE1 = _libraries['FIXME_STUB'].CNA_CVT_CON2_CVT_SCALE1
    CNA_CVT_CON2_CVT_SCALE1.restype = uint32_t
    CNA_CVT_CON2_CVT_SCALE1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON2_CVT_OFFSET1 = _libraries['FIXME_STUB'].CNA_CVT_CON2_CVT_OFFSET1
    CNA_CVT_CON2_CVT_OFFSET1.restype = uint32_t
    CNA_CVT_CON2_CVT_OFFSET1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON3_CVT_SCALE2 = _libraries['FIXME_STUB'].CNA_CVT_CON3_CVT_SCALE2
    CNA_CVT_CON3_CVT_SCALE2.restype = uint32_t
    CNA_CVT_CON3_CVT_SCALE2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON3_CVT_OFFSET2 = _libraries['FIXME_STUB'].CNA_CVT_CON3_CVT_OFFSET2
    CNA_CVT_CON3_CVT_OFFSET2.restype = uint32_t
    CNA_CVT_CON3_CVT_OFFSET2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON4_CVT_SCALE3 = _libraries['FIXME_STUB'].CNA_CVT_CON4_CVT_SCALE3
    CNA_CVT_CON4_CVT_SCALE3.restype = uint32_t
    CNA_CVT_CON4_CVT_SCALE3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON4_CVT_OFFSET3 = _libraries['FIXME_STUB'].CNA_CVT_CON4_CVT_OFFSET3
    CNA_CVT_CON4_CVT_OFFSET3.restype = uint32_t
    CNA_CVT_CON4_CVT_OFFSET3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON0_FC_SKIP_DATA = _libraries['FIXME_STUB'].CNA_FC_CON0_FC_SKIP_DATA
    CNA_FC_CON0_FC_SKIP_DATA.restype = uint32_t
    CNA_FC_CON0_FC_SKIP_DATA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_FC_CON0_RESERVED_0
    CNA_FC_CON0_RESERVED_0.restype = uint32_t
    CNA_FC_CON0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON0_FC_SKIP_EN = _libraries['FIXME_STUB'].CNA_FC_CON0_FC_SKIP_EN
    CNA_FC_CON0_FC_SKIP_EN.restype = uint32_t
    CNA_FC_CON0_FC_SKIP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_FC_CON1_RESERVED_0
    CNA_FC_CON1_RESERVED_0.restype = uint32_t
    CNA_FC_CON1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON1_DATA_OFFSET = _libraries['FIXME_STUB'].CNA_FC_CON1_DATA_OFFSET
    CNA_FC_CON1_DATA_OFFSET.restype = uint32_t
    CNA_FC_CON1_DATA_OFFSET.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_PAD_CON0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_PAD_CON0_RESERVED_0
    CNA_PAD_CON0_RESERVED_0.restype = uint32_t
    CNA_PAD_CON0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_PAD_CON0_PAD_LEFT = _libraries['FIXME_STUB'].CNA_PAD_CON0_PAD_LEFT
    CNA_PAD_CON0_PAD_LEFT.restype = uint32_t
    CNA_PAD_CON0_PAD_LEFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_PAD_CON0_PAD_TOP = _libraries['FIXME_STUB'].CNA_PAD_CON0_PAD_TOP
    CNA_PAD_CON0_PAD_TOP.restype = uint32_t
    CNA_PAD_CON0_PAD_TOP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR = _libraries['FIXME_STUB'].CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR
    CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR.restype = uint32_t
    CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON2_RESERVED_0 = _libraries['FIXME_STUB'].CNA_FC_CON2_RESERVED_0
    CNA_FC_CON2_RESERVED_0.restype = uint32_t
    CNA_FC_CON2_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_CON2_WEIGHT_OFFSET = _libraries['FIXME_STUB'].CNA_FC_CON2_WEIGHT_OFFSET
    CNA_FC_CON2_WEIGHT_OFFSET.restype = uint32_t
    CNA_FC_CON2_WEIGHT_OFFSET.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON0_OV4K_BYPASS = _libraries['FIXME_STUB'].CNA_DMA_CON0_OV4K_BYPASS
    CNA_DMA_CON0_OV4K_BYPASS.restype = uint32_t
    CNA_DMA_CON0_OV4K_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DMA_CON0_RESERVED_0
    CNA_DMA_CON0_RESERVED_0.restype = uint32_t
    CNA_DMA_CON0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON0_WEIGHT_BURST_LEN = _libraries['FIXME_STUB'].CNA_DMA_CON0_WEIGHT_BURST_LEN
    CNA_DMA_CON0_WEIGHT_BURST_LEN.restype = uint32_t
    CNA_DMA_CON0_WEIGHT_BURST_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON0_RESERVED_1 = _libraries['FIXME_STUB'].CNA_DMA_CON0_RESERVED_1
    CNA_DMA_CON0_RESERVED_1.restype = uint32_t
    CNA_DMA_CON0_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON0_DATA_BURST_LEN = _libraries['FIXME_STUB'].CNA_DMA_CON0_DATA_BURST_LEN
    CNA_DMA_CON0_DATA_BURST_LEN.restype = uint32_t
    CNA_DMA_CON0_DATA_BURST_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DMA_CON1_RESERVED_0
    CNA_DMA_CON1_RESERVED_0.restype = uint32_t
    CNA_DMA_CON1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON1_LINE_STRIDE = _libraries['FIXME_STUB'].CNA_DMA_CON1_LINE_STRIDE
    CNA_DMA_CON1_LINE_STRIDE.restype = uint32_t
    CNA_DMA_CON1_LINE_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON2_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DMA_CON2_RESERVED_0
    CNA_DMA_CON2_RESERVED_0.restype = uint32_t
    CNA_DMA_CON2_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DMA_CON2_SURF_STRIDE = _libraries['FIXME_STUB'].CNA_DMA_CON2_SURF_STRIDE
    CNA_DMA_CON2_SURF_STRIDE.restype = uint32_t
    CNA_DMA_CON2_SURF_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE0_RESERVED_0 = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE0_RESERVED_0
    CNA_FC_DATA_SIZE0_RESERVED_0.restype = uint32_t
    CNA_FC_DATA_SIZE0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE0_DMA_WIDTH = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE0_DMA_WIDTH
    CNA_FC_DATA_SIZE0_DMA_WIDTH.restype = uint32_t
    CNA_FC_DATA_SIZE0_DMA_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE0_RESERVED_1 = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE0_RESERVED_1
    CNA_FC_DATA_SIZE0_RESERVED_1.restype = uint32_t
    CNA_FC_DATA_SIZE0_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE0_DMA_HEIGHT = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE0_DMA_HEIGHT
    CNA_FC_DATA_SIZE0_DMA_HEIGHT.restype = uint32_t
    CNA_FC_DATA_SIZE0_DMA_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE1_RESERVED_0 = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE1_RESERVED_0
    CNA_FC_DATA_SIZE1_RESERVED_0.restype = uint32_t
    CNA_FC_DATA_SIZE1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_FC_DATA_SIZE1_DMA_CHANNEL = _libraries['FIXME_STUB'].CNA_FC_DATA_SIZE1_DMA_CHANNEL
    CNA_FC_DATA_SIZE1_DMA_CHANNEL.restype = uint32_t
    CNA_FC_DATA_SIZE1_DMA_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_RESERVED_0 = _libraries['FIXME_STUB'].CNA_CLK_GATE_RESERVED_0
    CNA_CLK_GATE_RESERVED_0.restype = uint32_t
    CNA_CLK_GATE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE = _libraries['FIXME_STUB'].CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE
    CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE.restype = uint32_t
    CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_RESERVED_1 = _libraries['FIXME_STUB'].CNA_CLK_GATE_RESERVED_1
    CNA_CLK_GATE_RESERVED_1.restype = uint32_t
    CNA_CLK_GATE_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_CSC_DISABLE_CLKGATE = _libraries['FIXME_STUB'].CNA_CLK_GATE_CSC_DISABLE_CLKGATE
    CNA_CLK_GATE_CSC_DISABLE_CLKGATE.restype = uint32_t
    CNA_CLK_GATE_CSC_DISABLE_CLKGATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE = _libraries['FIXME_STUB'].CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE
    CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE.restype = uint32_t
    CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE = _libraries['FIXME_STUB'].CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE
    CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE.restype = uint32_t
    CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_CTRL_RESERVED_0 = _libraries['FIXME_STUB'].CNA_DCOMP_CTRL_RESERVED_0
    CNA_DCOMP_CTRL_RESERVED_0.restype = uint32_t
    CNA_DCOMP_CTRL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_CTRL_WT_DEC_BYPASS = _libraries['FIXME_STUB'].CNA_DCOMP_CTRL_WT_DEC_BYPASS
    CNA_DCOMP_CTRL_WT_DEC_BYPASS.restype = uint32_t
    CNA_DCOMP_CTRL_WT_DEC_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_CTRL_DECOMP_CONTROL = _libraries['FIXME_STUB'].CNA_DCOMP_CTRL_DECOMP_CONTROL
    CNA_DCOMP_CTRL_DECOMP_CONTROL.restype = uint32_t
    CNA_DCOMP_CTRL_DECOMP_CONTROL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_REGNUM_DCOMP_REGNUM = _libraries['FIXME_STUB'].CNA_DCOMP_REGNUM_DCOMP_REGNUM
    CNA_DCOMP_REGNUM_DCOMP_REGNUM.restype = uint32_t
    CNA_DCOMP_REGNUM_DCOMP_REGNUM.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0 = _libraries['FIXME_STUB'].CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0
    CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0.restype = uint32_t
    CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0
    CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0.restype = uint32_t
    CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1
    CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1.restype = uint32_t
    CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2
    CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2.restype = uint32_t
    CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3
    CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3.restype = uint32_t
    CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4
    CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4.restype = uint32_t
    CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5
    CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5.restype = uint32_t
    CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6
    CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6.restype = uint32_t
    CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7
    CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7.restype = uint32_t
    CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8
    CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8.restype = uint32_t
    CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9
    CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9.restype = uint32_t
    CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10
    CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10.restype = uint32_t
    CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11
    CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11.restype = uint32_t
    CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12
    CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12.restype = uint32_t
    CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13
    CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13.restype = uint32_t
    CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14
    CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14.restype = uint32_t
    CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15 = _libraries['FIXME_STUB'].CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15
    CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15.restype = uint32_t
    CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_CVT_CON5_PER_CHANNEL_CVT_EN = _libraries['FIXME_STUB'].CNA_CVT_CON5_PER_CHANNEL_CVT_EN
    CNA_CVT_CON5_PER_CHANNEL_CVT_EN.restype = uint32_t
    CNA_CVT_CON5_PER_CHANNEL_CVT_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CNA_PAD_CON1_PAD_VALUE = _libraries['FIXME_STUB'].CNA_PAD_CON1_PAD_VALUE
    CNA_PAD_CON1_PAD_VALUE.restype = uint32_t
    CNA_PAD_CON1_PAD_VALUE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].CORE_S_STATUS_RESERVED_0
    CORE_S_STATUS_RESERVED_0.restype = uint32_t
    CORE_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].CORE_S_STATUS_STATUS_1
    CORE_S_STATUS_STATUS_1.restype = uint32_t
    CORE_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].CORE_S_STATUS_RESERVED_1
    CORE_S_STATUS_RESERVED_1.restype = uint32_t
    CORE_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].CORE_S_STATUS_STATUS_0
    CORE_S_STATUS_STATUS_0.restype = uint32_t
    CORE_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].CORE_S_POINTER_RESERVED_0
    CORE_S_POINTER_RESERVED_0.restype = uint32_t
    CORE_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].CORE_S_POINTER_EXECUTER
    CORE_S_POINTER_EXECUTER.restype = uint32_t
    CORE_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].CORE_S_POINTER_RESERVED_1
    CORE_S_POINTER_RESERVED_1.restype = uint32_t
    CORE_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].CORE_S_POINTER_EXECUTER_PP_CLEAR
    CORE_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    CORE_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].CORE_S_POINTER_POINTER_PP_CLEAR
    CORE_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    CORE_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].CORE_S_POINTER_POINTER_PP_MODE
    CORE_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    CORE_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].CORE_S_POINTER_EXECUTER_PP_EN
    CORE_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    CORE_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].CORE_S_POINTER_POINTER_PP_EN
    CORE_S_POINTER_POINTER_PP_EN.restype = uint32_t
    CORE_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_S_POINTER_POINTER = _libraries['FIXME_STUB'].CORE_S_POINTER_POINTER
    CORE_S_POINTER_POINTER.restype = uint32_t
    CORE_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].CORE_OPERATION_ENABLE_RESERVED_0
    CORE_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    CORE_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].CORE_OPERATION_ENABLE_OP_EN
    CORE_OPERATION_ENABLE_OP_EN.restype = uint32_t
    CORE_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MAC_GATING_RESERVED_0 = _libraries['FIXME_STUB'].CORE_MAC_GATING_RESERVED_0
    CORE_MAC_GATING_RESERVED_0.restype = uint32_t
    CORE_MAC_GATING_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MAC_GATING_SLCG_OP_EN = _libraries['FIXME_STUB'].CORE_MAC_GATING_SLCG_OP_EN
    CORE_MAC_GATING_SLCG_OP_EN.restype = uint32_t
    CORE_MAC_GATING_SLCG_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_RESERVED_0 = _libraries['FIXME_STUB'].CORE_MISC_CFG_RESERVED_0
    CORE_MISC_CFG_RESERVED_0.restype = uint32_t
    CORE_MISC_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_SOFT_GATING = _libraries['FIXME_STUB'].CORE_MISC_CFG_SOFT_GATING
    CORE_MISC_CFG_SOFT_GATING.restype = uint32_t
    CORE_MISC_CFG_SOFT_GATING.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_RESERVED_1 = _libraries['FIXME_STUB'].CORE_MISC_CFG_RESERVED_1
    CORE_MISC_CFG_RESERVED_1.restype = uint32_t
    CORE_MISC_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_PROC_PRECISION = _libraries['FIXME_STUB'].CORE_MISC_CFG_PROC_PRECISION
    CORE_MISC_CFG_PROC_PRECISION.restype = uint32_t
    CORE_MISC_CFG_PROC_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_RESERVED_2 = _libraries['FIXME_STUB'].CORE_MISC_CFG_RESERVED_2
    CORE_MISC_CFG_RESERVED_2.restype = uint32_t
    CORE_MISC_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_DW_EN = _libraries['FIXME_STUB'].CORE_MISC_CFG_DW_EN
    CORE_MISC_CFG_DW_EN.restype = uint32_t
    CORE_MISC_CFG_DW_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_MISC_CFG_QD_EN = _libraries['FIXME_STUB'].CORE_MISC_CFG_QD_EN
    CORE_MISC_CFG_QD_EN.restype = uint32_t
    CORE_MISC_CFG_QD_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT = _libraries['FIXME_STUB'].CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT
    CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT.restype = uint32_t
    CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH = _libraries['FIXME_STUB'].CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH
    CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH.restype = uint32_t
    CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_DATAOUT_SIZE_1_RESERVED_0 = _libraries['FIXME_STUB'].CORE_DATAOUT_SIZE_1_RESERVED_0
    CORE_DATAOUT_SIZE_1_RESERVED_0.restype = uint32_t
    CORE_DATAOUT_SIZE_1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL = _libraries['FIXME_STUB'].CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL
    CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL.restype = uint32_t
    CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_CLIP_TRUNCATE_RESERVED_0 = _libraries['FIXME_STUB'].CORE_CLIP_TRUNCATE_RESERVED_0
    CORE_CLIP_TRUNCATE_RESERVED_0.restype = uint32_t
    CORE_CLIP_TRUNCATE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_CLIP_TRUNCATE_ROUND_TYPE = _libraries['FIXME_STUB'].CORE_CLIP_TRUNCATE_ROUND_TYPE
    CORE_CLIP_TRUNCATE_ROUND_TYPE.restype = uint32_t
    CORE_CLIP_TRUNCATE_ROUND_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_CLIP_TRUNCATE_RESERVED_1 = _libraries['FIXME_STUB'].CORE_CLIP_TRUNCATE_RESERVED_1
    CORE_CLIP_TRUNCATE_RESERVED_1.restype = uint32_t
    CORE_CLIP_TRUNCATE_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    CORE_CLIP_TRUNCATE_CLIP_TRUNCATE = _libraries['FIXME_STUB'].CORE_CLIP_TRUNCATE_CLIP_TRUNCATE
    CORE_CLIP_TRUNCATE_CLIP_TRUNCATE.restype = uint32_t
    CORE_CLIP_TRUNCATE_CLIP_TRUNCATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].DPU_S_STATUS_RESERVED_0
    DPU_S_STATUS_RESERVED_0.restype = uint32_t
    DPU_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].DPU_S_STATUS_STATUS_1
    DPU_S_STATUS_STATUS_1.restype = uint32_t
    DPU_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].DPU_S_STATUS_RESERVED_1
    DPU_S_STATUS_RESERVED_1.restype = uint32_t
    DPU_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].DPU_S_STATUS_STATUS_0
    DPU_S_STATUS_STATUS_0.restype = uint32_t
    DPU_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].DPU_S_POINTER_RESERVED_0
    DPU_S_POINTER_RESERVED_0.restype = uint32_t
    DPU_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].DPU_S_POINTER_EXECUTER
    DPU_S_POINTER_EXECUTER.restype = uint32_t
    DPU_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].DPU_S_POINTER_RESERVED_1
    DPU_S_POINTER_RESERVED_1.restype = uint32_t
    DPU_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].DPU_S_POINTER_EXECUTER_PP_CLEAR
    DPU_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    DPU_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].DPU_S_POINTER_POINTER_PP_CLEAR
    DPU_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    DPU_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].DPU_S_POINTER_POINTER_PP_MODE
    DPU_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    DPU_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].DPU_S_POINTER_EXECUTER_PP_EN
    DPU_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    DPU_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].DPU_S_POINTER_POINTER_PP_EN
    DPU_S_POINTER_POINTER_PP_EN.restype = uint32_t
    DPU_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_S_POINTER_POINTER = _libraries['FIXME_STUB'].DPU_S_POINTER_POINTER
    DPU_S_POINTER_POINTER.restype = uint32_t
    DPU_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].DPU_OPERATION_ENABLE_RESERVED_0
    DPU_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    DPU_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].DPU_OPERATION_ENABLE_OP_EN
    DPU_OPERATION_ENABLE_OP_EN.restype = uint32_t
    DPU_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_COMB_USE = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_COMB_USE
    DPU_FEATURE_MODE_CFG_COMB_USE.restype = uint32_t
    DPU_FEATURE_MODE_CFG_COMB_USE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_TP_EN = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_TP_EN
    DPU_FEATURE_MODE_CFG_TP_EN.restype = uint32_t
    DPU_FEATURE_MODE_CFG_TP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_RGP_TYPE = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_RGP_TYPE
    DPU_FEATURE_MODE_CFG_RGP_TYPE.restype = uint32_t
    DPU_FEATURE_MODE_CFG_RGP_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_NONALIGN = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_NONALIGN
    DPU_FEATURE_MODE_CFG_NONALIGN.restype = uint32_t
    DPU_FEATURE_MODE_CFG_NONALIGN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_SURF_LEN = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_SURF_LEN
    DPU_FEATURE_MODE_CFG_SURF_LEN.restype = uint32_t
    DPU_FEATURE_MODE_CFG_SURF_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_BURST_LEN = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_BURST_LEN
    DPU_FEATURE_MODE_CFG_BURST_LEN.restype = uint32_t
    DPU_FEATURE_MODE_CFG_BURST_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_CONV_MODE = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_CONV_MODE
    DPU_FEATURE_MODE_CFG_CONV_MODE.restype = uint32_t
    DPU_FEATURE_MODE_CFG_CONV_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_OUTPUT_MODE = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_OUTPUT_MODE
    DPU_FEATURE_MODE_CFG_OUTPUT_MODE.restype = uint32_t
    DPU_FEATURE_MODE_CFG_OUTPUT_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_FEATURE_MODE_CFG_FLYING_MODE = _libraries['FIXME_STUB'].DPU_FEATURE_MODE_CFG_FLYING_MODE
    DPU_FEATURE_MODE_CFG_FLYING_MODE.restype = uint32_t
    DPU_FEATURE_MODE_CFG_FLYING_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_OUT_PRECISION = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_OUT_PRECISION
    DPU_DATA_FORMAT_OUT_PRECISION.restype = uint32_t
    DPU_DATA_FORMAT_OUT_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_IN_PRECISION = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_IN_PRECISION
    DPU_DATA_FORMAT_IN_PRECISION.restype = uint32_t
    DPU_DATA_FORMAT_IN_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_EW_TRUNCATE_NEG = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_EW_TRUNCATE_NEG
    DPU_DATA_FORMAT_EW_TRUNCATE_NEG.restype = uint32_t
    DPU_DATA_FORMAT_EW_TRUNCATE_NEG.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG
    DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG.restype = uint32_t
    DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG
    DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG.restype = uint32_t
    DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_MC_SURF_OUT = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_MC_SURF_OUT
    DPU_DATA_FORMAT_MC_SURF_OUT.restype = uint32_t
    DPU_DATA_FORMAT_MC_SURF_OUT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_FORMAT_PROC_PRECISION = _libraries['FIXME_STUB'].DPU_DATA_FORMAT_PROC_PRECISION
    DPU_DATA_FORMAT_PROC_PRECISION.restype = uint32_t
    DPU_DATA_FORMAT_PROC_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OFFSET_PEND_RESERVED_0 = _libraries['FIXME_STUB'].DPU_OFFSET_PEND_RESERVED_0
    DPU_OFFSET_PEND_RESERVED_0.restype = uint32_t
    DPU_OFFSET_PEND_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OFFSET_PEND_OFFSET_PEND = _libraries['FIXME_STUB'].DPU_OFFSET_PEND_OFFSET_PEND
    DPU_OFFSET_PEND_OFFSET_PEND.restype = uint32_t
    DPU_OFFSET_PEND_OFFSET_PEND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DST_BASE_ADDR_DST_BASE_ADDR = _libraries['FIXME_STUB'].DPU_DST_BASE_ADDR_DST_BASE_ADDR
    DPU_DST_BASE_ADDR_DST_BASE_ADDR.restype = uint32_t
    DPU_DST_BASE_ADDR_DST_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DST_SURF_STRIDE_DST_SURF_STRIDE = _libraries['FIXME_STUB'].DPU_DST_SURF_STRIDE_DST_SURF_STRIDE
    DPU_DST_SURF_STRIDE_DST_SURF_STRIDE.restype = uint32_t
    DPU_DST_SURF_STRIDE_DST_SURF_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DST_SURF_STRIDE_RESERVED_0 = _libraries['FIXME_STUB'].DPU_DST_SURF_STRIDE_RESERVED_0
    DPU_DST_SURF_STRIDE_RESERVED_0.restype = uint32_t
    DPU_DST_SURF_STRIDE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_WIDTH_RESERVED_0
    DPU_DATA_CUBE_WIDTH_RESERVED_0.restype = uint32_t
    DPU_DATA_CUBE_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_WIDTH_WIDTH = _libraries['FIXME_STUB'].DPU_DATA_CUBE_WIDTH_WIDTH
    DPU_DATA_CUBE_WIDTH_WIDTH.restype = uint32_t
    DPU_DATA_CUBE_WIDTH_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_HEIGHT_RESERVED_0
    DPU_DATA_CUBE_HEIGHT_RESERVED_0.restype = uint32_t
    DPU_DATA_CUBE_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_HEIGHT_MINMAX_CTL = _libraries['FIXME_STUB'].DPU_DATA_CUBE_HEIGHT_MINMAX_CTL
    DPU_DATA_CUBE_HEIGHT_MINMAX_CTL.restype = uint32_t
    DPU_DATA_CUBE_HEIGHT_MINMAX_CTL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_HEIGHT_RESERVED_1 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_HEIGHT_RESERVED_1
    DPU_DATA_CUBE_HEIGHT_RESERVED_1.restype = uint32_t
    DPU_DATA_CUBE_HEIGHT_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_HEIGHT_HEIGHT = _libraries['FIXME_STUB'].DPU_DATA_CUBE_HEIGHT_HEIGHT
    DPU_DATA_CUBE_HEIGHT_HEIGHT.restype = uint32_t
    DPU_DATA_CUBE_HEIGHT_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_0 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_0
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_0.restype = uint32_t
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1.restype = uint32_t
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_1 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_1
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_1.restype = uint32_t
    DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0.restype = uint32_t
    DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_CHANNEL_RESERVED_0 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_CHANNEL_RESERVED_0
    DPU_DATA_CUBE_CHANNEL_RESERVED_0.restype = uint32_t
    DPU_DATA_CUBE_CHANNEL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL = _libraries['FIXME_STUB'].DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL
    DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL.restype = uint32_t
    DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_CHANNEL_RESERVED_1 = _libraries['FIXME_STUB'].DPU_DATA_CUBE_CHANNEL_RESERVED_1
    DPU_DATA_CUBE_CHANNEL_RESERVED_1.restype = uint32_t
    DPU_DATA_CUBE_CHANNEL_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_DATA_CUBE_CHANNEL_CHANNEL = _libraries['FIXME_STUB'].DPU_DATA_CUBE_CHANNEL_CHANNEL
    DPU_DATA_CUBE_CHANNEL_CHANNEL.restype = uint32_t
    DPU_DATA_CUBE_CHANNEL_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BS_CFG_RESERVED_0
    DPU_BS_CFG_RESERVED_0.restype = uint32_t
    DPU_BS_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_ALU_ALGO = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_ALU_ALGO
    DPU_BS_CFG_BS_ALU_ALGO.restype = uint32_t
    DPU_BS_CFG_BS_ALU_ALGO.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_BS_CFG_RESERVED_1
    DPU_BS_CFG_RESERVED_1.restype = uint32_t
    DPU_BS_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_ALU_SRC = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_ALU_SRC
    DPU_BS_CFG_BS_ALU_SRC.restype = uint32_t
    DPU_BS_CFG_BS_ALU_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_RELUX_EN = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_RELUX_EN
    DPU_BS_CFG_BS_RELUX_EN.restype = uint32_t
    DPU_BS_CFG_BS_RELUX_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_RELU_BYPASS = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_RELU_BYPASS
    DPU_BS_CFG_BS_RELU_BYPASS.restype = uint32_t
    DPU_BS_CFG_BS_RELU_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_MUL_PRELU = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_MUL_PRELU
    DPU_BS_CFG_BS_MUL_PRELU.restype = uint32_t
    DPU_BS_CFG_BS_MUL_PRELU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_MUL_BYPASS = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_MUL_BYPASS
    DPU_BS_CFG_BS_MUL_BYPASS.restype = uint32_t
    DPU_BS_CFG_BS_MUL_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_RESERVED_2 = _libraries['FIXME_STUB'].DPU_BS_CFG_RESERVED_2
    DPU_BS_CFG_RESERVED_2.restype = uint32_t
    DPU_BS_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_ALU_BYPASS = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_ALU_BYPASS
    DPU_BS_CFG_BS_ALU_BYPASS.restype = uint32_t
    DPU_BS_CFG_BS_ALU_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_CFG_BS_BYPASS = _libraries['FIXME_STUB'].DPU_BS_CFG_BS_BYPASS
    DPU_BS_CFG_BS_BYPASS.restype = uint32_t
    DPU_BS_CFG_BS_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_ALU_CFG_BS_ALU_OPERAND = _libraries['FIXME_STUB'].DPU_BS_ALU_CFG_BS_ALU_OPERAND
    DPU_BS_ALU_CFG_BS_ALU_OPERAND.restype = uint32_t
    DPU_BS_ALU_CFG_BS_ALU_OPERAND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_BS_MUL_OPERAND = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_BS_MUL_OPERAND
    DPU_BS_MUL_CFG_BS_MUL_OPERAND.restype = uint32_t
    DPU_BS_MUL_CFG_BS_MUL_OPERAND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_RESERVED_0
    DPU_BS_MUL_CFG_RESERVED_0.restype = uint32_t
    DPU_BS_MUL_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE
    DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE.restype = uint32_t
    DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_RESERVED_1
    DPU_BS_MUL_CFG_RESERVED_1.restype = uint32_t
    DPU_BS_MUL_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_BS_TRUNCATE_SRC = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_BS_TRUNCATE_SRC
    DPU_BS_MUL_CFG_BS_TRUNCATE_SRC.restype = uint32_t
    DPU_BS_MUL_CFG_BS_TRUNCATE_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_MUL_CFG_BS_MUL_SRC = _libraries['FIXME_STUB'].DPU_BS_MUL_CFG_BS_MUL_SRC
    DPU_BS_MUL_CFG_BS_MUL_SRC.restype = uint32_t
    DPU_BS_MUL_CFG_BS_MUL_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT = _libraries['FIXME_STUB'].DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT
    DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT.restype = uint32_t
    DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_RGP_CNTER = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_RGP_CNTER
    DPU_BS_OW_CFG_RGP_CNTER.restype = uint32_t
    DPU_BS_OW_CFG_RGP_CNTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_TP_ORG_EN = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_TP_ORG_EN
    DPU_BS_OW_CFG_TP_ORG_EN.restype = uint32_t
    DPU_BS_OW_CFG_TP_ORG_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_RESERVED_0
    DPU_BS_OW_CFG_RESERVED_0.restype = uint32_t
    DPU_BS_OW_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_SIZE_E_2 = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_SIZE_E_2
    DPU_BS_OW_CFG_SIZE_E_2.restype = uint32_t
    DPU_BS_OW_CFG_SIZE_E_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_SIZE_E_1 = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_SIZE_E_1
    DPU_BS_OW_CFG_SIZE_E_1.restype = uint32_t
    DPU_BS_OW_CFG_SIZE_E_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_SIZE_E_0 = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_SIZE_E_0
    DPU_BS_OW_CFG_SIZE_E_0.restype = uint32_t
    DPU_BS_OW_CFG_SIZE_E_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_OD_BYPASS = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_OD_BYPASS
    DPU_BS_OW_CFG_OD_BYPASS.restype = uint32_t
    DPU_BS_OW_CFG_OD_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_CFG_OW_SRC = _libraries['FIXME_STUB'].DPU_BS_OW_CFG_OW_SRC
    DPU_BS_OW_CFG_OW_SRC.restype = uint32_t
    DPU_BS_OW_CFG_OW_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_OP_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BS_OW_OP_RESERVED_0
    DPU_BS_OW_OP_RESERVED_0.restype = uint32_t
    DPU_BS_OW_OP_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BS_OW_OP_OW_OP = _libraries['FIXME_STUB'].DPU_BS_OW_OP_OW_OP
    DPU_BS_OW_OP_OW_OP.restype = uint32_t
    DPU_BS_OW_OP_OW_OP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_0_RESERVED_0 = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_0_RESERVED_0
    DPU_WDMA_SIZE_0_RESERVED_0.restype = uint32_t
    DPU_WDMA_SIZE_0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_0_TP_PRECISION = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_0_TP_PRECISION
    DPU_WDMA_SIZE_0_TP_PRECISION.restype = uint32_t
    DPU_WDMA_SIZE_0_TP_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_0_SIZE_C_WDMA = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_0_SIZE_C_WDMA
    DPU_WDMA_SIZE_0_SIZE_C_WDMA.restype = uint32_t
    DPU_WDMA_SIZE_0_SIZE_C_WDMA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_0_RESERVED_1 = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_0_RESERVED_1
    DPU_WDMA_SIZE_0_RESERVED_1.restype = uint32_t
    DPU_WDMA_SIZE_0_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_0_CHANNEL_WDMA = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_0_CHANNEL_WDMA
    DPU_WDMA_SIZE_0_CHANNEL_WDMA.restype = uint32_t
    DPU_WDMA_SIZE_0_CHANNEL_WDMA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_1_RESERVED_0 = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_1_RESERVED_0
    DPU_WDMA_SIZE_1_RESERVED_0.restype = uint32_t
    DPU_WDMA_SIZE_1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_1_HEIGHT_WDMA = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_1_HEIGHT_WDMA
    DPU_WDMA_SIZE_1_HEIGHT_WDMA.restype = uint32_t
    DPU_WDMA_SIZE_1_HEIGHT_WDMA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_1_RESERVED_1 = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_1_RESERVED_1
    DPU_WDMA_SIZE_1_RESERVED_1.restype = uint32_t
    DPU_WDMA_SIZE_1_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_WDMA_SIZE_1_WIDTH_WDMA = _libraries['FIXME_STUB'].DPU_WDMA_SIZE_1_WIDTH_WDMA
    DPU_WDMA_SIZE_1_WIDTH_WDMA.restype = uint32_t
    DPU_WDMA_SIZE_1_WIDTH_WDMA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BN_CFG_RESERVED_0
    DPU_BN_CFG_RESERVED_0.restype = uint32_t
    DPU_BN_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_ALU_ALGO = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_ALU_ALGO
    DPU_BN_CFG_BN_ALU_ALGO.restype = uint32_t
    DPU_BN_CFG_BN_ALU_ALGO.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_BN_CFG_RESERVED_1
    DPU_BN_CFG_RESERVED_1.restype = uint32_t
    DPU_BN_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_ALU_SRC = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_ALU_SRC
    DPU_BN_CFG_BN_ALU_SRC.restype = uint32_t
    DPU_BN_CFG_BN_ALU_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_RELUX_EN = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_RELUX_EN
    DPU_BN_CFG_BN_RELUX_EN.restype = uint32_t
    DPU_BN_CFG_BN_RELUX_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_RELU_BYPASS = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_RELU_BYPASS
    DPU_BN_CFG_BN_RELU_BYPASS.restype = uint32_t
    DPU_BN_CFG_BN_RELU_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_MUL_PRELU = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_MUL_PRELU
    DPU_BN_CFG_BN_MUL_PRELU.restype = uint32_t
    DPU_BN_CFG_BN_MUL_PRELU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_MUL_BYPASS = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_MUL_BYPASS
    DPU_BN_CFG_BN_MUL_BYPASS.restype = uint32_t
    DPU_BN_CFG_BN_MUL_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_RESERVED_2 = _libraries['FIXME_STUB'].DPU_BN_CFG_RESERVED_2
    DPU_BN_CFG_RESERVED_2.restype = uint32_t
    DPU_BN_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_ALU_BYPASS = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_ALU_BYPASS
    DPU_BN_CFG_BN_ALU_BYPASS.restype = uint32_t
    DPU_BN_CFG_BN_ALU_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_CFG_BN_BYPASS = _libraries['FIXME_STUB'].DPU_BN_CFG_BN_BYPASS
    DPU_BN_CFG_BN_BYPASS.restype = uint32_t
    DPU_BN_CFG_BN_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_ALU_CFG_BN_ALU_OPERAND = _libraries['FIXME_STUB'].DPU_BN_ALU_CFG_BN_ALU_OPERAND
    DPU_BN_ALU_CFG_BN_ALU_OPERAND.restype = uint32_t
    DPU_BN_ALU_CFG_BN_ALU_OPERAND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_BN_MUL_OPERAND = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_BN_MUL_OPERAND
    DPU_BN_MUL_CFG_BN_MUL_OPERAND.restype = uint32_t
    DPU_BN_MUL_CFG_BN_MUL_OPERAND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_RESERVED_0
    DPU_BN_MUL_CFG_RESERVED_0.restype = uint32_t
    DPU_BN_MUL_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE
    DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE.restype = uint32_t
    DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_RESERVED_1
    DPU_BN_MUL_CFG_RESERVED_1.restype = uint32_t
    DPU_BN_MUL_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_BN_TRUNCATE_SRC = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_BN_TRUNCATE_SRC
    DPU_BN_MUL_CFG_BN_TRUNCATE_SRC.restype = uint32_t
    DPU_BN_MUL_CFG_BN_TRUNCATE_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_MUL_CFG_BN_MUL_SRC = _libraries['FIXME_STUB'].DPU_BN_MUL_CFG_BN_MUL_SRC
    DPU_BN_MUL_CFG_BN_MUL_SRC.restype = uint32_t
    DPU_BN_MUL_CFG_BN_MUL_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT = _libraries['FIXME_STUB'].DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT
    DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT.restype = uint32_t
    DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_CVT_TYPE = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_CVT_TYPE
    DPU_EW_CFG_EW_CVT_TYPE.restype = uint32_t
    DPU_EW_CFG_EW_CVT_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_CVT_ROUND = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_CVT_ROUND
    DPU_EW_CFG_EW_CVT_ROUND.restype = uint32_t
    DPU_EW_CFG_EW_CVT_ROUND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_DATA_MODE = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_DATA_MODE
    DPU_EW_CFG_EW_DATA_MODE.restype = uint32_t
    DPU_EW_CFG_EW_DATA_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_EW_CFG_RESERVED_0
    DPU_EW_CFG_RESERVED_0.restype = uint32_t
    DPU_EW_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EDATA_SIZE = _libraries['FIXME_STUB'].DPU_EW_CFG_EDATA_SIZE
    DPU_EW_CFG_EDATA_SIZE.restype = uint32_t
    DPU_EW_CFG_EDATA_SIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_EQUAL_EN = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_EQUAL_EN
    DPU_EW_CFG_EW_EQUAL_EN.restype = uint32_t
    DPU_EW_CFG_EW_EQUAL_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_BINARY_EN = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_BINARY_EN
    DPU_EW_CFG_EW_BINARY_EN.restype = uint32_t
    DPU_EW_CFG_EW_BINARY_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_ALU_ALGO = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_ALU_ALGO
    DPU_EW_CFG_EW_ALU_ALGO.restype = uint32_t
    DPU_EW_CFG_EW_ALU_ALGO.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_EW_CFG_RESERVED_1
    DPU_EW_CFG_RESERVED_1.restype = uint32_t
    DPU_EW_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_RELUX_EN = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_RELUX_EN
    DPU_EW_CFG_EW_RELUX_EN.restype = uint32_t
    DPU_EW_CFG_EW_RELUX_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_RELU_BYPASS = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_RELU_BYPASS
    DPU_EW_CFG_EW_RELU_BYPASS.restype = uint32_t
    DPU_EW_CFG_EW_RELU_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_OP_CVT_BYPASS = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_OP_CVT_BYPASS
    DPU_EW_CFG_EW_OP_CVT_BYPASS.restype = uint32_t
    DPU_EW_CFG_EW_OP_CVT_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_LUT_BYPASS = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_LUT_BYPASS
    DPU_EW_CFG_EW_LUT_BYPASS.restype = uint32_t
    DPU_EW_CFG_EW_LUT_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_OP_SRC = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_OP_SRC
    DPU_EW_CFG_EW_OP_SRC.restype = uint32_t
    DPU_EW_CFG_EW_OP_SRC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_MUL_PRELU = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_MUL_PRELU
    DPU_EW_CFG_EW_MUL_PRELU.restype = uint32_t
    DPU_EW_CFG_EW_MUL_PRELU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_RESERVED_2 = _libraries['FIXME_STUB'].DPU_EW_CFG_RESERVED_2
    DPU_EW_CFG_RESERVED_2.restype = uint32_t
    DPU_EW_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_OP_TYPE = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_OP_TYPE
    DPU_EW_CFG_EW_OP_TYPE.restype = uint32_t
    DPU_EW_CFG_EW_OP_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_OP_BYPASS = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_OP_BYPASS
    DPU_EW_CFG_EW_OP_BYPASS.restype = uint32_t
    DPU_EW_CFG_EW_OP_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CFG_EW_BYPASS = _libraries['FIXME_STUB'].DPU_EW_CFG_EW_BYPASS
    DPU_EW_CFG_EW_BYPASS.restype = uint32_t
    DPU_EW_CFG_EW_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET = _libraries['FIXME_STUB'].DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET
    DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET.restype = uint32_t
    DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE = _libraries['FIXME_STUB'].DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE
    DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE.restype = uint32_t
    DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT = _libraries['FIXME_STUB'].DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT.restype = uint32_t
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE = _libraries['FIXME_STUB'].DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE.restype = uint32_t
    DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT = _libraries['FIXME_STUB'].DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT
    DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT.restype = uint32_t
    DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET = _libraries['FIXME_STUB'].DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET
    DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET.restype = uint32_t
    DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SCALE_RESERVED_0 = _libraries['FIXME_STUB'].DPU_OUT_CVT_SCALE_RESERVED_0
    DPU_OUT_CVT_SCALE_RESERVED_0.restype = uint32_t
    DPU_OUT_CVT_SCALE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SCALE_FP32TOFP16_EN = _libraries['FIXME_STUB'].DPU_OUT_CVT_SCALE_FP32TOFP16_EN
    DPU_OUT_CVT_SCALE_FP32TOFP16_EN.restype = uint32_t
    DPU_OUT_CVT_SCALE_FP32TOFP16_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SCALE_OUT_CVT_SCALE = _libraries['FIXME_STUB'].DPU_OUT_CVT_SCALE_OUT_CVT_SCALE
    DPU_OUT_CVT_SCALE_OUT_CVT_SCALE.restype = uint32_t
    DPU_OUT_CVT_SCALE_OUT_CVT_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SHIFT_CVT_TYPE = _libraries['FIXME_STUB'].DPU_OUT_CVT_SHIFT_CVT_TYPE
    DPU_OUT_CVT_SHIFT_CVT_TYPE.restype = uint32_t
    DPU_OUT_CVT_SHIFT_CVT_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SHIFT_CVT_ROUND = _libraries['FIXME_STUB'].DPU_OUT_CVT_SHIFT_CVT_ROUND
    DPU_OUT_CVT_SHIFT_CVT_ROUND.restype = uint32_t
    DPU_OUT_CVT_SHIFT_CVT_ROUND.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SHIFT_RESERVED_0 = _libraries['FIXME_STUB'].DPU_OUT_CVT_SHIFT_RESERVED_0
    DPU_OUT_CVT_SHIFT_RESERVED_0.restype = uint32_t
    DPU_OUT_CVT_SHIFT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SHIFT_MINUS_EXP = _libraries['FIXME_STUB'].DPU_OUT_CVT_SHIFT_MINUS_EXP
    DPU_OUT_CVT_SHIFT_MINUS_EXP.restype = uint32_t
    DPU_OUT_CVT_SHIFT_MINUS_EXP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT = _libraries['FIXME_STUB'].DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT
    DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT.restype = uint32_t
    DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_0_EW_OPERAND_0 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_0_EW_OPERAND_0
    DPU_EW_OP_VALUE_0_EW_OPERAND_0.restype = uint32_t
    DPU_EW_OP_VALUE_0_EW_OPERAND_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_1_EW_OPERAND_1 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_1_EW_OPERAND_1
    DPU_EW_OP_VALUE_1_EW_OPERAND_1.restype = uint32_t
    DPU_EW_OP_VALUE_1_EW_OPERAND_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_2_EW_OPERAND_2 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_2_EW_OPERAND_2
    DPU_EW_OP_VALUE_2_EW_OPERAND_2.restype = uint32_t
    DPU_EW_OP_VALUE_2_EW_OPERAND_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_3_EW_OPERAND_3 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_3_EW_OPERAND_3
    DPU_EW_OP_VALUE_3_EW_OPERAND_3.restype = uint32_t
    DPU_EW_OP_VALUE_3_EW_OPERAND_3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_4_EW_OPERAND_4 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_4_EW_OPERAND_4
    DPU_EW_OP_VALUE_4_EW_OPERAND_4.restype = uint32_t
    DPU_EW_OP_VALUE_4_EW_OPERAND_4.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_5_EW_OPERAND_5 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_5_EW_OPERAND_5
    DPU_EW_OP_VALUE_5_EW_OPERAND_5.restype = uint32_t
    DPU_EW_OP_VALUE_5_EW_OPERAND_5.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_6_EW_OPERAND_6 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_6_EW_OPERAND_6
    DPU_EW_OP_VALUE_6_EW_OPERAND_6.restype = uint32_t
    DPU_EW_OP_VALUE_6_EW_OPERAND_6.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_EW_OP_VALUE_7_EW_OPERAND_7 = _libraries['FIXME_STUB'].DPU_EW_OP_VALUE_7_EW_OPERAND_7
    DPU_EW_OP_VALUE_7_EW_OPERAND_7.restype = uint32_t
    DPU_EW_OP_VALUE_7_EW_OPERAND_7.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_SURFACE_ADD_SURF_ADD = _libraries['FIXME_STUB'].DPU_SURFACE_ADD_SURF_ADD
    DPU_SURFACE_ADD_SURF_ADD.restype = uint32_t
    DPU_SURFACE_ADD_SURF_ADD.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_SURFACE_ADD_RESERVED_0 = _libraries['FIXME_STUB'].DPU_SURFACE_ADD_RESERVED_0
    DPU_SURFACE_ADD_RESERVED_0.restype = uint32_t
    DPU_SURFACE_ADD_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_CFG_RESERVED_0
    DPU_LUT_ACCESS_CFG_RESERVED_0.restype = uint32_t
    DPU_LUT_ACCESS_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE
    DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE.restype = uint32_t
    DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_CFG_LUT_TABLE_ID = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_CFG_LUT_TABLE_ID
    DPU_LUT_ACCESS_CFG_LUT_TABLE_ID.restype = uint32_t
    DPU_LUT_ACCESS_CFG_LUT_TABLE_ID.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_CFG_RESERVED_1
    DPU_LUT_ACCESS_CFG_RESERVED_1.restype = uint32_t
    DPU_LUT_ACCESS_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_CFG_LUT_ADDR = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_CFG_LUT_ADDR
    DPU_LUT_ACCESS_CFG_LUT_ADDR.restype = uint32_t
    DPU_LUT_ACCESS_CFG_LUT_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_DATA_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_DATA_RESERVED_0
    DPU_LUT_ACCESS_DATA_RESERVED_0.restype = uint32_t
    DPU_LUT_ACCESS_DATA_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA = _libraries['FIXME_STUB'].DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA
    DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA.restype = uint32_t
    DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_CFG_RESERVED_0
    DPU_LUT_CFG_RESERVED_0.restype = uint32_t
    DPU_LUT_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_CAL_SEL = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_CAL_SEL
    DPU_LUT_CFG_LUT_CAL_SEL.restype = uint32_t
    DPU_LUT_CFG_LUT_CAL_SEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_HYBRID_PRIORITY = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_HYBRID_PRIORITY
    DPU_LUT_CFG_LUT_HYBRID_PRIORITY.restype = uint32_t
    DPU_LUT_CFG_LUT_HYBRID_PRIORITY.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_OFLOW_PRIORITY = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_OFLOW_PRIORITY
    DPU_LUT_CFG_LUT_OFLOW_PRIORITY.restype = uint32_t
    DPU_LUT_CFG_LUT_OFLOW_PRIORITY.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_UFLOW_PRIORITY = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_UFLOW_PRIORITY
    DPU_LUT_CFG_LUT_UFLOW_PRIORITY.restype = uint32_t
    DPU_LUT_CFG_LUT_UFLOW_PRIORITY.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_LO_LE_MUX = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_LO_LE_MUX
    DPU_LUT_CFG_LUT_LO_LE_MUX.restype = uint32_t
    DPU_LUT_CFG_LUT_LO_LE_MUX.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_EXPAND_EN = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_EXPAND_EN
    DPU_LUT_CFG_LUT_EXPAND_EN.restype = uint32_t
    DPU_LUT_CFG_LUT_EXPAND_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_CFG_LUT_ROAD_SEL = _libraries['FIXME_STUB'].DPU_LUT_CFG_LUT_ROAD_SEL
    DPU_LUT_CFG_LUT_ROAD_SEL.restype = uint32_t
    DPU_LUT_CFG_LUT_ROAD_SEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_INFO_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_INFO_RESERVED_0
    DPU_LUT_INFO_RESERVED_0.restype = uint32_t
    DPU_LUT_INFO_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_INFO_LUT_LO_INDEX_SELECT = _libraries['FIXME_STUB'].DPU_LUT_INFO_LUT_LO_INDEX_SELECT
    DPU_LUT_INFO_LUT_LO_INDEX_SELECT.restype = uint32_t
    DPU_LUT_INFO_LUT_LO_INDEX_SELECT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_INFO_LUT_LE_INDEX_SELECT = _libraries['FIXME_STUB'].DPU_LUT_INFO_LUT_LE_INDEX_SELECT
    DPU_LUT_INFO_LUT_LE_INDEX_SELECT.restype = uint32_t
    DPU_LUT_INFO_LUT_LE_INDEX_SELECT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_INFO_RESERVED_1 = _libraries['FIXME_STUB'].DPU_LUT_INFO_RESERVED_1
    DPU_LUT_INFO_RESERVED_1.restype = uint32_t
    DPU_LUT_INFO_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_START_LUT_LE_START = _libraries['FIXME_STUB'].DPU_LUT_LE_START_LUT_LE_START
    DPU_LUT_LE_START_LUT_LE_START.restype = uint32_t
    DPU_LUT_LE_START_LUT_LE_START.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_END_LUT_LE_END = _libraries['FIXME_STUB'].DPU_LUT_LE_END_LUT_LE_END
    DPU_LUT_LE_END_LUT_LE_END.restype = uint32_t
    DPU_LUT_LE_END_LUT_LE_END.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_START_LUT_LO_START = _libraries['FIXME_STUB'].DPU_LUT_LO_START_LUT_LO_START
    DPU_LUT_LO_START_LUT_LO_START.restype = uint32_t
    DPU_LUT_LO_START_LUT_LO_START.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_END_LUT_LO_END = _libraries['FIXME_STUB'].DPU_LUT_LO_END_LUT_LO_END
    DPU_LUT_LO_END_LUT_LO_END.restype = uint32_t
    DPU_LUT_LO_END_LUT_LO_END.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE = _libraries['FIXME_STUB'].DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE.restype = uint32_t
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE = _libraries['FIXME_STUB'].DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE.restype = uint32_t
    DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_SLOPE_SHIFT_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_LE_SLOPE_SHIFT_RESERVED_0
    DPU_LUT_LE_SLOPE_SHIFT_RESERVED_0.restype = uint32_t
    DPU_LUT_LE_SLOPE_SHIFT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT = _libraries['FIXME_STUB'].DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT.restype = uint32_t
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT = _libraries['FIXME_STUB'].DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT.restype = uint32_t
    DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE = _libraries['FIXME_STUB'].DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE.restype = uint32_t
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE = _libraries['FIXME_STUB'].DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE.restype = uint32_t
    DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_SLOPE_SHIFT_RESERVED_0 = _libraries['FIXME_STUB'].DPU_LUT_LO_SLOPE_SHIFT_RESERVED_0
    DPU_LUT_LO_SLOPE_SHIFT_RESERVED_0.restype = uint32_t
    DPU_LUT_LO_SLOPE_SHIFT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT = _libraries['FIXME_STUB'].DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT.restype = uint32_t
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT = _libraries['FIXME_STUB'].DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT.restype = uint32_t
    DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_STATUS_RESERVED_0
    DPU_RDMA_RDMA_S_STATUS_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_STATUS_STATUS_1
    DPU_RDMA_RDMA_S_STATUS_STATUS_1.restype = uint32_t
    DPU_RDMA_RDMA_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_STATUS_RESERVED_1
    DPU_RDMA_RDMA_S_STATUS_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_STATUS_STATUS_0
    DPU_RDMA_RDMA_S_STATUS_STATUS_0.restype = uint32_t
    DPU_RDMA_RDMA_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_RESERVED_0
    DPU_RDMA_RDMA_S_POINTER_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_EXECUTER
    DPU_RDMA_RDMA_S_POINTER_EXECUTER.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_RESERVED_1
    DPU_RDMA_RDMA_S_POINTER_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_S_POINTER_POINTER = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_S_POINTER_POINTER
    DPU_RDMA_RDMA_S_POINTER_POINTER.restype = uint32_t
    DPU_RDMA_RDMA_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0
    DPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN
    DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN.restype = uint32_t
    DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_WIDTH_RESERVED_0
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_0
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_1
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_RESERVED_0
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL.restype = uint32_t
    DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR
    DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_0
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE
    DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE.restype = uint32_t
    DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_1
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR
    DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_0
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE
    DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE.restype = uint32_t
    DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_1
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR
    DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_RESERVED_0
    DPU_RDMA_RDMA_ERDMA_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS
    DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE.restype = uint32_t
    DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR
    DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE
    DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE.restype = uint32_t
    DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_EW_SURF_STRIDE_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_EW_SURF_STRIDE_RESERVED_0
    DPU_RDMA_RDMA_EW_SURF_STRIDE_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_EW_SURF_STRIDE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_RESERVED_0
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE.restype = uint32_t
    DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR
    DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_RESERVED_0
    DPU_RDMA_RDMA_SRC_DMA_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD
    DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN
    DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH.restype = uint32_t
    DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR
    DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR.restype = uint32_t
    DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_SURF_NOTCH_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_SURF_NOTCH_RESERVED_0
    DPU_RDMA_RDMA_SURF_NOTCH_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_SURF_NOTCH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE
    DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE.restype = uint32_t
    DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_PAD_CFG_RESERVED_0
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_PAD_CFG_PAD_TOP = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_PAD_CFG_PAD_TOP
    DPU_RDMA_RDMA_PAD_CFG_PAD_TOP.restype = uint32_t
    DPU_RDMA_RDMA_PAD_CFG_PAD_TOP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_1 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_PAD_CFG_RESERVED_1
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_1.restype = uint32_t
    DPU_RDMA_RDMA_PAD_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT
    DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT.restype = uint32_t
    DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_WEIGHT_E_WEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_WEIGHT_E_WEIGHT
    DPU_RDMA_RDMA_WEIGHT_E_WEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_WEIGHT_E_WEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_WEIGHT_N_WEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_WEIGHT_N_WEIGHT
    DPU_RDMA_RDMA_WEIGHT_N_WEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_WEIGHT_N_WEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_WEIGHT_B_WEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_WEIGHT_B_WEIGHT
    DPU_RDMA_RDMA_WEIGHT_B_WEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_WEIGHT_B_WEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_WEIGHT_M_WEIGHT = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_WEIGHT_M_WEIGHT
    DPU_RDMA_RDMA_WEIGHT_M_WEIGHT.restype = uint32_t
    DPU_RDMA_RDMA_WEIGHT_M_WEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH
    DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH.restype = uint32_t
    DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DPU_RDMA_RDMA_EW_SURF_NOTCH_RESERVED_0 = _libraries['FIXME_STUB'].DPU_RDMA_RDMA_EW_SURF_NOTCH_RESERVED_0
    DPU_RDMA_RDMA_EW_SURF_NOTCH_RESERVED_0.restype = uint32_t
    DPU_RDMA_RDMA_EW_SURF_NOTCH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].PPU_S_STATUS_RESERVED_0
    PPU_S_STATUS_RESERVED_0.restype = uint32_t
    PPU_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].PPU_S_STATUS_STATUS_1
    PPU_S_STATUS_STATUS_1.restype = uint32_t
    PPU_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].PPU_S_STATUS_RESERVED_1
    PPU_S_STATUS_RESERVED_1.restype = uint32_t
    PPU_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].PPU_S_STATUS_STATUS_0
    PPU_S_STATUS_STATUS_0.restype = uint32_t
    PPU_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].PPU_S_POINTER_RESERVED_0
    PPU_S_POINTER_RESERVED_0.restype = uint32_t
    PPU_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].PPU_S_POINTER_EXECUTER
    PPU_S_POINTER_EXECUTER.restype = uint32_t
    PPU_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].PPU_S_POINTER_RESERVED_1
    PPU_S_POINTER_RESERVED_1.restype = uint32_t
    PPU_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].PPU_S_POINTER_EXECUTER_PP_CLEAR
    PPU_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    PPU_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].PPU_S_POINTER_POINTER_PP_CLEAR
    PPU_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    PPU_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].PPU_S_POINTER_POINTER_PP_MODE
    PPU_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    PPU_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].PPU_S_POINTER_EXECUTER_PP_EN
    PPU_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    PPU_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].PPU_S_POINTER_POINTER_PP_EN
    PPU_S_POINTER_POINTER_PP_EN.restype = uint32_t
    PPU_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_S_POINTER_POINTER = _libraries['FIXME_STUB'].PPU_S_POINTER_POINTER
    PPU_S_POINTER_POINTER.restype = uint32_t
    PPU_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].PPU_OPERATION_ENABLE_RESERVED_0
    PPU_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    PPU_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].PPU_OPERATION_ENABLE_OP_EN
    PPU_OPERATION_ENABLE_OP_EN.restype = uint32_t
    PPU_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_WIDTH_RESERVED_0
    PPU_DATA_CUBE_IN_WIDTH_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_IN_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH
    PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH.restype = uint32_t
    PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_HEIGHT_RESERVED_0
    PPU_DATA_CUBE_IN_HEIGHT_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_IN_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT
    PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT.restype = uint32_t
    PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_CHANNEL_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_CHANNEL_RESERVED_0
    PPU_DATA_CUBE_IN_CHANNEL_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_IN_CHANNEL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL = _libraries['FIXME_STUB'].PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL
    PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL.restype = uint32_t
    PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_WIDTH_RESERVED_0
    PPU_DATA_CUBE_OUT_WIDTH_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_OUT_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH
    PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH.restype = uint32_t
    PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_HEIGHT_RESERVED_0
    PPU_DATA_CUBE_OUT_HEIGHT_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_OUT_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT
    PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT.restype = uint32_t
    PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_CHANNEL_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_CHANNEL_RESERVED_0
    PPU_DATA_CUBE_OUT_CHANNEL_RESERVED_0.restype = uint32_t
    PPU_DATA_CUBE_OUT_CHANNEL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL = _libraries['FIXME_STUB'].PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL
    PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL.restype = uint32_t
    PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_RESERVED_0 = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_RESERVED_0
    PPU_OPERATION_MODE_CFG_RESERVED_0.restype = uint32_t
    PPU_OPERATION_MODE_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_INDEX_EN = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_INDEX_EN
    PPU_OPERATION_MODE_CFG_INDEX_EN.restype = uint32_t
    PPU_OPERATION_MODE_CFG_INDEX_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_RESERVED_1 = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_RESERVED_1
    PPU_OPERATION_MODE_CFG_RESERVED_1.restype = uint32_t
    PPU_OPERATION_MODE_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_NOTCH_ADDR = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_NOTCH_ADDR
    PPU_OPERATION_MODE_CFG_NOTCH_ADDR.restype = uint32_t
    PPU_OPERATION_MODE_CFG_NOTCH_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_RESERVED_2 = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_RESERVED_2
    PPU_OPERATION_MODE_CFG_RESERVED_2.restype = uint32_t
    PPU_OPERATION_MODE_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_USE_CNT = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_USE_CNT
    PPU_OPERATION_MODE_CFG_USE_CNT.restype = uint32_t
    PPU_OPERATION_MODE_CFG_USE_CNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_FLYING_MODE = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_FLYING_MODE
    PPU_OPERATION_MODE_CFG_FLYING_MODE.restype = uint32_t
    PPU_OPERATION_MODE_CFG_FLYING_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_RESERVED_3 = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_RESERVED_3
    PPU_OPERATION_MODE_CFG_RESERVED_3.restype = uint32_t
    PPU_OPERATION_MODE_CFG_RESERVED_3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_OPERATION_MODE_CFG_POOLING_METHOD = _libraries['FIXME_STUB'].PPU_OPERATION_MODE_CFG_POOLING_METHOD
    PPU_OPERATION_MODE_CFG_POOLING_METHOD.restype = uint32_t
    PPU_OPERATION_MODE_CFG_POOLING_METHOD.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_RESERVED_0 = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_RESERVED_0
    PPU_POOLING_KERNEL_CFG_RESERVED_0.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_RESERVED_1 = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_RESERVED_1
    PPU_POOLING_KERNEL_CFG_RESERVED_1.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT
    PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_RESERVED_2 = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_RESERVED_2
    PPU_POOLING_KERNEL_CFG_RESERVED_2.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH = _libraries['FIXME_STUB'].PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH
    PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH.restype = uint32_t
    PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RECIP_KERNEL_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RECIP_KERNEL_WIDTH_RESERVED_0
    PPU_RECIP_KERNEL_WIDTH_RESERVED_0.restype = uint32_t
    PPU_RECIP_KERNEL_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH = _libraries['FIXME_STUB'].PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH
    PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH.restype = uint32_t
    PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RECIP_KERNEL_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RECIP_KERNEL_HEIGHT_RESERVED_0
    PPU_RECIP_KERNEL_HEIGHT_RESERVED_0.restype = uint32_t
    PPU_RECIP_KERNEL_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT = _libraries['FIXME_STUB'].PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT
    PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT.restype = uint32_t
    PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_RESERVED_0 = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_RESERVED_0
    PPU_POOLING_PADDING_CFG_RESERVED_0.restype = uint32_t
    PPU_POOLING_PADDING_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_PAD_BOTTOM = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_PAD_BOTTOM
    PPU_POOLING_PADDING_CFG_PAD_BOTTOM.restype = uint32_t
    PPU_POOLING_PADDING_CFG_PAD_BOTTOM.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_RESERVED_1 = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_RESERVED_1
    PPU_POOLING_PADDING_CFG_RESERVED_1.restype = uint32_t
    PPU_POOLING_PADDING_CFG_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_PAD_RIGHT = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_PAD_RIGHT
    PPU_POOLING_PADDING_CFG_PAD_RIGHT.restype = uint32_t
    PPU_POOLING_PADDING_CFG_PAD_RIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_RESERVED_2 = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_RESERVED_2
    PPU_POOLING_PADDING_CFG_RESERVED_2.restype = uint32_t
    PPU_POOLING_PADDING_CFG_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_PAD_TOP = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_PAD_TOP
    PPU_POOLING_PADDING_CFG_PAD_TOP.restype = uint32_t
    PPU_POOLING_PADDING_CFG_PAD_TOP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_RESERVED_3 = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_RESERVED_3
    PPU_POOLING_PADDING_CFG_RESERVED_3.restype = uint32_t
    PPU_POOLING_PADDING_CFG_RESERVED_3.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_POOLING_PADDING_CFG_PAD_LEFT = _libraries['FIXME_STUB'].PPU_POOLING_PADDING_CFG_PAD_LEFT
    PPU_POOLING_PADDING_CFG_PAD_LEFT.restype = uint32_t
    PPU_POOLING_PADDING_CFG_PAD_LEFT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0 = _libraries['FIXME_STUB'].PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0
    PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0.restype = uint32_t
    PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_PADDING_VALUE_2_CFG_RESERVED_0 = _libraries['FIXME_STUB'].PPU_PADDING_VALUE_2_CFG_RESERVED_0
    PPU_PADDING_VALUE_2_CFG_RESERVED_0.restype = uint32_t
    PPU_PADDING_VALUE_2_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1 = _libraries['FIXME_STUB'].PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1
    PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1.restype = uint32_t
    PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DST_BASE_ADDR_DST_BASE_ADDR = _libraries['FIXME_STUB'].PPU_DST_BASE_ADDR_DST_BASE_ADDR
    PPU_DST_BASE_ADDR_DST_BASE_ADDR.restype = uint32_t
    PPU_DST_BASE_ADDR_DST_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DST_BASE_ADDR_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DST_BASE_ADDR_RESERVED_0
    PPU_DST_BASE_ADDR_RESERVED_0.restype = uint32_t
    PPU_DST_BASE_ADDR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DST_SURF_STRIDE_DST_SURF_STRIDE = _libraries['FIXME_STUB'].PPU_DST_SURF_STRIDE_DST_SURF_STRIDE
    PPU_DST_SURF_STRIDE_DST_SURF_STRIDE.restype = uint32_t
    PPU_DST_SURF_STRIDE_DST_SURF_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DST_SURF_STRIDE_RESERVED_0 = _libraries['FIXME_STUB'].PPU_DST_SURF_STRIDE_RESERVED_0
    PPU_DST_SURF_STRIDE_RESERVED_0.restype = uint32_t
    PPU_DST_SURF_STRIDE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_FORMAT_INDEX_ADD = _libraries['FIXME_STUB'].PPU_DATA_FORMAT_INDEX_ADD
    PPU_DATA_FORMAT_INDEX_ADD.restype = uint32_t
    PPU_DATA_FORMAT_INDEX_ADD.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_FORMAT_DPU_FLYIN = _libraries['FIXME_STUB'].PPU_DATA_FORMAT_DPU_FLYIN
    PPU_DATA_FORMAT_DPU_FLYIN.restype = uint32_t
    PPU_DATA_FORMAT_DPU_FLYIN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_DATA_FORMAT_PROC_PRECISION = _libraries['FIXME_STUB'].PPU_DATA_FORMAT_PROC_PRECISION
    PPU_DATA_FORMAT_PROC_PRECISION.restype = uint32_t
    PPU_DATA_FORMAT_PROC_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_SURF_LEN = _libraries['FIXME_STUB'].PPU_MISC_CTRL_SURF_LEN
    PPU_MISC_CTRL_SURF_LEN.restype = uint32_t
    PPU_MISC_CTRL_SURF_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_RESERVED_0 = _libraries['FIXME_STUB'].PPU_MISC_CTRL_RESERVED_0
    PPU_MISC_CTRL_RESERVED_0.restype = uint32_t
    PPU_MISC_CTRL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_MC_SURF_OUT = _libraries['FIXME_STUB'].PPU_MISC_CTRL_MC_SURF_OUT
    PPU_MISC_CTRL_MC_SURF_OUT.restype = uint32_t
    PPU_MISC_CTRL_MC_SURF_OUT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_NONALIGN = _libraries['FIXME_STUB'].PPU_MISC_CTRL_NONALIGN
    PPU_MISC_CTRL_NONALIGN.restype = uint32_t
    PPU_MISC_CTRL_NONALIGN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_RESERVED_1 = _libraries['FIXME_STUB'].PPU_MISC_CTRL_RESERVED_1
    PPU_MISC_CTRL_RESERVED_1.restype = uint32_t
    PPU_MISC_CTRL_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_MISC_CTRL_BURST_LEN = _libraries['FIXME_STUB'].PPU_MISC_CTRL_BURST_LEN
    PPU_MISC_CTRL_BURST_LEN.restype = uint32_t
    PPU_MISC_CTRL_BURST_LEN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_STATUS_RESERVED_0
    PPU_RDMA_RDMA_S_STATUS_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_S_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_STATUS_STATUS_1 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_STATUS_STATUS_1
    PPU_RDMA_RDMA_S_STATUS_STATUS_1.restype = uint32_t
    PPU_RDMA_RDMA_S_STATUS_STATUS_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_STATUS_RESERVED_1
    PPU_RDMA_RDMA_S_STATUS_RESERVED_1.restype = uint32_t
    PPU_RDMA_RDMA_S_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_STATUS_STATUS_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_STATUS_STATUS_0
    PPU_RDMA_RDMA_S_STATUS_STATUS_0.restype = uint32_t
    PPU_RDMA_RDMA_S_STATUS_STATUS_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_RESERVED_0
    PPU_RDMA_RDMA_S_POINTER_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_EXECUTER = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_EXECUTER
    PPU_RDMA_RDMA_S_POINTER_EXECUTER.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_EXECUTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_RESERVED_1 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_RESERVED_1
    PPU_RDMA_RDMA_S_POINTER_RESERVED_1.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_S_POINTER_POINTER = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_S_POINTER_POINTER
    PPU_RDMA_RDMA_S_POINTER_POINTER.restype = uint32_t
    PPU_RDMA_RDMA_S_POINTER_POINTER.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0
    PPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN
    PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN.restype = uint32_t
    PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_WIDTH_RESERVED_0
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_HEIGHT_RESERVED_0
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_CHANNEL_RESERVED_0
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL.restype = uint32_t
    PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR
    PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR.restype = uint32_t
    PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE.restype = uint32_t
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_SRC_LINE_STRIDE_RESERVED_0
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_SRC_LINE_STRIDE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE.restype = uint32_t
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_SRC_SURF_STRIDE_RESERVED_0
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_SRC_SURF_STRIDE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_DATA_FORMAT_RESERVED_0 = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_DATA_FORMAT_RESERVED_0
    PPU_RDMA_RDMA_DATA_FORMAT_RESERVED_0.restype = uint32_t
    PPU_RDMA_RDMA_DATA_FORMAT_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION = _libraries['FIXME_STUB'].PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION
    PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION.restype = uint32_t
    PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_OUTSTANDING_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_OUTSTANDING_RESERVED_0
    DDMA_CFG_OUTSTANDING_RESERVED_0.restype = uint32_t
    DDMA_CFG_OUTSTANDING_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_OUTSTANDING_WR_OS_CNT = _libraries['FIXME_STUB'].DDMA_CFG_OUTSTANDING_WR_OS_CNT
    DDMA_CFG_OUTSTANDING_WR_OS_CNT.restype = uint32_t
    DDMA_CFG_OUTSTANDING_WR_OS_CNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_OUTSTANDING_RD_OS_CNT = _libraries['FIXME_STUB'].DDMA_CFG_OUTSTANDING_RD_OS_CNT
    DDMA_CFG_OUTSTANDING_RD_OS_CNT.restype = uint32_t
    DDMA_CFG_OUTSTANDING_RD_OS_CNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP
    DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP.restype = uint32_t
    DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU
    DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU.restype = uint32_t
    DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL
    DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL.restype = uint32_t
    DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE
    DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE.restype = uint32_t
    DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_WR_WEIGHT_0_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_WR_WEIGHT_0_RESERVED_0
    DDMA_WR_WEIGHT_0_RESERVED_0.restype = uint32_t
    DDMA_WR_WEIGHT_0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP = _libraries['FIXME_STUB'].DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP
    DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP.restype = uint32_t
    DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU = _libraries['FIXME_STUB'].DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU
    DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU.restype = uint32_t
    DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_ID_ERROR_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_ID_ERROR_RESERVED_0
    DDMA_CFG_ID_ERROR_RESERVED_0.restype = uint32_t
    DDMA_CFG_ID_ERROR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_ID_ERROR_WR_RESP_ID = _libraries['FIXME_STUB'].DDMA_CFG_ID_ERROR_WR_RESP_ID
    DDMA_CFG_ID_ERROR_WR_RESP_ID.restype = uint32_t
    DDMA_CFG_ID_ERROR_WR_RESP_ID.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_ID_ERROR_RESERVED_1 = _libraries['FIXME_STUB'].DDMA_CFG_ID_ERROR_RESERVED_1
    DDMA_CFG_ID_ERROR_RESERVED_1.restype = uint32_t
    DDMA_CFG_ID_ERROR_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_ID_ERROR_RD_RESP_ID = _libraries['FIXME_STUB'].DDMA_CFG_ID_ERROR_RD_RESP_ID
    DDMA_CFG_ID_ERROR_RD_RESP_ID.restype = uint32_t
    DDMA_CFG_ID_ERROR_RD_RESP_ID.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_1_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_1_RESERVED_0
    DDMA_RD_WEIGHT_1_RESERVED_0.restype = uint32_t
    DDMA_RD_WEIGHT_1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_RD_WEIGHT_1_RD_WEIGHT_PC = _libraries['FIXME_STUB'].DDMA_RD_WEIGHT_1_RD_WEIGHT_PC
    DDMA_RD_WEIGHT_1_RD_WEIGHT_PC.restype = uint32_t
    DDMA_RD_WEIGHT_1_RD_WEIGHT_PC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_FIFO_CLR_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_FIFO_CLR_RESERVED_0
    DDMA_CFG_DMA_FIFO_CLR_RESERVED_0.restype = uint32_t
    DDMA_CFG_DMA_FIFO_CLR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR = _libraries['FIXME_STUB'].DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR
    DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR.restype = uint32_t
    DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_RESERVED_0
    DDMA_CFG_DMA_ARB_RESERVED_0.restype = uint32_t
    DDMA_CFG_DMA_ARB_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL
    DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL.restype = uint32_t
    DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL
    DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL.restype = uint32_t
    DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_RESERVED_1 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_RESERVED_1
    DDMA_CFG_DMA_ARB_RESERVED_1.restype = uint32_t
    DDMA_CFG_DMA_ARB_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_WR_FIX_ARB = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_WR_FIX_ARB
    DDMA_CFG_DMA_ARB_WR_FIX_ARB.restype = uint32_t
    DDMA_CFG_DMA_ARB_WR_FIX_ARB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_RESERVED_2 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_RESERVED_2
    DDMA_CFG_DMA_ARB_RESERVED_2.restype = uint32_t
    DDMA_CFG_DMA_ARB_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_ARB_RD_FIX_ARB = _libraries['FIXME_STUB'].DDMA_CFG_DMA_ARB_RD_FIX_ARB
    DDMA_CFG_DMA_ARB_RD_FIX_ARB.restype = uint32_t
    DDMA_CFG_DMA_ARB_RD_FIX_ARB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RESERVED_0
    DDMA_CFG_DMA_RD_QOS_RESERVED_0.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RD_PC_QOS = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RD_PC_QOS
    DDMA_CFG_DMA_RD_QOS_RD_PC_QOS.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RD_PC_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS
    DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS
    DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS
    DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS
    DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS.restype = uint32_t
    DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RESERVED_0
    DDMA_CFG_DMA_RD_CFG_RESERVED_0.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RD_ARLOCK = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RD_ARLOCK
    DDMA_CFG_DMA_RD_CFG_RD_ARLOCK.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RD_ARLOCK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RD_ARCACHE = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RD_ARCACHE
    DDMA_CFG_DMA_RD_CFG_RD_ARCACHE.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RD_ARCACHE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RD_ARPROT = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RD_ARPROT
    DDMA_CFG_DMA_RD_CFG_RD_ARPROT.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RD_ARPROT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RD_ARBURST = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RD_ARBURST
    DDMA_CFG_DMA_RD_CFG_RD_ARBURST.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RD_ARBURST.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_RD_CFG_RD_ARSIZE = _libraries['FIXME_STUB'].DDMA_CFG_DMA_RD_CFG_RD_ARSIZE
    DDMA_CFG_DMA_RD_CFG_RD_ARSIZE.restype = uint32_t
    DDMA_CFG_DMA_RD_CFG_RD_ARSIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_RESERVED_0
    DDMA_CFG_DMA_WR_CFG_RESERVED_0.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_WR_AWLOCK = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_WR_AWLOCK
    DDMA_CFG_DMA_WR_CFG_WR_AWLOCK.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_WR_AWLOCK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_WR_AWCACHE = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_WR_AWCACHE
    DDMA_CFG_DMA_WR_CFG_WR_AWCACHE.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_WR_AWCACHE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_WR_AWPROT = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_WR_AWPROT
    DDMA_CFG_DMA_WR_CFG_WR_AWPROT.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_WR_AWPROT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_WR_AWBURST = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_WR_AWBURST
    DDMA_CFG_DMA_WR_CFG_WR_AWBURST.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_WR_AWBURST.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WR_CFG_WR_AWSIZE = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WR_CFG_WR_AWSIZE
    DDMA_CFG_DMA_WR_CFG_WR_AWSIZE.restype = uint32_t
    DDMA_CFG_DMA_WR_CFG_WR_AWSIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_DMA_WSTRB_WR_WSTRB = _libraries['FIXME_STUB'].DDMA_CFG_DMA_WSTRB_WR_WSTRB
    DDMA_CFG_DMA_WSTRB_WR_WSTRB.restype = uint32_t
    DDMA_CFG_DMA_WSTRB_WR_WSTRB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].DDMA_CFG_STATUS_RESERVED_0
    DDMA_CFG_STATUS_RESERVED_0.restype = uint32_t
    DDMA_CFG_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_STATUS_IDEL = _libraries['FIXME_STUB'].DDMA_CFG_STATUS_IDEL
    DDMA_CFG_STATUS_IDEL.restype = uint32_t
    DDMA_CFG_STATUS_IDEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    DDMA_CFG_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].DDMA_CFG_STATUS_RESERVED_1
    DDMA_CFG_STATUS_RESERVED_1.restype = uint32_t
    DDMA_CFG_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_OUTSTANDING_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_OUTSTANDING_RESERVED_0
    SDMA_CFG_OUTSTANDING_RESERVED_0.restype = uint32_t
    SDMA_CFG_OUTSTANDING_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_OUTSTANDING_WR_OS_CNT = _libraries['FIXME_STUB'].SDMA_CFG_OUTSTANDING_WR_OS_CNT
    SDMA_CFG_OUTSTANDING_WR_OS_CNT.restype = uint32_t
    SDMA_CFG_OUTSTANDING_WR_OS_CNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_OUTSTANDING_RD_OS_CNT = _libraries['FIXME_STUB'].SDMA_CFG_OUTSTANDING_RD_OS_CNT
    SDMA_CFG_OUTSTANDING_RD_OS_CNT.restype = uint32_t
    SDMA_CFG_OUTSTANDING_RD_OS_CNT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP
    SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP.restype = uint32_t
    SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU
    SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU.restype = uint32_t
    SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL
    SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL.restype = uint32_t
    SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE
    SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE.restype = uint32_t
    SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_WR_WEIGHT_0_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_WR_WEIGHT_0_RESERVED_0
    SDMA_WR_WEIGHT_0_RESERVED_0.restype = uint32_t
    SDMA_WR_WEIGHT_0_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP = _libraries['FIXME_STUB'].SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP
    SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP.restype = uint32_t
    SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU = _libraries['FIXME_STUB'].SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU
    SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU.restype = uint32_t
    SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_ID_ERROR_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_ID_ERROR_RESERVED_0
    SDMA_CFG_ID_ERROR_RESERVED_0.restype = uint32_t
    SDMA_CFG_ID_ERROR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_ID_ERROR_WR_RESP_ID = _libraries['FIXME_STUB'].SDMA_CFG_ID_ERROR_WR_RESP_ID
    SDMA_CFG_ID_ERROR_WR_RESP_ID.restype = uint32_t
    SDMA_CFG_ID_ERROR_WR_RESP_ID.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_ID_ERROR_RESERVED_1 = _libraries['FIXME_STUB'].SDMA_CFG_ID_ERROR_RESERVED_1
    SDMA_CFG_ID_ERROR_RESERVED_1.restype = uint32_t
    SDMA_CFG_ID_ERROR_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_ID_ERROR_RD_RESP_ID = _libraries['FIXME_STUB'].SDMA_CFG_ID_ERROR_RD_RESP_ID
    SDMA_CFG_ID_ERROR_RD_RESP_ID.restype = uint32_t
    SDMA_CFG_ID_ERROR_RD_RESP_ID.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_1_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_1_RESERVED_0
    SDMA_RD_WEIGHT_1_RESERVED_0.restype = uint32_t
    SDMA_RD_WEIGHT_1_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_RD_WEIGHT_1_RD_WEIGHT_PC = _libraries['FIXME_STUB'].SDMA_RD_WEIGHT_1_RD_WEIGHT_PC
    SDMA_RD_WEIGHT_1_RD_WEIGHT_PC.restype = uint32_t
    SDMA_RD_WEIGHT_1_RD_WEIGHT_PC.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_FIFO_CLR_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_FIFO_CLR_RESERVED_0
    SDMA_CFG_DMA_FIFO_CLR_RESERVED_0.restype = uint32_t
    SDMA_CFG_DMA_FIFO_CLR_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR = _libraries['FIXME_STUB'].SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR
    SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR.restype = uint32_t
    SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_RESERVED_0
    SDMA_CFG_DMA_ARB_RESERVED_0.restype = uint32_t
    SDMA_CFG_DMA_ARB_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL
    SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL.restype = uint32_t
    SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL
    SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL.restype = uint32_t
    SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_RESERVED_1 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_RESERVED_1
    SDMA_CFG_DMA_ARB_RESERVED_1.restype = uint32_t
    SDMA_CFG_DMA_ARB_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_WR_FIX_ARB = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_WR_FIX_ARB
    SDMA_CFG_DMA_ARB_WR_FIX_ARB.restype = uint32_t
    SDMA_CFG_DMA_ARB_WR_FIX_ARB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_RESERVED_2 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_RESERVED_2
    SDMA_CFG_DMA_ARB_RESERVED_2.restype = uint32_t
    SDMA_CFG_DMA_ARB_RESERVED_2.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_ARB_RD_FIX_ARB = _libraries['FIXME_STUB'].SDMA_CFG_DMA_ARB_RD_FIX_ARB
    SDMA_CFG_DMA_ARB_RD_FIX_ARB.restype = uint32_t
    SDMA_CFG_DMA_ARB_RD_FIX_ARB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RESERVED_0
    SDMA_CFG_DMA_RD_QOS_RESERVED_0.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RD_PC_QOS = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RD_PC_QOS
    SDMA_CFG_DMA_RD_QOS_RD_PC_QOS.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RD_PC_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS
    SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS
    SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS
    SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS
    SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS.restype = uint32_t
    SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RESERVED_0
    SDMA_CFG_DMA_RD_CFG_RESERVED_0.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RD_ARLOCK = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RD_ARLOCK
    SDMA_CFG_DMA_RD_CFG_RD_ARLOCK.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RD_ARLOCK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RD_ARCACHE = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RD_ARCACHE
    SDMA_CFG_DMA_RD_CFG_RD_ARCACHE.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RD_ARCACHE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RD_ARPROT = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RD_ARPROT
    SDMA_CFG_DMA_RD_CFG_RD_ARPROT.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RD_ARPROT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RD_ARBURST = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RD_ARBURST
    SDMA_CFG_DMA_RD_CFG_RD_ARBURST.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RD_ARBURST.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_RD_CFG_RD_ARSIZE = _libraries['FIXME_STUB'].SDMA_CFG_DMA_RD_CFG_RD_ARSIZE
    SDMA_CFG_DMA_RD_CFG_RD_ARSIZE.restype = uint32_t
    SDMA_CFG_DMA_RD_CFG_RD_ARSIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_RESERVED_0
    SDMA_CFG_DMA_WR_CFG_RESERVED_0.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_WR_AWLOCK = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_WR_AWLOCK
    SDMA_CFG_DMA_WR_CFG_WR_AWLOCK.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_WR_AWLOCK.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_WR_AWCACHE = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_WR_AWCACHE
    SDMA_CFG_DMA_WR_CFG_WR_AWCACHE.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_WR_AWCACHE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_WR_AWPROT = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_WR_AWPROT
    SDMA_CFG_DMA_WR_CFG_WR_AWPROT.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_WR_AWPROT.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_WR_AWBURST = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_WR_AWBURST
    SDMA_CFG_DMA_WR_CFG_WR_AWBURST.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_WR_AWBURST.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WR_CFG_WR_AWSIZE = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WR_CFG_WR_AWSIZE
    SDMA_CFG_DMA_WR_CFG_WR_AWSIZE.restype = uint32_t
    SDMA_CFG_DMA_WR_CFG_WR_AWSIZE.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_DMA_WSTRB_WR_WSTRB = _libraries['FIXME_STUB'].SDMA_CFG_DMA_WSTRB_WR_WSTRB
    SDMA_CFG_DMA_WSTRB_WR_WSTRB.restype = uint32_t
    SDMA_CFG_DMA_WSTRB_WR_WSTRB.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_STATUS_RESERVED_0 = _libraries['FIXME_STUB'].SDMA_CFG_STATUS_RESERVED_0
    SDMA_CFG_STATUS_RESERVED_0.restype = uint32_t
    SDMA_CFG_STATUS_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_STATUS_IDEL = _libraries['FIXME_STUB'].SDMA_CFG_STATUS_IDEL
    SDMA_CFG_STATUS_IDEL.restype = uint32_t
    SDMA_CFG_STATUS_IDEL.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    SDMA_CFG_STATUS_RESERVED_1 = _libraries['FIXME_STUB'].SDMA_CFG_STATUS_RESERVED_1
    SDMA_CFG_STATUS_RESERVED_1.restype = uint32_t
    SDMA_CFG_STATUS_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_RESERVED_0 = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_RESERVED_0
    GLOBAL_OPERATION_ENABLE_RESERVED_0.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_RESERVED_0.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN
    GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_PPU_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_PPU_OP_EN
    GLOBAL_OPERATION_ENABLE_PPU_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_PPU_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN
    GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_DPU_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_DPU_OP_EN
    GLOBAL_OPERATION_ENABLE_DPU_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_DPU_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_CORE_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_CORE_OP_EN
    GLOBAL_OPERATION_ENABLE_CORE_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_CORE_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_RESERVED_1 = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_RESERVED_1
    GLOBAL_OPERATION_ENABLE_RESERVED_1.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_RESERVED_1.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    GLOBAL_OPERATION_ENABLE_CNA_OP_EN = _libraries['FIXME_STUB'].GLOBAL_OPERATION_ENABLE_CNA_OP_EN
    GLOBAL_OPERATION_ENABLE_CNA_OP_EN.restype = uint32_t
    GLOBAL_OPERATION_ENABLE_CNA_OP_EN.argtypes = [uint32_t]
except AttributeError:
    pass
try:
    rkt_get_target = _libraries['FIXME_STUB'].rkt_get_target
    rkt_get_target.restype = uint32_t
    rkt_get_target.argtypes = [uint32_t]
except AttributeError:
    pass
__all__ = \
    ['CNA', 'CNA_CBUF_CON0_DATA_BANK', 'CNA_CBUF_CON0_DATA_REUSE',
    'CNA_CBUF_CON0_FC_DATA_BANK', 'CNA_CBUF_CON0_RESERVED_0',
    'CNA_CBUF_CON0_RESERVED_1', 'CNA_CBUF_CON0_WEIGHT_BANK',
    'CNA_CBUF_CON0_WEIGHT_REUSE', 'CNA_CBUF_CON1_DATA_ENTRIES',
    'CNA_CBUF_CON1_RESERVED_0',
    'CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE',
    'CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE',
    'CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE',
    'CNA_CLK_GATE_CSC_DISABLE_CLKGATE', 'CNA_CLK_GATE_RESERVED_0',
    'CNA_CLK_GATE_RESERVED_1', 'CNA_CONV_CON1_ARGB_IN',
    'CNA_CONV_CON1_CONV_MODE', 'CNA_CONV_CON1_DECONV',
    'CNA_CONV_CON1_GROUP_LINE_OFF', 'CNA_CONV_CON1_IN_PRECISION',
    'CNA_CONV_CON1_NONALIGN_DMA', 'CNA_CONV_CON1_PROC_PRECISION',
    'CNA_CONV_CON1_RESERVED_0', 'CNA_CONV_CON1_RESERVED_1',
    'CNA_CONV_CON1_RESERVED_2', 'CNA_CONV_CON2_CMD_FIFO_SRST',
    'CNA_CONV_CON2_CSC_DO_EN', 'CNA_CONV_CON2_CSC_WO_EN',
    'CNA_CONV_CON2_FEATURE_GRAINS', 'CNA_CONV_CON2_KERNEL_GROUP',
    'CNA_CONV_CON2_RESERVED_0', 'CNA_CONV_CON2_RESERVED_1',
    'CNA_CONV_CON2_RESERVED_2', 'CNA_CONV_CON3_ATROUS_X_DILATION',
    'CNA_CONV_CON3_ATROUS_Y_DILATION', 'CNA_CONV_CON3_CONV_X_STRIDE',
    'CNA_CONV_CON3_CONV_Y_STRIDE', 'CNA_CONV_CON3_DECONV_X_STRIDE',
    'CNA_CONV_CON3_DECONV_Y_STRIDE', 'CNA_CONV_CON3_NN_MODE',
    'CNA_CONV_CON3_RESERVED_0', 'CNA_CONV_CON3_RESERVED_1',
    'CNA_CONV_CON3_RESERVED_2', 'CNA_CONV_CON3_RESERVED_3',
    'CNA_CVT_CON0_CVT_BYPASS', 'CNA_CVT_CON0_CVT_TRUNCATE_0',
    'CNA_CVT_CON0_CVT_TRUNCATE_1', 'CNA_CVT_CON0_CVT_TRUNCATE_2',
    'CNA_CVT_CON0_CVT_TRUNCATE_3', 'CNA_CVT_CON0_CVT_TYPE',
    'CNA_CVT_CON0_DATA_SIGN', 'CNA_CVT_CON0_RESERVED_0',
    'CNA_CVT_CON0_ROUND_TYPE', 'CNA_CVT_CON1_CVT_OFFSET0',
    'CNA_CVT_CON1_CVT_SCALE0', 'CNA_CVT_CON2_CVT_OFFSET1',
    'CNA_CVT_CON2_CVT_SCALE1', 'CNA_CVT_CON3_CVT_OFFSET2',
    'CNA_CVT_CON3_CVT_SCALE2', 'CNA_CVT_CON4_CVT_OFFSET3',
    'CNA_CVT_CON4_CVT_SCALE3', 'CNA_CVT_CON5_PER_CHANNEL_CVT_EN',
    'CNA_DATA_SIZE0_DATAIN_HEIGHT', 'CNA_DATA_SIZE0_DATAIN_WIDTH',
    'CNA_DATA_SIZE0_RESERVED_0', 'CNA_DATA_SIZE0_RESERVED_1',
    'CNA_DATA_SIZE1_DATAIN_CHANNEL',
    'CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL', 'CNA_DATA_SIZE1_RESERVED_0',
    'CNA_DATA_SIZE2_DATAOUT_WIDTH', 'CNA_DATA_SIZE2_RESERVED_0',
    'CNA_DATA_SIZE3_DATAOUT_ATOMICS', 'CNA_DATA_SIZE3_RESERVED_0',
    'CNA_DATA_SIZE3_SURF_MODE', 'CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0',
    'CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0',
    'CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10',
    'CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11',
    'CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12',
    'CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13',
    'CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14',
    'CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15',
    'CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1',
    'CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2',
    'CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3',
    'CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4',
    'CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5',
    'CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6',
    'CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7',
    'CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8',
    'CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9',
    'CNA_DCOMP_CTRL_DECOMP_CONTROL', 'CNA_DCOMP_CTRL_RESERVED_0',
    'CNA_DCOMP_CTRL_WT_DEC_BYPASS', 'CNA_DCOMP_REGNUM_DCOMP_REGNUM',
    'CNA_DMA_CON0_DATA_BURST_LEN', 'CNA_DMA_CON0_OV4K_BYPASS',
    'CNA_DMA_CON0_RESERVED_0', 'CNA_DMA_CON0_RESERVED_1',
    'CNA_DMA_CON0_WEIGHT_BURST_LEN', 'CNA_DMA_CON1_LINE_STRIDE',
    'CNA_DMA_CON1_RESERVED_0', 'CNA_DMA_CON2_RESERVED_0',
    'CNA_DMA_CON2_SURF_STRIDE', 'CNA_FC_CON0_FC_SKIP_DATA',
    'CNA_FC_CON0_FC_SKIP_EN', 'CNA_FC_CON0_RESERVED_0',
    'CNA_FC_CON1_DATA_OFFSET', 'CNA_FC_CON1_RESERVED_0',
    'CNA_FC_CON2_RESERVED_0', 'CNA_FC_CON2_WEIGHT_OFFSET',
    'CNA_FC_DATA_SIZE0_DMA_HEIGHT', 'CNA_FC_DATA_SIZE0_DMA_WIDTH',
    'CNA_FC_DATA_SIZE0_RESERVED_0', 'CNA_FC_DATA_SIZE0_RESERVED_1',
    'CNA_FC_DATA_SIZE1_DMA_CHANNEL', 'CNA_FC_DATA_SIZE1_RESERVED_0',
    'CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR',
    'CNA_OPERATION_ENABLE_OP_EN', 'CNA_OPERATION_ENABLE_RESERVED_0',
    'CNA_PAD_CON0_PAD_LEFT', 'CNA_PAD_CON0_PAD_TOP',
    'CNA_PAD_CON0_RESERVED_0', 'CNA_PAD_CON1_PAD_VALUE',
    'CNA_S_POINTER_EXECUTER', 'CNA_S_POINTER_EXECUTER_PP_CLEAR',
    'CNA_S_POINTER_EXECUTER_PP_EN', 'CNA_S_POINTER_POINTER',
    'CNA_S_POINTER_POINTER_PP_CLEAR', 'CNA_S_POINTER_POINTER_PP_EN',
    'CNA_S_POINTER_POINTER_PP_MODE', 'CNA_S_POINTER_RESERVED_0',
    'CNA_S_POINTER_RESERVED_1', 'CNA_S_STATUS_RESERVED_0',
    'CNA_S_STATUS_RESERVED_1', 'CNA_S_STATUS_STATUS_0',
    'CNA_S_STATUS_STATUS_1', 'CNA_WEIGHT_SIZE0_WEIGHT_BYTES',
    'CNA_WEIGHT_SIZE1_RESERVED_0',
    'CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL',
    'CNA_WEIGHT_SIZE2_RESERVED_0', 'CNA_WEIGHT_SIZE2_RESERVED_1',
    'CNA_WEIGHT_SIZE2_RESERVED_2', 'CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT',
    'CNA_WEIGHT_SIZE2_WEIGHT_KERNELS',
    'CNA_WEIGHT_SIZE2_WEIGHT_WIDTH', 'CORE',
    'CORE_CLIP_TRUNCATE_CLIP_TRUNCATE',
    'CORE_CLIP_TRUNCATE_RESERVED_0', 'CORE_CLIP_TRUNCATE_RESERVED_1',
    'CORE_CLIP_TRUNCATE_ROUND_TYPE',
    'CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT',
    'CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH',
    'CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL',
    'CORE_DATAOUT_SIZE_1_RESERVED_0', 'CORE_MAC_GATING_RESERVED_0',
    'CORE_MAC_GATING_SLCG_OP_EN', 'CORE_MISC_CFG_DW_EN',
    'CORE_MISC_CFG_PROC_PRECISION', 'CORE_MISC_CFG_QD_EN',
    'CORE_MISC_CFG_RESERVED_0', 'CORE_MISC_CFG_RESERVED_1',
    'CORE_MISC_CFG_RESERVED_2', 'CORE_MISC_CFG_SOFT_GATING',
    'CORE_OPERATION_ENABLE_OP_EN', 'CORE_OPERATION_ENABLE_RESERVED_0',
    'CORE_S_POINTER_EXECUTER', 'CORE_S_POINTER_EXECUTER_PP_CLEAR',
    'CORE_S_POINTER_EXECUTER_PP_EN', 'CORE_S_POINTER_POINTER',
    'CORE_S_POINTER_POINTER_PP_CLEAR', 'CORE_S_POINTER_POINTER_PP_EN',
    'CORE_S_POINTER_POINTER_PP_MODE', 'CORE_S_POINTER_RESERVED_0',
    'CORE_S_POINTER_RESERVED_1', 'CORE_S_STATUS_RESERVED_0',
    'CORE_S_STATUS_RESERVED_1', 'CORE_S_STATUS_STATUS_0',
    'CORE_S_STATUS_STATUS_1', 'DDMA',
    'DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL', 'DDMA_CFG_DMA_ARB_RD_FIX_ARB',
    'DDMA_CFG_DMA_ARB_RESERVED_0', 'DDMA_CFG_DMA_ARB_RESERVED_1',
    'DDMA_CFG_DMA_ARB_RESERVED_2', 'DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL',
    'DDMA_CFG_DMA_ARB_WR_FIX_ARB',
    'DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR',
    'DDMA_CFG_DMA_FIFO_CLR_RESERVED_0',
    'DDMA_CFG_DMA_RD_CFG_RD_ARBURST',
    'DDMA_CFG_DMA_RD_CFG_RD_ARCACHE', 'DDMA_CFG_DMA_RD_CFG_RD_ARLOCK',
    'DDMA_CFG_DMA_RD_CFG_RD_ARPROT', 'DDMA_CFG_DMA_RD_CFG_RD_ARSIZE',
    'DDMA_CFG_DMA_RD_CFG_RESERVED_0',
    'DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS',
    'DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS',
    'DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS',
    'DDMA_CFG_DMA_RD_QOS_RD_PC_QOS', 'DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS',
    'DDMA_CFG_DMA_RD_QOS_RESERVED_0',
    'DDMA_CFG_DMA_WR_CFG_RESERVED_0',
    'DDMA_CFG_DMA_WR_CFG_WR_AWBURST',
    'DDMA_CFG_DMA_WR_CFG_WR_AWCACHE', 'DDMA_CFG_DMA_WR_CFG_WR_AWLOCK',
    'DDMA_CFG_DMA_WR_CFG_WR_AWPROT', 'DDMA_CFG_DMA_WR_CFG_WR_AWSIZE',
    'DDMA_CFG_DMA_WSTRB_WR_WSTRB', 'DDMA_CFG_ID_ERROR_RD_RESP_ID',
    'DDMA_CFG_ID_ERROR_RESERVED_0', 'DDMA_CFG_ID_ERROR_RESERVED_1',
    'DDMA_CFG_ID_ERROR_WR_RESP_ID', 'DDMA_CFG_OUTSTANDING_RD_OS_CNT',
    'DDMA_CFG_OUTSTANDING_RESERVED_0',
    'DDMA_CFG_OUTSTANDING_WR_OS_CNT', 'DDMA_CFG_STATUS_IDEL',
    'DDMA_CFG_STATUS_RESERVED_0', 'DDMA_CFG_STATUS_RESERVED_1',
    'DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU',
    'DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE',
    'DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL',
    'DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP', 'DDMA_RD_WEIGHT_1_RD_WEIGHT_PC',
    'DDMA_RD_WEIGHT_1_RESERVED_0', 'DDMA_WR_WEIGHT_0_RESERVED_0',
    'DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU',
    'DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP', 'DPU',
    'DPU_BN_ALU_CFG_BN_ALU_OPERAND', 'DPU_BN_CFG_BN_ALU_ALGO',
    'DPU_BN_CFG_BN_ALU_BYPASS', 'DPU_BN_CFG_BN_ALU_SRC',
    'DPU_BN_CFG_BN_BYPASS', 'DPU_BN_CFG_BN_MUL_BYPASS',
    'DPU_BN_CFG_BN_MUL_PRELU', 'DPU_BN_CFG_BN_RELUX_EN',
    'DPU_BN_CFG_BN_RELU_BYPASS', 'DPU_BN_CFG_RESERVED_0',
    'DPU_BN_CFG_RESERVED_1', 'DPU_BN_CFG_RESERVED_2',
    'DPU_BN_MUL_CFG_BN_MUL_OPERAND',
    'DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE', 'DPU_BN_MUL_CFG_BN_MUL_SRC',
    'DPU_BN_MUL_CFG_BN_TRUNCATE_SRC', 'DPU_BN_MUL_CFG_RESERVED_0',
    'DPU_BN_MUL_CFG_RESERVED_1',
    'DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT',
    'DPU_BS_ALU_CFG_BS_ALU_OPERAND', 'DPU_BS_CFG_BS_ALU_ALGO',
    'DPU_BS_CFG_BS_ALU_BYPASS', 'DPU_BS_CFG_BS_ALU_SRC',
    'DPU_BS_CFG_BS_BYPASS', 'DPU_BS_CFG_BS_MUL_BYPASS',
    'DPU_BS_CFG_BS_MUL_PRELU', 'DPU_BS_CFG_BS_RELUX_EN',
    'DPU_BS_CFG_BS_RELU_BYPASS', 'DPU_BS_CFG_RESERVED_0',
    'DPU_BS_CFG_RESERVED_1', 'DPU_BS_CFG_RESERVED_2',
    'DPU_BS_MUL_CFG_BS_MUL_OPERAND',
    'DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE', 'DPU_BS_MUL_CFG_BS_MUL_SRC',
    'DPU_BS_MUL_CFG_BS_TRUNCATE_SRC', 'DPU_BS_MUL_CFG_RESERVED_0',
    'DPU_BS_MUL_CFG_RESERVED_1', 'DPU_BS_OW_CFG_OD_BYPASS',
    'DPU_BS_OW_CFG_OW_SRC', 'DPU_BS_OW_CFG_RESERVED_0',
    'DPU_BS_OW_CFG_RGP_CNTER', 'DPU_BS_OW_CFG_SIZE_E_0',
    'DPU_BS_OW_CFG_SIZE_E_1', 'DPU_BS_OW_CFG_SIZE_E_2',
    'DPU_BS_OW_CFG_TP_ORG_EN', 'DPU_BS_OW_OP_OW_OP',
    'DPU_BS_OW_OP_RESERVED_0',
    'DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT',
    'DPU_DATA_CUBE_CHANNEL_CHANNEL',
    'DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL',
    'DPU_DATA_CUBE_CHANNEL_RESERVED_0',
    'DPU_DATA_CUBE_CHANNEL_RESERVED_1', 'DPU_DATA_CUBE_HEIGHT_HEIGHT',
    'DPU_DATA_CUBE_HEIGHT_MINMAX_CTL',
    'DPU_DATA_CUBE_HEIGHT_RESERVED_0',
    'DPU_DATA_CUBE_HEIGHT_RESERVED_1',
    'DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0',
    'DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1',
    'DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_0',
    'DPU_DATA_CUBE_NOTCH_ADDR_RESERVED_1',
    'DPU_DATA_CUBE_WIDTH_RESERVED_0', 'DPU_DATA_CUBE_WIDTH_WIDTH',
    'DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG',
    'DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG',
    'DPU_DATA_FORMAT_EW_TRUNCATE_NEG', 'DPU_DATA_FORMAT_IN_PRECISION',
    'DPU_DATA_FORMAT_MC_SURF_OUT', 'DPU_DATA_FORMAT_OUT_PRECISION',
    'DPU_DATA_FORMAT_PROC_PRECISION',
    'DPU_DST_BASE_ADDR_DST_BASE_ADDR',
    'DPU_DST_SURF_STRIDE_DST_SURF_STRIDE',
    'DPU_DST_SURF_STRIDE_RESERVED_0', 'DPU_EW_CFG_EDATA_SIZE',
    'DPU_EW_CFG_EW_ALU_ALGO', 'DPU_EW_CFG_EW_BINARY_EN',
    'DPU_EW_CFG_EW_BYPASS', 'DPU_EW_CFG_EW_CVT_ROUND',
    'DPU_EW_CFG_EW_CVT_TYPE', 'DPU_EW_CFG_EW_DATA_MODE',
    'DPU_EW_CFG_EW_EQUAL_EN', 'DPU_EW_CFG_EW_LUT_BYPASS',
    'DPU_EW_CFG_EW_MUL_PRELU', 'DPU_EW_CFG_EW_OP_BYPASS',
    'DPU_EW_CFG_EW_OP_CVT_BYPASS', 'DPU_EW_CFG_EW_OP_SRC',
    'DPU_EW_CFG_EW_OP_TYPE', 'DPU_EW_CFG_EW_RELUX_EN',
    'DPU_EW_CFG_EW_RELU_BYPASS', 'DPU_EW_CFG_RESERVED_0',
    'DPU_EW_CFG_RESERVED_1', 'DPU_EW_CFG_RESERVED_2',
    'DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET',
    'DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE',
    'DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT',
    'DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE',
    'DPU_EW_OP_VALUE_0_EW_OPERAND_0',
    'DPU_EW_OP_VALUE_1_EW_OPERAND_1',
    'DPU_EW_OP_VALUE_2_EW_OPERAND_2',
    'DPU_EW_OP_VALUE_3_EW_OPERAND_3',
    'DPU_EW_OP_VALUE_4_EW_OPERAND_4',
    'DPU_EW_OP_VALUE_5_EW_OPERAND_5',
    'DPU_EW_OP_VALUE_6_EW_OPERAND_6',
    'DPU_EW_OP_VALUE_7_EW_OPERAND_7',
    'DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT',
    'DPU_FEATURE_MODE_CFG_BURST_LEN', 'DPU_FEATURE_MODE_CFG_COMB_USE',
    'DPU_FEATURE_MODE_CFG_CONV_MODE',
    'DPU_FEATURE_MODE_CFG_FLYING_MODE',
    'DPU_FEATURE_MODE_CFG_NONALIGN',
    'DPU_FEATURE_MODE_CFG_OUTPUT_MODE',
    'DPU_FEATURE_MODE_CFG_RGP_TYPE', 'DPU_FEATURE_MODE_CFG_SURF_LEN',
    'DPU_FEATURE_MODE_CFG_TP_EN',
    'DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE',
    'DPU_LUT_ACCESS_CFG_LUT_ADDR', 'DPU_LUT_ACCESS_CFG_LUT_TABLE_ID',
    'DPU_LUT_ACCESS_CFG_RESERVED_0', 'DPU_LUT_ACCESS_CFG_RESERVED_1',
    'DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA',
    'DPU_LUT_ACCESS_DATA_RESERVED_0', 'DPU_LUT_CFG_LUT_CAL_SEL',
    'DPU_LUT_CFG_LUT_EXPAND_EN', 'DPU_LUT_CFG_LUT_HYBRID_PRIORITY',
    'DPU_LUT_CFG_LUT_LO_LE_MUX', 'DPU_LUT_CFG_LUT_OFLOW_PRIORITY',
    'DPU_LUT_CFG_LUT_ROAD_SEL', 'DPU_LUT_CFG_LUT_UFLOW_PRIORITY',
    'DPU_LUT_CFG_RESERVED_0', 'DPU_LUT_INFO_LUT_LE_INDEX_SELECT',
    'DPU_LUT_INFO_LUT_LO_INDEX_SELECT', 'DPU_LUT_INFO_RESERVED_0',
    'DPU_LUT_INFO_RESERVED_1', 'DPU_LUT_LE_END_LUT_LE_END',
    'DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE',
    'DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE',
    'DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT',
    'DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT',
    'DPU_LUT_LE_SLOPE_SHIFT_RESERVED_0',
    'DPU_LUT_LE_START_LUT_LE_START', 'DPU_LUT_LO_END_LUT_LO_END',
    'DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE',
    'DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE',
    'DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT',
    'DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT',
    'DPU_LUT_LO_SLOPE_SHIFT_RESERVED_0',
    'DPU_LUT_LO_START_LUT_LO_START', 'DPU_OFFSET_PEND_OFFSET_PEND',
    'DPU_OFFSET_PEND_RESERVED_0', 'DPU_OPERATION_ENABLE_OP_EN',
    'DPU_OPERATION_ENABLE_RESERVED_0',
    'DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET',
    'DPU_OUT_CVT_SCALE_FP32TOFP16_EN',
    'DPU_OUT_CVT_SCALE_OUT_CVT_SCALE', 'DPU_OUT_CVT_SCALE_RESERVED_0',
    'DPU_OUT_CVT_SHIFT_CVT_ROUND', 'DPU_OUT_CVT_SHIFT_CVT_TYPE',
    'DPU_OUT_CVT_SHIFT_MINUS_EXP', 'DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT',
    'DPU_OUT_CVT_SHIFT_RESERVED_0', 'DPU_RDMA',
    'DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR',
    'DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE',
    'DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_BRDMA_CFG_RESERVED_1',
    'DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR',
    'DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL',
    'DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_RESERVED_0',
    'DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR',
    'DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT',
    'DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_0',
    'DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_RESERVED_1',
    'DPU_RDMA_RDMA_DATA_CUBE_WIDTH_RESERVED_0',
    'DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH',
    'DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE',
    'DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE',
    'DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE',
    'DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN',
    'DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE',
    'DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS',
    'DPU_RDMA_RDMA_ERDMA_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR',
    'DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH',
    'DPU_RDMA_RDMA_EW_SURF_NOTCH_RESERVED_0',
    'DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE',
    'DPU_RDMA_RDMA_EW_SURF_STRIDE_RESERVED_0',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION',
    'DPU_RDMA_RDMA_FEATURE_MODE_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE',
    'DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_NRDMA_CFG_RESERVED_1',
    'DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN',
    'DPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0',
    'DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT', 'DPU_RDMA_RDMA_PAD_CFG_PAD_TOP',
    'DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE',
    'DPU_RDMA_RDMA_PAD_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_PAD_CFG_RESERVED_1',
    'DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_RESERVED_0',
    'DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN',
    'DPU_RDMA_RDMA_SURF_NOTCH_RESERVED_0',
    'DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR',
    'DPU_RDMA_RDMA_S_POINTER_EXECUTER',
    'DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR',
    'DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN',
    'DPU_RDMA_RDMA_S_POINTER_POINTER',
    'DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR',
    'DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN',
    'DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE',
    'DPU_RDMA_RDMA_S_POINTER_RESERVED_0',
    'DPU_RDMA_RDMA_S_POINTER_RESERVED_1',
    'DPU_RDMA_RDMA_S_STATUS_RESERVED_0',
    'DPU_RDMA_RDMA_S_STATUS_RESERVED_1',
    'DPU_RDMA_RDMA_S_STATUS_STATUS_0',
    'DPU_RDMA_RDMA_S_STATUS_STATUS_1',
    'DPU_RDMA_RDMA_WEIGHT_B_WEIGHT', 'DPU_RDMA_RDMA_WEIGHT_E_WEIGHT',
    'DPU_RDMA_RDMA_WEIGHT_M_WEIGHT', 'DPU_RDMA_RDMA_WEIGHT_N_WEIGHT',
    'DPU_SURFACE_ADD_RESERVED_0', 'DPU_SURFACE_ADD_SURF_ADD',
    'DPU_S_POINTER_EXECUTER', 'DPU_S_POINTER_EXECUTER_PP_CLEAR',
    'DPU_S_POINTER_EXECUTER_PP_EN', 'DPU_S_POINTER_POINTER',
    'DPU_S_POINTER_POINTER_PP_CLEAR', 'DPU_S_POINTER_POINTER_PP_EN',
    'DPU_S_POINTER_POINTER_PP_MODE', 'DPU_S_POINTER_RESERVED_0',
    'DPU_S_POINTER_RESERVED_1', 'DPU_S_STATUS_RESERVED_0',
    'DPU_S_STATUS_RESERVED_1', 'DPU_S_STATUS_STATUS_0',
    'DPU_S_STATUS_STATUS_1', 'DPU_WDMA_SIZE_0_CHANNEL_WDMA',
    'DPU_WDMA_SIZE_0_RESERVED_0', 'DPU_WDMA_SIZE_0_RESERVED_1',
    'DPU_WDMA_SIZE_0_SIZE_C_WDMA', 'DPU_WDMA_SIZE_0_TP_PRECISION',
    'DPU_WDMA_SIZE_1_HEIGHT_WDMA', 'DPU_WDMA_SIZE_1_RESERVED_0',
    'DPU_WDMA_SIZE_1_RESERVED_1', 'DPU_WDMA_SIZE_1_WIDTH_WDMA',
    'GLOBAL', 'GLOBAL_OPERATION_ENABLE_CNA_OP_EN',
    'GLOBAL_OPERATION_ENABLE_CORE_OP_EN',
    'GLOBAL_OPERATION_ENABLE_DPU_OP_EN',
    'GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN',
    'GLOBAL_OPERATION_ENABLE_PPU_OP_EN',
    'GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN',
    'GLOBAL_OPERATION_ENABLE_RESERVED_0',
    'GLOBAL_OPERATION_ENABLE_RESERVED_1', 'PC',
    'PC_BASE_ADDRESS_PC_SEL', 'PC_BASE_ADDRESS_PC_SOURCE_ADDR',
    'PC_BASE_ADDRESS_RESERVED_0', 'PC_INTERRUPT_CLEAR_RESERVED_0',
    'PC_INTERRUPT_MASK_RESERVED_0',
    'PC_INTERRUPT_RAW_STATUS_RESERVED_0',
    'PC_INTERRUPT_STATUS_RESERVED_0', 'PC_OPERATION_ENABLE_OP_EN',
    'PC_OPERATION_ENABLE_RESERVED_0',
    'PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT',
    'PC_REGISTER_AMOUNTS_RESERVED_0', 'PC_TASK_CON_RESERVED_0',
    'PC_TASK_CON_TASK_COUNT_CLEAR', 'PC_TASK_CON_TASK_NUMBER',
    'PC_TASK_CON_TASK_PP_EN', 'PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR',
    'PC_TASK_DMA_BASE_ADDR_RESERVED_0', 'PC_TASK_STATUS_RESERVED_0',
    'PC_TASK_STATUS_TASK_STATUS', 'PC_VERSION_NUM_VERSION_NUM',
    'PC_VERSION_VERSION', 'PPU',
    'PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL',
    'PPU_DATA_CUBE_IN_CHANNEL_RESERVED_0',
    'PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT',
    'PPU_DATA_CUBE_IN_HEIGHT_RESERVED_0',
    'PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH',
    'PPU_DATA_CUBE_IN_WIDTH_RESERVED_0',
    'PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL',
    'PPU_DATA_CUBE_OUT_CHANNEL_RESERVED_0',
    'PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT',
    'PPU_DATA_CUBE_OUT_HEIGHT_RESERVED_0',
    'PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH',
    'PPU_DATA_CUBE_OUT_WIDTH_RESERVED_0', 'PPU_DATA_FORMAT_DPU_FLYIN',
    'PPU_DATA_FORMAT_INDEX_ADD', 'PPU_DATA_FORMAT_PROC_PRECISION',
    'PPU_DST_BASE_ADDR_DST_BASE_ADDR', 'PPU_DST_BASE_ADDR_RESERVED_0',
    'PPU_DST_SURF_STRIDE_DST_SURF_STRIDE',
    'PPU_DST_SURF_STRIDE_RESERVED_0', 'PPU_MISC_CTRL_BURST_LEN',
    'PPU_MISC_CTRL_MC_SURF_OUT', 'PPU_MISC_CTRL_NONALIGN',
    'PPU_MISC_CTRL_RESERVED_0', 'PPU_MISC_CTRL_RESERVED_1',
    'PPU_MISC_CTRL_SURF_LEN', 'PPU_OPERATION_ENABLE_OP_EN',
    'PPU_OPERATION_ENABLE_RESERVED_0',
    'PPU_OPERATION_MODE_CFG_FLYING_MODE',
    'PPU_OPERATION_MODE_CFG_INDEX_EN',
    'PPU_OPERATION_MODE_CFG_NOTCH_ADDR',
    'PPU_OPERATION_MODE_CFG_POOLING_METHOD',
    'PPU_OPERATION_MODE_CFG_RESERVED_0',
    'PPU_OPERATION_MODE_CFG_RESERVED_1',
    'PPU_OPERATION_MODE_CFG_RESERVED_2',
    'PPU_OPERATION_MODE_CFG_RESERVED_3',
    'PPU_OPERATION_MODE_CFG_USE_CNT',
    'PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0',
    'PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1',
    'PPU_PADDING_VALUE_2_CFG_RESERVED_0',
    'PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT',
    'PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT',
    'PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH',
    'PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH',
    'PPU_POOLING_KERNEL_CFG_RESERVED_0',
    'PPU_POOLING_KERNEL_CFG_RESERVED_1',
    'PPU_POOLING_KERNEL_CFG_RESERVED_2',
    'PPU_POOLING_PADDING_CFG_PAD_BOTTOM',
    'PPU_POOLING_PADDING_CFG_PAD_LEFT',
    'PPU_POOLING_PADDING_CFG_PAD_RIGHT',
    'PPU_POOLING_PADDING_CFG_PAD_TOP',
    'PPU_POOLING_PADDING_CFG_RESERVED_0',
    'PPU_POOLING_PADDING_CFG_RESERVED_1',
    'PPU_POOLING_PADDING_CFG_RESERVED_2',
    'PPU_POOLING_PADDING_CFG_RESERVED_3', 'PPU_RDMA',
    'PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL',
    'PPU_RDMA_RDMA_CUBE_IN_CHANNEL_RESERVED_0',
    'PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT',
    'PPU_RDMA_RDMA_CUBE_IN_HEIGHT_RESERVED_0',
    'PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH',
    'PPU_RDMA_RDMA_CUBE_IN_WIDTH_RESERVED_0',
    'PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION',
    'PPU_RDMA_RDMA_DATA_FORMAT_RESERVED_0',
    'PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN',
    'PPU_RDMA_RDMA_OPERATION_ENABLE_RESERVED_0',
    'PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR',
    'PPU_RDMA_RDMA_SRC_LINE_STRIDE_RESERVED_0',
    'PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE',
    'PPU_RDMA_RDMA_SRC_SURF_STRIDE_RESERVED_0',
    'PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE',
    'PPU_RDMA_RDMA_S_POINTER_EXECUTER',
    'PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR',
    'PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN',
    'PPU_RDMA_RDMA_S_POINTER_POINTER',
    'PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR',
    'PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN',
    'PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE',
    'PPU_RDMA_RDMA_S_POINTER_RESERVED_0',
    'PPU_RDMA_RDMA_S_POINTER_RESERVED_1',
    'PPU_RDMA_RDMA_S_STATUS_RESERVED_0',
    'PPU_RDMA_RDMA_S_STATUS_RESERVED_1',
    'PPU_RDMA_RDMA_S_STATUS_STATUS_0',
    'PPU_RDMA_RDMA_S_STATUS_STATUS_1',
    'PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT',
    'PPU_RECIP_KERNEL_HEIGHT_RESERVED_0',
    'PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH',
    'PPU_RECIP_KERNEL_WIDTH_RESERVED_0', 'PPU_S_POINTER_EXECUTER',
    'PPU_S_POINTER_EXECUTER_PP_CLEAR', 'PPU_S_POINTER_EXECUTER_PP_EN',
    'PPU_S_POINTER_POINTER', 'PPU_S_POINTER_POINTER_PP_CLEAR',
    'PPU_S_POINTER_POINTER_PP_EN', 'PPU_S_POINTER_POINTER_PP_MODE',
    'PPU_S_POINTER_RESERVED_0', 'PPU_S_POINTER_RESERVED_1',
    'PPU_S_STATUS_RESERVED_0', 'PPU_S_STATUS_RESERVED_1',
    'PPU_S_STATUS_STATUS_0', 'PPU_S_STATUS_STATUS_1', 'SDMA',
    'SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL', 'SDMA_CFG_DMA_ARB_RD_FIX_ARB',
    'SDMA_CFG_DMA_ARB_RESERVED_0', 'SDMA_CFG_DMA_ARB_RESERVED_1',
    'SDMA_CFG_DMA_ARB_RESERVED_2', 'SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL',
    'SDMA_CFG_DMA_ARB_WR_FIX_ARB',
    'SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR',
    'SDMA_CFG_DMA_FIFO_CLR_RESERVED_0',
    'SDMA_CFG_DMA_RD_CFG_RD_ARBURST',
    'SDMA_CFG_DMA_RD_CFG_RD_ARCACHE', 'SDMA_CFG_DMA_RD_CFG_RD_ARLOCK',
    'SDMA_CFG_DMA_RD_CFG_RD_ARPROT', 'SDMA_CFG_DMA_RD_CFG_RD_ARSIZE',
    'SDMA_CFG_DMA_RD_CFG_RESERVED_0',
    'SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS',
    'SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS',
    'SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS',
    'SDMA_CFG_DMA_RD_QOS_RD_PC_QOS', 'SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS',
    'SDMA_CFG_DMA_RD_QOS_RESERVED_0',
    'SDMA_CFG_DMA_WR_CFG_RESERVED_0',
    'SDMA_CFG_DMA_WR_CFG_WR_AWBURST',
    'SDMA_CFG_DMA_WR_CFG_WR_AWCACHE', 'SDMA_CFG_DMA_WR_CFG_WR_AWLOCK',
    'SDMA_CFG_DMA_WR_CFG_WR_AWPROT', 'SDMA_CFG_DMA_WR_CFG_WR_AWSIZE',
    'SDMA_CFG_DMA_WSTRB_WR_WSTRB', 'SDMA_CFG_ID_ERROR_RD_RESP_ID',
    'SDMA_CFG_ID_ERROR_RESERVED_0', 'SDMA_CFG_ID_ERROR_RESERVED_1',
    'SDMA_CFG_ID_ERROR_WR_RESP_ID', 'SDMA_CFG_OUTSTANDING_RD_OS_CNT',
    'SDMA_CFG_OUTSTANDING_RESERVED_0',
    'SDMA_CFG_OUTSTANDING_WR_OS_CNT', 'SDMA_CFG_STATUS_IDEL',
    'SDMA_CFG_STATUS_RESERVED_0', 'SDMA_CFG_STATUS_RESERVED_1',
    'SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU',
    'SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE',
    'SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL',
    'SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP', 'SDMA_RD_WEIGHT_1_RD_WEIGHT_PC',
    'SDMA_RD_WEIGHT_1_RESERVED_0', 'SDMA_WR_WEIGHT_0_RESERVED_0',
    'SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU',
    'SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP', 'rkt_get_target', 'target',
    'uint32_t']
