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
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

_libraries['librknnrt.so'] = ctypes.CDLL('/usr/lib/librknnrt.so')


rknn_context = ctypes.c_uint64

# values for enumeration '_rknn_query_cmd'
_rknn_query_cmd__enumvalues = {
    0: 'RKNN_QUERY_IN_OUT_NUM',
    1: 'RKNN_QUERY_INPUT_ATTR',
    2: 'RKNN_QUERY_OUTPUT_ATTR',
    3: 'RKNN_QUERY_PERF_DETAIL',
    4: 'RKNN_QUERY_PERF_RUN',
    5: 'RKNN_QUERY_SDK_VERSION',
    6: 'RKNN_QUERY_MEM_SIZE',
    7: 'RKNN_QUERY_CUSTOM_STRING',
    8: 'RKNN_QUERY_NATIVE_INPUT_ATTR',
    9: 'RKNN_QUERY_NATIVE_OUTPUT_ATTR',
    8: 'RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR',
    9: 'RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR',
    10: 'RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR',
    11: 'RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR',
    12: 'RKNN_QUERY_DEVICE_MEM_INFO',
    13: 'RKNN_QUERY_INPUT_DYNAMIC_RANGE',
    14: 'RKNN_QUERY_CURRENT_INPUT_ATTR',
    15: 'RKNN_QUERY_CURRENT_OUTPUT_ATTR',
    16: 'RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR',
    17: 'RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR',
    18: 'RKNN_QUERY_CMD_MAX',
}
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 3
RKNN_QUERY_PERF_RUN = 4
RKNN_QUERY_SDK_VERSION = 5
RKNN_QUERY_MEM_SIZE = 6
RKNN_QUERY_CUSTOM_STRING = 7
RKNN_QUERY_NATIVE_INPUT_ATTR = 8
RKNN_QUERY_NATIVE_OUTPUT_ATTR = 9
RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR = 8
RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR = 9
RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR = 10
RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR = 11
RKNN_QUERY_DEVICE_MEM_INFO = 12
RKNN_QUERY_INPUT_DYNAMIC_RANGE = 13
RKNN_QUERY_CURRENT_INPUT_ATTR = 14
RKNN_QUERY_CURRENT_OUTPUT_ATTR = 15
RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR = 16
RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR = 17
RKNN_QUERY_CMD_MAX = 18
_rknn_query_cmd = ctypes.c_uint32 # enum
rknn_query_cmd = _rknn_query_cmd
rknn_query_cmd__enumvalues = _rknn_query_cmd__enumvalues

# values for enumeration '_rknn_tensor_type'
_rknn_tensor_type__enumvalues = {
    0: 'RKNN_TENSOR_FLOAT32',
    1: 'RKNN_TENSOR_FLOAT16',
    2: 'RKNN_TENSOR_INT8',
    3: 'RKNN_TENSOR_UINT8',
    4: 'RKNN_TENSOR_INT16',
    5: 'RKNN_TENSOR_UINT16',
    6: 'RKNN_TENSOR_INT32',
    7: 'RKNN_TENSOR_UINT32',
    8: 'RKNN_TENSOR_INT64',
    9: 'RKNN_TENSOR_BOOL',
    10: 'RKNN_TENSOR_INT4',
    11: 'RKNN_TENSOR_BFLOAT16',
    12: 'RKNN_TENSOR_TYPE_MAX',
}
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_UINT16 = 5
RKNN_TENSOR_INT32 = 6
RKNN_TENSOR_UINT32 = 7
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_BOOL = 9
RKNN_TENSOR_INT4 = 10
RKNN_TENSOR_BFLOAT16 = 11
RKNN_TENSOR_TYPE_MAX = 12
_rknn_tensor_type = ctypes.c_uint32 # enum
rknn_tensor_type = _rknn_tensor_type
rknn_tensor_type__enumvalues = _rknn_tensor_type__enumvalues
try:
    get_type_string = _libraries['FIXME_STUB'].get_type_string
    get_type_string.restype = ctypes.POINTER(ctypes.c_ubyte)
    get_type_string.argtypes = [rknn_tensor_type]
except AttributeError:
    pass

# values for enumeration '_rknn_tensor_qnt_type'
_rknn_tensor_qnt_type__enumvalues = {
    0: 'RKNN_TENSOR_QNT_NONE',
    1: 'RKNN_TENSOR_QNT_DFP',
    2: 'RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC',
    3: 'RKNN_TENSOR_QNT_MAX',
}
RKNN_TENSOR_QNT_NONE = 0
RKNN_TENSOR_QNT_DFP = 1
RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC = 2
RKNN_TENSOR_QNT_MAX = 3
_rknn_tensor_qnt_type = ctypes.c_uint32 # enum
rknn_tensor_qnt_type = _rknn_tensor_qnt_type
rknn_tensor_qnt_type__enumvalues = _rknn_tensor_qnt_type__enumvalues
try:
    get_qnt_type_string = _libraries['FIXME_STUB'].get_qnt_type_string
    get_qnt_type_string.restype = ctypes.POINTER(ctypes.c_ubyte)
    get_qnt_type_string.argtypes = [rknn_tensor_qnt_type]
except AttributeError:
    pass

# values for enumeration '_rknn_tensor_format'
_rknn_tensor_format__enumvalues = {
    0: 'RKNN_TENSOR_NCHW',
    1: 'RKNN_TENSOR_NHWC',
    2: 'RKNN_TENSOR_NC1HWC2',
    3: 'RKNN_TENSOR_UNDEFINED',
    4: 'RKNN_TENSOR_FORMAT_MAX',
}
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NC1HWC2 = 2
RKNN_TENSOR_UNDEFINED = 3
RKNN_TENSOR_FORMAT_MAX = 4
_rknn_tensor_format = ctypes.c_uint32 # enum
rknn_tensor_format = _rknn_tensor_format
rknn_tensor_format__enumvalues = _rknn_tensor_format__enumvalues

# values for enumeration '_rknn_core_mask'
_rknn_core_mask__enumvalues = {
    0: 'RKNN_NPU_CORE_AUTO',
    1: 'RKNN_NPU_CORE_0',
    2: 'RKNN_NPU_CORE_1',
    4: 'RKNN_NPU_CORE_2',
    3: 'RKNN_NPU_CORE_0_1',
    7: 'RKNN_NPU_CORE_0_1_2',
    65535: 'RKNN_NPU_CORE_ALL',
    65536: 'RKNN_NPU_CORE_UNDEFINED',
}
RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0 = 1
RKNN_NPU_CORE_1 = 2
RKNN_NPU_CORE_2 = 4
RKNN_NPU_CORE_0_1 = 3
RKNN_NPU_CORE_0_1_2 = 7
RKNN_NPU_CORE_ALL = 65535
RKNN_NPU_CORE_UNDEFINED = 65536
_rknn_core_mask = ctypes.c_uint32 # enum
rknn_core_mask = _rknn_core_mask
rknn_core_mask__enumvalues = _rknn_core_mask__enumvalues
try:
    get_format_string = _libraries['FIXME_STUB'].get_format_string
    get_format_string.restype = ctypes.POINTER(ctypes.c_ubyte)
    get_format_string.argtypes = [rknn_tensor_format]
except AttributeError:
    pass
class struct__rknn_input_output_num(Structure):
    pass

struct__rknn_input_output_num._pack_ = 1 # source:False
struct__rknn_input_output_num._fields_ = [
    ('n_input', ctypes.c_uint32),
    ('n_output', ctypes.c_uint32),
]

rknn_input_output_num = struct__rknn_input_output_num
class struct__rknn_tensor_attr(Structure):
    pass

struct__rknn_tensor_attr._pack_ = 1 # source:False
struct__rknn_tensor_attr._fields_ = [
    ('index', ctypes.c_uint32),
    ('n_dims', ctypes.c_uint32),
    ('dims', ctypes.c_uint32 * 16),
    ('name', ctypes.c_ubyte * 256),
    ('n_elems', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('fmt', rknn_tensor_format),
    ('type', rknn_tensor_type),
    ('qnt_type', rknn_tensor_qnt_type),
    ('fl', ctypes.c_byte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('zp', ctypes.c_int32),
    ('scale', ctypes.c_float),
    ('w_stride', ctypes.c_uint32),
    ('size_with_stride', ctypes.c_uint32),
    ('pass_through', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('h_stride', ctypes.c_uint32),
]

rknn_tensor_attr = struct__rknn_tensor_attr
class struct__rknn_input_range(Structure):
    pass

struct__rknn_input_range._pack_ = 1 # source:False
struct__rknn_input_range._fields_ = [
    ('index', ctypes.c_uint32),
    ('shape_number', ctypes.c_uint32),
    ('fmt', rknn_tensor_format),
    ('name', ctypes.c_ubyte * 256),
    ('dyn_range', ctypes.c_uint32 * 16 * 512),
    ('n_dims', ctypes.c_uint32),
]

rknn_input_range = struct__rknn_input_range
class struct__rknn_perf_detail(Structure):
    pass

struct__rknn_perf_detail._pack_ = 1 # source:False
struct__rknn_perf_detail._fields_ = [
    ('perf_data', ctypes.POINTER(ctypes.c_ubyte)),
    ('data_len', ctypes.c_uint64),
]

rknn_perf_detail = struct__rknn_perf_detail
class struct__rknn_perf_run(Structure):
    pass

struct__rknn_perf_run._pack_ = 1 # source:False
struct__rknn_perf_run._fields_ = [
    ('run_duration', ctypes.c_int64),
]

rknn_perf_run = struct__rknn_perf_run
class struct__rknn_sdk_version(Structure):
    pass

struct__rknn_sdk_version._pack_ = 1 # source:False
struct__rknn_sdk_version._fields_ = [
    ('api_version', ctypes.c_ubyte * 256),
    ('drv_version', ctypes.c_ubyte * 256),
]

rknn_sdk_version = struct__rknn_sdk_version
class struct__rknn_mem_size(Structure):
    pass

struct__rknn_mem_size._pack_ = 1 # source:False
struct__rknn_mem_size._fields_ = [
    ('total_weight_size', ctypes.c_uint32),
    ('total_internal_size', ctypes.c_uint32),
    ('total_dma_allocated_size', ctypes.c_uint64),
    ('total_sram_size', ctypes.c_uint32),
    ('free_sram_size', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 10),
]

rknn_mem_size = struct__rknn_mem_size
class struct__rknn_custom_string(Structure):
    pass

struct__rknn_custom_string._pack_ = 1 # source:False
struct__rknn_custom_string._fields_ = [
    ('string', ctypes.c_ubyte * 1024),
]

rknn_custom_string = struct__rknn_custom_string

# values for enumeration '_rknn_tensor_mem_flags'
_rknn_tensor_mem_flags__enumvalues = {
    1: 'RKNN_TENSOR_MEMORY_FLAGS_ALLOC_INSIDE',
    2: 'RKNN_TENSOR_MEMORY_FLAGS_FROM_FD',
    3: 'RKNN_TENSOR_MEMORY_FLAGS_FROM_PHYS',
    4: 'RKNN_TENSOR_MEMORY_FLAGS_UNKNOWN',
}
RKNN_TENSOR_MEMORY_FLAGS_ALLOC_INSIDE = 1
RKNN_TENSOR_MEMORY_FLAGS_FROM_FD = 2
RKNN_TENSOR_MEMORY_FLAGS_FROM_PHYS = 3
RKNN_TENSOR_MEMORY_FLAGS_UNKNOWN = 4
_rknn_tensor_mem_flags = ctypes.c_uint32 # enum
rknn_tensor_mem_flags = _rknn_tensor_mem_flags
rknn_tensor_mem_flags__enumvalues = _rknn_tensor_mem_flags__enumvalues

# values for enumeration '_rknn_mem_alloc_flags'
_rknn_mem_alloc_flags__enumvalues = {
    0: 'RKNN_FLAG_MEMORY_FLAGS_DEFAULT',
    1: 'RKNN_FLAG_MEMORY_CACHEABLE',
    2: 'RKNN_FLAG_MEMORY_NON_CACHEABLE',
    4: 'RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM',
}
RKNN_FLAG_MEMORY_FLAGS_DEFAULT = 0
RKNN_FLAG_MEMORY_CACHEABLE = 1
RKNN_FLAG_MEMORY_NON_CACHEABLE = 2
RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM = 4
_rknn_mem_alloc_flags = ctypes.c_uint32 # enum
rknn_mem_alloc_flags = _rknn_mem_alloc_flags
rknn_mem_alloc_flags__enumvalues = _rknn_mem_alloc_flags__enumvalues

# values for enumeration '_rknn_mem_sync_mode'
_rknn_mem_sync_mode__enumvalues = {
    1: 'RKNN_MEMORY_SYNC_TO_DEVICE',
    2: 'RKNN_MEMORY_SYNC_FROM_DEVICE',
    3: 'RKNN_MEMORY_SYNC_BIDIRECTIONAL',
}
RKNN_MEMORY_SYNC_TO_DEVICE = 1
RKNN_MEMORY_SYNC_FROM_DEVICE = 2
RKNN_MEMORY_SYNC_BIDIRECTIONAL = 3
_rknn_mem_sync_mode = ctypes.c_uint32 # enum
rknn_mem_sync_mode = _rknn_mem_sync_mode
rknn_mem_sync_mode__enumvalues = _rknn_mem_sync_mode__enumvalues
class struct__rknn_tensor_memory(Structure):
    pass

struct__rknn_tensor_memory._pack_ = 1 # source:False
struct__rknn_tensor_memory._fields_ = [
    ('virt_addr', ctypes.POINTER(None)),
    ('phys_addr', ctypes.c_uint64),
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_int32),
    ('size', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('priv_data', ctypes.POINTER(None)),
]

rknn_tensor_mem = struct__rknn_tensor_memory
class struct__rknn_input(Structure):
    pass

struct__rknn_input._pack_ = 1 # source:False
struct__rknn_input._fields_ = [
    ('index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('buf', ctypes.POINTER(None)),
    ('size', ctypes.c_uint32),
    ('pass_through', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('type', rknn_tensor_type),
    ('fmt', rknn_tensor_format),
]

rknn_input = struct__rknn_input
class struct__rknn_output(Structure):
    pass

struct__rknn_output._pack_ = 1 # source:False
struct__rknn_output._fields_ = [
    ('want_float', ctypes.c_ubyte),
    ('is_prealloc', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('index', ctypes.c_uint32),
    ('buf', ctypes.POINTER(None)),
    ('size', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

rknn_output = struct__rknn_output
class struct__rknn_init_extend(Structure):
    pass

struct__rknn_init_extend._pack_ = 1 # source:False
struct__rknn_init_extend._fields_ = [
    ('ctx', ctypes.c_uint64),
    ('real_model_offset', ctypes.c_int32),
    ('real_model_size', ctypes.c_uint32),
    ('model_buffer_fd', ctypes.c_int32),
    ('model_buffer_flags', ctypes.c_uint32),
    ('reserved', ctypes.c_ubyte * 112),
]

rknn_init_extend = struct__rknn_init_extend
class struct__rknn_run_extend(Structure):
    pass

struct__rknn_run_extend._pack_ = 1 # source:False
struct__rknn_run_extend._fields_ = [
    ('frame_id', ctypes.c_uint64),
    ('non_block', ctypes.c_int32),
    ('timeout_ms', ctypes.c_int32),
    ('fence_fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

rknn_run_extend = struct__rknn_run_extend
class struct__rknn_output_extend(Structure):
    pass

struct__rknn_output_extend._pack_ = 1 # source:False
struct__rknn_output_extend._fields_ = [
    ('frame_id', ctypes.c_uint64),
]

rknn_output_extend = struct__rknn_output_extend
uint32_t = ctypes.c_uint32
try:
    rknn_init = _libraries['librknnrt.so'].rknn_init
    rknn_init.restype = ctypes.c_int32
    rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None), uint32_t, uint32_t, ctypes.POINTER(struct__rknn_init_extend)]
except AttributeError:
    pass
try:
    rknn_dup_context = _libraries['librknnrt.so'].rknn_dup_context
    rknn_dup_context.restype = ctypes.c_int32
    rknn_dup_context.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    rknn_destroy = _libraries['librknnrt.so'].rknn_destroy
    rknn_destroy.restype = ctypes.c_int32
    rknn_destroy.argtypes = [rknn_context]
except AttributeError:
    pass
try:
    rknn_query = _libraries['librknnrt.so'].rknn_query
    rknn_query.restype = ctypes.c_int32
    rknn_query.argtypes = [rknn_context, rknn_query_cmd, ctypes.POINTER(None), uint32_t]
except AttributeError:
    pass
try:
    rknn_inputs_set = _libraries['librknnrt.so'].rknn_inputs_set
    rknn_inputs_set.restype = ctypes.c_int32
    rknn_inputs_set.argtypes = [rknn_context, uint32_t, struct__rknn_input * 2]
except AttributeError:
    pass
try:
    rknn_set_batch_core_num = _libraries['librknnrt.so'].rknn_set_batch_core_num
    rknn_set_batch_core_num.restype = ctypes.c_int32
    rknn_set_batch_core_num.argtypes = [rknn_context, ctypes.c_int32]
except AttributeError:
    pass
try:
    rknn_set_core_mask = _libraries['librknnrt.so'].rknn_set_core_mask
    rknn_set_core_mask.restype = ctypes.c_int32
    rknn_set_core_mask.argtypes = [rknn_context, rknn_core_mask]
except AttributeError:
    pass
try:
    rknn_run = _libraries['librknnrt.so'].rknn_run
    rknn_run.restype = ctypes.c_int32
    rknn_run.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_run_extend)]
except AttributeError:
    pass
try:
    rknn_wait = _libraries['librknnrt.so'].rknn_wait
    rknn_wait.restype = ctypes.c_int32
    rknn_wait.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_run_extend)]
except AttributeError:
    pass
try:
    rknn_outputs_get = _libraries['librknnrt.so'].rknn_outputs_get
    rknn_outputs_get.restype = ctypes.c_int32
    rknn_outputs_get.argtypes = [rknn_context, uint32_t, struct__rknn_output * 2, ctypes.POINTER(struct__rknn_output_extend)]
except AttributeError:
    pass
try:
    rknn_outputs_release = _libraries['librknnrt.so'].rknn_outputs_release
    rknn_outputs_release.restype = ctypes.c_int32
    rknn_outputs_release.argtypes = [rknn_context, uint32_t, struct__rknn_output * 0]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    rknn_create_mem_from_phys = _libraries['librknnrt.so'].rknn_create_mem_from_phys
    rknn_create_mem_from_phys.restype = ctypes.POINTER(struct__rknn_tensor_memory)
    rknn_create_mem_from_phys.argtypes = [rknn_context, uint64_t, ctypes.POINTER(None), uint32_t]
except AttributeError:
    pass
int32_t = ctypes.c_int32
try:
    rknn_create_mem_from_fd = _libraries['librknnrt.so'].rknn_create_mem_from_fd
    rknn_create_mem_from_fd.restype = ctypes.POINTER(struct__rknn_tensor_memory)
    rknn_create_mem_from_fd.argtypes = [rknn_context, int32_t, ctypes.POINTER(None), uint32_t, int32_t]
except AttributeError:
    pass
try:
    rknn_create_mem_from_mb_blk = _libraries['FIXME_STUB'].rknn_create_mem_from_mb_blk
    rknn_create_mem_from_mb_blk.restype = ctypes.POINTER(struct__rknn_tensor_memory)
    rknn_create_mem_from_mb_blk.argtypes = [rknn_context, ctypes.POINTER(None), int32_t]
except AttributeError:
    pass
try:
    rknn_create_mem = _libraries['librknnrt.so'].rknn_create_mem
    rknn_create_mem.restype = ctypes.POINTER(struct__rknn_tensor_memory)
    rknn_create_mem.argtypes = [rknn_context, uint32_t]
except AttributeError:
    pass
try:
    rknn_create_mem2 = _libraries['librknnrt.so'].rknn_create_mem2
    rknn_create_mem2.restype = ctypes.POINTER(struct__rknn_tensor_memory)
    rknn_create_mem2.argtypes = [rknn_context, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    rknn_destroy_mem = _libraries['librknnrt.so'].rknn_destroy_mem
    rknn_destroy_mem.restype = ctypes.c_int32
    rknn_destroy_mem.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_memory)]
except AttributeError:
    pass
try:
    rknn_set_weight_mem = _libraries['librknnrt.so'].rknn_set_weight_mem
    rknn_set_weight_mem.restype = ctypes.c_int32
    rknn_set_weight_mem.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_memory)]
except AttributeError:
    pass
try:
    rknn_set_internal_mem = _libraries['librknnrt.so'].rknn_set_internal_mem
    rknn_set_internal_mem.restype = ctypes.c_int32
    rknn_set_internal_mem.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_memory)]
except AttributeError:
    pass
try:
    rknn_set_io_mem = _libraries['librknnrt.so'].rknn_set_io_mem
    rknn_set_io_mem.restype = ctypes.c_int32
    rknn_set_io_mem.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_memory), ctypes.POINTER(struct__rknn_tensor_attr)]
except AttributeError:
    pass
try:
    rknn_set_input_shape = _libraries['librknnrt.so'].rknn_set_input_shape
    rknn_set_input_shape.restype = ctypes.c_int32
    rknn_set_input_shape.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_attr)]
except AttributeError:
    pass
try:
    rknn_set_input_shapes = _libraries['librknnrt.so'].rknn_set_input_shapes
    rknn_set_input_shapes.restype = ctypes.c_int32
    rknn_set_input_shapes.argtypes = [rknn_context, uint32_t, struct__rknn_tensor_attr * 0]
except AttributeError:
    pass
try:
    rknn_mem_sync = _libraries['librknnrt.so'].rknn_mem_sync
    rknn_mem_sync.restype = ctypes.c_int32
    rknn_mem_sync.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_tensor_memory), rknn_mem_sync_mode]
except AttributeError:
    pass
rknn_matmul_ctx = ctypes.c_uint64

# values for enumeration '_rknn_matmul_quant_type'
_rknn_matmul_quant_type__enumvalues = {
    0: 'RKNN_QUANT_TYPE_PER_LAYER_SYM',
    1: 'RKNN_QUANT_TYPE_PER_LAYER_ASYM',
    2: 'RKNN_QUANT_TYPE_PER_CHANNEL_SYM',
    3: 'RKNN_QUANT_TYPE_PER_CHANNEL_ASYM',
    4: 'RKNN_QUANT_TYPE_PER_GROUP_SYM',
    5: 'RKNN_QUANT_TYPE_PER_GROUP_ASYM',
}
RKNN_QUANT_TYPE_PER_LAYER_SYM = 0
RKNN_QUANT_TYPE_PER_LAYER_ASYM = 1
RKNN_QUANT_TYPE_PER_CHANNEL_SYM = 2
RKNN_QUANT_TYPE_PER_CHANNEL_ASYM = 3
RKNN_QUANT_TYPE_PER_GROUP_SYM = 4
RKNN_QUANT_TYPE_PER_GROUP_ASYM = 5
_rknn_matmul_quant_type = ctypes.c_uint32 # enum
rknn_matmul_quant_type = _rknn_matmul_quant_type
rknn_matmul_quant_type__enumvalues = _rknn_matmul_quant_type__enumvalues
class struct__rknn_quant_params(Structure):
    pass

struct__rknn_quant_params._pack_ = 1 # source:False
struct__rknn_quant_params._fields_ = [
    ('name', ctypes.c_ubyte * 256),
    ('scale', ctypes.POINTER(ctypes.c_float)),
    ('scale_len', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('zp', ctypes.POINTER(ctypes.c_int32)),
    ('zp_len', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

rknn_quant_params = struct__rknn_quant_params

# values for enumeration '_rknn_matmul_type'
_rknn_matmul_type__enumvalues = {
    1: 'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32',
    2: 'RKNN_INT8_MM_INT8_TO_INT32',
    3: 'RKNN_INT8_MM_INT8_TO_INT8',
    4: 'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16',
    5: 'RKNN_FLOAT16_MM_INT8_TO_FLOAT32',
    6: 'RKNN_FLOAT16_MM_INT8_TO_FLOAT16',
    7: 'RKNN_FLOAT16_MM_INT4_TO_FLOAT32',
    8: 'RKNN_FLOAT16_MM_INT4_TO_FLOAT16',
    9: 'RKNN_INT8_MM_INT8_TO_FLOAT32',
    10: 'RKNN_INT4_MM_INT4_TO_INT16',
    11: 'RKNN_INT8_MM_INT4_TO_INT32',
    12: 'RKNN_FLOAT16_MM_INT4_TO_BFLOAT16',
    15: 'RKNN_INT8_MM_INT4_TO_FLOAT16',
}
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1
RKNN_INT8_MM_INT8_TO_INT32 = 2
RKNN_INT8_MM_INT8_TO_INT8 = 3
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4
RKNN_FLOAT16_MM_INT8_TO_FLOAT32 = 5
RKNN_FLOAT16_MM_INT8_TO_FLOAT16 = 6
RKNN_FLOAT16_MM_INT4_TO_FLOAT32 = 7
RKNN_FLOAT16_MM_INT4_TO_FLOAT16 = 8
RKNN_INT8_MM_INT8_TO_FLOAT32 = 9
RKNN_INT4_MM_INT4_TO_INT16 = 10
RKNN_INT8_MM_INT4_TO_INT32 = 11
RKNN_FLOAT16_MM_INT4_TO_BFLOAT16 = 12
RKNN_INT8_MM_INT4_TO_FLOAT16 = 15
_rknn_matmul_type = ctypes.c_uint32 # enum
rknn_matmul_type = _rknn_matmul_type
rknn_matmul_type__enumvalues = _rknn_matmul_type__enumvalues
try:
    get_matmul_type_string = _libraries['FIXME_STUB'].get_matmul_type_string
    get_matmul_type_string.restype = ctypes.POINTER(ctypes.c_ubyte)
    get_matmul_type_string.argtypes = [rknn_matmul_type]
except AttributeError:
    pass
class struct__rknn_matmul_tensor_attr(Structure):
    pass

struct__rknn_matmul_tensor_attr._pack_ = 1 # source:False
struct__rknn_matmul_tensor_attr._fields_ = [
    ('name', ctypes.c_ubyte * 256),
    ('n_dims', ctypes.c_uint32),
    ('dims', ctypes.c_uint32 * 16),
    ('size', ctypes.c_uint32),
    ('type', rknn_tensor_type),
]

rknn_matmul_tensor_attr = struct__rknn_matmul_tensor_attr
class struct__rknn_matmul_io_attr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('A', rknn_matmul_tensor_attr),
    ('B', rknn_matmul_tensor_attr),
    ('C', rknn_matmul_tensor_attr),
     ]

rknn_matmul_io_attr = struct__rknn_matmul_io_attr
class struct__rknn_matmul_shape(Structure):
    pass

struct__rknn_matmul_shape._pack_ = 1 # source:False
struct__rknn_matmul_shape._fields_ = [
    ('M', ctypes.c_int32),
    ('K', ctypes.c_int32),
    ('N', ctypes.c_int32),
]

rknn_matmul_shape = struct__rknn_matmul_shape

# values for enumeration 'c__EA_rknn_matmul_layout'
c__EA_rknn_matmul_layout__enumvalues = {
    0: 'RKNN_MM_LAYOUT_NORM',
    1: 'RKNN_MM_LAYOUT_NATIVE',
    2: 'RKNN_MM_LAYOUT_TP_NORM',
}
RKNN_MM_LAYOUT_NORM = 0
RKNN_MM_LAYOUT_NATIVE = 1
RKNN_MM_LAYOUT_TP_NORM = 2
c__EA_rknn_matmul_layout = ctypes.c_uint32 # enum
rknn_matmul_layout = c__EA_rknn_matmul_layout
rknn_matmul_layout__enumvalues = c__EA_rknn_matmul_layout__enumvalues
class struct_rknn_matmul_info_t(Structure):
    pass

struct_rknn_matmul_info_t._pack_ = 1 # source:False
struct_rknn_matmul_info_t._fields_ = [
    ('M', ctypes.c_int32),
    ('K', ctypes.c_int32),
    ('N', ctypes.c_int32),
    ('type', rknn_matmul_type),
    ('B_layout', ctypes.c_int16),
    ('B_quant_type', ctypes.c_int16),
    ('AC_layout', ctypes.c_int16),
    ('AC_quant_type', ctypes.c_int16),
    ('iommu_domain_id', ctypes.c_int32),
    ('group_size', ctypes.c_int16),
    ('reserved', ctypes.c_byte * 34),
]

rknn_matmul_info = struct_rknn_matmul_info_t
try:
    rknn_matmul_create = _libraries['librknnrt.so'].rknn_matmul_create
    rknn_matmul_create.restype = ctypes.c_int32
    rknn_matmul_create.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_rknn_matmul_info_t), ctypes.POINTER(struct__rknn_matmul_io_attr)]
except AttributeError:
    pass
try:
    rknn_matmul_create_dynamic_shape = _libraries['librknnrt.so'].rknn_matmul_create_dynamic_shape
    rknn_matmul_create_dynamic_shape.restype = ctypes.c_int32
    rknn_matmul_create_dynamic_shape.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_rknn_matmul_info_t), ctypes.c_int32, struct__rknn_matmul_shape * 0, struct__rknn_matmul_io_attr * 0]
except AttributeError:
    pass
try:
    rknn_matmul_set_io_mem = _libraries['librknnrt.so'].rknn_matmul_set_io_mem
    rknn_matmul_set_io_mem.restype = ctypes.c_int32
    rknn_matmul_set_io_mem.argtypes = [rknn_matmul_ctx, ctypes.POINTER(struct__rknn_tensor_memory), ctypes.POINTER(struct__rknn_matmul_tensor_attr)]
except AttributeError:
    pass
try:
    rknn_matmul_set_core_mask = _libraries['librknnrt.so'].rknn_matmul_set_core_mask
    rknn_matmul_set_core_mask.restype = ctypes.c_int32
    rknn_matmul_set_core_mask.argtypes = [rknn_matmul_ctx, rknn_core_mask]
except AttributeError:
    pass
try:
    rknn_matmul_set_quant_params = _libraries['librknnrt.so'].rknn_matmul_set_quant_params
    rknn_matmul_set_quant_params.restype = ctypes.c_int32
    rknn_matmul_set_quant_params.argtypes = [rknn_matmul_ctx, ctypes.POINTER(struct__rknn_quant_params)]
except AttributeError:
    pass
try:
    rknn_matmul_get_quant_params = _libraries['librknnrt.so'].rknn_matmul_get_quant_params
    rknn_matmul_get_quant_params.restype = ctypes.c_int32
    rknn_matmul_get_quant_params.argtypes = [rknn_matmul_ctx, ctypes.POINTER(struct__rknn_quant_params), ctypes.POINTER(ctypes.c_float)]
except AttributeError:
    pass
try:
    rknn_matmul_set_dynamic_shape = _libraries['librknnrt.so'].rknn_matmul_set_dynamic_shape
    rknn_matmul_set_dynamic_shape.restype = ctypes.c_int32
    rknn_matmul_set_dynamic_shape.argtypes = [rknn_matmul_ctx, ctypes.POINTER(struct__rknn_matmul_shape)]
except AttributeError:
    pass
try:
    rknn_matmul_run = _libraries['librknnrt.so'].rknn_matmul_run
    rknn_matmul_run.restype = ctypes.c_int32
    rknn_matmul_run.argtypes = [rknn_matmul_ctx]
except AttributeError:
    pass
try:
    rknn_matmul_destroy = _libraries['librknnrt.so'].rknn_matmul_destroy
    rknn_matmul_destroy.restype = ctypes.c_int32
    rknn_matmul_destroy.argtypes = [rknn_matmul_ctx]
except AttributeError:
    pass
try:
    rknn_B_normal_layout_to_native_layout = _libraries['librknnrt.so'].rknn_B_normal_layout_to_native_layout
    rknn_B_normal_layout_to_native_layout.restype = ctypes.c_int32
    rknn_B_normal_layout_to_native_layout.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(struct_rknn_matmul_info_t)]
except AttributeError:
    pass
rknn_custom_op_interal_context = ctypes.c_uint64

# values for enumeration '_rknn_target_type'
_rknn_target_type__enumvalues = {
    1: 'RKNN_TARGET_TYPE_CPU',
    2: 'RKNN_TARGET_TYPE_GPU',
    3: 'RKNN_TARGET_TYPE_MAX',
}
RKNN_TARGET_TYPE_CPU = 1
RKNN_TARGET_TYPE_GPU = 2
RKNN_TARGET_TYPE_MAX = 3
_rknn_target_type = ctypes.c_uint32 # enum
rknn_target_type = _rknn_target_type
rknn_target_type__enumvalues = _rknn_target_type__enumvalues
class struct__rknn_gpu_op_context(Structure):
    pass

struct__rknn_gpu_op_context._pack_ = 1 # source:False
struct__rknn_gpu_op_context._fields_ = [
    ('cl_context', ctypes.POINTER(None)),
    ('cl_command_queue', ctypes.POINTER(None)),
    ('cl_kernel', ctypes.POINTER(None)),
]

rknn_gpu_op_context = struct__rknn_gpu_op_context
class struct__rknn_custom_op_context(Structure):
    pass

struct__rknn_custom_op_context._pack_ = 1 # source:False
struct__rknn_custom_op_context._fields_ = [
    ('target', rknn_target_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('internal_ctx', ctypes.c_uint64),
    ('gpu_ctx', rknn_gpu_op_context),
    ('priv_data', ctypes.POINTER(None)),
]

rknn_custom_op_context = struct__rknn_custom_op_context
class struct__rknn_custom_op_tensor(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('attr', rknn_tensor_attr),
    ('mem', rknn_tensor_mem),
     ]

rknn_custom_op_tensor = struct__rknn_custom_op_tensor
class struct__rknn_custom_op_attr(Structure):
    pass

struct__rknn_custom_op_attr._pack_ = 1 # source:False
struct__rknn_custom_op_attr._fields_ = [
    ('name', ctypes.c_ubyte * 256),
    ('dtype', rknn_tensor_type),
    ('n_elems', ctypes.c_uint32),
    ('data', ctypes.POINTER(None)),
]

rknn_custom_op_attr = struct__rknn_custom_op_attr
class struct__rknn_custom_op(Structure):
    pass

struct__rknn_custom_op._pack_ = 1 # source:False
struct__rknn_custom_op._fields_ = [
    ('version', ctypes.c_uint32),
    ('target', rknn_target_type),
    ('op_type', ctypes.c_ubyte * 256),
    ('cl_kernel_name', ctypes.c_ubyte * 256),
    ('cl_kernel_source', ctypes.POINTER(ctypes.c_ubyte)),
    ('cl_source_size', ctypes.c_uint64),
    ('cl_build_options', ctypes.c_ubyte * 256),
    ('init', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct__rknn_custom_op_context), ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32, ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32)),
    ('prepare', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct__rknn_custom_op_context), ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32, ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32)),
    ('compute', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct__rknn_custom_op_context), ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32, ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32)),
    
    ('compute_native', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct__rknn_custom_op_context), ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32, ctypes.POINTER(struct__rknn_custom_op_tensor), ctypes.c_uint32)),
    ('destroy', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct__rknn_custom_op_context))),
]

rknn_custom_op = struct__rknn_custom_op
get_custom_op_func = ctypes.CFUNCTYPE(ctypes.POINTER(struct__rknn_custom_op))
try:
    rknn_register_custom_ops = _libraries['librknnrt.so'].rknn_register_custom_ops
    rknn_register_custom_ops.restype = ctypes.c_int32
    rknn_register_custom_ops.argtypes = [rknn_context, ctypes.POINTER(struct__rknn_custom_op), uint32_t]
except AttributeError:
    pass
try:
    rknn_custom_op_get_op_attr = _libraries['librknnrt.so'].rknn_custom_op_get_op_attr
    rknn_custom_op_get_op_attr.restype = None
    rknn_custom_op_get_op_attr.argtypes = [ctypes.POINTER(struct__rknn_custom_op_context), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(struct__rknn_custom_op_attr)]
except AttributeError:
    pass
__all__ = \
    ['RKNN_FLAG_MEMORY_CACHEABLE', 'RKNN_FLAG_MEMORY_FLAGS_DEFAULT',
    'RKNN_FLAG_MEMORY_NON_CACHEABLE',
    'RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM',
    'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16',
    'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32',
    'RKNN_FLOAT16_MM_INT4_TO_BFLOAT16',
    'RKNN_FLOAT16_MM_INT4_TO_FLOAT16',
    'RKNN_FLOAT16_MM_INT4_TO_FLOAT32',
    'RKNN_FLOAT16_MM_INT8_TO_FLOAT16',
    'RKNN_FLOAT16_MM_INT8_TO_FLOAT32', 'RKNN_INT4_MM_INT4_TO_INT16',
    'RKNN_INT8_MM_INT4_TO_FLOAT16', 'RKNN_INT8_MM_INT4_TO_INT32',
    'RKNN_INT8_MM_INT8_TO_FLOAT32', 'RKNN_INT8_MM_INT8_TO_INT32',
    'RKNN_INT8_MM_INT8_TO_INT8', 'RKNN_MEMORY_SYNC_BIDIRECTIONAL',
    'RKNN_MEMORY_SYNC_FROM_DEVICE', 'RKNN_MEMORY_SYNC_TO_DEVICE',
    'RKNN_MM_LAYOUT_NATIVE', 'RKNN_MM_LAYOUT_NORM',
    'RKNN_MM_LAYOUT_TP_NORM', 'RKNN_NPU_CORE_0', 'RKNN_NPU_CORE_0_1',
    'RKNN_NPU_CORE_0_1_2', 'RKNN_NPU_CORE_1', 'RKNN_NPU_CORE_2',
    'RKNN_NPU_CORE_ALL', 'RKNN_NPU_CORE_AUTO',
    'RKNN_NPU_CORE_UNDEFINED', 'RKNN_QUANT_TYPE_PER_CHANNEL_ASYM',
    'RKNN_QUANT_TYPE_PER_CHANNEL_SYM',
    'RKNN_QUANT_TYPE_PER_GROUP_ASYM', 'RKNN_QUANT_TYPE_PER_GROUP_SYM',
    'RKNN_QUANT_TYPE_PER_LAYER_ASYM', 'RKNN_QUANT_TYPE_PER_LAYER_SYM',
    'RKNN_QUERY_CMD_MAX', 'RKNN_QUERY_CURRENT_INPUT_ATTR',
    'RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR',
    'RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR',
    'RKNN_QUERY_CURRENT_OUTPUT_ATTR', 'RKNN_QUERY_CUSTOM_STRING',
    'RKNN_QUERY_DEVICE_MEM_INFO', 'RKNN_QUERY_INPUT_ATTR',
    'RKNN_QUERY_INPUT_DYNAMIC_RANGE', 'RKNN_QUERY_IN_OUT_NUM',
    'RKNN_QUERY_MEM_SIZE', 'RKNN_QUERY_NATIVE_INPUT_ATTR',
    'RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR',
    'RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR',
    'RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR',
    'RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR',
    'RKNN_QUERY_NATIVE_OUTPUT_ATTR', 'RKNN_QUERY_OUTPUT_ATTR',
    'RKNN_QUERY_PERF_DETAIL', 'RKNN_QUERY_PERF_RUN',
    'RKNN_QUERY_SDK_VERSION', 'RKNN_TARGET_TYPE_CPU',
    'RKNN_TARGET_TYPE_GPU', 'RKNN_TARGET_TYPE_MAX',
    'RKNN_TENSOR_BFLOAT16', 'RKNN_TENSOR_BOOL', 'RKNN_TENSOR_FLOAT16',
    'RKNN_TENSOR_FLOAT32', 'RKNN_TENSOR_FORMAT_MAX',
    'RKNN_TENSOR_INT16', 'RKNN_TENSOR_INT32', 'RKNN_TENSOR_INT4',
    'RKNN_TENSOR_INT64', 'RKNN_TENSOR_INT8',
    'RKNN_TENSOR_MEMORY_FLAGS_ALLOC_INSIDE',
    'RKNN_TENSOR_MEMORY_FLAGS_FROM_FD',
    'RKNN_TENSOR_MEMORY_FLAGS_FROM_PHYS',
    'RKNN_TENSOR_MEMORY_FLAGS_UNKNOWN', 'RKNN_TENSOR_NC1HWC2',
    'RKNN_TENSOR_NCHW', 'RKNN_TENSOR_NHWC',
    'RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC', 'RKNN_TENSOR_QNT_DFP',
    'RKNN_TENSOR_QNT_MAX', 'RKNN_TENSOR_QNT_NONE',
    'RKNN_TENSOR_TYPE_MAX', 'RKNN_TENSOR_UINT16',
    'RKNN_TENSOR_UINT32', 'RKNN_TENSOR_UINT8',
    'RKNN_TENSOR_UNDEFINED', '_rknn_core_mask',
    '_rknn_matmul_quant_type', '_rknn_matmul_type',
    '_rknn_mem_alloc_flags', '_rknn_mem_sync_mode', '_rknn_query_cmd',
    '_rknn_target_type', '_rknn_tensor_format',
    '_rknn_tensor_mem_flags', '_rknn_tensor_qnt_type',
    '_rknn_tensor_type', 'c__EA_rknn_matmul_layout',
    'get_custom_op_func', 'get_format_string',
    'get_matmul_type_string', 'get_qnt_type_string',
    'get_type_string', 'int32_t',
    'rknn_B_normal_layout_to_native_layout', 'rknn_context',
    'rknn_core_mask', 'rknn_core_mask__enumvalues', 'rknn_create_mem',
    'rknn_create_mem2', 'rknn_create_mem_from_fd',
    'rknn_create_mem_from_mb_blk', 'rknn_create_mem_from_phys',
    'rknn_custom_op', 'rknn_custom_op_attr', 'rknn_custom_op_context',
    'rknn_custom_op_get_op_attr', 'rknn_custom_op_interal_context',
    'rknn_custom_op_tensor', 'rknn_custom_string', 'rknn_destroy',
    'rknn_destroy_mem', 'rknn_dup_context', 'rknn_gpu_op_context',
    'rknn_init', 'rknn_init_extend', 'rknn_input',
    'rknn_input_output_num', 'rknn_input_range', 'rknn_inputs_set',
    'rknn_matmul_create', 'rknn_matmul_create_dynamic_shape',
    'rknn_matmul_ctx', 'rknn_matmul_destroy',
    'rknn_matmul_get_quant_params', 'rknn_matmul_info',
    'rknn_matmul_io_attr', 'rknn_matmul_layout',
    'rknn_matmul_layout__enumvalues', 'rknn_matmul_quant_type',
    'rknn_matmul_quant_type__enumvalues', 'rknn_matmul_run',
    'rknn_matmul_set_core_mask', 'rknn_matmul_set_dynamic_shape',
    'rknn_matmul_set_io_mem', 'rknn_matmul_set_quant_params',
    'rknn_matmul_shape', 'rknn_matmul_tensor_attr',
    'rknn_matmul_type', 'rknn_matmul_type__enumvalues',
    'rknn_mem_alloc_flags', 'rknn_mem_alloc_flags__enumvalues',
    'rknn_mem_size', 'rknn_mem_sync', 'rknn_mem_sync_mode',
    'rknn_mem_sync_mode__enumvalues', 'rknn_output',
    'rknn_output_extend', 'rknn_outputs_get', 'rknn_outputs_release',
    'rknn_perf_detail', 'rknn_perf_run', 'rknn_quant_params',
    'rknn_query', 'rknn_query_cmd', 'rknn_query_cmd__enumvalues',
    'rknn_register_custom_ops', 'rknn_run', 'rknn_run_extend',
    'rknn_sdk_version', 'rknn_set_batch_core_num',
    'rknn_set_core_mask', 'rknn_set_input_shape',
    'rknn_set_input_shapes', 'rknn_set_internal_mem',
    'rknn_set_io_mem', 'rknn_set_weight_mem', 'rknn_target_type',
    'rknn_target_type__enumvalues', 'rknn_tensor_attr',
    'rknn_tensor_format', 'rknn_tensor_format__enumvalues',
    'rknn_tensor_mem', 'rknn_tensor_mem_flags',
    'rknn_tensor_mem_flags__enumvalues', 'rknn_tensor_qnt_type',
    'rknn_tensor_qnt_type__enumvalues', 'rknn_tensor_type',
    'rknn_tensor_type__enumvalues', 'rknn_wait',
    'struct__rknn_custom_op', 'struct__rknn_custom_op_attr',
    'struct__rknn_custom_op_context', 'struct__rknn_custom_op_tensor',
    'struct__rknn_custom_string', 'struct__rknn_gpu_op_context',
    'struct__rknn_init_extend', 'struct__rknn_input',
    'struct__rknn_input_output_num', 'struct__rknn_input_range',
    'struct__rknn_matmul_io_attr', 'struct__rknn_matmul_shape',
    'struct__rknn_matmul_tensor_attr', 'struct__rknn_mem_size',
    'struct__rknn_output', 'struct__rknn_output_extend',
    'struct__rknn_perf_detail', 'struct__rknn_perf_run',
    'struct__rknn_quant_params', 'struct__rknn_run_extend',
    'struct__rknn_sdk_version', 'struct__rknn_tensor_attr',
    'struct__rknn_tensor_memory', 'struct_rknn_matmul_info_t',
    'uint32_t', 'uint64_t']
