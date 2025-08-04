# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os



import functools
from tinygrad.runtime.support.hcq import FileIOInterface

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd, __payload=None, **kwargs):
  ret = __fd.ioctl((__idir<<30) | (ctypes.sizeof(made := (__payload or __user_struct(**kwargs)))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)

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



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



_DRM_MODE_H = True # macro
_DRM_H_ = True # macro
DRM_NAME = "drm" # macro
DRM_MIN_ORDER = 5 # macro
DRM_MAX_ORDER = 22 # macro
DRM_RAM_PERCENT = 10 # macro
_DRM_LOCK_HELD = 0x80000000 # macro
_DRM_LOCK_CONT = 0x40000000 # macro
def _DRM_LOCK_IS_HELD(lock):  # macro
   return ((lock)&0x80000000)
def _DRM_LOCK_IS_CONT(lock):  # macro
   return ((lock)&0x40000000)
def _DRM_LOCKING_CONTEXT(lock):  # macro
   return ((lock)&~(0x80000000|0x40000000))
_DRM_VBLANK_HIGH_CRTC_SHIFT = 1 # macro
_DRM_PRE_MODESET = 1 # macro
_DRM_POST_MODESET = 2 # macro
DRM_CAP_DUMB_BUFFER = 0x1 # macro
DRM_CAP_VBLANK_HIGH_CRTC = 0x2 # macro
DRM_CAP_DUMB_PREFERRED_DEPTH = 0x3 # macro
DRM_CAP_DUMB_PREFER_SHADOW = 0x4 # macro
DRM_CAP_PRIME = 0x5 # macro
DRM_PRIME_CAP_IMPORT = 0x1 # macro
DRM_PRIME_CAP_EXPORT = 0x2 # macro
DRM_CAP_TIMESTAMP_MONOTONIC = 0x6 # macro
DRM_CAP_ASYNC_PAGE_FLIP = 0x7 # macro
DRM_CAP_CURSOR_WIDTH = 0x8 # macro
DRM_CAP_CURSOR_HEIGHT = 0x9 # macro
DRM_CAP_ADDFB2_MODIFIERS = 0x10 # macro
DRM_CAP_PAGE_FLIP_TARGET = 0x11 # macro
DRM_CAP_CRTC_IN_VBLANK_EVENT = 0x12 # macro
DRM_CAP_SYNCOBJ = 0x13 # macro
DRM_CLIENT_CAP_STEREO_3D = 1 # macro
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2 # macro
DRM_CLIENT_CAP_ATOMIC = 3 # macro
DRM_CLIENT_CAP_ASPECT_RATIO = 4 # macro
DRM_CLIENT_CAP_WRITEBACK_CONNECTORS = 5 # macro
# DRM_RDWR = O_RDWR # macro
# DRM_CLOEXEC = O_CLOEXEC # macro
DRM_SYNCOBJ_CREATE_SIGNALED = (1<<0) # macro
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE = (1<<0) # macro
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1<<0) # macro
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1<<0) # macro
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1<<1) # macro
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001 # macro
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002 # macro
DRM_IOCTL_BASE = 'd' # macro
def DRM_IO(nr):  # macro
   return _IO('d',nr)
def DRM_IOR(nr, type):  # macro
   return _IOR('d',nr,type)
def DRM_IOW(nr, type):  # macro
   return _IOW('d',nr,type)
def DRM_IOWR(nr, type):  # macro
   return _IOWR('d',nr,type)
DRM_IOCTL_SET_MASTER = DRM_IO ( 0x1e ) # macro
DRM_IOCTL_DROP_MASTER = DRM_IO ( 0x1f ) # macro
DRM_IOCTL_AGP_ACQUIRE = DRM_IO ( 0x30 ) # macro
DRM_IOCTL_AGP_RELEASE = DRM_IO ( 0x31 ) # macro
# DRM_IOCTL_WAIT_VBLANK = DRM_IOWR ( 0x3a , drm_wait_vblank ) # macro
DRM_COMMAND_BASE = 0x40 # macro
DRM_COMMAND_END = 0xA0 # macro
DRM_EVENT_VBLANK = 0x01 # macro
DRM_EVENT_FLIP_COMPLETE = 0x02 # macro
DRM_EVENT_CRTC_SEQUENCE = 0x03 # macro
DRM_DISPLAY_INFO_LEN = 32 # macro
DRM_CONNECTOR_NAME_LEN = 32 # macro
DRM_DISPLAY_MODE_LEN = 32 # macro
DRM_PROP_NAME_LEN = 32 # macro
DRM_MODE_TYPE_BUILTIN = (1<<0) # macro
DRM_MODE_TYPE_CLOCK_C = ((1<<1)|(1<<0)) # macro
DRM_MODE_TYPE_CRTC_C = ((1<<2)|(1<<0)) # macro
DRM_MODE_TYPE_PREFERRED = (1<<3) # macro
DRM_MODE_TYPE_DEFAULT = (1<<4) # macro
DRM_MODE_TYPE_USERDEF = (1<<5) # macro
DRM_MODE_TYPE_DRIVER = (1<<6) # macro
DRM_MODE_TYPE_ALL = ((1<<3)|(1<<5)|(1<<6)) # macro
DRM_MODE_FLAG_PHSYNC = (1<<0) # macro
DRM_MODE_FLAG_NHSYNC = (1<<1) # macro
DRM_MODE_FLAG_PVSYNC = (1<<2) # macro
DRM_MODE_FLAG_NVSYNC = (1<<3) # macro
DRM_MODE_FLAG_INTERLACE = (1<<4) # macro
DRM_MODE_FLAG_DBLSCAN = (1<<5) # macro
DRM_MODE_FLAG_CSYNC = (1<<6) # macro
DRM_MODE_FLAG_PCSYNC = (1<<7) # macro
DRM_MODE_FLAG_NCSYNC = (1<<8) # macro
DRM_MODE_FLAG_HSKEW = (1<<9) # macro
DRM_MODE_FLAG_BCAST = (1<<10) # macro
DRM_MODE_FLAG_PIXMUX = (1<<11) # macro
DRM_MODE_FLAG_DBLCLK = (1<<12) # macro
DRM_MODE_FLAG_CLKDIV2 = (1<<13) # macro
DRM_MODE_FLAG_3D_MASK = (0x1f<<14) # macro
DRM_MODE_FLAG_3D_NONE = (0<<14) # macro
DRM_MODE_FLAG_3D_FRAME_PACKING = (1<<14) # macro
DRM_MODE_FLAG_3D_FIELD_ALTERNATIVE = (2<<14) # macro
DRM_MODE_FLAG_3D_LINE_ALTERNATIVE = (3<<14) # macro
DRM_MODE_FLAG_3D_SIDE_BY_SIDE_FULL = (4<<14) # macro
DRM_MODE_FLAG_3D_L_DEPTH = (5<<14) # macro
DRM_MODE_FLAG_3D_L_DEPTH_GFX_GFX_DEPTH = (6<<14) # macro
DRM_MODE_FLAG_3D_TOP_AND_BOTTOM = (7<<14) # macro
DRM_MODE_FLAG_3D_SIDE_BY_SIDE_HALF = (8<<14) # macro
DRM_MODE_PICTURE_ASPECT_NONE = 0 # macro
DRM_MODE_PICTURE_ASPECT_4_3 = 1 # macro
DRM_MODE_PICTURE_ASPECT_16_9 = 2 # macro
DRM_MODE_PICTURE_ASPECT_64_27 = 3 # macro
DRM_MODE_PICTURE_ASPECT_256_135 = 4 # macro
DRM_MODE_CONTENT_TYPE_NO_DATA = 0 # macro
DRM_MODE_CONTENT_TYPE_GRAPHICS = 1 # macro
DRM_MODE_CONTENT_TYPE_PHOTO = 2 # macro
DRM_MODE_CONTENT_TYPE_CINEMA = 3 # macro
DRM_MODE_CONTENT_TYPE_GAME = 4 # macro
DRM_MODE_FLAG_PIC_AR_MASK = (0x0F<<19) # macro
DRM_MODE_FLAG_PIC_AR_NONE = (0<<19) # macro
DRM_MODE_FLAG_PIC_AR_4_3 = (1<<19) # macro
DRM_MODE_FLAG_PIC_AR_16_9 = (2<<19) # macro
DRM_MODE_FLAG_PIC_AR_64_27 = (3<<19) # macro
DRM_MODE_FLAG_PIC_AR_256_135 = (4<<19) # macro
DRM_MODE_FLAG_ALL = ((1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<4)|(1<<5)|(1<<6)|(1<<7)|(1<<8)|(1<<9)|(1<<12)|(1<<13)|(0x1f<<14)) # macro
DRM_MODE_DPMS_ON = 0 # macro
DRM_MODE_DPMS_STANDBY = 1 # macro
DRM_MODE_DPMS_SUSPEND = 2 # macro
DRM_MODE_DPMS_OFF = 3 # macro
DRM_MODE_SCALE_NONE = 0 # macro
DRM_MODE_SCALE_FULLSCREEN = 1 # macro
DRM_MODE_SCALE_CENTER = 2 # macro
DRM_MODE_SCALE_ASPECT = 3 # macro
DRM_MODE_DITHERING_OFF = 0 # macro
DRM_MODE_DITHERING_ON = 1 # macro
DRM_MODE_DITHERING_AUTO = 2 # macro
DRM_MODE_DIRTY_OFF = 0 # macro
DRM_MODE_DIRTY_ON = 1 # macro
DRM_MODE_DIRTY_ANNOTATE = 2 # macro
DRM_MODE_LINK_STATUS_GOOD = 0 # macro
DRM_MODE_LINK_STATUS_BAD = 1 # macro
DRM_MODE_ROTATE_0 = (1<<0) # macro
DRM_MODE_ROTATE_90 = (1<<1) # macro
DRM_MODE_ROTATE_180 = (1<<2) # macro
DRM_MODE_ROTATE_270 = (1<<3) # macro
DRM_MODE_ROTATE_MASK = ((1<<0)|(1<<1)|(1<<2)|(1<<3)) # macro
DRM_MODE_REFLECT_X = (1<<4) # macro
DRM_MODE_REFLECT_Y = (1<<5) # macro
DRM_MODE_REFLECT_MASK = ((1<<4)|(1<<5)) # macro
DRM_MODE_CONTENT_PROTECTION_UNDESIRED = 0 # macro
DRM_MODE_CONTENT_PROTECTION_DESIRED = 1 # macro
DRM_MODE_CONTENT_PROTECTION_ENABLED = 2 # macro
DRM_MODE_PRESENT_TOP_FIELD = (1<<0) # macro
DRM_MODE_PRESENT_BOTTOM_FIELD = (1<<1) # macro
DRM_MODE_ENCODER_NONE = 0 # macro
DRM_MODE_ENCODER_DAC = 1 # macro
DRM_MODE_ENCODER_TMDS = 2 # macro
DRM_MODE_ENCODER_LVDS = 3 # macro
DRM_MODE_ENCODER_TVDAC = 4 # macro
DRM_MODE_ENCODER_VIRTUAL = 5 # macro
DRM_MODE_ENCODER_DSI = 6 # macro
DRM_MODE_ENCODER_DPMST = 7 # macro
DRM_MODE_ENCODER_DPI = 8 # macro
DRM_MODE_CONNECTOR_Unknown = 0 # macro
DRM_MODE_CONNECTOR_VGA = 1 # macro
DRM_MODE_CONNECTOR_DVII = 2 # macro
DRM_MODE_CONNECTOR_DVID = 3 # macro
DRM_MODE_CONNECTOR_DVIA = 4 # macro
DRM_MODE_CONNECTOR_Composite = 5 # macro
DRM_MODE_CONNECTOR_SVIDEO = 6 # macro
DRM_MODE_CONNECTOR_LVDS = 7 # macro
DRM_MODE_CONNECTOR_Component = 8 # macro
DRM_MODE_CONNECTOR_9PinDIN = 9 # macro
DRM_MODE_CONNECTOR_DisplayPort = 10 # macro
DRM_MODE_CONNECTOR_HDMIA = 11 # macro
DRM_MODE_CONNECTOR_HDMIB = 12 # macro
DRM_MODE_CONNECTOR_TV = 13 # macro
DRM_MODE_CONNECTOR_eDP = 14 # macro
DRM_MODE_CONNECTOR_VIRTUAL = 15 # macro
DRM_MODE_CONNECTOR_DSI = 16 # macro
DRM_MODE_CONNECTOR_DPI = 17 # macro
DRM_MODE_CONNECTOR_WRITEBACK = 18 # macro
DRM_MODE_PROP_PENDING = (1<<0) # macro
DRM_MODE_PROP_RANGE = (1<<1) # macro
DRM_MODE_PROP_IMMUTABLE = (1<<2) # macro
DRM_MODE_PROP_ENUM = (1<<3) # macro
DRM_MODE_PROP_BLOB = (1<<4) # macro
DRM_MODE_PROP_BITMASK = (1<<5) # macro
DRM_MODE_PROP_LEGACY_TYPE = ((1<<1)|(1<<3)|(1<<4)|(1<<5)) # macro
DRM_MODE_PROP_EXTENDED_TYPE = 0x0000ffc0 # macro
def DRM_MODE_PROP_TYPE(n):  # macro
   return ((n)<<6)
DRM_MODE_PROP_OBJECT = DRM_MODE_PROP_TYPE ( 1 ) # macro
DRM_MODE_PROP_SIGNED_RANGE = DRM_MODE_PROP_TYPE ( 2 ) # macro
DRM_MODE_PROP_ATOMIC = 0x80000000 # macro
DRM_MODE_OBJECT_CRTC = 0xcccccccc # macro
DRM_MODE_OBJECT_CONNECTOR = 0xc0c0c0c0 # macro
DRM_MODE_OBJECT_ENCODER = 0xe0e0e0e0 # macro
DRM_MODE_OBJECT_MODE = 0xdededede # macro
DRM_MODE_OBJECT_PROPERTY = 0xb0b0b0b0 # macro
DRM_MODE_OBJECT_FB = 0xfbfbfbfb # macro
DRM_MODE_OBJECT_BLOB = 0xbbbbbbbb # macro
DRM_MODE_OBJECT_PLANE = 0xeeeeeeee # macro
DRM_MODE_OBJECT_ANY = 0 # macro
DRM_MODE_FB_INTERLACED = (1<<0) # macro
DRM_MODE_FB_MODIFIERS = (1<<1) # macro
DRM_MODE_FB_DIRTY_ANNOTATE_COPY = 0x01 # macro
DRM_MODE_FB_DIRTY_ANNOTATE_FILL = 0x02 # macro
DRM_MODE_FB_DIRTY_FLAGS = 0x03 # macro
DRM_MODE_FB_DIRTY_MAX_CLIPS = 256 # macro
DRM_MODE_CURSOR_BO = 0x01 # macro
DRM_MODE_CURSOR_MOVE = 0x02 # macro
DRM_MODE_CURSOR_FLAGS = 0x03 # macro
DRM_MODE_PAGE_FLIP_EVENT = 0x01 # macro
DRM_MODE_PAGE_FLIP_ASYNC = 0x02 # macro
DRM_MODE_PAGE_FLIP_TARGET_ABSOLUTE = 0x4 # macro
DRM_MODE_PAGE_FLIP_TARGET_RELATIVE = 0x8 # macro
DRM_MODE_PAGE_FLIP_TARGET = (0x4|0x8) # macro
DRM_MODE_PAGE_FLIP_FLAGS = (0x01|0x02|(0x4|0x8)) # macro
DRM_MODE_ATOMIC_TEST_ONLY = 0x0100 # macro
DRM_MODE_ATOMIC_NONBLOCK = 0x0200 # macro
DRM_MODE_ATOMIC_ALLOW_MODESET = 0x0400 # macro
DRM_MODE_ATOMIC_FLAGS = (0x01|0x02|0x0100|0x0200|0x0400) # macro
FORMAT_BLOB_CURRENT = 1 # macro
drm_handle_t = ctypes.c_uint32
drm_context_t = ctypes.c_uint32
drm_drawable_t = ctypes.c_uint32
drm_magic_t = ctypes.c_uint32
class struct_drm_clip_rect(Structure):
    pass

struct_drm_clip_rect._pack_ = 1 # source:False
struct_drm_clip_rect._fields_ = [
    ('x1', ctypes.c_uint16),
    ('y1', ctypes.c_uint16),
    ('x2', ctypes.c_uint16),
    ('y2', ctypes.c_uint16),
]

class struct_drm_drawable_info(Structure):
    pass

struct_drm_drawable_info._pack_ = 1 # source:False
struct_drm_drawable_info._fields_ = [
    ('num_rects', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('rects', ctypes.POINTER(struct_drm_clip_rect)),
]

class struct_drm_tex_region(Structure):
    pass

struct_drm_tex_region._pack_ = 1 # source:False
struct_drm_tex_region._fields_ = [
    ('next', ctypes.c_ubyte),
    ('prev', ctypes.c_ubyte),
    ('in_use', ctypes.c_ubyte),
    ('padding', ctypes.c_ubyte),
    ('age', ctypes.c_uint32),
]

class struct_drm_hw_lock(Structure):
    pass

struct_drm_hw_lock._pack_ = 1 # source:False
struct_drm_hw_lock._fields_ = [
    ('lock', ctypes.c_uint32),
    ('padding', ctypes.c_ubyte * 60),
]

class struct_drm_version(Structure):
    pass

struct_drm_version._pack_ = 1 # source:False
struct_drm_version._fields_ = [
    ('version_major', ctypes.c_int32),
    ('version_minor', ctypes.c_int32),
    ('version_patchlevel', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('name_len', ctypes.c_uint64),
    ('name', ctypes.POINTER(ctypes.c_ubyte)),
    ('date_len', ctypes.c_uint64),
    ('date', ctypes.POINTER(ctypes.c_ubyte)),
    ('desc_len', ctypes.c_uint64),
    ('desc', ctypes.POINTER(ctypes.c_ubyte)),
]

DRM_IOCTL_VERSION = DRM_IOWR ( 0x00 , struct_drm_version ) # macro (from list)
class struct_drm_unique(Structure):
    pass

struct_drm_unique._pack_ = 1 # source:False
struct_drm_unique._fields_ = [
    ('unique_len', ctypes.c_uint64),
    ('unique', ctypes.POINTER(ctypes.c_ubyte)),
]

DRM_IOCTL_GET_UNIQUE = DRM_IOWR ( 0x01 , struct_drm_unique ) # macro (from list)
DRM_IOCTL_SET_UNIQUE = DRM_IOW ( 0x10 , struct_drm_unique ) # macro (from list)
class struct_drm_list(Structure):
    pass

struct_drm_list._pack_ = 1 # source:False
struct_drm_list._fields_ = [
    ('count', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('version', ctypes.POINTER(struct_drm_version)),
]

class struct_drm_block(Structure):
    pass

struct_drm_block._pack_ = 1 # source:False
struct_drm_block._fields_ = [
    ('unused', ctypes.c_int32),
]

DRM_IOCTL_BLOCK = DRM_IOWR ( 0x12 , struct_drm_block ) # macro (from list)
DRM_IOCTL_UNBLOCK = DRM_IOWR ( 0x13 , struct_drm_block ) # macro (from list)

# values for enumeration 'drm_control_func'
drm_control_func__enumvalues = {
    0: '_DRM_ADD_COMMAND',
    1: '_DRM_RM_COMMAND',
    2: '_DRM_INST_HANDLER',
    3: '_DRM_UNINST_HANDLER',
}
_DRM_ADD_COMMAND = 0
_DRM_RM_COMMAND = 1
_DRM_INST_HANDLER = 2
_DRM_UNINST_HANDLER = 3
drm_control_func = ctypes.c_uint32 # enum
class struct_drm_control(Structure):
    pass

struct_drm_control._pack_ = 1 # source:False
struct_drm_control._fields_ = [
    ('func', drm_control_func),
    ('irq', ctypes.c_int32),
]

DRM_IOCTL_CONTROL = DRM_IOW ( 0x14 , struct_drm_control ) # macro (from list)

# values for enumeration 'drm_map_type'
drm_map_type__enumvalues = {
    0: '_DRM_FRAME_BUFFER',
    1: '_DRM_REGISTERS',
    2: '_DRM_SHM',
    3: '_DRM_AGP',
    4: '_DRM_SCATTER_GATHER',
    5: '_DRM_CONSISTENT',
}
_DRM_FRAME_BUFFER = 0
_DRM_REGISTERS = 1
_DRM_SHM = 2
_DRM_AGP = 3
_DRM_SCATTER_GATHER = 4
_DRM_CONSISTENT = 5
drm_map_type = ctypes.c_uint32 # enum

# values for enumeration 'drm_map_flags'
drm_map_flags__enumvalues = {
    1: '_DRM_RESTRICTED',
    2: '_DRM_READ_ONLY',
    4: '_DRM_LOCKED',
    8: '_DRM_KERNEL',
    16: '_DRM_WRITE_COMBINING',
    32: '_DRM_CONTAINS_LOCK',
    64: '_DRM_REMOVABLE',
    128: '_DRM_DRIVER',
}
_DRM_RESTRICTED = 1
_DRM_READ_ONLY = 2
_DRM_LOCKED = 4
_DRM_KERNEL = 8
_DRM_WRITE_COMBINING = 16
_DRM_CONTAINS_LOCK = 32
_DRM_REMOVABLE = 64
_DRM_DRIVER = 128
drm_map_flags = ctypes.c_uint32 # enum
class struct_drm_ctx_priv_map(Structure):
    pass

struct_drm_ctx_priv_map._pack_ = 1 # source:False
struct_drm_ctx_priv_map._fields_ = [
    ('ctx_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', ctypes.POINTER(None)),
]

DRM_IOCTL_SET_SAREA_CTX = DRM_IOW ( 0x1c , struct_drm_ctx_priv_map ) # macro (from list)
DRM_IOCTL_GET_SAREA_CTX = DRM_IOWR ( 0x1d , struct_drm_ctx_priv_map ) # macro (from list)
class struct_drm_map(Structure):
    pass

struct_drm_map._pack_ = 1 # source:False
struct_drm_map._fields_ = [
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('type', drm_map_type),
    ('flags', drm_map_flags),
    ('handle', ctypes.POINTER(None)),
    ('mtrr', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

DRM_IOCTL_GET_MAP = DRM_IOWR ( 0x04 , struct_drm_map ) # macro (from list)
DRM_IOCTL_ADD_MAP = DRM_IOWR ( 0x15 , struct_drm_map ) # macro (from list)
DRM_IOCTL_RM_MAP = DRM_IOW ( 0x1b , struct_drm_map ) # macro (from list)
class struct_drm_client(Structure):
    pass

struct_drm_client._pack_ = 1 # source:False
struct_drm_client._fields_ = [
    ('idx', ctypes.c_int32),
    ('auth', ctypes.c_int32),
    ('pid', ctypes.c_uint64),
    ('uid', ctypes.c_uint64),
    ('magic', ctypes.c_uint64),
    ('iocs', ctypes.c_uint64),
]

DRM_IOCTL_GET_CLIENT = DRM_IOWR ( 0x05 , struct_drm_client ) # macro (from list)

# values for enumeration 'drm_stat_type'
drm_stat_type__enumvalues = {
    0: '_DRM_STAT_LOCK',
    1: '_DRM_STAT_OPENS',
    2: '_DRM_STAT_CLOSES',
    3: '_DRM_STAT_IOCTLS',
    4: '_DRM_STAT_LOCKS',
    5: '_DRM_STAT_UNLOCKS',
    6: '_DRM_STAT_VALUE',
    7: '_DRM_STAT_BYTE',
    8: '_DRM_STAT_COUNT',
    9: '_DRM_STAT_IRQ',
    10: '_DRM_STAT_PRIMARY',
    11: '_DRM_STAT_SECONDARY',
    12: '_DRM_STAT_DMA',
    13: '_DRM_STAT_SPECIAL',
    14: '_DRM_STAT_MISSED',
}
_DRM_STAT_LOCK = 0
_DRM_STAT_OPENS = 1
_DRM_STAT_CLOSES = 2
_DRM_STAT_IOCTLS = 3
_DRM_STAT_LOCKS = 4
_DRM_STAT_UNLOCKS = 5
_DRM_STAT_VALUE = 6
_DRM_STAT_BYTE = 7
_DRM_STAT_COUNT = 8
_DRM_STAT_IRQ = 9
_DRM_STAT_PRIMARY = 10
_DRM_STAT_SECONDARY = 11
_DRM_STAT_DMA = 12
_DRM_STAT_SPECIAL = 13
_DRM_STAT_MISSED = 14
drm_stat_type = ctypes.c_uint32 # enum
class struct_drm_stats(Structure):
    pass

class struct_drm_stats_0(Structure):
    pass

struct_drm_stats_0._pack_ = 1 # source:False
struct_drm_stats_0._fields_ = [
    ('value', ctypes.c_uint64),
    ('type', drm_stat_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_drm_stats._pack_ = 1 # source:False
struct_drm_stats._fields_ = [
    ('count', ctypes.c_uint64),
    ('data', struct_drm_stats_0 * 15),
]

DRM_IOCTL_GET_STATS = DRM_IOR ( 0x06 , struct_drm_stats ) # macro (from list)

# values for enumeration 'drm_lock_flags'
drm_lock_flags__enumvalues = {
    1: '_DRM_LOCK_READY',
    2: '_DRM_LOCK_QUIESCENT',
    4: '_DRM_LOCK_FLUSH',
    8: '_DRM_LOCK_FLUSH_ALL',
    16: '_DRM_HALT_ALL_QUEUES',
    32: '_DRM_HALT_CUR_QUEUES',
}
_DRM_LOCK_READY = 1
_DRM_LOCK_QUIESCENT = 2
_DRM_LOCK_FLUSH = 4
_DRM_LOCK_FLUSH_ALL = 8
_DRM_HALT_ALL_QUEUES = 16
_DRM_HALT_CUR_QUEUES = 32
drm_lock_flags = ctypes.c_uint32 # enum
class struct_drm_lock(Structure):
    pass

struct_drm_lock._pack_ = 1 # source:False
struct_drm_lock._fields_ = [
    ('context', ctypes.c_int32),
    ('flags', drm_lock_flags),
]

DRM_IOCTL_LOCK = DRM_IOW ( 0x2a , struct_drm_lock ) # macro (from list)
DRM_IOCTL_UNLOCK = DRM_IOW ( 0x2b , struct_drm_lock ) # macro (from list)
DRM_IOCTL_FINISH = DRM_IOW ( 0x2c , struct_drm_lock ) # macro (from list)

# values for enumeration 'drm_dma_flags'
drm_dma_flags__enumvalues = {
    1: '_DRM_DMA_BLOCK',
    2: '_DRM_DMA_WHILE_LOCKED',
    4: '_DRM_DMA_PRIORITY',
    16: '_DRM_DMA_WAIT',
    32: '_DRM_DMA_SMALLER_OK',
    64: '_DRM_DMA_LARGER_OK',
}
_DRM_DMA_BLOCK = 1
_DRM_DMA_WHILE_LOCKED = 2
_DRM_DMA_PRIORITY = 4
_DRM_DMA_WAIT = 16
_DRM_DMA_SMALLER_OK = 32
_DRM_DMA_LARGER_OK = 64
drm_dma_flags = ctypes.c_uint32 # enum

# values for enumeration 'drm_buf_desc_flags'
drm_buf_desc_flags__enumvalues = {
    1: '_DRM_PAGE_ALIGN',
    2: '_DRM_AGP_BUFFER',
    4: '_DRM_SG_BUFFER',
    8: '_DRM_FB_BUFFER',
    16: '_DRM_PCI_BUFFER_RO',
}
_DRM_PAGE_ALIGN = 1
_DRM_AGP_BUFFER = 2
_DRM_SG_BUFFER = 4
_DRM_FB_BUFFER = 8
_DRM_PCI_BUFFER_RO = 16
drm_buf_desc_flags = ctypes.c_uint32 # enum
class struct_drm_buf_desc(Structure):
    pass

struct_drm_buf_desc._pack_ = 1 # source:False
struct_drm_buf_desc._fields_ = [
    ('count', ctypes.c_int32),
    ('size', ctypes.c_int32),
    ('low_mark', ctypes.c_int32),
    ('high_mark', ctypes.c_int32),
    ('flags', drm_buf_desc_flags),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('agp_start', ctypes.c_uint64),
]

DRM_IOCTL_ADD_BUFS = DRM_IOWR ( 0x16 , struct_drm_buf_desc ) # macro (from list)
DRM_IOCTL_MARK_BUFS = DRM_IOW ( 0x17 , struct_drm_buf_desc ) # macro (from list)
class struct_drm_buf_info(Structure):
    pass

struct_drm_buf_info._pack_ = 1 # source:False
struct_drm_buf_info._fields_ = [
    ('count', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('list', ctypes.POINTER(struct_drm_buf_desc)),
]

DRM_IOCTL_INFO_BUFS = DRM_IOWR ( 0x18 , struct_drm_buf_info ) # macro (from list)
class struct_drm_buf_free(Structure):
    pass

struct_drm_buf_free._pack_ = 1 # source:False
struct_drm_buf_free._fields_ = [
    ('count', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('list', ctypes.POINTER(ctypes.c_int32)),
]

DRM_IOCTL_FREE_BUFS = DRM_IOW ( 0x1a , struct_drm_buf_free ) # macro (from list)
class struct_drm_buf_pub(Structure):
    pass

struct_drm_buf_pub._pack_ = 1 # source:False
struct_drm_buf_pub._fields_ = [
    ('idx', ctypes.c_int32),
    ('total', ctypes.c_int32),
    ('used', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('address', ctypes.POINTER(None)),
]

class struct_drm_buf_map(Structure):
    pass

struct_drm_buf_map._pack_ = 1 # source:False
struct_drm_buf_map._fields_ = [
    ('count', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('virtual', ctypes.POINTER(None)),
    ('list', ctypes.POINTER(struct_drm_buf_pub)),
]

DRM_IOCTL_MAP_BUFS = DRM_IOWR ( 0x19 , struct_drm_buf_map ) # macro (from list)
class struct_drm_dma(Structure):
    pass

struct_drm_dma._pack_ = 1 # source:False
struct_drm_dma._fields_ = [
    ('context', ctypes.c_int32),
    ('send_count', ctypes.c_int32),
    ('send_indices', ctypes.POINTER(ctypes.c_int32)),
    ('send_sizes', ctypes.POINTER(ctypes.c_int32)),
    ('flags', drm_dma_flags),
    ('request_count', ctypes.c_int32),
    ('request_size', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('request_indices', ctypes.POINTER(ctypes.c_int32)),
    ('request_sizes', ctypes.POINTER(ctypes.c_int32)),
    ('granted_count', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

DRM_IOCTL_DMA = DRM_IOWR ( 0x29 , struct_drm_dma ) # macro (from list)

# values for enumeration 'drm_ctx_flags'
drm_ctx_flags__enumvalues = {
    1: '_DRM_CONTEXT_PRESERVED',
    2: '_DRM_CONTEXT_2DONLY',
}
_DRM_CONTEXT_PRESERVED = 1
_DRM_CONTEXT_2DONLY = 2
drm_ctx_flags = ctypes.c_uint32 # enum
class struct_drm_ctx(Structure):
    pass

struct_drm_ctx._pack_ = 1 # source:False
struct_drm_ctx._fields_ = [
    ('handle', ctypes.c_uint32),
    ('flags', drm_ctx_flags),
]

DRM_IOCTL_ADD_CTX = DRM_IOWR ( 0x20 , struct_drm_ctx ) # macro (from list)
DRM_IOCTL_RM_CTX = DRM_IOWR ( 0x21 , struct_drm_ctx ) # macro (from list)
DRM_IOCTL_MOD_CTX = DRM_IOW ( 0x22 , struct_drm_ctx ) # macro (from list)
DRM_IOCTL_GET_CTX = DRM_IOWR ( 0x23 , struct_drm_ctx ) # macro (from list)
DRM_IOCTL_SWITCH_CTX = DRM_IOW ( 0x24 , struct_drm_ctx ) # macro (from list)
DRM_IOCTL_NEW_CTX = DRM_IOW ( 0x25 , struct_drm_ctx ) # macro (from list)
class struct_drm_ctx_res(Structure):
    pass

struct_drm_ctx_res._pack_ = 1 # source:False
struct_drm_ctx_res._fields_ = [
    ('count', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('contexts', ctypes.POINTER(struct_drm_ctx)),
]

DRM_IOCTL_RES_CTX = DRM_IOWR ( 0x26 , struct_drm_ctx_res ) # macro (from list)
class struct_drm_draw(Structure):
    pass

struct_drm_draw._pack_ = 1 # source:False
struct_drm_draw._fields_ = [
    ('handle', ctypes.c_uint32),
]

DRM_IOCTL_ADD_DRAW = DRM_IOWR ( 0x27 , struct_drm_draw ) # macro (from list)
DRM_IOCTL_RM_DRAW = DRM_IOWR ( 0x28 , struct_drm_draw ) # macro (from list)

# values for enumeration 'c__EA_drm_drawable_info_type_t'
c__EA_drm_drawable_info_type_t__enumvalues = {
    0: 'DRM_DRAWABLE_CLIPRECTS',
}
DRM_DRAWABLE_CLIPRECTS = 0
c__EA_drm_drawable_info_type_t = ctypes.c_uint32 # enum
drm_drawable_info_type_t = c__EA_drm_drawable_info_type_t
drm_drawable_info_type_t__enumvalues = c__EA_drm_drawable_info_type_t__enumvalues
class struct_drm_update_draw(Structure):
    pass

struct_drm_update_draw._pack_ = 1 # source:False
struct_drm_update_draw._fields_ = [
    ('handle', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('num', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('data', ctypes.c_uint64),
]

DRM_IOCTL_UPDATE_DRAW = DRM_IOW ( 0x3f , struct_drm_update_draw ) # macro (from list)
class struct_drm_auth(Structure):
    pass

struct_drm_auth._pack_ = 1 # source:False
struct_drm_auth._fields_ = [
    ('magic', ctypes.c_uint32),
]

DRM_IOCTL_GET_MAGIC = DRM_IOR ( 0x02 , struct_drm_auth ) # macro (from list)
DRM_IOCTL_AUTH_MAGIC = DRM_IOW ( 0x11 , struct_drm_auth ) # macro (from list)
class struct_drm_irq_busid(Structure):
    pass

struct_drm_irq_busid._pack_ = 1 # source:False
struct_drm_irq_busid._fields_ = [
    ('irq', ctypes.c_int32),
    ('busnum', ctypes.c_int32),
    ('devnum', ctypes.c_int32),
    ('funcnum', ctypes.c_int32),
]

DRM_IOCTL_IRQ_BUSID = DRM_IOWR ( 0x03 , struct_drm_irq_busid ) # macro (from list)

# values for enumeration 'drm_vblank_seq_type'
drm_vblank_seq_type__enumvalues = {
    0: '_DRM_VBLANK_ABSOLUTE',
    1: '_DRM_VBLANK_RELATIVE',
    62: '_DRM_VBLANK_HIGH_CRTC_MASK',
    67108864: '_DRM_VBLANK_EVENT',
    134217728: '_DRM_VBLANK_FLIP',
    268435456: '_DRM_VBLANK_NEXTONMISS',
    536870912: '_DRM_VBLANK_SECONDARY',
    1073741824: '_DRM_VBLANK_SIGNAL',
}
_DRM_VBLANK_ABSOLUTE = 0
_DRM_VBLANK_RELATIVE = 1
_DRM_VBLANK_HIGH_CRTC_MASK = 62
_DRM_VBLANK_EVENT = 67108864
_DRM_VBLANK_FLIP = 134217728
_DRM_VBLANK_NEXTONMISS = 268435456
_DRM_VBLANK_SECONDARY = 536870912
_DRM_VBLANK_SIGNAL = 1073741824
drm_vblank_seq_type = ctypes.c_uint32 # enum
_DRM_VBLANK_TYPES_MASK = (_DRM_VBLANK_ABSOLUTE|_DRM_VBLANK_RELATIVE) # macro
_DRM_VBLANK_FLAGS_MASK = (_DRM_VBLANK_EVENT|_DRM_VBLANK_SIGNAL|_DRM_VBLANK_SECONDARY|_DRM_VBLANK_NEXTONMISS) # macro
class struct_drm_wait_vblank_request(Structure):
    pass

struct_drm_wait_vblank_request._pack_ = 1 # source:False
struct_drm_wait_vblank_request._fields_ = [
    ('type', drm_vblank_seq_type),
    ('sequence', ctypes.c_uint32),
    ('signal', ctypes.c_uint64),
]

class struct_drm_wait_vblank_reply(Structure):
    pass

struct_drm_wait_vblank_reply._pack_ = 1 # source:False
struct_drm_wait_vblank_reply._fields_ = [
    ('type', drm_vblank_seq_type),
    ('sequence', ctypes.c_uint32),
    ('tval_sec', ctypes.c_int64),
    ('tval_usec', ctypes.c_int64),
]

class union_drm_wait_vblank(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('request', struct_drm_wait_vblank_request),
    ('reply', struct_drm_wait_vblank_reply),
     ]

class struct_drm_modeset_ctl(Structure):
    pass

struct_drm_modeset_ctl._pack_ = 1 # source:False
struct_drm_modeset_ctl._fields_ = [
    ('crtc', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
]

DRM_IOCTL_MODESET_CTL = DRM_IOW ( 0x08 , struct_drm_modeset_ctl ) # macro (from list)
class struct_drm_agp_mode(Structure):
    pass

struct_drm_agp_mode._pack_ = 1 # source:False
struct_drm_agp_mode._fields_ = [
    ('mode', ctypes.c_uint64),
]

DRM_IOCTL_AGP_ENABLE = DRM_IOW ( 0x32 , struct_drm_agp_mode ) # macro (from list)
class struct_drm_agp_buffer(Structure):
    pass

struct_drm_agp_buffer._pack_ = 1 # source:False
struct_drm_agp_buffer._fields_ = [
    ('size', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('type', ctypes.c_uint64),
    ('physical', ctypes.c_uint64),
]

DRM_IOCTL_AGP_ALLOC = DRM_IOWR ( 0x34 , struct_drm_agp_buffer ) # macro (from list)
DRM_IOCTL_AGP_FREE = DRM_IOW ( 0x35 , struct_drm_agp_buffer ) # macro (from list)
class struct_drm_agp_binding(Structure):
    pass

struct_drm_agp_binding._pack_ = 1 # source:False
struct_drm_agp_binding._fields_ = [
    ('handle', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

DRM_IOCTL_AGP_BIND = DRM_IOW ( 0x36 , struct_drm_agp_binding ) # macro (from list)
DRM_IOCTL_AGP_UNBIND = DRM_IOW ( 0x37 , struct_drm_agp_binding ) # macro (from list)
class struct_drm_agp_info(Structure):
    pass

struct_drm_agp_info._pack_ = 1 # source:False
struct_drm_agp_info._fields_ = [
    ('agp_version_major', ctypes.c_int32),
    ('agp_version_minor', ctypes.c_int32),
    ('mode', ctypes.c_uint64),
    ('aperture_base', ctypes.c_uint64),
    ('aperture_size', ctypes.c_uint64),
    ('memory_allowed', ctypes.c_uint64),
    ('memory_used', ctypes.c_uint64),
    ('id_vendor', ctypes.c_uint16),
    ('id_device', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

DRM_IOCTL_AGP_INFO = DRM_IOR ( 0x33 , struct_drm_agp_info ) # macro (from list)
class struct_drm_scatter_gather(Structure):
    pass

struct_drm_scatter_gather._pack_ = 1 # source:False
struct_drm_scatter_gather._fields_ = [
    ('size', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
]

DRM_IOCTL_SG_ALLOC = DRM_IOWR ( 0x38 , struct_drm_scatter_gather ) # macro (from list)
DRM_IOCTL_SG_FREE = DRM_IOW ( 0x39 , struct_drm_scatter_gather ) # macro (from list)
class struct_drm_set_version(Structure):
    pass

struct_drm_set_version._pack_ = 1 # source:False
struct_drm_set_version._fields_ = [
    ('drm_di_major', ctypes.c_int32),
    ('drm_di_minor', ctypes.c_int32),
    ('drm_dd_major', ctypes.c_int32),
    ('drm_dd_minor', ctypes.c_int32),
]

DRM_IOCTL_SET_VERSION = DRM_IOWR ( 0x07 , struct_drm_set_version ) # macro (from list)
class struct_drm_gem_close(Structure):
    pass

struct_drm_gem_close._pack_ = 1 # source:False
struct_drm_gem_close._fields_ = [
    ('handle', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_GEM_CLOSE = DRM_IOW ( 0x09 , struct_drm_gem_close ) # macro (from list)
class struct_drm_gem_flink(Structure):
    pass

struct_drm_gem_flink._pack_ = 1 # source:False
struct_drm_gem_flink._fields_ = [
    ('handle', ctypes.c_uint32),
    ('name', ctypes.c_uint32),
]

DRM_IOCTL_GEM_FLINK = DRM_IOWR ( 0x0a , struct_drm_gem_flink ) # macro (from list)
class struct_drm_gem_open(Structure):
    pass

struct_drm_gem_open._pack_ = 1 # source:False
struct_drm_gem_open._fields_ = [
    ('name', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
]

DRM_IOCTL_GEM_OPEN = DRM_IOWR ( 0x0b , struct_drm_gem_open ) # macro (from list)
class struct_drm_get_cap(Structure):
    pass

struct_drm_get_cap._pack_ = 1 # source:False
struct_drm_get_cap._fields_ = [
    ('capability', ctypes.c_uint64),
    ('value', ctypes.c_uint64),
]

DRM_IOCTL_GET_CAP = DRM_IOWR ( 0x0c , struct_drm_get_cap ) # macro (from list)
class struct_drm_set_client_cap(Structure):
    pass

struct_drm_set_client_cap._pack_ = 1 # source:False
struct_drm_set_client_cap._fields_ = [
    ('capability', ctypes.c_uint64),
    ('value', ctypes.c_uint64),
]

DRM_IOCTL_SET_CLIENT_CAP = DRM_IOW ( 0x0d , struct_drm_set_client_cap ) # macro (from list)
class struct_drm_prime_handle(Structure):
    pass

struct_drm_prime_handle._pack_ = 1 # source:False
struct_drm_prime_handle._fields_ = [
    ('handle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('fd', ctypes.c_int32),
]

DRM_IOCTL_PRIME_HANDLE_TO_FD = DRM_IOWR ( 0x2d , struct_drm_prime_handle ) # macro (from list)
DRM_IOCTL_PRIME_FD_TO_HANDLE = DRM_IOWR ( 0x2e , struct_drm_prime_handle ) # macro (from list)
class struct_drm_syncobj_create(Structure):
    pass

struct_drm_syncobj_create._pack_ = 1 # source:False
struct_drm_syncobj_create._fields_ = [
    ('handle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR ( 0xBF , struct_drm_syncobj_create ) # macro (from list)
class struct_drm_syncobj_destroy(Structure):
    pass

struct_drm_syncobj_destroy._pack_ = 1 # source:False
struct_drm_syncobj_destroy._fields_ = [
    ('handle', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR ( 0xC0 , struct_drm_syncobj_destroy ) # macro (from list)
class struct_drm_syncobj_handle(Structure):
    pass

struct_drm_syncobj_handle._pack_ = 1 # source:False
struct_drm_syncobj_handle._fields_ = [
    ('handle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('fd', ctypes.c_int32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR ( 0xC1 , struct_drm_syncobj_handle ) # macro (from list)
DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR ( 0xC2 , struct_drm_syncobj_handle ) # macro (from list)
class struct_drm_syncobj_wait(Structure):
    pass

struct_drm_syncobj_wait._pack_ = 1 # source:False
struct_drm_syncobj_wait._fields_ = [
    ('handles', ctypes.c_uint64),
    ('timeout_nsec', ctypes.c_int64),
    ('count_handles', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('first_signaled', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR ( 0xC3 , struct_drm_syncobj_wait ) # macro (from list)
class struct_drm_syncobj_array(Structure):
    pass

struct_drm_syncobj_array._pack_ = 1 # source:False
struct_drm_syncobj_array._fields_ = [
    ('handles', ctypes.c_uint64),
    ('count_handles', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR ( 0xC4 , struct_drm_syncobj_array ) # macro (from list)
DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR ( 0xC5 , struct_drm_syncobj_array ) # macro (from list)
class struct_drm_crtc_get_sequence(Structure):
    pass

struct_drm_crtc_get_sequence._pack_ = 1 # source:False
struct_drm_crtc_get_sequence._fields_ = [
    ('crtc_id', ctypes.c_uint32),
    ('active', ctypes.c_uint32),
    ('sequence', ctypes.c_uint64),
    ('sequence_ns', ctypes.c_int64),
]

DRM_IOCTL_CRTC_GET_SEQUENCE = DRM_IOWR ( 0x3b , struct_drm_crtc_get_sequence ) # macro (from list)
class struct_drm_crtc_queue_sequence(Structure):
    pass

struct_drm_crtc_queue_sequence._pack_ = 1 # source:False
struct_drm_crtc_queue_sequence._fields_ = [
    ('crtc_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('sequence', ctypes.c_uint64),
    ('user_data', ctypes.c_uint64),
]

DRM_IOCTL_CRTC_QUEUE_SEQUENCE = DRM_IOWR ( 0x3c , struct_drm_crtc_queue_sequence ) # macro (from list)
class struct_drm_event(Structure):
    pass

struct_drm_event._pack_ = 1 # source:False
struct_drm_event._fields_ = [
    ('type', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

class struct_drm_event_vblank(Structure):
    pass

struct_drm_event_vblank._pack_ = 1 # source:False
struct_drm_event_vblank._fields_ = [
    ('base', struct_drm_event),
    ('user_data', ctypes.c_uint64),
    ('tv_sec', ctypes.c_uint32),
    ('tv_usec', ctypes.c_uint32),
    ('sequence', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
]

class struct_drm_event_crtc_sequence(Structure):
    pass

struct_drm_event_crtc_sequence._pack_ = 1 # source:False
struct_drm_event_crtc_sequence._fields_ = [
    ('base', struct_drm_event),
    ('user_data', ctypes.c_uint64),
    ('time_ns', ctypes.c_int64),
    ('sequence', ctypes.c_uint64),
]

drm_clip_rect_t = struct_drm_clip_rect
drm_drawable_info_t = struct_drm_drawable_info
drm_tex_region_t = struct_drm_tex_region
drm_hw_lock_t = struct_drm_hw_lock
drm_version_t = struct_drm_version
drm_unique_t = struct_drm_unique
drm_list_t = struct_drm_list
drm_block_t = struct_drm_block
drm_control_t = struct_drm_control
drm_map_type_t = drm_map_type
drm_map_type_t__enumvalues = drm_map_type__enumvalues
drm_map_flags_t = drm_map_flags
drm_map_flags_t__enumvalues = drm_map_flags__enumvalues
drm_ctx_priv_map_t = struct_drm_ctx_priv_map
drm_map_t = struct_drm_map
drm_client_t = struct_drm_client
drm_stat_type_t = drm_stat_type
drm_stat_type_t__enumvalues = drm_stat_type__enumvalues
drm_stats_t = struct_drm_stats
drm_lock_flags_t = drm_lock_flags
drm_lock_flags_t__enumvalues = drm_lock_flags__enumvalues
drm_control_func_t = drm_control_func
drm_control_func_t__enumvalues = drm_control_func__enumvalues
drm_lock_t = struct_drm_lock
drm_dma_flags_t = drm_dma_flags
drm_dma_flags_t__enumvalues = drm_dma_flags__enumvalues
drm_buf_desc_t = struct_drm_buf_desc
drm_buf_desc_flags_t = drm_buf_desc_flags
drm_buf_desc_flags_t__enumvalues = drm_buf_desc_flags__enumvalues
drm_buf_info_t = struct_drm_buf_info
drm_buf_free_t = struct_drm_buf_free
drm_buf_pub_t = struct_drm_buf_pub
drm_buf_map_t = struct_drm_buf_map
drm_dma_t = struct_drm_dma
drm_wait_vblank_t = union_drm_wait_vblank
drm_agp_mode_t = struct_drm_agp_mode
drm_ctx_flags_t = drm_ctx_flags
drm_ctx_flags_t__enumvalues = drm_ctx_flags__enumvalues
drm_ctx_t = struct_drm_ctx
drm_ctx_res_t = struct_drm_ctx_res
drm_draw_t = struct_drm_draw
drm_update_draw_t = struct_drm_update_draw
drm_auth_t = struct_drm_auth
drm_irq_busid_t = struct_drm_irq_busid
drm_vblank_seq_type_t = drm_vblank_seq_type
drm_vblank_seq_type_t__enumvalues = drm_vblank_seq_type__enumvalues
drm_agp_buffer_t = struct_drm_agp_buffer
drm_agp_binding_t = struct_drm_agp_binding
drm_agp_info_t = struct_drm_agp_info
drm_scatter_gather_t = struct_drm_scatter_gather
drm_set_version_t = struct_drm_set_version
class struct_drm_mode_modeinfo(Structure):
    pass

struct_drm_mode_modeinfo._pack_ = 1 # source:False
struct_drm_mode_modeinfo._fields_ = [
    ('clock', ctypes.c_uint32),
    ('hdisplay', ctypes.c_uint16),
    ('hsync_start', ctypes.c_uint16),
    ('hsync_end', ctypes.c_uint16),
    ('htotal', ctypes.c_uint16),
    ('hskew', ctypes.c_uint16),
    ('vdisplay', ctypes.c_uint16),
    ('vsync_start', ctypes.c_uint16),
    ('vsync_end', ctypes.c_uint16),
    ('vtotal', ctypes.c_uint16),
    ('vscan', ctypes.c_uint16),
    ('vrefresh', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('name', ctypes.c_ubyte * 32),
]

class struct_drm_mode_card_res(Structure):
    pass

struct_drm_mode_card_res._pack_ = 1 # source:False
struct_drm_mode_card_res._fields_ = [
    ('fb_id_ptr', ctypes.c_uint64),
    ('crtc_id_ptr', ctypes.c_uint64),
    ('connector_id_ptr', ctypes.c_uint64),
    ('encoder_id_ptr', ctypes.c_uint64),
    ('count_fbs', ctypes.c_uint32),
    ('count_crtcs', ctypes.c_uint32),
    ('count_connectors', ctypes.c_uint32),
    ('count_encoders', ctypes.c_uint32),
    ('min_width', ctypes.c_uint32),
    ('max_width', ctypes.c_uint32),
    ('min_height', ctypes.c_uint32),
    ('max_height', ctypes.c_uint32),
]

DRM_IOCTL_MODE_GETRESOURCES = DRM_IOWR ( 0xA0 , struct_drm_mode_card_res ) # macro (from list)
class struct_drm_mode_crtc(Structure):
    pass

struct_drm_mode_crtc._pack_ = 1 # source:False
struct_drm_mode_crtc._fields_ = [
    ('set_connectors_ptr', ctypes.c_uint64),
    ('count_connectors', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('fb_id', ctypes.c_uint32),
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('gamma_size', ctypes.c_uint32),
    ('mode_valid', ctypes.c_uint32),
    ('mode', struct_drm_mode_modeinfo),
]

DRM_IOCTL_MODE_GETCRTC = DRM_IOWR ( 0xA1 , struct_drm_mode_crtc ) # macro (from list)
DRM_IOCTL_MODE_SETCRTC = DRM_IOWR ( 0xA2 , struct_drm_mode_crtc ) # macro (from list)
class struct_drm_mode_set_plane(Structure):
    pass

struct_drm_mode_set_plane._pack_ = 1 # source:False
struct_drm_mode_set_plane._fields_ = [
    ('plane_id', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('fb_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('crtc_x', ctypes.c_int32),
    ('crtc_y', ctypes.c_int32),
    ('crtc_w', ctypes.c_uint32),
    ('crtc_h', ctypes.c_uint32),
    ('src_x', ctypes.c_uint32),
    ('src_y', ctypes.c_uint32),
    ('src_h', ctypes.c_uint32),
    ('src_w', ctypes.c_uint32),
]

DRM_IOCTL_MODE_SETPLANE = DRM_IOWR ( 0xB7 , struct_drm_mode_set_plane ) # macro (from list)
class struct_drm_mode_get_plane(Structure):
    pass

struct_drm_mode_get_plane._pack_ = 1 # source:False
struct_drm_mode_get_plane._fields_ = [
    ('plane_id', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('fb_id', ctypes.c_uint32),
    ('possible_crtcs', ctypes.c_uint32),
    ('gamma_size', ctypes.c_uint32),
    ('count_format_types', ctypes.c_uint32),
    ('format_type_ptr', ctypes.c_uint64),
]

DRM_IOCTL_MODE_GETPLANE = DRM_IOWR ( 0xB6 , struct_drm_mode_get_plane ) # macro (from list)
class struct_drm_mode_get_plane_res(Structure):
    pass

struct_drm_mode_get_plane_res._pack_ = 1 # source:False
struct_drm_mode_get_plane_res._fields_ = [
    ('plane_id_ptr', ctypes.c_uint64),
    ('count_planes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

DRM_IOCTL_MODE_GETPLANERESOURCES = DRM_IOWR ( 0xB5 , struct_drm_mode_get_plane_res ) # macro (from list)
class struct_drm_mode_get_encoder(Structure):
    pass

struct_drm_mode_get_encoder._pack_ = 1 # source:False
struct_drm_mode_get_encoder._fields_ = [
    ('encoder_id', ctypes.c_uint32),
    ('encoder_type', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('possible_crtcs', ctypes.c_uint32),
    ('possible_clones', ctypes.c_uint32),
]

DRM_IOCTL_MODE_GETENCODER = DRM_IOWR ( 0xA6 , struct_drm_mode_get_encoder ) # macro (from list)

# values for enumeration 'drm_mode_subconnector'
drm_mode_subconnector__enumvalues = {
    0: 'DRM_MODE_SUBCONNECTOR_Automatic',
    0: 'DRM_MODE_SUBCONNECTOR_Unknown',
    3: 'DRM_MODE_SUBCONNECTOR_DVID',
    4: 'DRM_MODE_SUBCONNECTOR_DVIA',
    5: 'DRM_MODE_SUBCONNECTOR_Composite',
    6: 'DRM_MODE_SUBCONNECTOR_SVIDEO',
    8: 'DRM_MODE_SUBCONNECTOR_Component',
    9: 'DRM_MODE_SUBCONNECTOR_SCART',
}
DRM_MODE_SUBCONNECTOR_Automatic = 0
DRM_MODE_SUBCONNECTOR_Unknown = 0
DRM_MODE_SUBCONNECTOR_DVID = 3
DRM_MODE_SUBCONNECTOR_DVIA = 4
DRM_MODE_SUBCONNECTOR_Composite = 5
DRM_MODE_SUBCONNECTOR_SVIDEO = 6
DRM_MODE_SUBCONNECTOR_Component = 8
DRM_MODE_SUBCONNECTOR_SCART = 9
drm_mode_subconnector = ctypes.c_uint32 # enum
class struct_drm_mode_get_connector(Structure):
    pass

struct_drm_mode_get_connector._pack_ = 1 # source:False
struct_drm_mode_get_connector._fields_ = [
    ('encoders_ptr', ctypes.c_uint64),
    ('modes_ptr', ctypes.c_uint64),
    ('props_ptr', ctypes.c_uint64),
    ('prop_values_ptr', ctypes.c_uint64),
    ('count_modes', ctypes.c_uint32),
    ('count_props', ctypes.c_uint32),
    ('count_encoders', ctypes.c_uint32),
    ('encoder_id', ctypes.c_uint32),
    ('connector_id', ctypes.c_uint32),
    ('connector_type', ctypes.c_uint32),
    ('connector_type_id', ctypes.c_uint32),
    ('connection', ctypes.c_uint32),
    ('mm_width', ctypes.c_uint32),
    ('mm_height', ctypes.c_uint32),
    ('subpixel', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

DRM_IOCTL_MODE_GETCONNECTOR = DRM_IOWR ( 0xA7 , struct_drm_mode_get_connector ) # macro (from list)
class struct_drm_mode_property_enum(Structure):
    pass

struct_drm_mode_property_enum._pack_ = 1 # source:False
struct_drm_mode_property_enum._fields_ = [
    ('value', ctypes.c_uint64),
    ('name', ctypes.c_ubyte * 32),
]

class struct_drm_mode_get_property(Structure):
    pass

struct_drm_mode_get_property._pack_ = 1 # source:False
struct_drm_mode_get_property._fields_ = [
    ('values_ptr', ctypes.c_uint64),
    ('enum_blob_ptr', ctypes.c_uint64),
    ('prop_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('name', ctypes.c_ubyte * 32),
    ('count_values', ctypes.c_uint32),
    ('count_enum_blobs', ctypes.c_uint32),
]

DRM_IOCTL_MODE_GETPROPERTY = DRM_IOWR ( 0xAA , struct_drm_mode_get_property ) # macro (from list)
class struct_drm_mode_connector_set_property(Structure):
    pass

struct_drm_mode_connector_set_property._pack_ = 1 # source:False
struct_drm_mode_connector_set_property._fields_ = [
    ('value', ctypes.c_uint64),
    ('prop_id', ctypes.c_uint32),
    ('connector_id', ctypes.c_uint32),
]

DRM_IOCTL_MODE_SETPROPERTY = DRM_IOWR ( 0xAB , struct_drm_mode_connector_set_property ) # macro (from list)
class struct_drm_mode_obj_get_properties(Structure):
    pass

struct_drm_mode_obj_get_properties._pack_ = 1 # source:False
struct_drm_mode_obj_get_properties._fields_ = [
    ('props_ptr', ctypes.c_uint64),
    ('prop_values_ptr', ctypes.c_uint64),
    ('count_props', ctypes.c_uint32),
    ('obj_id', ctypes.c_uint32),
    ('obj_type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

DRM_IOCTL_MODE_OBJ_GETPROPERTIES = DRM_IOWR ( 0xB9 , struct_drm_mode_obj_get_properties ) # macro (from list)
class struct_drm_mode_obj_set_property(Structure):
    pass

struct_drm_mode_obj_set_property._pack_ = 1 # source:False
struct_drm_mode_obj_set_property._fields_ = [
    ('value', ctypes.c_uint64),
    ('prop_id', ctypes.c_uint32),
    ('obj_id', ctypes.c_uint32),
    ('obj_type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

DRM_IOCTL_MODE_OBJ_SETPROPERTY = DRM_IOWR ( 0xBA , struct_drm_mode_obj_set_property ) # macro (from list)
class struct_drm_mode_get_blob(Structure):
    pass

struct_drm_mode_get_blob._pack_ = 1 # source:False
struct_drm_mode_get_blob._fields_ = [
    ('blob_id', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
    ('data', ctypes.c_uint64),
]

DRM_IOCTL_MODE_GETPROPBLOB = DRM_IOWR ( 0xAC , struct_drm_mode_get_blob ) # macro (from list)
class struct_drm_mode_fb_cmd(Structure):
    pass

struct_drm_mode_fb_cmd._pack_ = 1 # source:False
struct_drm_mode_fb_cmd._fields_ = [
    ('fb_id', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
    ('bpp', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
]

DRM_IOCTL_MODE_GETFB = DRM_IOWR ( 0xAD , struct_drm_mode_fb_cmd ) # macro (from list)
DRM_IOCTL_MODE_ADDFB = DRM_IOWR ( 0xAE , struct_drm_mode_fb_cmd ) # macro (from list)
class struct_drm_mode_fb_cmd2(Structure):
    pass

struct_drm_mode_fb_cmd2._pack_ = 1 # source:False
struct_drm_mode_fb_cmd2._fields_ = [
    ('fb_id', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pixel_format', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('handles', ctypes.c_uint32 * 4),
    ('pitches', ctypes.c_uint32 * 4),
    ('offsets', ctypes.c_uint32 * 4),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('modifier', ctypes.c_uint64 * 4),
]

DRM_IOCTL_MODE_ADDFB2 = DRM_IOWR ( 0xB8 , struct_drm_mode_fb_cmd2 ) # macro (from list)
class struct_drm_mode_fb_dirty_cmd(Structure):
    pass

struct_drm_mode_fb_dirty_cmd._pack_ = 1 # source:False
struct_drm_mode_fb_dirty_cmd._fields_ = [
    ('fb_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('color', ctypes.c_uint32),
    ('num_clips', ctypes.c_uint32),
    ('clips_ptr', ctypes.c_uint64),
]

DRM_IOCTL_MODE_DIRTYFB = DRM_IOWR ( 0xB1 , struct_drm_mode_fb_dirty_cmd ) # macro (from list)
class struct_drm_mode_mode_cmd(Structure):
    pass

struct_drm_mode_mode_cmd._pack_ = 1 # source:False
struct_drm_mode_mode_cmd._fields_ = [
    ('connector_id', ctypes.c_uint32),
    ('mode', struct_drm_mode_modeinfo),
]

DRM_IOCTL_MODE_ATTACHMODE = DRM_IOWR ( 0xA8 , struct_drm_mode_mode_cmd ) # macro (from list)
DRM_IOCTL_MODE_DETACHMODE = DRM_IOWR ( 0xA9 , struct_drm_mode_mode_cmd ) # macro (from list)
class struct_drm_mode_cursor(Structure):
    pass

struct_drm_mode_cursor._pack_ = 1 # source:False
struct_drm_mode_cursor._fields_ = [
    ('flags', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
]

DRM_IOCTL_MODE_CURSOR = DRM_IOWR ( 0xA3 , struct_drm_mode_cursor ) # macro (from list)
class struct_drm_mode_cursor2(Structure):
    pass

struct_drm_mode_cursor2._pack_ = 1 # source:False
struct_drm_mode_cursor2._fields_ = [
    ('flags', ctypes.c_uint32),
    ('crtc_id', ctypes.c_uint32),
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('hot_x', ctypes.c_int32),
    ('hot_y', ctypes.c_int32),
]

DRM_IOCTL_MODE_CURSOR2 = DRM_IOWR ( 0xBB , struct_drm_mode_cursor2 ) # macro (from list)
class struct_drm_mode_crtc_lut(Structure):
    pass

struct_drm_mode_crtc_lut._pack_ = 1 # source:False
struct_drm_mode_crtc_lut._fields_ = [
    ('crtc_id', ctypes.c_uint32),
    ('gamma_size', ctypes.c_uint32),
    ('red', ctypes.c_uint64),
    ('green', ctypes.c_uint64),
    ('blue', ctypes.c_uint64),
]

DRM_IOCTL_MODE_GETGAMMA = DRM_IOWR ( 0xA4 , struct_drm_mode_crtc_lut ) # macro (from list)
DRM_IOCTL_MODE_SETGAMMA = DRM_IOWR ( 0xA5 , struct_drm_mode_crtc_lut ) # macro (from list)
class struct_drm_color_ctm(Structure):
    pass

struct_drm_color_ctm._pack_ = 1 # source:False
struct_drm_color_ctm._fields_ = [
    ('matrix', ctypes.c_uint64 * 9),
]

class struct_drm_color_lut(Structure):
    pass

struct_drm_color_lut._pack_ = 1 # source:False
struct_drm_color_lut._fields_ = [
    ('red', ctypes.c_uint16),
    ('green', ctypes.c_uint16),
    ('blue', ctypes.c_uint16),
    ('reserved', ctypes.c_uint16),
]

class struct_drm_mode_crtc_page_flip(Structure):
    pass

struct_drm_mode_crtc_page_flip._pack_ = 1 # source:False
struct_drm_mode_crtc_page_flip._fields_ = [
    ('crtc_id', ctypes.c_uint32),
    ('fb_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('user_data', ctypes.c_uint64),
]

DRM_IOCTL_MODE_PAGE_FLIP = DRM_IOWR ( 0xB0 , struct_drm_mode_crtc_page_flip ) # macro (from list)
class struct_drm_mode_crtc_page_flip_target(Structure):
    pass

struct_drm_mode_crtc_page_flip_target._pack_ = 1 # source:False
struct_drm_mode_crtc_page_flip_target._fields_ = [
    ('crtc_id', ctypes.c_uint32),
    ('fb_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('sequence', ctypes.c_uint32),
    ('user_data', ctypes.c_uint64),
]

class struct_drm_mode_create_dumb(Structure):
    pass

struct_drm_mode_create_dumb._pack_ = 1 # source:False
struct_drm_mode_create_dumb._fields_ = [
    ('height', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('bpp', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('handle', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
]

DRM_IOCTL_MODE_CREATE_DUMB = DRM_IOWR ( 0xB2 , struct_drm_mode_create_dumb ) # macro (from list)
class struct_drm_mode_map_dumb(Structure):
    pass

struct_drm_mode_map_dumb._pack_ = 1 # source:False
struct_drm_mode_map_dumb._fields_ = [
    ('handle', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
]

DRM_IOCTL_MODE_MAP_DUMB = DRM_IOWR ( 0xB3 , struct_drm_mode_map_dumb ) # macro (from list)
class struct_drm_mode_destroy_dumb(Structure):
    pass

struct_drm_mode_destroy_dumb._pack_ = 1 # source:False
struct_drm_mode_destroy_dumb._fields_ = [
    ('handle', ctypes.c_uint32),
]

DRM_IOCTL_MODE_DESTROY_DUMB = DRM_IOWR ( 0xB4 , struct_drm_mode_destroy_dumb ) # macro (from list)
class struct_drm_mode_atomic(Structure):
    pass

struct_drm_mode_atomic._pack_ = 1 # source:False
struct_drm_mode_atomic._fields_ = [
    ('flags', ctypes.c_uint32),
    ('count_objs', ctypes.c_uint32),
    ('objs_ptr', ctypes.c_uint64),
    ('count_props_ptr', ctypes.c_uint64),
    ('props_ptr', ctypes.c_uint64),
    ('prop_values_ptr', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
    ('user_data', ctypes.c_uint64),
]

DRM_IOCTL_MODE_ATOMIC = DRM_IOWR ( 0xBC , struct_drm_mode_atomic ) # macro (from list)
class struct_drm_format_modifier_blob(Structure):
    pass

struct_drm_format_modifier_blob._pack_ = 1 # source:False
struct_drm_format_modifier_blob._fields_ = [
    ('version', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('count_formats', ctypes.c_uint32),
    ('formats_offset', ctypes.c_uint32),
    ('count_modifiers', ctypes.c_uint32),
    ('modifiers_offset', ctypes.c_uint32),
]

class struct_drm_format_modifier(Structure):
    pass

struct_drm_format_modifier._pack_ = 1 # source:False
struct_drm_format_modifier._fields_ = [
    ('formats', ctypes.c_uint64),
    ('offset', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
    ('modifier', ctypes.c_uint64),
]

class struct_drm_mode_create_blob(Structure):
    pass

struct_drm_mode_create_blob._pack_ = 1 # source:False
struct_drm_mode_create_blob._fields_ = [
    ('data', ctypes.c_uint64),
    ('length', ctypes.c_uint32),
    ('blob_id', ctypes.c_uint32),
]

DRM_IOCTL_MODE_CREATEPROPBLOB = DRM_IOWR ( 0xBD , struct_drm_mode_create_blob ) # macro (from list)
class struct_drm_mode_destroy_blob(Structure):
    pass

struct_drm_mode_destroy_blob._pack_ = 1 # source:False
struct_drm_mode_destroy_blob._fields_ = [
    ('blob_id', ctypes.c_uint32),
]

DRM_IOCTL_MODE_DESTROYPROPBLOB = DRM_IOWR ( 0xBE , struct_drm_mode_destroy_blob ) # macro (from list)
class struct_drm_mode_create_lease(Structure):
    pass

struct_drm_mode_create_lease._pack_ = 1 # source:False
struct_drm_mode_create_lease._fields_ = [
    ('object_ids', ctypes.c_uint64),
    ('object_count', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('lessee_id', ctypes.c_uint32),
    ('fd', ctypes.c_uint32),
]

DRM_IOCTL_MODE_CREATE_LEASE = DRM_IOWR ( 0xC6 , struct_drm_mode_create_lease ) # macro (from list)
class struct_drm_mode_list_lessees(Structure):
    pass

struct_drm_mode_list_lessees._pack_ = 1 # source:False
struct_drm_mode_list_lessees._fields_ = [
    ('count_lessees', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
    ('lessees_ptr', ctypes.c_uint64),
]

DRM_IOCTL_MODE_LIST_LESSEES = DRM_IOWR ( 0xC7 , struct_drm_mode_list_lessees ) # macro (from list)
class struct_drm_mode_get_lease(Structure):
    pass

struct_drm_mode_get_lease._pack_ = 1 # source:False
struct_drm_mode_get_lease._fields_ = [
    ('count_objects', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
    ('objects_ptr', ctypes.c_uint64),
]

DRM_IOCTL_MODE_GET_LEASE = DRM_IOWR ( 0xC8 , struct_drm_mode_get_lease ) # macro (from list)
class struct_drm_mode_revoke_lease(Structure):
    pass

struct_drm_mode_revoke_lease._pack_ = 1 # source:False
struct_drm_mode_revoke_lease._fields_ = [
    ('lessee_id', ctypes.c_uint32),
]

DRM_IOCTL_MODE_REVOKE_LEASE = DRM_IOWR ( 0xC9 , struct_drm_mode_revoke_lease ) # macro (from list)
__user = True # macro
# __packed = ((packed)) # macro
RKNPU_OFFSET_VERSION = 0x0 # macro
RKNPU_OFFSET_VERSION_NUM = 0x4 # macro
RKNPU_OFFSET_PC_OP_EN = 0x8 # macro
RKNPU_OFFSET_PC_DATA_ADDR = 0x10 # macro
RKNPU_OFFSET_PC_DATA_AMOUNT = 0x14 # macro
RKNPU_OFFSET_PC_TASK_CONTROL = 0x30 # macro
RKNPU_OFFSET_PC_DMA_BASE_ADDR = 0x34 # macro
RKNPU_OFFSET_PC_TASK_STATUS = 0x3c # macro
RKNPU_OFFSET_INT_MASK = 0x20 # macro
RKNPU_OFFSET_INT_CLEAR = 0x24 # macro
RKNPU_OFFSET_INT_STATUS = 0x28 # macro
RKNPU_OFFSET_INT_RAW_STATUS = 0x2c # macro
RKNPU_OFFSET_CLR_ALL_RW_AMOUNT = 0x8010 # macro
RKNPU_OFFSET_DT_WR_AMOUNT = 0x8034 # macro
RKNPU_OFFSET_DT_RD_AMOUNT = 0x8038 # macro
RKNPU_OFFSET_WT_RD_AMOUNT = 0x803c # macro
RKNPU_OFFSET_ENABLE_MASK = 0xf008 # macro
RKNPU_INT_CLEAR = 0x1ffff # macro
RKNPU_PC_DATA_EXTRA_AMOUNT = 4 # macro
def RKNPU_STR_HELPER(x):  # macro
   return #x
def RKNPU_GET_DRV_VERSION_CODE(MAJOR, MINOR, PATCHLEVEL):  # macro
   return (MAJOR*10000+MINOR*100+PATCHLEVEL)
def RKNPU_GET_DRV_VERSION_MAJOR(CODE):  # macro
   return (CODE/10000)
def RKNPU_GET_DRV_VERSION_MINOR(CODE):  # macro
   return ((CODE%10000)/100)
def RKNPU_GET_DRV_VERSION_PATCHLEVEL(CODE):  # macro
   return (CODE%100)
RKNPU_ACTION = 0x00 # macro
RKNPU_SUBMIT = 0x01 # macro
RKNPU_MEM_CREATE = 0x02 # macro
RKNPU_MEM_MAP = 0x03 # macro
RKNPU_MEM_DESTROY = 0x04 # macro
RKNPU_MEM_SYNC = 0x05 # macro
RKNPU_IOC_MAGIC = 'r' # macro
def RKNPU_IOW(nr, type):  # macro
   return _IOW('r',nr,type)
def RKNPU_IOR(nr, type):  # macro
   return _IOR('r',nr,type)
def RKNPU_IOWR(nr, type):  # macro
   return _IOWR('r',nr,type)

# values for enumeration 'e_rknpu_mem_type'
e_rknpu_mem_type__enumvalues = {
    0: 'RKNPU_MEM_CONTIGUOUS',
    1: 'RKNPU_MEM_NON_CONTIGUOUS',
    0: 'RKNPU_MEM_NON_CACHEABLE',
    2: 'RKNPU_MEM_CACHEABLE',
    4: 'RKNPU_MEM_WRITE_COMBINE',
    8: 'RKNPU_MEM_KERNEL_MAPPING',
    16: 'RKNPU_MEM_IOMMU',
    32: 'RKNPU_MEM_ZEROING',
    64: 'RKNPU_MEM_SECURE',
    128: 'RKNPU_MEM_NON_DMA32',
    256: 'RKNPU_MEM_TRY_ALLOC_SRAM',
    511: 'RKNPU_MEM_MASK',
}
RKNPU_MEM_CONTIGUOUS = 0
RKNPU_MEM_NON_CONTIGUOUS = 1
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_MEM_CACHEABLE = 2
RKNPU_MEM_WRITE_COMBINE = 4
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_IOMMU = 16
RKNPU_MEM_ZEROING = 32
RKNPU_MEM_SECURE = 64
RKNPU_MEM_NON_DMA32 = 128
RKNPU_MEM_TRY_ALLOC_SRAM = 256
RKNPU_MEM_MASK = 511
e_rknpu_mem_type = ctypes.c_uint32 # enum

# values for enumeration 'e_rknpu_mem_sync_mode'
e_rknpu_mem_sync_mode__enumvalues = {
    1: 'RKNPU_MEM_SYNC_TO_DEVICE',
    2: 'RKNPU_MEM_SYNC_FROM_DEVICE',
    3: 'RKNPU_MEM_SYNC_MASK',
}
RKNPU_MEM_SYNC_TO_DEVICE = 1
RKNPU_MEM_SYNC_FROM_DEVICE = 2
RKNPU_MEM_SYNC_MASK = 3
e_rknpu_mem_sync_mode = ctypes.c_uint32 # enum

# values for enumeration 'e_rknpu_job_mode'
e_rknpu_job_mode__enumvalues = {
    0: 'RKNPU_JOB_SLAVE',
    1: 'RKNPU_JOB_PC',
    0: 'RKNPU_JOB_BLOCK',
    2: 'RKNPU_JOB_NONBLOCK',
    4: 'RKNPU_JOB_PINGPONG',
    8: 'RKNPU_JOB_FENCE_IN',
    16: 'RKNPU_JOB_FENCE_OUT',
    31: 'RKNPU_JOB_MASK',
}
RKNPU_JOB_SLAVE = 0
RKNPU_JOB_PC = 1
RKNPU_JOB_BLOCK = 0
RKNPU_JOB_NONBLOCK = 2
RKNPU_JOB_PINGPONG = 4
RKNPU_JOB_FENCE_IN = 8
RKNPU_JOB_FENCE_OUT = 16
RKNPU_JOB_MASK = 31
e_rknpu_job_mode = ctypes.c_uint32 # enum

# values for enumeration 'e_rknpu_action'
e_rknpu_action__enumvalues = {
    0: 'RKNPU_GET_HW_VERSION',
    1: 'RKNPU_GET_DRV_VERSION',
    2: 'RKNPU_GET_FREQ',
    3: 'RKNPU_SET_FREQ',
    4: 'RKNPU_GET_VOLT',
    5: 'RKNPU_SET_VOLT',
    6: 'RKNPU_ACT_RESET',
    7: 'RKNPU_GET_BW_PRIORITY',
    8: 'RKNPU_SET_BW_PRIORITY',
    9: 'RKNPU_GET_BW_EXPECT',
    10: 'RKNPU_SET_BW_EXPECT',
    11: 'RKNPU_GET_BW_TW',
    12: 'RKNPU_SET_BW_TW',
    13: 'RKNPU_ACT_CLR_TOTAL_RW_AMOUNT',
    14: 'RKNPU_GET_DT_WR_AMOUNT',
    15: 'RKNPU_GET_DT_RD_AMOUNT',
    16: 'RKNPU_GET_WT_RD_AMOUNT',
    17: 'RKNPU_GET_TOTAL_RW_AMOUNT',
    18: 'RKNPU_GET_IOMMU_EN',
    19: 'RKNPU_SET_PROC_NICE',
    20: 'RKNPU_POWER_ON',
    21: 'RKNPU_POWER_OFF',
    22: 'RKNPU_GET_TOTAL_SRAM_SIZE',
    23: 'RKNPU_GET_FREE_SRAM_SIZE',
}
RKNPU_GET_HW_VERSION = 0
RKNPU_GET_DRV_VERSION = 1
RKNPU_GET_FREQ = 2
RKNPU_SET_FREQ = 3
RKNPU_GET_VOLT = 4
RKNPU_SET_VOLT = 5
RKNPU_ACT_RESET = 6
RKNPU_GET_BW_PRIORITY = 7
RKNPU_SET_BW_PRIORITY = 8
RKNPU_GET_BW_EXPECT = 9
RKNPU_SET_BW_EXPECT = 10
RKNPU_GET_BW_TW = 11
RKNPU_SET_BW_TW = 12
RKNPU_ACT_CLR_TOTAL_RW_AMOUNT = 13
RKNPU_GET_DT_WR_AMOUNT = 14
RKNPU_GET_DT_RD_AMOUNT = 15
RKNPU_GET_WT_RD_AMOUNT = 16
RKNPU_GET_TOTAL_RW_AMOUNT = 17
RKNPU_GET_IOMMU_EN = 18
RKNPU_SET_PROC_NICE = 19
RKNPU_POWER_ON = 20
RKNPU_POWER_OFF = 21
RKNPU_GET_TOTAL_SRAM_SIZE = 22
RKNPU_GET_FREE_SRAM_SIZE = 23
e_rknpu_action = ctypes.c_uint32 # enum
class struct_rknpu_mem_create(Structure):
    pass

struct_rknpu_mem_create._pack_ = 1 # source:False
struct_rknpu_mem_create._fields_ = [
    ('handle', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('obj_addr', ctypes.c_uint64),
    ('dma_addr', ctypes.c_uint64),
    ('sram_size', ctypes.c_uint64),
]

DRM_IOCTL_RKNPU_MEM_CREATE = DRM_IOWR ( 0x40 + 0x02 , struct_rknpu_mem_create ) # macro (from list)
IOCTL_RKNPU_MEM_CREATE = RKNPU_IOWR ( 0x02 , struct_rknpu_mem_create ) # macro (from list)
class struct_rknpu_mem_map(Structure):
    pass

struct_rknpu_mem_map._pack_ = 1 # source:False
struct_rknpu_mem_map._fields_ = [
    ('handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
]

DRM_IOCTL_RKNPU_MEM_MAP = DRM_IOWR ( 0x40 + 0x03 , struct_rknpu_mem_map ) # macro (from list)
IOCTL_RKNPU_MEM_MAP = RKNPU_IOWR ( 0x03 , struct_rknpu_mem_map ) # macro (from list)
class struct_rknpu_mem_destroy(Structure):
    pass

struct_rknpu_mem_destroy._pack_ = 1 # source:False
struct_rknpu_mem_destroy._fields_ = [
    ('handle', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('obj_addr', ctypes.c_uint64),
]

DRM_IOCTL_RKNPU_MEM_DESTROY = DRM_IOWR ( 0x40 + 0x04 , struct_rknpu_mem_destroy ) # macro (from list)
IOCTL_RKNPU_MEM_DESTROY = RKNPU_IOWR ( 0x04 , struct_rknpu_mem_destroy ) # macro (from list)
class struct_rknpu_mem_sync(Structure):
    pass

struct_rknpu_mem_sync._pack_ = 1 # source:False
struct_rknpu_mem_sync._fields_ = [
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('obj_addr', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

DRM_IOCTL_RKNPU_MEM_SYNC = DRM_IOWR ( 0x40 + 0x05 , struct_rknpu_mem_sync ) # macro (from list)
IOCTL_RKNPU_MEM_SYNC = RKNPU_IOWR ( 0x05 , struct_rknpu_mem_sync ) # macro (from list)
class struct_rknpu_task(Structure):
    pass

struct_rknpu_task._pack_ = 1 # source:True
struct_rknpu_task._fields_ = [
    ('flags', ctypes.c_uint32),
    ('op_idx', ctypes.c_uint32),
    ('enable_mask', ctypes.c_uint32),
    ('int_mask', ctypes.c_uint32),
    ('int_clear', ctypes.c_uint32),
    ('int_status', ctypes.c_uint32),
    ('regcfg_amount', ctypes.c_uint32),
    ('regcfg_offset', ctypes.c_uint32),
    ('regcmd_addr', ctypes.c_uint64),
]

class struct_rknpu_subcore_task(Structure):
    pass

struct_rknpu_subcore_task._pack_ = 1 # source:False
struct_rknpu_subcore_task._fields_ = [
    ('task_start', ctypes.c_uint32),
    ('task_number', ctypes.c_uint32),
]

class struct_rknpu_submit(Structure):
    pass

struct_rknpu_submit._pack_ = 1 # source:False
struct_rknpu_submit._fields_ = [
    ('flags', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('task_start', ctypes.c_uint32),
    ('task_number', ctypes.c_uint32),
    ('task_counter', ctypes.c_uint32),
    ('priority', ctypes.c_int32),
    ('task_obj_addr', ctypes.c_uint64),
    ('regcfg_obj_addr', ctypes.c_uint64),
    ('task_base_addr', ctypes.c_uint64),
    ('user_data', ctypes.c_uint64),
    ('core_mask', ctypes.c_uint32),
    ('fence_fd', ctypes.c_int32),
    ('subcore_task', struct_rknpu_subcore_task * 5),
]

DRM_IOCTL_RKNPU_SUBMIT = DRM_IOWR ( 0x40 + 0x01 , struct_rknpu_submit ) # macro (from list)
IOCTL_RKNPU_SUBMIT = RKNPU_IOWR ( 0x01 , struct_rknpu_submit ) # macro (from list)
class struct_rknpu_action(Structure):
    pass

struct_rknpu_action._pack_ = 1 # source:False
struct_rknpu_action._fields_ = [
    ('flags', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

DRM_IOCTL_RKNPU_ACTION = DRM_IOWR ( 0x40 + 0x00 , struct_rknpu_action ) # macro (from list)
IOCTL_RKNPU_ACTION = RKNPU_IOWR ( 0x00 , struct_rknpu_action ) # macro (from list)
DMA_HEAP_IOC_MAGIC = 'H' # macro
DMA_BUF_SYNC_READ = (1<<0) # macro
DMA_BUF_SYNC_WRITE = (2<<0) # macro
DMA_BUF_SYNC_RW = ((1<<0)|(2<<0)) # macro
DMA_BUF_SYNC_START = (0<<2) # macro
DMA_BUF_SYNC_END = (1<<2) # macro
DMA_BUF_BASE = 'b' # macro
CMA_HEAP_SIZE = (1024*1024) # macro
GGML_RKNPU2_MAX_MATMUL_KERNELS = 16 # macro
class struct_ggml_sync_data_pack(Structure):
    pass

struct_ggml_sync_data_pack._pack_ = 1 # source:False
struct_ggml_sync_data_pack._fields_ = [
    ('data', ctypes.c_uint64),
]

DMA_BUF_IOCTL_SYNC = _IOW ( 'b' , 0 , struct_ggml_sync_data_pack ) # macro (from list)
class struct_ggml_rknpu2_data_pack(Structure):
    pass

class struct__rknn_tensor_memory(Structure):
    pass


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
struct_ggml_rknpu2_data_pack._pack_ = 1 # source:False
struct_ggml_rknpu2_data_pack._fields_ = [
    ('type', _rknn_tensor_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('ordered_data', ctypes.POINTER(None)),
    ('initialized', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('B', ctypes.POINTER(struct__rknn_tensor_memory)),
]

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

class struct_ggml_rknpu2_matmul_kernel(Structure):
    pass

class struct_rknn_matmul_info_t(Structure):
    pass


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
struct_rknn_matmul_info_t._pack_ = 1 # source:False
struct_rknn_matmul_info_t._fields_ = [
    ('M', ctypes.c_int32),
    ('K', ctypes.c_int32),
    ('N', ctypes.c_int32),
    ('type', _rknn_matmul_type),
    ('B_layout', ctypes.c_int16),
    ('B_quant_type', ctypes.c_int16),
    ('AC_layout', ctypes.c_int16),
    ('AC_quant_type', ctypes.c_int16),
    ('iommu_domain_id', ctypes.c_int32),
    ('group_size', ctypes.c_int16),
    ('reserved', ctypes.c_byte * 34),
]

class struct__rknn_matmul_io_attr(Structure):
    pass

class struct__rknn_matmul_tensor_attr(Structure):
    pass

struct__rknn_matmul_tensor_attr._pack_ = 1 # source:False
struct__rknn_matmul_tensor_attr._fields_ = [
    ('name', ctypes.c_ubyte * 256),
    ('n_dims', ctypes.c_uint32),
    ('dims', ctypes.c_uint32 * 16),
    ('size', ctypes.c_uint32),
    ('type', _rknn_tensor_type),
]

struct__rknn_matmul_io_attr._pack_ = 1 # source:False
struct__rknn_matmul_io_attr._fields_ = [
    ('A', struct__rknn_matmul_tensor_attr),
    ('B', struct__rknn_matmul_tensor_attr),
    ('C', struct__rknn_matmul_tensor_attr),
]

struct_ggml_rknpu2_matmul_kernel._pack_ = 1 # source:False
struct_ggml_rknpu2_matmul_kernel._fields_ = [
    ('matmul_info', struct_rknn_matmul_info_t),
    ('matmul_ctx', ctypes.c_uint64),
    ('matmul_io_attr', struct__rknn_matmul_io_attr),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('A', ctypes.POINTER(struct__rknn_tensor_memory)),
    ('C', ctypes.POINTER(struct__rknn_tensor_memory)),
]

class struct_dma_heap_allocation_data(Structure):
    pass

struct_dma_heap_allocation_data._pack_ = 1 # source:False
struct_dma_heap_allocation_data._fields_ = [
    ('len', ctypes.c_uint64),
    ('fd', ctypes.c_uint32),
    ('fd_flags', ctypes.c_uint32),
    ('heap_flags', ctypes.c_uint64),
]

DMA_HEAP_IOCTL_ALLOC = _IOWR ( 'H' , 0x0 , struct_dma_heap_allocation_data ) # macro (from list)
NPU_CNA_H = True # macro
class struct_npu_cna_desc(Structure):
    pass

struct_npu_cna_desc._pack_ = 1 # source:False
struct_npu_cna_desc._fields_ = [
    ('enable', ctypes.c_ubyte),
    ('conv_mode', ctypes.c_ubyte),
    ('in_precision', ctypes.c_ubyte),
    ('proc_precision', ctypes.c_ubyte),
    ('kernel_groups', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('feature_grains', ctypes.c_uint16),
    ('conv_y_stride', ctypes.c_ubyte),
    ('conv_x_stride', ctypes.c_ubyte),
    ('datain_width', ctypes.c_uint16),
    ('datain_height', ctypes.c_uint16),
    ('datain_channel', ctypes.c_uint16),
    ('dataout_width', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('dataout_atomics', ctypes.c_uint32),
    ('weight_bytes', ctypes.c_uint32),
    ('weight_bytes_per_kernel', ctypes.c_uint32),
    ('weight_width', ctypes.c_ubyte),
    ('weight_height', ctypes.c_ubyte),
    ('weight_kernels', ctypes.c_uint16),
    ('weight_bank', ctypes.c_ubyte),
    ('data_bank', ctypes.c_ubyte),
    ('data_entries', ctypes.c_uint16),
    ('data_sign', ctypes.c_ubyte),
    ('cvt_type', ctypes.c_ubyte),
    ('cvt_bypass', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte),
    ('cvt_scale0', ctypes.c_uint16),
    ('cvt_scale1', ctypes.c_uint16),
    ('cvt_scale2', ctypes.c_uint16),
    ('cvt_scale3', ctypes.c_uint16),
    ('fc_skip_en', ctypes.c_ubyte),
    ('PADDING_3', ctypes.c_ubyte),
    ('data_offset', ctypes.c_uint16),
    ('pad_left', ctypes.c_ubyte),
    ('pad_top', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 2),
    ('feature_base_addr', ctypes.c_uint32),
    ('weight_offset', ctypes.c_uint16),
    ('weight_burst_len', ctypes.c_ubyte),
    ('data_burst_len', ctypes.c_ubyte),
    ('line_stride', ctypes.c_uint32),
    ('surf_stride', ctypes.c_int32),
    ('dma_width', ctypes.c_uint16),
    ('dma_height', ctypes.c_uint16),
    ('dma_channel', ctypes.c_uint16),
    ('PADDING_5', ctypes.c_ubyte * 2),
    ('decompress_addr0', ctypes.c_uint32),
    ('dataout_height', ctypes.c_uint16),
    ('PADDING_6', ctypes.c_ubyte * 2),
]

npu_cna_desc = struct_npu_cna_desc
class struct_npu_core_desc(Structure):
    pass

struct_npu_core_desc._pack_ = 1 # source:False
struct_npu_core_desc._fields_ = [
    ('proc_precision', ctypes.c_ubyte),
    ('qd_en', ctypes.c_ubyte),
    ('dataout_height', ctypes.c_uint16),
    ('dataout_width', ctypes.c_uint16),
    ('dataout_channel', ctypes.c_uint16),
]

npu_core_desc = struct_npu_core_desc
class struct_nup_pc_desc(Structure):
    pass

struct_nup_pc_desc._pack_ = 1 # source:False
struct_nup_pc_desc._fields_ = [
    ('pc_source_addr', ctypes.c_uint32),
    ('pc_data_amount', ctypes.c_uint32),
]

npu_pc_desc = struct_nup_pc_desc
class struct_npu_cna_core_task(Structure):
    pass

struct_npu_cna_core_task._pack_ = 1 # source:False
struct_npu_cna_core_task._fields_ = [
    ('ops', ctypes.c_uint64 * 112),
]

npu_cna_core_task = struct_npu_cna_core_task
NPU_DPU_H = True # macro
class struct_npu_dpu_desc(Structure):
    pass

struct_npu_dpu_desc._pack_ = 1 # source:False
struct_npu_dpu_desc._fields_ = [
    ('burst_len', ctypes.c_ubyte),
    ('conv_mode', ctypes.c_ubyte),
    ('output_mode', ctypes.c_ubyte),
    ('flying_mode', ctypes.c_ubyte),
    ('out_precision', ctypes.c_ubyte),
    ('in_precision', ctypes.c_ubyte),
    ('proc_precision', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('dst_base_addr', ctypes.c_uint32),
    ('dst_surf_stride', ctypes.c_uint32),
    ('width', ctypes.c_uint16),
    ('height', ctypes.c_uint16),
    ('channel', ctypes.c_uint16),
    ('bs_bypass', ctypes.c_ubyte),
    ('bs_alu_bypass', ctypes.c_ubyte),
    ('bs_mul_bypass', ctypes.c_ubyte),
    ('bs_relu_bypass', ctypes.c_ubyte),
    ('od_bypass', ctypes.c_ubyte),
    ('size_e_2', ctypes.c_ubyte),
    ('size_e_1', ctypes.c_ubyte),
    ('size_e_0', ctypes.c_ubyte),
    ('channel_wdma', ctypes.c_uint16),
    ('height_wdma', ctypes.c_uint16),
    ('width_wdma', ctypes.c_uint16),
    ('bn_relu_bypass', ctypes.c_ubyte),
    ('bn_mul_bypass', ctypes.c_ubyte),
    ('bn_alu_bypass', ctypes.c_ubyte),
    ('bn_bypass', ctypes.c_ubyte),
    ('ew_bypass', ctypes.c_ubyte),
    ('ew_op_bypass', ctypes.c_ubyte),
    ('ew_lut_bypass', ctypes.c_ubyte),
    ('ew_op_cvt_bypass', ctypes.c_ubyte),
    ('ew_relu_bypass', ctypes.c_ubyte),
    ('fp32tofp16_en', ctypes.c_ubyte),
    ('out_cvt_scale', ctypes.c_uint16),
    ('surf_add', ctypes.c_uint32),
]

npu_dpu_desc = struct_npu_dpu_desc
NPU_HW_H = True # macro
PC_OPERATION_ENABLE = 0x0008 # macro
PC_BASE_ADDRESS = 0x0010 # macro
PC_REGISTER_AMOUNTS = 0x0014 # macro
CNA_S_POINTER = 0x1004 # macro
CNA_CONV_CON1 = 0x100C # macro
CNA_CONV_CON2 = 0x1010 # macro
CNA_CONV_CON3 = 0x1014 # macro
CNA_DATA_SIZE0 = 0x1020 # macro
CNA_DATA_SIZE1 = 0x1024 # macro
CNA_DATA_SIZE2 = 0x1028 # macro
CNA_DATA_SIZE3 = 0x102C # macro
CNA_WEIGHT_SIZE0 = 0x1030 # macro
CNA_WEIGHT_SIZE1 = 0x1034 # macro
CNA_WEIGHT_SIZE2 = 0x1038 # macro
CNA_CBUF_CON0 = 0x1040 # macro
CNA_CBUF_CON1 = 0x1044 # macro
CNA_CVT_CON0 = 0x104C # macro
CNA_CVT_CON1 = 0x1050 # macro
CNA_CVT_CON2 = 0x1054 # macro
CNA_CVT_CON3 = 0x1058 # macro
CNA_CVT_CON4 = 0x105C # macro
CNA_FC_CON0 = 0x1060 # macro
CNA_FC_CON1 = 0x1064 # macro
CNA_PAD_CON0 = 0x1068 # macro
CNA_FEATURE_DATA_ADDR = 0x1070 # macro
CNA_FC_CON2 = 0x1074 # macro
CNA_DMA_CON0 = 0x1078 # macro
CNA_DMA_CON1 = 0x107C # macro
CNA_DMA_CON2 = 0x1080 # macro
CNA_FC_DATA_SIZE0 = 0x1084 # macro
CNA_FC_DATA_SIZE1 = 0x1088 # macro
CNA_DCOMP_CTRL = 0x1100 # macro
CNA_DCOMP_REGNUM = 0x1104 # macro
CNA_DCOMP_ADDR0 = 0x1110 # macro
CNA_DCOMP_AMOUNT = 0x1140 # macro
CNA_DCOMP_AMOUNT1 = 0x1144 # macro
CNA_DCOMP_AMOUNT2 = 0x1148 # macro
CNA_DCOMP_AMOUNT3 = 0x114C # macro
CNA_DCOMP_AMOUNT4 = 0x1150 # macro
CNA_DCOMP_AMOUNT5 = 0x1154 # macro
CNA_DCOMP_AMOUNT6 = 0x1158 # macro
CNA_DCOMP_AMOUNT7 = 0x115C # macro
CNA_DCOMP_AMOUNT8 = 0x1160 # macro
CNA_DCOMP_AMOUNT9 = 0x1164 # macro
CNA_DCOMP_AMOUNT10 = 0x1168 # macro
CNA_DCOMP_AMOUNT11 = 0x116C # macro
CNA_DCOMP_AMOUNT12 = 0x1170 # macro
CNA_DCOMP_AMOUNT13 = 0x1174 # macro
CNA_DCOMP_AMOUNT14 = 0x1178 # macro
CNA_DCOMP_AMOUNT15 = 0x117C # macro
CNA_CVT_CON5 = 0x1180 # macro
CNA_PAD_CON1 = 0x1184 # macro
CORE_S_POINTER = 0x3004 # macro
CORE_MISC_CFG = 0x3010 # macro
CORE_DATAOUT_SIZE_0 = 0x3014 # macro
CORE_DATAOUT_SIZE_1 = 0x3018 # macro
CORE_CLIP_TRUNCATE = 0x301C # macro
CORE_3030 = 0x3030 # macro
DPU_S_POINTER = 0x4004 # macro
DPU_FEATURE_MODE_CFG = 0x400C # macro
DPU_DATA_FORMAT = 0x4010 # macro
DPU_OFFSET_PEND = 0x4014 # macro
DPU_DST_BASE_ADD = 0x4020 # macro
DPU_DST_SURF_STRIDE = 0x4024 # macro
DPU_DATA_CUBE_WIDTH = 0x4030 # macro
DPU_DATA_CUBE_HEIGHT = 0x4034 # macro
DPU_DATA_CUBE_NOTCH_ADDR = 0x4038 # macro
DPU_DATA_CUBE_CHANNEL = 0x403C # macro
DPU_BS_CFG = 0x4040 # macro
DPU_BS_ALU_CFG = 0x4044 # macro
DPU_BS_MUL_CFG = 0x4048 # macro
DPU_BS_RELUX_CMP_VALUE = 0x404C # macro
DPU_BS_OW_CFG = 0x4050 # macro
DPU_BS_OW_OP = 0x4054 # macro
DPU_WDMA_SIZE_0 = 0x4058 # macro
DPU_WDMA_SIZE_1 = 0x405C # macro
DPU_BN_CFG = 0x4060 # macro
DPU_BN_ALU_CFG = 0x4064 # macro
DPU_BN_MUL_CFG = 0x4068 # macro
DPU_BN_RELUX_CMP_VALUE = 0x406C # macro
DPU_EW_CFG = 0x4070 # macro
DPU_EW_CVT_OFFSET_VALUE = 0x4074 # macro
DPU_EW_CVT_SCALE_VALUE = 0x4078 # macro
DPU_EW_RELUX_CMP_VALUE = 0x407C # macro
DPU_OUT_CVT_OFFSET = 0x4080 # macro
DPU_OUT_CVT_SCALE = 0x4084 # macro
DPU_OUT_CVT_SHIFT = 0x4088 # macro
DPU_EW_OP_VALUE_0 = 0x4090 # macro
DPU_EW_OP_VALUE_1 = 0x4094 # macro
DPU_EW_OP_VALUE_2 = 0x4098 # macro
DPU_EW_OP_VALUE_3 = 0x409C # macro
DPU_EW_OP_VALUE_4 = 0x40A0 # macro
DPU_EW_OP_VALUE_5 = 0x40A4 # macro
DPU_EW_OP_VALUE_6 = 0x40A8 # macro
DPU_EW_OP_VALUE_7 = 0x40AC # macro
DPU_SURFACE_ADD = 0x40C0 # macro
DPU_40C4 = 0x40C4 # macro
DPU_LUT_ACCESS_CFG = 0x4100 # macro
DPU_LUT_ACCESS_DATA = 0x4104 # macro
DPU_LUT_CFG = 0x4108 # macro
DPU_LUT_INFO = 0x410C # macro
DPU_LUT_LE_START = 0x4110 # macro
DPU_LUT_LE_END = 0x4114 # macro
DPU_LUT_LO_START = 0x4118 # macro
DPU_LUT_LO_END = 0x411C # macro
DPU_LUT_LE_SLOPE_SCALE = 0x4120 # macro
DPU_LUT_LE_SLOPE_SHIFT = 0x4124 # macro
DPU_LUT_LO_SLOPE_SCALE = 0x4128 # macro
DPU_LUT_LO_SLOPE_SHIFT = 0x412C # macro
BLOCK_PC = 0x0100 # macro
BLOCK_CNA = 0x0200 # macro
BLOCK_CORE = 0x0800 # macro
BLOCK_DPU = 0x1000 # macro
BLOCK_DPU_RDMA = 0x2000 # macro
BLOCK_PPU = 0x4000 # macro
BLOCK_PPU_RDMA = 0x8000 # macro
PC_OP_01 = 0x01 # macro
PC_OP_40 = 0x40 # macro
PC_OP_ENABLE = 0x80 # macro
OP_REG_PC = (0x0100|0x01) # macro
OP_REG_CNA = (0x0200|0x01) # macro
OP_REG_CORE = (0x0800|0x01) # macro
OP_REG_DPU = (0x1000|0x01) # macro
OP_40 = (0x40|0x01) # macro
OP_ENABLE = (0x80|0x01) # macro
OP_NONE = 0x0 # macro
PC_ENABLE = 0x01 # macro
PC_ENABLE_CNA = 0x04 # macro
PC_ENABLE_DPU = 0x08 # macro
PC_ENABLE_PPU = 0x10 # macro
# def NPUOP(op, value, reg):  # macro
#    return (((uint64_t)(op&0xffff))<<48)|(((uint64_t)(value&0xffffffff))<<16)|(uint64_t)(reg&0xffff)
NPU_CBUF_BANK_SIZE = 32768 # macro
NPU_CBUF_BANKS = 12 # macro

# values for enumeration 'c__Ea_direct_convolution'
c__Ea_direct_convolution__enumvalues = {
    0: 'direct_convolution',
}
direct_convolution = 0
c__Ea_direct_convolution = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_precision_int8'
c__Ea_precision_int8__enumvalues = {
    0: 'precision_int8',
    2: 'precision_float16',
    4: 'precision_int32',
    5: 'precision_float32',
}
precision_int8 = 0
precision_float16 = 2
precision_int32 = 4
precision_float32 = 5
c__Ea_precision_int8 = ctypes.c_uint32 # enum
__all__ = \
    ['BLOCK_CNA', 'BLOCK_CORE', 'BLOCK_DPU', 'BLOCK_DPU_RDMA',
    'BLOCK_PC', 'BLOCK_PPU', 'BLOCK_PPU_RDMA', 'CMA_HEAP_SIZE',
    'CNA_CBUF_CON0', 'CNA_CBUF_CON1', 'CNA_CONV_CON1',
    'CNA_CONV_CON2', 'CNA_CONV_CON3', 'CNA_CVT_CON0', 'CNA_CVT_CON1',
    'CNA_CVT_CON2', 'CNA_CVT_CON3', 'CNA_CVT_CON4', 'CNA_CVT_CON5',
    'CNA_DATA_SIZE0', 'CNA_DATA_SIZE1', 'CNA_DATA_SIZE2',
    'CNA_DATA_SIZE3', 'CNA_DCOMP_ADDR0', 'CNA_DCOMP_AMOUNT',
    'CNA_DCOMP_AMOUNT1', 'CNA_DCOMP_AMOUNT10', 'CNA_DCOMP_AMOUNT11',
    'CNA_DCOMP_AMOUNT12', 'CNA_DCOMP_AMOUNT13', 'CNA_DCOMP_AMOUNT14',
    'CNA_DCOMP_AMOUNT15', 'CNA_DCOMP_AMOUNT2', 'CNA_DCOMP_AMOUNT3',
    'CNA_DCOMP_AMOUNT4', 'CNA_DCOMP_AMOUNT5', 'CNA_DCOMP_AMOUNT6',
    'CNA_DCOMP_AMOUNT7', 'CNA_DCOMP_AMOUNT8', 'CNA_DCOMP_AMOUNT9',
    'CNA_DCOMP_CTRL', 'CNA_DCOMP_REGNUM', 'CNA_DMA_CON0',
    'CNA_DMA_CON1', 'CNA_DMA_CON2', 'CNA_FC_CON0', 'CNA_FC_CON1',
    'CNA_FC_CON2', 'CNA_FC_DATA_SIZE0', 'CNA_FC_DATA_SIZE1',
    'CNA_FEATURE_DATA_ADDR', 'CNA_PAD_CON0', 'CNA_PAD_CON1',
    'CNA_S_POINTER', 'CNA_WEIGHT_SIZE0', 'CNA_WEIGHT_SIZE1',
    'CNA_WEIGHT_SIZE2', 'CORE_3030', 'CORE_CLIP_TRUNCATE',
    'CORE_DATAOUT_SIZE_0', 'CORE_DATAOUT_SIZE_1', 'CORE_MISC_CFG',
    'CORE_S_POINTER', 'DMA_BUF_BASE', 'DMA_BUF_SYNC_END',
    'DMA_BUF_SYNC_READ', 'DMA_BUF_SYNC_RW', 'DMA_BUF_SYNC_START',
    'DMA_BUF_SYNC_WRITE', 'DMA_HEAP_IOC_MAGIC', 'DPU_40C4',
    'DPU_BN_ALU_CFG', 'DPU_BN_CFG', 'DPU_BN_MUL_CFG',
    'DPU_BN_RELUX_CMP_VALUE', 'DPU_BS_ALU_CFG', 'DPU_BS_CFG',
    'DPU_BS_MUL_CFG', 'DPU_BS_OW_CFG', 'DPU_BS_OW_OP',
    'DPU_BS_RELUX_CMP_VALUE', 'DPU_DATA_CUBE_CHANNEL',
    'DPU_DATA_CUBE_HEIGHT', 'DPU_DATA_CUBE_NOTCH_ADDR',
    'DPU_DATA_CUBE_WIDTH', 'DPU_DATA_FORMAT', 'DPU_DST_BASE_ADD',
    'DPU_DST_SURF_STRIDE', 'DPU_EW_CFG', 'DPU_EW_CVT_OFFSET_VALUE',
    'DPU_EW_CVT_SCALE_VALUE', 'DPU_EW_OP_VALUE_0',
    'DPU_EW_OP_VALUE_1', 'DPU_EW_OP_VALUE_2', 'DPU_EW_OP_VALUE_3',
    'DPU_EW_OP_VALUE_4', 'DPU_EW_OP_VALUE_5', 'DPU_EW_OP_VALUE_6',
    'DPU_EW_OP_VALUE_7', 'DPU_EW_RELUX_CMP_VALUE',
    'DPU_FEATURE_MODE_CFG', 'DPU_LUT_ACCESS_CFG',
    'DPU_LUT_ACCESS_DATA', 'DPU_LUT_CFG', 'DPU_LUT_INFO',
    'DPU_LUT_LE_END', 'DPU_LUT_LE_SLOPE_SCALE',
    'DPU_LUT_LE_SLOPE_SHIFT', 'DPU_LUT_LE_START', 'DPU_LUT_LO_END',
    'DPU_LUT_LO_SLOPE_SCALE', 'DPU_LUT_LO_SLOPE_SHIFT',
    'DPU_LUT_LO_START', 'DPU_OFFSET_PEND', 'DPU_OUT_CVT_OFFSET',
    'DPU_OUT_CVT_SCALE', 'DPU_OUT_CVT_SHIFT', 'DPU_SURFACE_ADD',
    'DPU_S_POINTER', 'DPU_WDMA_SIZE_0', 'DPU_WDMA_SIZE_1',
    'DRM_CAP_ADDFB2_MODIFIERS', 'DRM_CAP_ASYNC_PAGE_FLIP',
    'DRM_CAP_CRTC_IN_VBLANK_EVENT', 'DRM_CAP_CURSOR_HEIGHT',
    'DRM_CAP_CURSOR_WIDTH', 'DRM_CAP_DUMB_BUFFER',
    'DRM_CAP_DUMB_PREFERRED_DEPTH', 'DRM_CAP_DUMB_PREFER_SHADOW',
    'DRM_CAP_PAGE_FLIP_TARGET', 'DRM_CAP_PRIME', 'DRM_CAP_SYNCOBJ',
    'DRM_CAP_TIMESTAMP_MONOTONIC', 'DRM_CAP_VBLANK_HIGH_CRTC',
    'DRM_CLIENT_CAP_ASPECT_RATIO', 'DRM_CLIENT_CAP_ATOMIC',
    'DRM_CLIENT_CAP_STEREO_3D', 'DRM_CLIENT_CAP_UNIVERSAL_PLANES',
    'DRM_CLIENT_CAP_WRITEBACK_CONNECTORS', 'DRM_COMMAND_BASE',
    'DRM_COMMAND_END', 'DRM_CONNECTOR_NAME_LEN',
    'DRM_CRTC_SEQUENCE_NEXT_ON_MISS', 'DRM_CRTC_SEQUENCE_RELATIVE',
    'DRM_DISPLAY_INFO_LEN', 'DRM_DISPLAY_MODE_LEN',
    'DRM_DRAWABLE_CLIPRECTS', 'DRM_EVENT_CRTC_SEQUENCE',
    'DRM_EVENT_FLIP_COMPLETE', 'DRM_EVENT_VBLANK',
    'DRM_IOCTL_AGP_ACQUIRE', 'DRM_IOCTL_AGP_RELEASE',
    'DRM_IOCTL_BASE', 'DRM_IOCTL_DROP_MASTER', 'DRM_IOCTL_SET_MASTER',
    'DRM_MAX_ORDER', 'DRM_MIN_ORDER', 'DRM_MODE_ATOMIC_ALLOW_MODESET',
    'DRM_MODE_ATOMIC_FLAGS', 'DRM_MODE_ATOMIC_NONBLOCK',
    'DRM_MODE_ATOMIC_TEST_ONLY', 'DRM_MODE_CONNECTOR_9PinDIN',
    'DRM_MODE_CONNECTOR_Component', 'DRM_MODE_CONNECTOR_Composite',
    'DRM_MODE_CONNECTOR_DPI', 'DRM_MODE_CONNECTOR_DSI',
    'DRM_MODE_CONNECTOR_DVIA', 'DRM_MODE_CONNECTOR_DVID',
    'DRM_MODE_CONNECTOR_DVII', 'DRM_MODE_CONNECTOR_DisplayPort',
    'DRM_MODE_CONNECTOR_HDMIA', 'DRM_MODE_CONNECTOR_HDMIB',
    'DRM_MODE_CONNECTOR_LVDS', 'DRM_MODE_CONNECTOR_SVIDEO',
    'DRM_MODE_CONNECTOR_TV', 'DRM_MODE_CONNECTOR_Unknown',
    'DRM_MODE_CONNECTOR_VGA', 'DRM_MODE_CONNECTOR_VIRTUAL',
    'DRM_MODE_CONNECTOR_WRITEBACK', 'DRM_MODE_CONNECTOR_eDP',
    'DRM_MODE_CONTENT_PROTECTION_DESIRED',
    'DRM_MODE_CONTENT_PROTECTION_ENABLED',
    'DRM_MODE_CONTENT_PROTECTION_UNDESIRED',
    'DRM_MODE_CONTENT_TYPE_CINEMA', 'DRM_MODE_CONTENT_TYPE_GAME',
    'DRM_MODE_CONTENT_TYPE_GRAPHICS', 'DRM_MODE_CONTENT_TYPE_NO_DATA',
    'DRM_MODE_CONTENT_TYPE_PHOTO', 'DRM_MODE_CURSOR_BO',
    'DRM_MODE_CURSOR_FLAGS', 'DRM_MODE_CURSOR_MOVE',
    'DRM_MODE_DIRTY_ANNOTATE', 'DRM_MODE_DIRTY_OFF',
    'DRM_MODE_DIRTY_ON', 'DRM_MODE_DITHERING_AUTO',
    'DRM_MODE_DITHERING_OFF', 'DRM_MODE_DITHERING_ON',
    'DRM_MODE_DPMS_OFF', 'DRM_MODE_DPMS_ON', 'DRM_MODE_DPMS_STANDBY',
    'DRM_MODE_DPMS_SUSPEND', 'DRM_MODE_ENCODER_DAC',
    'DRM_MODE_ENCODER_DPI', 'DRM_MODE_ENCODER_DPMST',
    'DRM_MODE_ENCODER_DSI', 'DRM_MODE_ENCODER_LVDS',
    'DRM_MODE_ENCODER_NONE', 'DRM_MODE_ENCODER_TMDS',
    'DRM_MODE_ENCODER_TVDAC', 'DRM_MODE_ENCODER_VIRTUAL',
    'DRM_MODE_FB_DIRTY_ANNOTATE_COPY',
    'DRM_MODE_FB_DIRTY_ANNOTATE_FILL', 'DRM_MODE_FB_DIRTY_FLAGS',
    'DRM_MODE_FB_DIRTY_MAX_CLIPS', 'DRM_MODE_FB_INTERLACED',
    'DRM_MODE_FB_MODIFIERS', 'DRM_MODE_FLAG_3D_FIELD_ALTERNATIVE',
    'DRM_MODE_FLAG_3D_FRAME_PACKING',
    'DRM_MODE_FLAG_3D_LINE_ALTERNATIVE', 'DRM_MODE_FLAG_3D_L_DEPTH',
    'DRM_MODE_FLAG_3D_L_DEPTH_GFX_GFX_DEPTH', 'DRM_MODE_FLAG_3D_MASK',
    'DRM_MODE_FLAG_3D_NONE', 'DRM_MODE_FLAG_3D_SIDE_BY_SIDE_FULL',
    'DRM_MODE_FLAG_3D_SIDE_BY_SIDE_HALF',
    'DRM_MODE_FLAG_3D_TOP_AND_BOTTOM', 'DRM_MODE_FLAG_ALL',
    'DRM_MODE_FLAG_BCAST', 'DRM_MODE_FLAG_CLKDIV2',
    'DRM_MODE_FLAG_CSYNC', 'DRM_MODE_FLAG_DBLCLK',
    'DRM_MODE_FLAG_DBLSCAN', 'DRM_MODE_FLAG_HSKEW',
    'DRM_MODE_FLAG_INTERLACE', 'DRM_MODE_FLAG_NCSYNC',
    'DRM_MODE_FLAG_NHSYNC', 'DRM_MODE_FLAG_NVSYNC',
    'DRM_MODE_FLAG_PCSYNC', 'DRM_MODE_FLAG_PHSYNC',
    'DRM_MODE_FLAG_PIC_AR_16_9', 'DRM_MODE_FLAG_PIC_AR_256_135',
    'DRM_MODE_FLAG_PIC_AR_4_3', 'DRM_MODE_FLAG_PIC_AR_64_27',
    'DRM_MODE_FLAG_PIC_AR_MASK', 'DRM_MODE_FLAG_PIC_AR_NONE',
    'DRM_MODE_FLAG_PIXMUX', 'DRM_MODE_FLAG_PVSYNC',
    'DRM_MODE_LINK_STATUS_BAD', 'DRM_MODE_LINK_STATUS_GOOD',
    'DRM_MODE_OBJECT_ANY', 'DRM_MODE_OBJECT_BLOB',
    'DRM_MODE_OBJECT_CONNECTOR', 'DRM_MODE_OBJECT_CRTC',
    'DRM_MODE_OBJECT_ENCODER', 'DRM_MODE_OBJECT_FB',
    'DRM_MODE_OBJECT_MODE', 'DRM_MODE_OBJECT_PLANE',
    'DRM_MODE_OBJECT_PROPERTY', 'DRM_MODE_PAGE_FLIP_ASYNC',
    'DRM_MODE_PAGE_FLIP_EVENT', 'DRM_MODE_PAGE_FLIP_FLAGS',
    'DRM_MODE_PAGE_FLIP_TARGET', 'DRM_MODE_PAGE_FLIP_TARGET_ABSOLUTE',
    'DRM_MODE_PAGE_FLIP_TARGET_RELATIVE',
    'DRM_MODE_PICTURE_ASPECT_16_9', 'DRM_MODE_PICTURE_ASPECT_256_135',
    'DRM_MODE_PICTURE_ASPECT_4_3', 'DRM_MODE_PICTURE_ASPECT_64_27',
    'DRM_MODE_PICTURE_ASPECT_NONE', 'DRM_MODE_PRESENT_BOTTOM_FIELD',
    'DRM_MODE_PRESENT_TOP_FIELD', 'DRM_MODE_PROP_ATOMIC',
    'DRM_MODE_PROP_BITMASK', 'DRM_MODE_PROP_BLOB',
    'DRM_MODE_PROP_ENUM', 'DRM_MODE_PROP_EXTENDED_TYPE',
    'DRM_MODE_PROP_IMMUTABLE', 'DRM_MODE_PROP_LEGACY_TYPE',
    'DRM_MODE_PROP_OBJECT', 'DRM_MODE_PROP_PENDING',
    'DRM_MODE_PROP_RANGE', 'DRM_MODE_PROP_SIGNED_RANGE',
    'DRM_MODE_REFLECT_MASK', 'DRM_MODE_REFLECT_X',
    'DRM_MODE_REFLECT_Y', 'DRM_MODE_ROTATE_0', 'DRM_MODE_ROTATE_180',
    'DRM_MODE_ROTATE_270', 'DRM_MODE_ROTATE_90',
    'DRM_MODE_ROTATE_MASK', 'DRM_MODE_SCALE_ASPECT',
    'DRM_MODE_SCALE_CENTER', 'DRM_MODE_SCALE_FULLSCREEN',
    'DRM_MODE_SCALE_NONE', 'DRM_MODE_SUBCONNECTOR_Automatic',
    'DRM_MODE_SUBCONNECTOR_Component',
    'DRM_MODE_SUBCONNECTOR_Composite', 'DRM_MODE_SUBCONNECTOR_DVIA',
    'DRM_MODE_SUBCONNECTOR_DVID', 'DRM_MODE_SUBCONNECTOR_SCART',
    'DRM_MODE_SUBCONNECTOR_SVIDEO', 'DRM_MODE_SUBCONNECTOR_Unknown',
    'DRM_MODE_TYPE_ALL', 'DRM_MODE_TYPE_BUILTIN',
    'DRM_MODE_TYPE_CLOCK_C', 'DRM_MODE_TYPE_CRTC_C',
    'DRM_MODE_TYPE_DEFAULT', 'DRM_MODE_TYPE_DRIVER',
    'DRM_MODE_TYPE_PREFERRED', 'DRM_MODE_TYPE_USERDEF', 'DRM_NAME',
    'DRM_PRIME_CAP_EXPORT', 'DRM_PRIME_CAP_IMPORT',
    'DRM_PROP_NAME_LEN', 'DRM_RAM_PERCENT',
    'DRM_SYNCOBJ_CREATE_SIGNALED',
    'DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE',
    'DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE',
    'DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL',
    'DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT', 'FORMAT_BLOB_CURRENT',
    'GGML_RKNPU2_MAX_MATMUL_KERNELS', 'NPU_CBUF_BANKS',
    'NPU_CBUF_BANK_SIZE', 'NPU_CNA_H', 'NPU_DPU_H', 'NPU_HW_H',
    'OP_40', 'OP_ENABLE', 'OP_NONE', 'OP_REG_CNA', 'OP_REG_CORE',
    'OP_REG_DPU', 'OP_REG_PC', 'PC_BASE_ADDRESS', 'PC_ENABLE',
    'PC_ENABLE_CNA', 'PC_ENABLE_DPU', 'PC_ENABLE_PPU',
    'PC_OPERATION_ENABLE', 'PC_OP_01', 'PC_OP_40', 'PC_OP_ENABLE',
    'PC_REGISTER_AMOUNTS', 'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16',
    'RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32',
    'RKNN_FLOAT16_MM_INT4_TO_BFLOAT16',
    'RKNN_FLOAT16_MM_INT4_TO_FLOAT16',
    'RKNN_FLOAT16_MM_INT4_TO_FLOAT32',
    'RKNN_FLOAT16_MM_INT8_TO_FLOAT16',
    'RKNN_FLOAT16_MM_INT8_TO_FLOAT32', 'RKNN_INT4_MM_INT4_TO_INT16',
    'RKNN_INT8_MM_INT4_TO_FLOAT16', 'RKNN_INT8_MM_INT4_TO_INT32',
    'RKNN_INT8_MM_INT8_TO_FLOAT32', 'RKNN_INT8_MM_INT8_TO_INT32',
    'RKNN_INT8_MM_INT8_TO_INT8', 'RKNN_TENSOR_BFLOAT16',
    'RKNN_TENSOR_BOOL', 'RKNN_TENSOR_FLOAT16', 'RKNN_TENSOR_FLOAT32',
    'RKNN_TENSOR_INT16', 'RKNN_TENSOR_INT32', 'RKNN_TENSOR_INT4',
    'RKNN_TENSOR_INT64', 'RKNN_TENSOR_INT8', 'RKNN_TENSOR_TYPE_MAX',
    'RKNN_TENSOR_UINT16', 'RKNN_TENSOR_UINT32', 'RKNN_TENSOR_UINT8',
    'RKNPU_ACTION', 'RKNPU_ACT_CLR_TOTAL_RW_AMOUNT',
    'RKNPU_ACT_RESET', 'RKNPU_GET_BW_EXPECT', 'RKNPU_GET_BW_PRIORITY',
    'RKNPU_GET_BW_TW', 'RKNPU_GET_DRV_VERSION',
    'RKNPU_GET_DT_RD_AMOUNT', 'RKNPU_GET_DT_WR_AMOUNT',
    'RKNPU_GET_FREE_SRAM_SIZE', 'RKNPU_GET_FREQ',
    'RKNPU_GET_HW_VERSION', 'RKNPU_GET_IOMMU_EN',
    'RKNPU_GET_TOTAL_RW_AMOUNT', 'RKNPU_GET_TOTAL_SRAM_SIZE',
    'RKNPU_GET_VOLT', 'RKNPU_GET_WT_RD_AMOUNT', 'RKNPU_INT_CLEAR',
    'RKNPU_IOC_MAGIC', 'RKNPU_JOB_BLOCK', 'RKNPU_JOB_FENCE_IN',
    'RKNPU_JOB_FENCE_OUT', 'RKNPU_JOB_MASK', 'RKNPU_JOB_NONBLOCK',
    'RKNPU_JOB_PC', 'RKNPU_JOB_PINGPONG', 'RKNPU_JOB_SLAVE',
    'RKNPU_MEM_CACHEABLE', 'RKNPU_MEM_CONTIGUOUS', 'RKNPU_MEM_CREATE',
    'RKNPU_MEM_DESTROY', 'RKNPU_MEM_IOMMU',
    'RKNPU_MEM_KERNEL_MAPPING', 'RKNPU_MEM_MAP', 'RKNPU_MEM_MASK',
    'RKNPU_MEM_NON_CACHEABLE', 'RKNPU_MEM_NON_CONTIGUOUS',
    'RKNPU_MEM_NON_DMA32', 'RKNPU_MEM_SECURE', 'RKNPU_MEM_SYNC',
    'RKNPU_MEM_SYNC_FROM_DEVICE', 'RKNPU_MEM_SYNC_MASK',
    'RKNPU_MEM_SYNC_TO_DEVICE', 'RKNPU_MEM_TRY_ALLOC_SRAM',
    'RKNPU_MEM_WRITE_COMBINE', 'RKNPU_MEM_ZEROING',
    'RKNPU_OFFSET_CLR_ALL_RW_AMOUNT', 'RKNPU_OFFSET_DT_RD_AMOUNT',
    'RKNPU_OFFSET_DT_WR_AMOUNT', 'RKNPU_OFFSET_ENABLE_MASK',
    'RKNPU_OFFSET_INT_CLEAR', 'RKNPU_OFFSET_INT_MASK',
    'RKNPU_OFFSET_INT_RAW_STATUS', 'RKNPU_OFFSET_INT_STATUS',
    'RKNPU_OFFSET_PC_DATA_ADDR', 'RKNPU_OFFSET_PC_DATA_AMOUNT',
    'RKNPU_OFFSET_PC_DMA_BASE_ADDR', 'RKNPU_OFFSET_PC_OP_EN',
    'RKNPU_OFFSET_PC_TASK_CONTROL', 'RKNPU_OFFSET_PC_TASK_STATUS',
    'RKNPU_OFFSET_VERSION', 'RKNPU_OFFSET_VERSION_NUM',
    'RKNPU_OFFSET_WT_RD_AMOUNT', 'RKNPU_PC_DATA_EXTRA_AMOUNT',
    'RKNPU_POWER_OFF', 'RKNPU_POWER_ON', 'RKNPU_SET_BW_EXPECT',
    'RKNPU_SET_BW_PRIORITY', 'RKNPU_SET_BW_TW', 'RKNPU_SET_FREQ',
    'RKNPU_SET_PROC_NICE', 'RKNPU_SET_VOLT', 'RKNPU_SUBMIT',
    '_DRM_ADD_COMMAND', '_DRM_AGP', '_DRM_AGP_BUFFER',
    '_DRM_CONSISTENT', '_DRM_CONTAINS_LOCK', '_DRM_CONTEXT_2DONLY',
    '_DRM_CONTEXT_PRESERVED', '_DRM_DMA_BLOCK', '_DRM_DMA_LARGER_OK',
    '_DRM_DMA_PRIORITY', '_DRM_DMA_SMALLER_OK', '_DRM_DMA_WAIT',
    '_DRM_DMA_WHILE_LOCKED', '_DRM_DRIVER', '_DRM_FB_BUFFER',
    '_DRM_FRAME_BUFFER', '_DRM_HALT_ALL_QUEUES',
    '_DRM_HALT_CUR_QUEUES', '_DRM_H_', '_DRM_INST_HANDLER',
    '_DRM_KERNEL', '_DRM_LOCKED', '_DRM_LOCK_CONT', '_DRM_LOCK_FLUSH',
    '_DRM_LOCK_FLUSH_ALL', '_DRM_LOCK_HELD', '_DRM_LOCK_QUIESCENT',
    '_DRM_LOCK_READY', '_DRM_MODE_H', '_DRM_PAGE_ALIGN',
    '_DRM_PCI_BUFFER_RO', '_DRM_POST_MODESET', '_DRM_PRE_MODESET',
    '_DRM_READ_ONLY', '_DRM_REGISTERS', '_DRM_REMOVABLE',
    '_DRM_RESTRICTED', '_DRM_RM_COMMAND', '_DRM_SCATTER_GATHER',
    '_DRM_SG_BUFFER', '_DRM_SHM', '_DRM_STAT_BYTE',
    '_DRM_STAT_CLOSES', '_DRM_STAT_COUNT', '_DRM_STAT_DMA',
    '_DRM_STAT_IOCTLS', '_DRM_STAT_IRQ', '_DRM_STAT_LOCK',
    '_DRM_STAT_LOCKS', '_DRM_STAT_MISSED', '_DRM_STAT_OPENS',
    '_DRM_STAT_PRIMARY', '_DRM_STAT_SECONDARY', '_DRM_STAT_SPECIAL',
    '_DRM_STAT_UNLOCKS', '_DRM_STAT_VALUE', '_DRM_UNINST_HANDLER',
    '_DRM_VBLANK_ABSOLUTE', '_DRM_VBLANK_EVENT',
    '_DRM_VBLANK_FLAGS_MASK', '_DRM_VBLANK_FLIP',
    '_DRM_VBLANK_HIGH_CRTC_MASK', '_DRM_VBLANK_HIGH_CRTC_SHIFT',
    '_DRM_VBLANK_NEXTONMISS', '_DRM_VBLANK_RELATIVE',
    '_DRM_VBLANK_SECONDARY', '_DRM_VBLANK_SIGNAL',
    '_DRM_VBLANK_TYPES_MASK', '_DRM_WRITE_COMBINING', '_IO', '_IOR',
    '_IOW', '_IOWR', '__user', '_rknn_matmul_type',
    '_rknn_tensor_type', 'c__EA_drm_drawable_info_type_t',
    'c__Ea_direct_convolution', 'c__Ea_precision_int8',
    'direct_convolution', 'drm_agp_binding_t', 'drm_agp_buffer_t',
    'drm_agp_info_t', 'drm_agp_mode_t', 'drm_auth_t', 'drm_block_t',
    'drm_buf_desc_flags', 'drm_buf_desc_flags_t',
    'drm_buf_desc_flags_t__enumvalues', 'drm_buf_desc_t',
    'drm_buf_free_t', 'drm_buf_info_t', 'drm_buf_map_t',
    'drm_buf_pub_t', 'drm_client_t', 'drm_clip_rect_t',
    'drm_context_t', 'drm_control_func', 'drm_control_func_t',
    'drm_control_func_t__enumvalues', 'drm_control_t',
    'drm_ctx_flags', 'drm_ctx_flags_t', 'drm_ctx_flags_t__enumvalues',
    'drm_ctx_priv_map_t', 'drm_ctx_res_t', 'drm_ctx_t',
    'drm_dma_flags', 'drm_dma_flags_t', 'drm_dma_flags_t__enumvalues',
    'drm_dma_t', 'drm_draw_t', 'drm_drawable_info_t',
    'drm_drawable_info_type_t',
    'drm_drawable_info_type_t__enumvalues', 'drm_drawable_t',
    'drm_handle_t', 'drm_hw_lock_t', 'drm_irq_busid_t', 'drm_list_t',
    'drm_lock_flags', 'drm_lock_flags_t',
    'drm_lock_flags_t__enumvalues', 'drm_lock_t', 'drm_magic_t',
    'drm_map_flags', 'drm_map_flags_t', 'drm_map_flags_t__enumvalues',
    'drm_map_t', 'drm_map_type', 'drm_map_type_t',
    'drm_map_type_t__enumvalues', 'drm_mode_subconnector',
    'drm_scatter_gather_t', 'drm_set_version_t', 'drm_stat_type',
    'drm_stat_type_t', 'drm_stat_type_t__enumvalues', 'drm_stats_t',
    'drm_tex_region_t', 'drm_unique_t', 'drm_update_draw_t',
    'drm_vblank_seq_type', 'drm_vblank_seq_type_t',
    'drm_vblank_seq_type_t__enumvalues', 'drm_version_t',
    'drm_wait_vblank_t', 'e_rknpu_action', 'e_rknpu_job_mode',
    'e_rknpu_mem_sync_mode', 'e_rknpu_mem_type', 'npu_cna_core_task',
    'npu_cna_desc', 'npu_core_desc', 'npu_dpu_desc', 'npu_pc_desc',
    'precision_float16', 'precision_float32', 'precision_int32',
    'precision_int8', 'struct__rknn_matmul_io_attr',
    'struct__rknn_matmul_tensor_attr', 'struct__rknn_tensor_memory',
    'struct_dma_heap_allocation_data', 'struct_drm_agp_binding',
    'struct_drm_agp_buffer', 'struct_drm_agp_info',
    'struct_drm_agp_mode', 'struct_drm_auth', 'struct_drm_block',
    'struct_drm_buf_desc', 'struct_drm_buf_free',
    'struct_drm_buf_info', 'struct_drm_buf_map', 'struct_drm_buf_pub',
    'struct_drm_client', 'struct_drm_clip_rect',
    'struct_drm_color_ctm', 'struct_drm_color_lut',
    'struct_drm_control', 'struct_drm_crtc_get_sequence',
    'struct_drm_crtc_queue_sequence', 'struct_drm_ctx',
    'struct_drm_ctx_priv_map', 'struct_drm_ctx_res', 'struct_drm_dma',
    'struct_drm_draw', 'struct_drm_drawable_info', 'struct_drm_event',
    'struct_drm_event_crtc_sequence', 'struct_drm_event_vblank',
    'struct_drm_format_modifier', 'struct_drm_format_modifier_blob',
    'struct_drm_gem_close', 'struct_drm_gem_flink',
    'struct_drm_gem_open', 'struct_drm_get_cap', 'struct_drm_hw_lock',
    'struct_drm_irq_busid', 'struct_drm_list', 'struct_drm_lock',
    'struct_drm_map', 'struct_drm_mode_atomic',
    'struct_drm_mode_card_res',
    'struct_drm_mode_connector_set_property',
    'struct_drm_mode_create_blob', 'struct_drm_mode_create_dumb',
    'struct_drm_mode_create_lease', 'struct_drm_mode_crtc',
    'struct_drm_mode_crtc_lut', 'struct_drm_mode_crtc_page_flip',
    'struct_drm_mode_crtc_page_flip_target', 'struct_drm_mode_cursor',
    'struct_drm_mode_cursor2', 'struct_drm_mode_destroy_blob',
    'struct_drm_mode_destroy_dumb', 'struct_drm_mode_fb_cmd',
    'struct_drm_mode_fb_cmd2', 'struct_drm_mode_fb_dirty_cmd',
    'struct_drm_mode_get_blob', 'struct_drm_mode_get_connector',
    'struct_drm_mode_get_encoder', 'struct_drm_mode_get_lease',
    'struct_drm_mode_get_plane', 'struct_drm_mode_get_plane_res',
    'struct_drm_mode_get_property', 'struct_drm_mode_list_lessees',
    'struct_drm_mode_map_dumb', 'struct_drm_mode_mode_cmd',
    'struct_drm_mode_modeinfo', 'struct_drm_mode_obj_get_properties',
    'struct_drm_mode_obj_set_property',
    'struct_drm_mode_property_enum', 'struct_drm_mode_revoke_lease',
    'struct_drm_mode_set_plane', 'struct_drm_modeset_ctl',
    'struct_drm_prime_handle', 'struct_drm_scatter_gather',
    'struct_drm_set_client_cap', 'struct_drm_set_version',
    'struct_drm_stats', 'struct_drm_stats_0',
    'struct_drm_syncobj_array', 'struct_drm_syncobj_create',
    'struct_drm_syncobj_destroy', 'struct_drm_syncobj_handle',
    'struct_drm_syncobj_wait', 'struct_drm_tex_region',
    'struct_drm_unique', 'struct_drm_update_draw',
    'struct_drm_version', 'struct_drm_wait_vblank_reply',
    'struct_drm_wait_vblank_request', 'struct_ggml_rknpu2_data_pack',
    'struct_ggml_rknpu2_matmul_kernel', 'struct_ggml_sync_data_pack',
    'struct_npu_cna_core_task', 'struct_npu_cna_desc',
    'struct_npu_core_desc', 'struct_npu_dpu_desc',
    'struct_nup_pc_desc', 'struct_rknn_matmul_info_t',
    'struct_rknpu_action', 'struct_rknpu_mem_create',
    'struct_rknpu_mem_destroy', 'struct_rknpu_mem_map',
    'struct_rknpu_mem_sync', 'struct_rknpu_subcore_task',
    'struct_rknpu_submit', 'struct_rknpu_task',
    'union_drm_wait_vblank']
