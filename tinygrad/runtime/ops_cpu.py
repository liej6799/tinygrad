from __future__ import annotations
import platform, subprocess, sys, ctypes, ctypes.util, functools, time, mmap, threading, queue, tempfile, os
from tinygrad.helpers import capstone_flatdump, getenv, from_mv, to_mv, OSX, mv_address, wait_cond, cpu_profile
from tinygrad.device import Compiler, BufferSpec, DMACPURef
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import sint

class CPUSignal(HCQSignal):
  def _sleep(self, time_spent_waiting_ms:int):
    if self.is_timeline and self.owner is not None: self.owner.tasks.join()

class ClangJITCompiler(Compiler):

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    # Remove bare metal flags for linking and add shared library flags
    link_args = ['-O2', '-fPIC', '-fno-math-errno', '-shared', '-fno-ident']
    link_arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output([getenv("CC", 'clang'), '-x', 'c', *link_args, *link_arch_args, '-I/data/Dev/tinygrad/extra/rockchip',
                                '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return output_file.read()


class CPUWorker(threading.Thread):
  def __init__(self, dev, tasks, thread_id):
    super().__init__()
    self.dev, self.tasks, self.thread_id, self.daemon = dev, tasks, thread_id, True

  def run(self):
    while True:
      cmd_iter = iter(self.tasks.get())
      for cmd in cmd_iter:
        args_cnt = next(cmd_iter)
        cmd(*[next(cmd_iter) for _ in range(args_cnt)])
      self.tasks.task_done()

class CPUComputeQueue(HWQueue):
  def _exec(self, prg, bufs, *args):
    prg.fxn(*map(ctypes.c_uint64, args[:bufs]), *map(ctypes.c_int64 if platform.machine() == "arm64" else ctypes.c_int32, args[bufs:]))
  def _signal(self, signal_addr, value): to_mv(signal_addr, 4).cast('I')[0] = value
  def _wait(self, signal_addr, value): wait_cond(lambda: to_mv(signal_addr, 4).cast('I')[0] >= value, timeout_ms=60000)
  def _timestamp(self, timestamp_addr): to_mv(timestamp_addr, 8).cast('Q')[0] = time.perf_counter_ns()
  def cmd(self, cmd, *args):
    self.q(cmd, len(args), *args)
    return self

  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    return self.cmd(self._exec, prg, len(args_state.bufs), *[x.va_addr for x in args_state.bufs], *args_state.vals)
  def wait(self, signal, value=0): return self.cmd(self._wait, signal.value_addr, value)
  def timestamp(self, signal): return self.cmd(self._timestamp, signal.timestamp_addr)
  def signal(self, signal, value:sint=0): return self.cmd(self._signal, signal.value_addr, value)
  def _submit(self, dev): dev.tasks.put(self._q[:])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram):
  rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or sys.platform == "win32" else 'libgcc_s.so.1')

  def __init__(self, dev, name:str, lib:bytes):
    """Initialize from shared library"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as tmp_lib:
      tmp_lib.write(lib)
      tmp_lib.flush()
      
      # Load as shared library
      self.dll = ctypes.CDLL(tmp_lib.name)
      self.fxn = getattr(self.dll, name, None)

    
    super().__init__(HCQArgsState, dev, name, kernargs_alloc_size=0)



  def __del__(self):
    if getattr(sys, 'is_finalizing', lambda: True)(): return


class CPUAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr: addr, buf = options.external_ptr, None
    elif sys.platform == "win32": addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview:
   self.dev.synchronize()
   return to_mv(src.va_addr, src.size)
  def _as_dmaref(self, buf):
    self.dev.synchronize()
    return DMACPURef(buf.va_addr, buf.size)
  def _copyin(self, dest, src:memoryview):
    self.dev.synchronize()
    with cpu_profile('TINY -> CPU', self.dev.device, is_copy=True): ctypes.memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src):
    self.dev.synchronize()
    with cpu_profile('CPU -> TINY', self.dev.device, is_copy=True): ctypes.memmove(from_mv(dest), src.va_addr, len(dest))
  def _map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")

class CPUDevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.tasks:queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()
    super().__init__(device, CPUAllocator(self), ClangRenderer(), ClangJITCompiler(), functools.partial(CPUProgram, self), CPUSignal, CPUComputeQueue)
