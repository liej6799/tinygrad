import ctypes, platform
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.helpers import OSX, getenv, capstone_flatdump, DEBUG
from tinygrad.renderer.llvmir import LLVMRenderer
import tinygrad.runtime.autogen.llvm as llvm
from tinygrad.runtime.support.elf import jit_loader

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret

class LLVMCompiler(Compiler):
  jit = True
  target_arch = {'arm64': 'AArch64', 'aarch64': 'AArch64', 'x86_64': 'X86', 'AMD64': 'X86'}[platform.machine()]
  def __init__(self, processor:str, feats:str):
    opt = 1
    super().__init__(f"compile_llvm_{self.target_arch}{'_jit' if self.jit else ''}{'_opt' if opt else ''}")


class HostLLVMCompiler(LLVMCompiler):
  def __init__(self):
    # +reserve-x18 here does the same thing as -ffixed-x18 in ops_cpu.py, see comments there for why it's needed on arm osx
    cpu, feats = ctypes.string_at(llvm.LLVMGetHostCPUName()), (b'+reserve-x18,' if OSX else b'') + ctypes.string_at(llvm.LLVMGetHostCPUFeatures())
    super().__init__(cpu.decode(), feats.decode())

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, LLVMRenderer(), HostLLVMCompiler(), CPUProgram)
