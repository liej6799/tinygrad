import platform, subprocess, sys
from tinygrad.helpers import capstone_flatdump, getenv
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer

class ClangJITCompiler(Compiler):

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    args = [ '-fPIC', '-shared', '-ffreestanding', '-L/usr/lib/librknnrt.so', '-lrknnrt', '-lstdc++']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    obj = subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    print('clang jit', [getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'])
    return jit_loader(obj)


class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangRenderer(), ClangJITCompiler(), CPUProgram)

CPUDevice = ClangDevice