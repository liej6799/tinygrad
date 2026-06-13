#!/usr/bin/env python3
# npu_ioctl_decode: decode RK3588 NPU (rknpu) DRM ioctls and check whether a program actually drives the NPU.
# strace shows these as raw `_IOC(...,0x64,..)` or even WRONG collision names (DRM_IOCTL_ARMADA_GEM_CREATE ...),
# because rknpu reuses the DRM_COMMAND_BASE range. this tool maps them back to the real DRM_IOCTL_RKNPU_* names.
#
# two modes:
#   decode:  python extra/rockchip/npu_ioctl_decode.py 0xc0686441 0xc0286442    # decode raw request number(s)
#   run:     python extra/rockchip/npu_ioctl_decode.py -- python extra/gemm/simple_matmul.py   # trace + summarize
# in run mode a non-zero SUBMIT count proves the NPU executed (each run_gemm submit = one matmul on the MAC array).
import sys, os, re, subprocess, tempfile, argparse
from collections import Counter

# from extra/rockchip/rknpu_ioctl.h: DRM_COMMAND_BASE(0x40) + RKNPU_<op>, all _IOWR('d'==0x64, nr, sizeof(struct)).
DRM_COMMAND_BASE = 0x40
RKNPU = {0x40:"ACTION", 0x41:"SUBMIT", 0x42:"MEM_CREATE", 0x43:"MEM_MAP", 0x44:"MEM_DESTROY", 0x45:"MEM_SYNC"}
DIR = {0:"NONE", 1:"W", 2:"R", 3:"WR"}                 # _IOC dir bits (READ=2|WRITE=1)

def decode_req(req:int):
  # split a 32-bit Linux _IOC() request: [dir:2][size:14][type:8][nr:8]
  nr, typ, size, d = req & 0xff, (req>>8)&0xff, (req>>16)&0x3fff, (req>>30)&0x3
  name = f"DRM_IOCTL_RKNPU_{RKNPU[nr]}" if typ == 0x64 and nr in RKNPU else None
  return name, nr, typ, size, d

def fmt(req:int):
  name, nr, typ, size, d = decode_req(req)
  who = name or f"(type=0x{typ:02x} nr=0x{nr:02x} -- not an rknpu ioctl)"
  return f"0x{req:08x}  {who:<28s} dir={DIR[d]:<2s} nr=0x{nr:02x} size={size}B"

# strace -y -X raw line: "[pid] ioctl(3</dev/dri/card1>, 0xc0686441, 0x...) = 0"  (ret may be -1 ERRNO ...)
LINE = re.compile(r"ioctl\((\d+)<([^>]+)>,\s*(0x[0-9a-fA-F]+)[^=]*=\s*(-?\d+)")

def run_and_trace(cmd:list[str], keep:str|None):
  if not (strace := _which("strace")):
    print("npu_ioctl_decode: `strace` not found (apt install strace) -- cannot trace.", file=sys.stderr); return 2
  out = keep or tempfile.mkstemp(prefix="npu_ioctl_", suffix=".strace")[1]
  # -f follow forks, -y annotate fds with their path, -X raw -> numeric request (so collisions don't mislead us)
  rc = subprocess.run([strace, "-f", "-y", "-X", "raw", "-e", "trace=ioctl,openat", "-o", out, *cmd]).returncode
  per_dev:dict[str, Counter] = {}
  errs = []
  with open(out, errors="ignore") as f:
    for ln in f:
      if not (m := LINE.search(ln)): continue
      path, req, ret = m.group(2), int(m.group(3), 16), int(m.group(4))
      name, nr, typ, _, _ = decode_req(req)
      if name is None and "/dev/dri/" not in path: continue            # ignore unrelated ioctls
      key = name or f"0x{req:08x}"
      per_dev.setdefault(path, Counter())[key] += 1
      if ret < 0: errs.append((path, key, ln.strip()))
  print(f"\n=== rknpu ioctl summary (exit={rc}) ===")
  npu_used = False
  for path in sorted(per_dev):
    c = per_dev[path]
    if not any(k.startswith("DRM_IOCTL_RKNPU") for k in c): continue   # only show NPU devices
    print(f"{path}:")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(f"   {v:5d}x {k}")
    npu_used |= c.get("DRM_IOCTL_RKNPU_SUBMIT", 0) > 0
  if errs:
    print("errors:")
    for p, k, ln in errs[:10]: print(f"   {k} on {p}: {ln}")
  subs = sum(c.get("DRM_IOCTL_RKNPU_SUBMIT", 0) for c in per_dev.values())
  print(f"\nverdict: {'NPU USED' if npu_used else 'NO NPU SUBMIT seen'}  ({subs} RKNPU_SUBMIT ioctl(s))")
  if not keep: os.unlink(out)
  else: print(f"(raw strace kept at {out})")
  return 0 if npu_used else 1

def _which(x):
  for p in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(c := os.path.join(p, x)): return c
  return None

def main(argv):
  ap = argparse.ArgumentParser(description="decode rknpu DRM ioctls / check that a program drives the NPU",
                               usage="%(prog)s 0xREQ [0xREQ...]   |   %(prog)s [--keep FILE] -- CMD [ARGS...]")
  ap.add_argument("--keep", metavar="FILE", help="keep the raw strace log at FILE (run mode)")
  if "--" in argv:
    i = argv.index("--"); args = ap.parse_args(argv[:i]); cmd = argv[i+1:]
    if not cmd: ap.error("no command after --")
    return run_and_trace(cmd, args.keep)
  args, reqs = ap.parse_known_args(argv)
  if not reqs:
    ap.print_help(); return 0
  for r in reqs: print(fmt(int(r, 16) if isinstance(r, str) and r.lower().startswith("0x") else int(r)))
  return 0

if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
