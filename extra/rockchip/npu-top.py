#!/usr/bin/env python3
# npu-top: a tiny `top` for the RK3588 NPU. samples the kernel's debugfs load/freq/volt nodes and shows
# per-core utilization live, so you can SEE whether a run (e.g. ops_rk / gemm.py) is actually on the NPU.
# usage:  python extra/rockchip/npu-top.py [-i SECONDS] [-n SAMPLES] [--once]
#   while a matmul runs in another shell, Core% jumps off 0 -> the GEMM is on the NPU MAC, not clang/CPU.
import os, re, sys, time, argparse, glob

DBG = "/sys/kernel/debug/rknpu"                       # primary source (per-core %), needs root on most boards
DEVFREQ = glob.glob("/sys/class/devfreq/*.npu")       # fallback: aggregate "load@freqHz"
CORE_RE = re.compile(r"Core(\d+):\s*(\d+)\s*%")        # "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"

def _read(path):
  try:
    with open(path) as f: return f.read().strip()
  except OSError: return None

def sample():
  # returns (cores[list[int]], freq_mhz|None, volt_mv|None, power|None). cores=[] if nothing readable.
  load = _read(f"{DBG}/load")
  cores = [int(v) for _, v in CORE_RE.findall(load)] if load else []
  freq = _read(f"{DBG}/freq"); volt = _read(f"{DBG}/volt"); power = _read(f"{DBG}/power")
  if not cores and DEVFREQ:                            # debugfs not accessible -> devfreq aggregate
    dl = _read(f"{DEVFREQ[0]}/load")                   # e.g. "100@1000000000Hz"
    if dl and "@" in dl:
      pct, _, hz = dl.partition("@"); cores = [int(pct)]; freq = freq or hz.rstrip("Hz")
    freq = freq or _read(f"{DEVFREQ[0]}/cur_freq")
  fmhz = int(freq)//1_000_000 if freq and freq.isdigit() else None
  vmv = int(volt)//1000 if volt and volt.isdigit() else None
  return cores, fmhz, vmv, power

def main():
  ap = argparse.ArgumentParser(description="live RK3588 NPU utilization monitor")
  ap.add_argument("-i", "--interval", type=float, default=0.5, help="seconds between samples (default 0.5)")
  ap.add_argument("-n", "--count", type=int, default=0, help="number of samples then exit (0 = forever)")
  ap.add_argument("--once", action="store_true", help="print a single sample and exit")
  args = ap.parse_args()

  cores0, _, _, _ = sample()
  if not cores0:
    print(f"npu-top: cannot read NPU load. tried {DBG}/load (try sudo) and {DEVFREQ or '(no devfreq *.npu)'}", file=sys.stderr)
    return 1
  ncore = len(cores0)
  tty = sys.stdout.isatty()
  peak = [0]*ncore
  hdr = "   time   " + "".join(f"  Core{i} " for i in range(ncore)) + "   max   freq   volt  power"
  print(hdr)
  n = 0
  try:
    while True:
      cores, fmhz, vmv, power = sample()
      cores = (cores + [0]*ncore)[:ncore]
      peak = [max(p, c) for p, c in zip(peak, cores)]
      mx = max(cores) if cores else 0
      row = (time.strftime("%H:%M:%S") + "  " + "".join(f"{c:5d}% " for c in cores) +
             f"{mx:5d}% {('%4dMHz'%fmhz) if fmhz else '   ?   '} {('%4dmV'%vmv) if vmv else '  ?  '}  {power or '?'}")
      # bar of the hottest core so activity is obvious at a glance
      bar = "#" * (mx*20//100)
      end = "\r" if (tty and not args.once and args.count == 0) else "\n"
      sys.stdout.write(row + f"  |{bar:<20}|" + end); sys.stdout.flush()
      n += 1
      if args.once or (args.count and n >= args.count): break
      time.sleep(args.interval)
  except KeyboardInterrupt: pass
  if tty and not args.once: sys.stdout.write("\n")
  print("peak per-core: " + ", ".join(f"Core{i}={p}%" for i, p in enumerate(peak)))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
