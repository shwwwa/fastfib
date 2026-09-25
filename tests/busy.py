import os, bisect, subprocess, collections
nm = subprocess.run([os.environ.get("NM", "nm"), "-n", "sprof.exe"], capture_output=True, text=True).stdout.split("\n")
syms = sorted((int(p[0], 16), p[2]) for p in (l.split() for l in nm) if len(p) == 3 and p[1] in "tT")
addrs = [a for a, _ in syms]
lines = open("sprof_out.txt").read().split()
base = int(lines[1], 16)
idle_fns = {"tp_worker", "pool_wait"}
hist = collections.Counter(); main_state = collections.Counter(); busy = 0; rnd_main = None
for h in lines[2:]:
    v = int(h, 16)
    if v == 0:
        hist[busy] += 1; busy = 0; continue
    rip = (v & ~(1 << 63)) - base + 0x140000000
    i = bisect.bisect_right(addrs, rip) - 1
    name = syms[i][1].split(".")[0] if 0 <= i < len(syms) - 1 and rip - syms[i][0] < 0x100000 else None
    working = name is not None and name not in idle_fns
    busy += working
tot = sum(hist.values())
if tot == 0:
    raise SystemExit("no sampling rounds: the run was too short (use a larger n or more reps)")
print(f"{tot} sampling rounds (16 threads each); threads doing real work per round:")
for k in sorted(hist): print(f"  {k:2d} busy: {100*hist[k]/tot:5.1f}% " + "#" * int(60*hist[k]/tot))
print(f"average busy threads: {sum(k*c for k,c in hist.items())/tot:.1f}")
