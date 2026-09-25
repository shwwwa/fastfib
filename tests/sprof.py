# maps sprof_out.txt addresses to functions via `nm -n sprof.exe`
import os, sys, bisect, subprocess, collections
nm = subprocess.run([os.environ.get("NM", "nm"), "-n", "sprof.exe"],
                    capture_output=True, text=True).stdout.split("\n")
syms = []
for l in nm:
    p = l.split()
    if len(p) == 3 and p[1] in "tT":
        syms.append((int(p[0], 16), p[2]))
addrs = [a for a, _ in syms]
lines = open("sprof_out.txt").read().split()
base = int(lines[1], 16)
imgbase = 0x140000000
cnt = collections.Counter()
busy = collections.Counter()
total = 0
for h in lines[2:]:
    v = int(h, 16)
    main = v >> 63
    rip = (v & ~(1 << 63)) - base + imgbase
    i = bisect.bisect_right(addrs, rip) - 1
    in_exe = 0 <= i and rip - syms[i][0] < 0x100000 and i < len(syms) - 1
    name = syms[i][1] if in_exe else "(outside exe: waiting / OS / C runtime)"
    name = name.split(".")[0]
    cnt[name] += 1
    total += 1
if total == 0:
    sys.exit("no samples: the run was too short for the sampler (use a larger n or more reps)")
work = sum(c for k, c in cnt.items() if not k.startswith("(outside"))
print(f"{total} samples over all threads; {work} ({100*work/total:.0f}%) inside our code")
print(f"{'function':42} {'% of work':>9} {'% all':>7}")
for k, c in cnt.most_common(25):
    w = f"{100*c/work:8.1f}%" if not k.startswith("(outside") else "        -"
    print(f"{k:42} {w} {100*c/total:6.1f}%")
