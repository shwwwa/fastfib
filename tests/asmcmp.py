# Compares two gcc .s files function by function. Local labels (.L123, .LC4)
# are renamed by order of first use inside each function, so reordering code
# or different label counters don't count as differences; constant-pool
# blocks (.LC*) are compared as a multiset of their contents.
# usage: python asmcmp.py a.s b.s
import sys, re, collections

def parse(path):
    funcs, consts, cur, body = {}, collections.Counter(), None, []
    def flush():
        if cur is None: return
        if cur.startswith(".LC"): consts["\n".join(body)] += 1
        else: funcs[cur] = body[:]
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^([A-Za-z_.$][\w.$]*):$", line)
        if m and not re.match(r"^\.L\d+$", m.group(1)):
            flush(); cur, body = m.group(1), []
            continue
        st = line.strip()
        if cur is None or st.startswith((".file", ".ident", ".def", "#", "/APP", "/NO_APP")): continue
        body.append(line)
    flush()
    return funcs, consts

def norm(body, consts_map):
    names = {}
    def ren(m):
        k = m.group(0)
        if k.startswith(".LC"): return consts_map.get(k, k)
        names.setdefault(k, f".L#{len(names)}")
        return names[k]
    return [re.sub(r"\.L(C)?\d+", ren, l) for l in body]

def const_names(path):  # .LCn -> its content, so references compare by value
    out, cur, body = {}, None, []
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^(\.LC\d+):$", line)
        if m or re.match(r"^[A-Za-z_.$][\w.$]*:$", line):
            if cur: out[cur] = "<" + "|".join(x.strip() for x in body if not x.strip().startswith(".section") and not x.strip().startswith(".align")) + ">"
            cur, body = (m.group(1) if m else None), []
            continue
        if cur and not line.strip().startswith((".def", ".ident")): body.append(line)
    if cur: out[cur] = "<" + "|".join(x.strip() for x in body) + ">"
    return out

(fa, ca), (fb, cb) = parse(sys.argv[1]), parse(sys.argv[2])
na, nb = const_names(sys.argv[1]), const_names(sys.argv[2])
same = diff = 0
for name in sorted(set(fa) | set(fb)):
    if name not in fa or name not in fb:
        print(f"only in {'b' if name not in fa else 'a'}: {name}")
        diff += 1
        continue
    if norm(fa[name], na) == norm(fb[name], nb): same += 1
    else:
        diff += 1
        a, b = norm(fa[name], na), norm(fb[name], nb)
        i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        print(f"DIFFERS: {name} ({len(a)} vs {len(b)} lines), first at line {i}: {a[i] if i < len(a) else ''!r} vs {b[i] if i < len(b) else ''!r}")
print(f"{same} functions identical, {diff} different; constant pools {'equal' if sorted(na.values()) == sorted(nb.values()) else 'DIFFER'}")
