#!/usr/bin/env python3
"""reviewer1_edit_4_step1_longevity_slope.py

Purpose : reviewer1_edit_4 step1 - per-turn latency trend of the 20,000-turn session
          (re-analysis of the submitted log; no new measurement)
Record  : TEST-LEDGER 107 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_4_step1_longevity_slope.py ieee/secVB1_20000turn_session.log
"""
import re, sys

LOG = sys.argv[1]

#------------------------------------------------------------
#Func : read_feed
#------------------------------------------------------------
def read_feed(path):
    pts = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\[feed\] (\d+)/20000 turns, (\d+) trims, (\d+)s", line)
        if m:
            pts.append((int(m.group(1)), int(m.group(3))))
    return pts

#------------------------------------------------------------
#Func : read_fifths
#------------------------------------------------------------
def read_fifths(path):
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"turns\s+(\d+)-\s*(\d+): p50=(\d+)ms p95=(\d+)ms", line)
        if m:
            out.append(tuple(int(x) for x in m.groups()))
    return out

#------------------------------------------------------------
#Func : block_means
#------------------------------------------------------------
def block_means(pts):
    per = []
    prev_t, prev_s = 0, 0
    for t, s in pts:
        per.append(((prev_t + t) / 2.0, (s - prev_s) / float(t - prev_t) * 1000.0))
        prev_t, prev_s = t, s
    return per

#------------------------------------------------------------
#Func : slope
#------------------------------------------------------------
def slope(per):
    n = len(per)
    mx = sum(x for x, _ in per) / n
    my = sum(y for _, y in per) / n
    return sum((x - mx) * (y - my) for x, y in per) / sum((x - mx) ** 2 for x, _ in per)

#------------------------------------------------------------
#Func : main
#------------------------------------------------------------
def main():
    for lo, hi, p50, p95 in read_fifths(LOG):
        print("server-side per-turn latency, turns %5d-%5d: p50 %d ms, p95 %d ms (printed in the log)" % (lo, hi, p50, p95))
    pts = read_feed(LOG)
    per = block_means(pts)
    ys = sorted(y for _, y in per)
    k = slope(per)
    print("client-side 100-turn block means: %d blocks, median %.0f ms, min %.0f ms, max %.0f ms" % (len(per), ys[len(ys) // 2], ys[0], ys[-1]))
    print("linear fit over all blocks: %+.2f ms per 1,000 turns (%+.1f ms over 20,000 turns)" % (k * 1000.0, k * 20000.0))
    first = pts[9][1] / 1000.0 * 1000.0
    last = (pts[-1][1] - pts[-11][1]) / 1000.0 * 1000.0
    print("mean per-turn time, first 1,000 turns %.0f ms, last 1,000 turns %.0f ms" % (first, last))

if __name__ == "__main__":
    main()
