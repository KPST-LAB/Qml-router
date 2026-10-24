#!/usr/bin/env python3
"""reviewer2_minor1_density_verified_only.py
Purpose : Reviewer 2, minor comment 1. Recomputes the Fig. 1(a) reconnect statistics from the
          per-reconnect raw file of reviewer1_edit_14 with only the reconnects that passed the
          exact check (Section IV excludes timings without a passing verification). The
          short-request throughput (short_rate) is copied unchanged: it is a separate stream of
          completed requests that carry no planted fact.
Usage   : python3 reviewer2_minor1_density_verified_only.py RAW_JSON OUT_JSON
"""
import json, math, sys

raw = json.load(open(sys.argv[1]))
out = {}
for mode, levels in raw.items():
    out[mode] = {}
    for n, row in levels.items():
        users = row["per_user"]
        lats = sorted(u["latency"] for u in users.values() if u["correct"])
        k = len(lats)
        median = lats[k // 2] if k % 2 else (lats[k // 2 - 1] + lats[k // 2]) / 2
        new = dict(row)
        new.update({"median": round(median, 4), "max": lats[-1], "min": lats[0],
                    "p95_nr": lats[math.ceil(0.95 * k) - 1],
                    "n_checked": len(users), "n_passed": k,
                    "excluded": [i for i, u in users.items() if not u["correct"]]})
        out[mode][n] = new
        print(f"{mode} N={n}: checked {len(users)}, passed {k}, median {row['median']} -> {new['median']}, "
              f"max {row['max']} -> {new['max']}, short_rate {row['short_rate']}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
