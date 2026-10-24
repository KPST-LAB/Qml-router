#!/usr/bin/env python3
"""plot_lme_recall.py

Purpose : Fig. 6(a) for the resubmission: recall_all@k of the lexical (BM25) and the embedding
Record  : TEST-LEDGER 98 (docs/3-status/TEST-LEDGER.md)
Usage   : plot_lme_recall.py CURVE_JSON OUT_STEM   (writes OUT_STEM.pdf and .png)
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "grid.linewidth": 0.4, "pdf.fonttype": 42})
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#d9d8d4"
d = json.load(open(sys.argv[1])); out = sys.argv[2]
ks = [3, 5, 8, 12, 24, 32, 50]
fig, ax = plt.subplots(figsize=(3.45, 2.4))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis="y", color=GRID); ax.set_axisbelow(True)
spec = [("msf", "lexical", BLUE, "-", "o", "multi-session, lexical"),
        ("msf", "embedding", BLUE, "--", "o", "multi-session, embedding"),
        ("trf", "lexical", ORANGE, "-", "s", "temporal-reasoning, lexical"),
        ("trf", "embedding", ORANGE, "--", "s", "temporal-reasoning, embedding")]
for cls, ranker, color, ls, mk, label in spec:
    n = d[cls]["n"]; y = [d[cls][ranker]["all"][str(k)] / n for k in ks]
    ax.plot(ks, y, color=color, linestyle=ls, marker=mk, markersize=3.4,
            markerfacecolor=(color if ranker == "lexical" else "white"), label=label)
ax.axvline(32, color="#777777", linestyle=":", linewidth=0.8)
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
ax.set_ylim(0.3, 1.02); ax.set_xlabel("rows injected per question, $k$"); ax.set_ylabel("recall_all@k")
ax.legend(frameon=False, loc="lower right", handlelength=2.6)
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("saved", out)
