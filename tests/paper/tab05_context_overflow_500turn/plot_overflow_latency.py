#!/usr/bin/env python3
"""plot_overflow_latency.py

Purpose : Figure for the 500-turn overflow run (Table 5): per-turn p50 latency at
Usage   : plot_overflow_latency.py OUT_STEM   (writes OUT_STEM.pdf and .png)
"""
import re, sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.join(os.environ.get("REPO", "."), "artifact", "paper", "logs", "")
NAME = {"llama": "llama.cpp\nLlama-3.2-3B\nRTX 4090", "vllm": "vLLM\nLlama-3.2-3B\nRTX 4090",
        "sglang": "SGLang\nLlama-3.2-3B\nRTX 4090", "qwen3-4b": "llama.cpp\nQwen3-4B\nRTX 4090"}
ORIN = "llama.cpp\nLlama-3.2-3B\nOrin"
ORDER = [NAME["llama"], NAME["vllm"], NAME["sglang"], ORIN, NAME["qwen3-4b"]]

#------------------------------------------------------------
#Func : load
#------------------------------------------------------------
def load(fn, orin=False):
    out = {}
    for part in re.split(r"#### file: ", open(P + fn, encoding="utf-8").read())[1:]:
        m = re.match(r"result_([a-z0-9-]+?)-(stock|qml)-t500\.log", part.split("\n", 1)[0])
        if not m:
            continue
        eng, condition = m.groups()
        p50 = {int(t): int(v.replace(",", "")) for t, v in re.findall(r"turn (\d+) checkpoint: block p50 ([0-9,]+) ms", part)}
        corr = re.findall(r"turn 500 checkpoint:.*?correct (\d+/\d+)", part)[0]
        out[(ORIN if orin else NAME[eng], condition)] = (p50[500], corr)
    return out

S = {}
S.update(load("tab05_500turn_4090.log")); S.update(load("tab05_500turn_orin.log", orin=True)); S.update(load("tab05_500turn_qwen4b.log"))
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
})
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#d9d8d4"
fig, ax = plt.subplots(figsize=(3.45, 1.7))
x = np.arange(len(ORDER)); w = 0.38
b = [S[(o, "stock")][0] for o in ORDER]; q = [S[(o, "qml")][0] for o in ORDER]
ax.bar(x - w / 2, b, w, color=ORANGE, label="baseline")
ax.bar(x + w / 2, q, w, color=BLUE, label="qml-router")
ax.set_yscale("log"); ax.set_ylim(100, 120000)
ax.set_yticks([100, 1000, 10000, 100000])
ax.minorticks_off()
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_ylabel("p50 at turn 500 (ms)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis="y", which="major", color=GRID)
ax.set_axisbelow(True)
for i, o in enumerate(ORDER):
    ax.text(x[i] - w / 2, b[i] * 1.15, f"0/{S[(o, 'stock')][1].split('/')[1]}", ha="center", va="bottom", fontsize=6, color="#555555")
    ax.text(x[i] + w / 2, q[i] * 1.45, S[(o, "qml")][1], ha="center", va="bottom", fontsize=6, color=BLUE, fontweight="bold")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=7, handlelength=1.2, columnspacing=1.0)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/overflow-latency"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("saved", out)
