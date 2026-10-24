"""plot_paper_figs.py

Purpose : Renders the paper's figures (vector PDF, 3.45 in column width, STIX serif fonts, marker+linestyle dual coding for grayscale printing) from benchmark result JSON files
Usage   : python3 plot_paper_figs.py DENSITY_JSON CROSSOVER_JSON DENSITY_OUT
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
})

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRID = "#d9d8d4"

STOCK = dict(color=ORANGE, marker="s", markersize=3.4, linestyle="--",
             label="baseline llama.cpp (recompute)")
QML = dict(color=BLUE, marker="o", markersize=3.4, linestyle="-",
           label="llama.cpp + qml-router (restore)")

#------------------------------------------------------------
#Func : style
#------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID)
    ax.set_axisbelow(True)

#------------------------------------------------------------
#Func : fig_density
#------------------------------------------------------------
def fig_density(results, out):
    data = json.load(open(results))
    levels = sorted(int(k) for k in data["qml"].keys())
    x = list(range(len(levels)))

    def s(mode, key):
        return [data[mode][str(n)][key] for n in levels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True,
                                   gridspec_kw={"hspace": 0.32})
    for ax in (ax1, ax2):
        style(ax)

    MED_S = dict(STOCK, linestyle=":", markerfacecolor="white", label=None)
    MED_Q = dict(QML, linestyle=":", markerfacecolor="white", label=None)
    ax1.plot(x, s("stock", "max"), **STOCK)
    ax1.plot(x, s("qml", "max"), **QML)
    med = "median" if "median" in data["qml"][str(levels[0])] else "p50"
    ax1.plot(x, s("stock", med), **MED_S)
    ax1.plot(x, s("qml", med), **MED_Q)
    ax1.set_yscale("log")
    ax1.set_ylabel("reconnect latency (s)\nslowest: solid, median: dotted")
    ax1.legend(frameon=False, loc="upper left", handlelength=2.4)
    ax1.annotate(f'{s("stock","max")[-1]:.1f}', (x[-1], s("stock","max")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    ax1.annotate(f'{s("qml","max")[-1]:.2f}', (x[-1], s("qml","max")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")

    ax2.plot(x, s("stock", "short_rate"), **STOCK)
    ax2.plot(x, s("qml", "short_rate"), **QML)
    ax2.set_ylabel("short-request throughput (req/s)")
    ax2.set_xlabel("concurrent reconnecting users $N$ (1 arrival / 4 s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in levels])
    ax2.annotate(f'{s("qml","short_rate")[-1]:.1f}', (x[-1], s("qml","short_rate")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    ax2.annotate(f'{s("stock","short_rate")[-1]:.1f}', (x[-1], s("stock","short_rate")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")

    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("saved", out)

#------------------------------------------------------------
#Func : fig_crossover
#------------------------------------------------------------
def fig_crossover(results, out):
    data = json.load(open(results))
    xs = [r["tokens"] / 1000 for r in data["qml"]]
    yq = [r["latency"] for r in data["qml"]]
    ys = [r["latency"] for r in data["stock"]]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    style(ax)
    ax.plot(xs, ys, **STOCK)
    ax.plot(xs, yq, **QML)
    ax.set_xlabel("conversation length at reconnect (K tokens)")
    ax.set_ylabel("time to first answer (s)")
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)
    ax.set_xscale("log", base=2)
    ticks = [2, 4, 8, 16, 32, 64]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-2, 5), fontsize=8, ha="right")
    ax.annotate(f"{yq[-1]:.1f}", (xs[-1], yq[-1]), textcoords="offset points",
                xytext=(-2, 5), fontsize=8, ha="right")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    fig_density(sys.argv[1], sys.argv[3])
    fig_crossover(sys.argv[2], sys.argv[4])
