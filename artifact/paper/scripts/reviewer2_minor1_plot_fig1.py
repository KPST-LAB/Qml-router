"""reviewer2_minor1_plot_fig1.py

Purpose : Renders Fig. 1 of the article, with each chart labeled (a), (b), (c) at its
          top-left corner and a four-entry legend in chart (a) (slowest and median for
          each system), so every line is named in the figure itself. Figure only; no
          measurement.
Usage   : python3 reviewer2_minor1_plot_fig1.py DENSITY_JSON CROSSOVER_JSON DENSITY_OUT CROSSOVER_OUT
          DENSITY_JSON  = data/reviewer2_minor1_density_verified_only.json
          CROSSOVER_JSON = logs/fig01_crossover.json (the result file, not the input in data/)
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
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
})

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRID = "#d9d8d4"

STOCK = dict(color=ORANGE, marker="s", markersize=3.4, linestyle="--")
QML = dict(color=BLUE, marker="o", markersize=3.4, linestyle="-")

#------------------------------------------------------------
#Func : style
#  in: ax -- matplotlib axes
# out: none; hides the top/right spines and draws a light y grid
#------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID)
    ax.set_axisbelow(True)

#------------------------------------------------------------
#Func : panel_label
#  in: ax -- axes to label; text -- "(a)", "(b)" or "(c)"
# out: none; writes the label just above the top-left corner of the axes
#------------------------------------------------------------
def panel_label(ax, text):
    ax.text(-0.02, 1.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")

#------------------------------------------------------------
#Func : fig_density
#  in: results -- verified-only density JSON; out -- output path without extension
# out: none; writes out.pdf and out.png (charts (a) and (b))
#------------------------------------------------------------
def fig_density(results, out):
    data = json.load(open(results))
    levels = sorted(int(k) for k in data["qml"].keys())
    x = list(range(len(levels)))

    def s(mode, key):
        return [data[mode][str(n)][key] for n in levels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.8), sharex=True,
                                   gridspec_kw={"hspace": 0.38})
    for ax in (ax1, ax2):
        style(ax)

    med = "median" if "median" in data["qml"][str(levels[0])] else "p50"
    ax1.plot(x, s("stock", "max"), label="baseline, slowest", **STOCK)
    ax1.plot(x, s("stock", med), label="baseline, median",
             **dict(STOCK, linestyle=":", markerfacecolor="white"))
    ax1.plot(x, s("qml", "max"), label="qml-router, slowest", **QML)
    ax1.plot(x, s("qml", med), label="qml-router, median",
             **dict(QML, linestyle=":", markerfacecolor="white"))
    ax1.set_yscale("log")
    ax1.set_ylim(0.2, 3000)
    ax1.set_ylabel("reconnect latency (s)")
    ax1.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.2,
               columnspacing=0.8, borderaxespad=0.1)
    ax1.annotate(f'{s("stock","max")[-1]:.1f}', (x[-1], s("stock","max")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    ax1.annotate(f'{s("qml","max")[-1]:.2f}', (x[-1], s("qml","max")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    panel_label(ax1, "(a)")

    ax2.plot(x, s("stock", "short_rate"), label="baseline", **STOCK)
    ax2.plot(x, s("qml", "short_rate"), label="qml-router", **QML)
    ax2.set_ylabel("short-request throughput (req/s)")
    ax2.set_xlabel("concurrent reconnecting users $N$ (1 arrival / 4 s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in levels])
    ax2.legend(frameon=False, loc="upper right", handlelength=2.2, borderaxespad=0.1)
    ax2.annotate(f'{s("qml","short_rate")[-1]:.1f}', (x[-1], s("qml","short_rate")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    ax2.annotate(f'{s("stock","short_rate")[-1]:.1f}', (x[-1], s("stock","short_rate")[-1]),
                 textcoords="offset points", xytext=(-2, 5), fontsize=8, ha="right")
    panel_label(ax2, "(b)")

    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("saved", out)

#------------------------------------------------------------
#Func : fig_crossover
#  in: results -- crossover JSON; out -- output path without extension
# out: none; writes out.pdf and out.png (chart (c))
#------------------------------------------------------------
def fig_crossover(results, out):
    data = json.load(open(results))
    xs = [r["tokens"] / 1000 for r in data["qml"]]
    yq = [r["latency"] for r in data["qml"]]
    ys = [r["latency"] for r in data["stock"]]

    # same figure size and grid as fig_density, chart in the top cell: the plot area of (c)
    # then has the same height as each of (a) and (b) when both images share one width
    fig, (ax, ax_off) = plt.subplots(2, 1, figsize=(3.45, 3.8),
                                     gridspec_kw={"hspace": 0.38})
    ax_off.set_visible(False)
    style(ax)
    ax.plot(xs, ys, label="baseline llama.cpp (recompute)", **STOCK)
    ax.plot(xs, yq, label="llama.cpp + qml-router (restore)", **QML)
    ax.set_xlabel("conversation length at reconnect (K tokens)")
    ax.set_ylabel("time to first answer (s)")
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)
    ax.set_xscale("log", base=2)
    ticks = [2, 4, 8, 16, 32, 64]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylim(-0.5, 16)
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-2, 5), fontsize=8, ha="right")
    ax.annotate(f"{yq[-1]:.1f}", (xs[-1], yq[-1]), textcoords="offset points",
                xytext=(-2, 5), fontsize=8, ha="right")
    panel_label(ax, "(c)")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    fig_density(sys.argv[1], sys.argv[3])
    fig_crossover(sys.argv[2], sys.argv[4])
