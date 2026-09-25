"""
Data charts, drawn at their exact render width.

Each chart also writes the values it plotted to paper/figures/<name>.json, and
check_figures.py recomputes those values from the raw logs, so a chart cannot
drift from the data it claims to show. No numbers are drawn inside the axes.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import stats as S  # noqa: E402

FIG = A.ROOT / "paper" / "figures"
COL, FULL = 3.45, 7.10
INK = "#1f2733"
PLACE_COLOR = {"model": "#245b81", "bus": "#2f7d68", "tool": "#d97706"}
PLACE_LABEL = {"model": "model client", "bus": "message bus", "tool": "tool boundary"}
ENTRY_LABEL = {"tool": "tool result", "user": "user task", "agent": "agent message", "memory": "memory"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def rate_bars(ax, groups, series, values, colors, labels, unreachable=frozenset()):
    """Grouped bars with Wilson intervals. values[g][s] = (k, n). A (g, s)
    cell in `unreachable` is drawn as a cross with no interval: the placement
    never received the injected text in time, so the zero is structural and
    not a measured rate."""
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(groups):
            if (g, s) in unreachable:
                ax.plot(i - 0.4 + w * (j + 0.5), 0.03, marker="x", color=colors[s], ms=4, mew=1.1)
                continue
            k, n = values[g][s]
            p = k / n if n else 0.0
            a, b = S.wilson(k, n) if n else (0.0, 0.0)
            xs.append(i - 0.4 + w * (j + 0.5))
            ys.append(p)
            lo.append(max(0.0, p - a))
            hi.append(max(0.0, b - p))
        ax.bar(xs, ys, w * 0.92, color=colors[s], label=labels[s])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])


def fig_catch(rows):
    """Share of attack runs in which each placement flags the injection in
    time, per entry point, for both guards."""
    atk = [r for r in rows if r["label"] == "attack"]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    for ax, g, title in zip(axes, ("llm", "deberta"), ("LLM guard", "DeBERTa")):
        vals, unreach = {}, set()
        for e in A.ENTRIES:
            a = [r for r in atk if r["entry"] == e]
            vals[e] = {pl: (sum(r[f"{pl}|{g}"]["catches"] for r in a), len(a)) for pl in A.PLACEMENTS}
            for pl in A.PLACEMENTS:
                if a and not any(r[f"{pl}|{g}"]["sees_in_time"] for r in a):
                    unreach.add((e, pl))
        rate_bars(ax, A.ENTRIES, A.PLACEMENTS, vals, PLACE_COLOR, PLACE_LABEL, unreach)
        out[g + "_unreachable"] = sorted(f"{e}|{pl}" for e, pl in unreach)
        ax.set_xticklabels([ENTRY_LABEL[e] for e in A.ENTRIES])
        ax.set_title(title, fontsize=8)
        out[g] = vals
    axes[0].set_ylabel("attack runs caught in time (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.55, rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_catch.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    (FIG / "fig_catch.json").write_text(json.dumps(out, indent=1))


GROUP_LABEL = {"attack": "attack", "matched": "matched\nbenign", "hn_imperative": "imperative\nhard neg.",
               "hn_vocab": "vocabulary\nhard neg.", "hn_identifier": "identifier\nhard neg."}


def fig_gateway(rows):
    """Gateway block rate per probe group, split by what triggered the block."""
    parts = [("carrier", "regex matched the carrier", "#8a93a2"),
             ("passage", "regex needed the passage", "#d97706"),
             ("ml", "DeBERTa classifier", "#245b81")]
    out = {}
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    for i, g in enumerate(A.GROUPS):
        x = [r for r in rows if r["group"] == g]
        n = len(x)
        cnt = {"carrier": sum(r["blocked"] and r["kind"] == "regex" and r["carrier_hit"] for r in x),
               "passage": sum(r["blocked"] and r["kind"] == "regex" and not r["carrier_hit"] for r in x),
               "ml": sum(r["blocked"] and r["kind"] == "ml" for r in x)}
        bottom = 0.0
        for key, label, color in parts:
            h = cnt[key] / n
            ax.bar(i, h, 0.62, bottom=bottom, color=color, label=label if i == 0 else None)
            bottom += h
        k = sum(r["blocked"] for r in x)
        lo, hi = S.wilson(k, n)
        ax.errorbar(i, k / n, yerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], fmt="none", ecolor=INK,
                    elinewidth=0.7, capsize=1.8)
        out[g] = {"n": n, "blocked": k, **cnt}
    ax.set_xticks(range(len(A.GROUPS)))
    ax.set_xticklabels([GROUP_LABEL[g] for g in A.GROUPS], fontsize=6.8)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("probes blocked (%)")
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(pad=0.55, rect=(0, 0, 1, 0.84))
    fig.savefig(FIG / "fig_gateway.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    (FIG / "fig_gateway.json").write_text(json.dumps(out, indent=1))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = A.phase2()
    if rows:
        fig_catch(rows)
    g = A.phase7()
    if g:
        fig_gateway(g)
    print("figures written")


if __name__ == "__main__":
    main()
