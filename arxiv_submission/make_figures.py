"""
Generate publication-quality figures for the semantic Stroop EMNLP paper.

All figures use a consistent academic style:
- Times-like serif for legibility under acl.sty
- ColorBrewer / Tol palettes for color-blind safety
- Vector PDF output
- Black-and-white grayscale-readable encoding (shape/hatch + color)

Outputs go to figures/ next to this script.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def find_data_dir() -> Path:
    """Return the CSV directory used to regenerate the manuscript figures."""
    env_path = os.environ.get("SEMANTIC_STROOP_CSV_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        HERE.parent / "semantic_stroop_final_reproduction_data_package" / "csv",
        HERE.parent / "data" / "csv",
        HERE / ".." / "semantic_stroop_final_reproduction_data_package" / "csv",
    ])

    required = "behavior_summary_by_model.csv"
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / required).exists():
            return candidate

    searched = "\n  - ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        "Could not find the semantic Stroop CSV data package. "
        "Set SEMANTIC_STROOP_CSV_DIR to the csv/ directory, or place the "
        "data package at ../semantic_stroop_final_reproduction_data_package/csv "
        "or ../data/csv.\nSearched:\n  - " + searched
    )


DATA = find_data_dir()
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Two-column EMNLP A4 width is about 16 cm = 6.30"; column width about 7.7 cm = 3.03"
COL_W = 3.05  # inches
FULL_W = 6.30  # inches

# --- Style -------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "patch.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": False,
})

# --- Color palette -----------------------------------------------------------
# Family colors: Tol's bright qualitative palette
FAMILY_COLOR = {
    "Qwen": "#4477AA",     # blue
    "Gemma": "#EE6677",    # red
    "OLMo": "#228833",     # green
    "Mistral": "#CCBB44",  # yellow
}
SCALE_TINT = {  # hue + brightness for instruct
    "base": 1.00,
    "instruct": 0.55,
}

# Mechanism source-group colors (sequential)
SOURCE_COLOR = {
    "definition_subject": "#4477AA",
    "definition_target":  "#EE6677",
    "query_word":         "#228833",
    "definition_subject__definition_target": "#AA3377",
    "definition_subject__query_word":        "#66CCEE",
    "definition_target__query_word":         "#CCBB44",
    "source_all":                            "#000000",
}

# Pretty model labels (in ordering used for forest)
MODEL_ORDER = [
    ("olmo_1b_base",        "OLMo-1B"),
    ("qwen25_15b_base",     "Qwen2.5-1.5B"),
    ("qwen25_15b_instruct", "Qwen2.5-1.5B-IT"),
    ("gemma2_2b_base",      "Gemma-2-2B"),
    ("gemma2_2b_instruct",  "Gemma-2-2B-IT"),
    ("qwen25_7b_base",      "Qwen2.5-7B"),
    ("qwen25_7b_instruct",  "Qwen2.5-7B-IT"),
    ("mistral7b_base",      "Mistral-7B"),
    ("mistral7b_instruct",  "Mistral-7B-IT"),
    ("gemma2_9b_base",      "Gemma-2-9B"),
    ("gemma2_9b_instruct",  "Gemma-2-9B-IT"),
]
MODEL_KEY_TO_LABEL = dict(MODEL_ORDER)

MECH_MODEL_ORDER = [
    ("olmo_1b_base",        "OLMo-1B"),
    ("qwen25_15b_base",     "Qwen2.5-1.5B"),
    ("qwen25_15b_instruct", "Qwen2.5-1.5B-IT"),
    ("gemma2_2b_base",      "Gemma-2-2B"),
    ("gemma2_2b_instruct",  "Gemma-2-2B-IT"),
]


def family_of(model_key: str) -> str:
    if model_key.startswith("qwen"):
        return "Qwen"
    if model_key.startswith("gemma"):
        return "Gemma"
    if model_key.startswith("olmo"):
        return "OLMo"
    if model_key.startswith("mistral"):
        return "Mistral"
    return "?"


def adjust_brightness(hex_color, factor):
    """Mix toward white (factor>1) or toward black (factor<1)."""
    c = mpl.colors.to_rgb(hex_color)
    if factor < 1:
        return tuple(ch * factor for ch in c)
    t = factor - 1
    return tuple(ch + (1 - ch) * t for ch in c)


def save(fig, name, pad_inches=None):
    pdf = OUT / f"{name}.pdf"
    if pad_inches is None:
        fig.savefig(pdf)
    else:
        fig.savefig(pdf, pad_inches=pad_inches)
    plt.close(fig)
    print(f"  wrote {pdf}")


# =============================================================================
# Figure 1: paradigm schematic.
#
# Figure 1 is hand-authored (figures/fig1_paradigm.pdf) using the jaguar /
# car / animal polysemy example with values from Mistral-7B base under the
# game-rule prompt. The script does not regenerate it; if you need to update
# the figure, edit the PDF directly.
# =============================================================================


# =============================================================================
# Figure 2: model-level forest plot
# =============================================================================
def fig2_forest():
    df = pd.read_csv(DATA / "behavior_summary_by_model.csv")
    df = df[df["score_mode"] == "sum_logprob"].copy()

    rows = []
    for mk, label in MODEL_ORDER:
        r = df[df["model_key"] == mk]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        rows.append((mk, label, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]),
                     float(r["params_b"]), str(r["tuning"])))
    rows = rows[::-1]  # so first is at top

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    ax.axvline(0, color="#888", linewidth=0.6, linestyle="--", zorder=0)

    ys = np.arange(len(rows))
    for i, (mk, label, mean, lo, hi, params, tuning) in enumerate(rows):
        fam = family_of(mk)
        color = FAMILY_COLOR[fam]
        face = "white" if tuning == "base" else color
        edge = color
        marker = "o" if tuning == "base" else "s"
        ax.errorbar([mean], [ys[i]], xerr=[[mean - lo], [hi - mean]],
                    fmt=marker, color=color, markeredgecolor=edge,
                    markerfacecolor=face, ecolor=color,
                    elinewidth=0.9, capsize=2, capthick=0.6,
                    markersize=4.0, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlabel(r"Stroop interference $\Delta$ (sum log-prob)")
    ax.set_xlim(0, 3.4)
    ax.tick_params(axis="y", length=0)
    for s in ("left",):
        ax.spines[s].set_color("#bbbbbb")

    # Compact legend: family colors + base/IT marker convention.
    # Placed BELOW the axes (under the x-axis label).
    legend_handles = [
        Line2D([0], [0], marker="o", color=c, markerfacecolor=c,
               markeredgecolor=c, linestyle="None", markersize=4.0, label=fam)
        for fam, c in FAMILY_COLOR.items()
    ] + [
        Line2D([0], [0], marker="o", color="#444",
               markerfacecolor="white", markeredgecolor="#444",
               linestyle="None", markersize=4.0, label="base"),
        Line2D([0], [0], marker="s", color="#444",
               markerfacecolor="#444", markeredgecolor="#444",
               linestyle="None", markersize=4.0, label="instruct"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.27), ncol=6,
              frameon=False, columnspacing=0.9, handletextpad=0.3,
              fontsize=6.5, borderaxespad=0.0)
    fig.subplots_adjust(bottom=0.24)

    save(fig, "fig2_model_forest")


# =============================================================================
# Figure 3: external validity heatmaps (conflict_type × model, prompt_style × model)
# =============================================================================
def _heatmap(ax, mat, row_labels, col_labels, vmin, vmax, cmap):
    # Use TwoSlopeNorm so 0 stays at the white center of the diverging
    # colormap even when the colorbar bounds are asymmetric ([-1, +6]).
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Annotate. With TwoSlopeNorm centered at 0, darkness grows away from 0
    # on each side; use |v|/half-range on that side to decide text color.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            half = -vmin if v < 0 else vmax
            tc = "white" if abs(v) / half > 0.55 else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.0, color=tc)
    return im


def fig3_heatmaps():
    df_ct = pd.read_csv(DATA / "behavior_summary_by_model_conflict_type.csv")
    df_ct = df_ct[df_ct["score_mode"] == "sum_logprob"].copy()
    df_ps = pd.read_csv(DATA / "behavior_summary_by_model_prompt_style.csv")
    df_ps = df_ps[df_ps["score_mode"] == "sum_logprob"].copy()

    model_keys = [mk for mk, _ in MODEL_ORDER]
    model_labels = [lbl for _, lbl in MODEL_ORDER]

    conflict_order = ["antonym_remapping", "arbitrary_semantic_remapping",
                      "polysemy_entity_remapping", "domain_definition_remapping"]
    conflict_labels = ["Antonym", "Arbitrary", "Polysemy", "Domain-def."]

    style_order = ["game_rule", "glossary", "technical_document", "definition_scope"]
    style_labels = ["Game rule", "Glossary", "Tech doc", "Scoped def."]

    # Rows = category (4), cols = model (11). Two panels stacked vertically
    # so the figure is short and wide — fits the two-column ACL page width.
    M_ct = np.full((len(conflict_order), len(model_keys)), np.nan)
    for j, mk in enumerate(model_keys):
        for i, ct in enumerate(conflict_order):
            r = df_ct[(df_ct["model_key"] == mk) & (df_ct["conflict_type"] == ct)]
            if len(r):
                M_ct[i, j] = float(r["mean"].iloc[0])

    M_ps = np.full((len(style_order), len(model_keys)), np.nan)
    for j, mk in enumerate(model_keys):
        for i, ps in enumerate(style_order):
            r = df_ps[(df_ps["model_key"] == mk) & (df_ps["prompt_style"] == ps)]
            if len(r):
                M_ps[i, j] = float(r["mean"].iloc[0])

    # Use a diverging colormap centered at 0. The negative half is much
    # smaller than the positive half (only one cell is negative,
    # antonym × Gemma-2-9B-IT at -0.63), so we asymmetric-clip the colorbar
    # to [-1, +6]: the single negative cell is still identifiable as the
    # only blue cell, while the positive half gets the contrast it needs.
    vmin, vmax = -1.0, 6.0
    cmap = mpl.cm.get_cmap("RdBu_r")

    # Two panels stacked vertically with a shared colorbar on the right.
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 2.45),
                             constrained_layout=True)
    im0 = _heatmap(axes[0], M_ct, conflict_labels, model_labels, vmin, vmax, cmap)
    axes[0].set_title("(a) Effect by conflict family", loc="left",
                      fontsize=8.5, weight="bold", pad=4)
    # Top panel: hide x-tick labels (model labels appear under the bottom panel).
    axes[0].tick_params(axis="x", labelbottom=False)
    im1 = _heatmap(axes[1], M_ps, style_labels, model_labels, vmin, vmax, cmap)
    axes[1].set_title("(b) Effect by prompt style", loc="left",
                      fontsize=8.5, weight="bold", pad=4)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.015, aspect=18)
    cbar.set_label(r"$\Delta$ (sum log-prob)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)

    save(fig, "fig3_heatmaps")


# =============================================================================
# Figure 4: lexical-prior strength predicts interference (binned scatter + fit)
# =============================================================================
def fig4_lexical_prior():
    df = pd.read_csv(DATA / "behavior_item_prompt_effects.csv")
    # Aggregate at item-prompt level for sum_logprob only
    df = df[df["score_mode"] == "sum_logprob"].copy()
    df = df.dropna(subset=["lexical_distractor_advantage_sum", "stroop_effect"])
    x = df["lexical_distractor_advantage_sum"].astype(float).values
    y = df["stroop_effect"].astype(float).values
    fams = df["family"].values

    # Bin by lexical advantage decile
    qs = np.quantile(x, np.linspace(0, 1, 11))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    bins = np.digitize(x, qs) - 1
    bins = np.clip(bins, 0, 9)
    bin_x, bin_y, bin_se = [], [], []
    for k in range(10):
        m = bins == k
        if m.sum() < 5:
            continue
        bin_x.append(np.mean(x[m]))
        bin_y.append(np.mean(y[m]))
        bin_se.append(np.std(y[m], ddof=1) / np.sqrt(m.sum()))
    bin_x, bin_y, bin_se = map(np.array, (bin_x, bin_y, bin_se))

    # Linear fit on the raw item-prompt data
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
    ys = slope * xs + intercept

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    # Light scatter cloud (subsampled, by family)
    rng = np.random.default_rng(7)
    n = len(x)
    idx = rng.choice(n, size=min(2500, n), replace=False)
    for fam, c in FAMILY_COLOR.items():
        m = (fams[idx] == fam)
        ax.scatter(x[idx][m], y[idx][m], s=2, color=c, alpha=0.10, linewidth=0)
    # Binned means
    ax.errorbar(bin_x, bin_y, yerr=bin_se, fmt="o", color="#222222",
                markerfacecolor="white", markeredgecolor="#222222",
                markersize=3.4, capsize=1.6, capthick=0.5, elinewidth=0.6,
                linewidth=0, zorder=4, label="decile mean ±1 SE")
    # Linear fit
    ax.plot(xs, ys, color="#990000", linewidth=1.0, zorder=5,
            label=f"OLS slope = {slope:.3f}")
    ax.axhline(0, color="#888", linestyle=":", linewidth=0.5, zorder=1)

    ax.set_xlabel(r"Lexical-prior advantage  $\log P(\mathrm{distractor})-\log P(\mathrm{target})$"
                  + "\n(ordinary-prior prompt)")
    ax.set_ylabel(r"Stroop interference $\Delta$")
    ax.set_xlim(np.percentile(x, 0.5), np.percentile(x, 99.5))
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4)

    save(fig, "fig4_lexical_prior")


# =============================================================================
# Figure 5: mechanism core (full width, 2 panels)
#  (a) layer-wise recovery curves for source groups (one model: gemma2_2b_base)
#  (b) source_all vs best_pair (and weakest singletons) across 5 mechanism models
# =============================================================================
def fig5_mechanism_core():
    df_lay = pd.read_csv(DATA / "mechanism_source_residual_patches.csv")
    df_sum = pd.read_csv(DATA / "mechanism_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.45),
                             gridspec_kw={"width_ratios": [1.2, 1.0]},
                             constrained_layout=True)

    # --- Panel A: layer-wise recovery curves for one representative model ---
    # Use the residual-stream hook the full triplet selects in the summary
    # (not always pre — Gemma-2-2B base now selects mid).
    rep_model = "gemma2_2b_base"
    rep_hook = df_sum[df_sum["model_key"] == rep_model
                      ]["best_source_component"].iloc[0]
    sub = df_lay[(df_lay["model_key"] == rep_model) &
                 (df_lay["component"] == rep_hook)].copy()
    n_layers = int(sub["layer"].max()) + 1
    rep_hook_label = {
        "hook_resid_pre": "pre",
        "hook_resid_mid": "mid",
        "hook_resid_post": "post",
    }.get(rep_hook, rep_hook)

    # Encode group type by linestyle (in addition to color, for legibility
    # when the seven curves overlap and to make singleton/pair/triplet
    # visually distinguishable without consulting the legend):
    #   singleton  -> solid (thin)
    #   pair       -> dashed
    #   triplet    -> solid + bold
    SINGLETONS = {"definition_target", "query_word", "definition_subject"}
    PAIRS = {
        "definition_target__query_word",
        "definition_subject__definition_target",
        "definition_subject__query_word",
    }
    plot_order = [
        ("definition_target", "def_target"),
        ("query_word", "query_word"),
        ("definition_subject", "def_subject"),
        ("definition_target__query_word", "def_target + query"),
        ("definition_subject__definition_target", "def_subj + def_target"),
        ("definition_subject__query_word", "def_subj + query"),
        ("source_all", "Full triplet"),
    ]

    ax = axes[0]
    for grp, label in plot_order:
        s = sub[sub["group"] == grp].sort_values("layer")
        if len(s) == 0:
            continue
        c = SOURCE_COLOR[grp]
        if grp == "source_all":
            ls, lw, alpha = "-", 1.8, 1.0
        elif grp in SINGLETONS:
            ls, lw, alpha = "-", 0.9, 0.9
        else:  # PAIRS
            ls, lw, alpha = (0, (4, 1.5)), 1.0, 0.9
        ax.plot(s["layer"], s["recovery"], color=c, linewidth=lw,
                linestyle=ls, alpha=alpha, label=label,
                marker="o" if grp == "source_all" else None,
                markersize=2.4 if grp == "source_all" else 0)

    ax.set_xlabel(f"layer (residual stream, {rep_hook_label})")
    ax.set_ylabel("recovery $R$")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.2, 1.25)
    ax.set_title("(a) Layer-wise patching by source group" +
                 f" ({MODEL_KEY_TO_LABEL[rep_model]})",
                 loc="left", fontsize=8.5, weight="bold", pad=4)
    ax.legend(loc="upper right", ncol=1,
              fontsize=5.6, frameon=False, handlelength=1.4,
              labelspacing=0.20, handletextpad=0.4,
              borderaxespad=0.2)

    # --- Panel B: source_all vs best_pair across 5 mechanism models ---
    ax = axes[1]
    rows = []
    for mk, lbl in MECH_MODEL_ORDER:
        r = df_sum[df_sum["model_key"] == mk].iloc[0]
        rows.append((lbl, float(r["best_source_recovery"]), float(r["best_pair_recovery"])))

    labels = [r[0] for r in rows]
    src = [r[1] for r in rows]
    pair = [r[2] for r in rows]

    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h/2, pair, height=h, color="#cccccc", edgecolor="#666",
            linewidth=0.5, label="best pair")
    ax.barh(y + h/2, src, height=h, color="#222222", edgecolor="#222",
            linewidth=0.5, label="Full triplet")
    # Annotate values on right
    for i, (p, s) in enumerate(zip(pair, src)):
        ax.text(p + 0.02, y[i] - h/2, f"{p:.2f}", va="center", fontsize=6.2, color="#444")
        ax.text(s + 0.02, y[i] + h/2, f"{s:.2f}", va="center", fontsize=6.2, color="#222")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.30)
    ax.set_xlabel("recovery $R$")
    ax.set_title("(b) Triplet exceeds best pair in every model",
                 loc="left", fontsize=8.5, weight="bold", pad=4)
    ax.tick_params(axis="y", length=0)
    # Push the top of the data range UP (more negative y after invert_yaxis)
    # so a legend can fit in the empty strip ABOVE the OLMo-1B bars without
    # the title being displaced.
    ymin_now, ymax_now = ax.get_ylim()  # invert_yaxis was already called
    # ax.invert_yaxis() makes ymin > ymax; the visual top corresponds to
    # the smaller of the two. Add headroom there.
    ax.set_ylim(top=min(ymin_now, ymax_now) - 0.55)
    # Inline legend at upper-CENTER, INSIDE the axes (anchor < 1.0 so the
    # title is not pushed up). Two entries side-by-side keeps it compact.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99),
              ncol=2, frameon=True, framealpha=0.92,
              edgecolor="#bbbbbb", facecolor="white",
              fontsize=6.4, handlelength=1.4, handletextpad=0.4,
              columnspacing=1.0, borderpad=0.35)

    save(fig, "fig5_mechanism_core")


# =============================================================================
# Figure 6: mechanism evidence (column width)
#  Matched-triplet recovery vs item-mismatched controls per model.
#  The output filename keeps the historical fig7_evidence.pdf name used by
#  main.tex; in the compiled manuscript it is numbered as Figure 6.
# =============================================================================
def fig7_evidence():
    # Single-panel column-width figure: matched triplet vs item-mismatched
    # controls. The reader zero-ablation panel (formerly panel b) has been
    # moved to Appendix F Table tab:mediation; the body cites that table.
    df_rnd = pd.read_csv(DATA / "mechanism_random_controls.csv")

    fig, ax = plt.subplots(figsize=(COL_W, 1.75), constrained_layout=True)

    rng = np.random.default_rng(0)
    for i, (mk, lbl) in enumerate(MECH_MODEL_ORDER):
        sub = df_rnd[df_rnd["model_key"] == mk]
        random = sub["random_recovery"].values
        matched = float(sub["matched_recovery"].iloc[0])
        # jittered scatter for random
        jit = rng.normal(0, 0.06, size=len(random))
        ax.scatter(np.full_like(random, i, dtype=float) + jit, random,
                   s=3, color="#888", alpha=0.5, linewidth=0)
        # mean of random as horizontal tick
        rmean = random.mean()
        ax.plot([i - 0.18, i + 0.18], [rmean, rmean],
                color="#555", linewidth=0.9)
        # matched as colored diamond
        ax.scatter([i], [matched], s=40, marker="D",
                   color=FAMILY_COLOR[family_of(mk)],
                   edgecolor="black", linewidth=0.5, zorder=4)

    ax.axhline(0, color="#aaaaaa", linestyle=":", linewidth=0.5)
    ax.axhline(1, color="#aaaaaa", linestyle=":", linewidth=0.5)
    ax.set_xticks(range(len(MECH_MODEL_ORDER)))
    ax.set_xticklabels([lbl for _, lbl in MECH_MODEL_ORDER],
                       rotation=20, ha="right", fontsize=6.8)
    ax.set_ylabel("recovery $R$")
    # Add headroom above the matched diamonds (which sit near R=1.0) so the
    # legend can live inside the axes with a clear vertical gap above the
    # data row.
    ymin_now, ymax_now = ax.get_ylim()
    ax.set_ylim(ymin_now, max(ymax_now, 1.05) + 1.30)
    leg_handles = [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#444", markeredgecolor="black",
               markersize=5, label="matched triplet"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#888", markeredgecolor="none",
               markersize=4, label="item-mismatched"),
    ]
    ax.legend(handles=leg_handles,
              loc="upper center", bbox_to_anchor=(0.5, 1.00),
              ncol=2, frameon=False,
              fontsize=6.5, handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0.0)

    save(fig, "fig7_evidence")


# =============================================================================
# Optional diagnostic: logit decomposition (column width).
# The current manuscript reports this decomposition as Table 3, so this helper
# is kept for ad hoc diagnostics and is not called by main().
# =============================================================================
def fig6_logit_decomp():
    df = pd.read_csv(DATA / "mechanism_logit_decomposition.csv")
    rows = []
    for mk, lbl in MECH_MODEL_ORDER:
        r = df[df["model_key"] == mk].iloc[0]
        rows.append((lbl, mk,
                     float(r["target_logit_delta"]),
                     float(r["distractor_logit_delta"]),
                     float(r["logit_diff_delta"])))

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    y = np.arange(len(rows))
    h = 0.36
    tar = [r[2] for r in rows]
    dis = [r[3] for r in rows]
    diff = [r[4] for r in rows]

    # Set xlim wide enough for the most negative bar/label and to leave room
    # on the right for the margin annotation. Some models can have a slightly
    # positive target_logit_delta (e.g. Gemma-2-2B), in which case its label
    # sits to the right of zero and the margin annotation must not collide
    # with it.
    XMIN, XMAX = -14.0, 7.5
    PAD = 0.45  # horizontal padding between bar end and value text

    ax.barh(y - h/2, tar, height=h, color="#EE6677",
            edgecolor="#7a3540", linewidth=0.4, label="target $\\Delta\\ell$")
    ax.barh(y + h/2, dis, height=h, color="#4477AA",
            edgecolor="#264a72", linewidth=0.4, label="distractor $\\Delta\\ell$")

    for i, (t, d, df_v) in enumerate(zip(tar, dis, diff)):
        # Place each value label on the outside end of its bar so it never
        # overlaps the bar itself, regardless of sign.
        if t >= 0:
            ax.text(t + PAD, y[i] - h/2, f"{t:+.2f}",
                    va="center", ha="left", fontsize=6.0, color="#5a2630")
        else:
            ax.text(t - PAD, y[i] - h/2, f"{t:+.2f}",
                    va="center", ha="right", fontsize=6.0, color="#5a2630")
        if d >= 0:
            ax.text(d + PAD, y[i] + h/2, f"{d:+.2f}",
                    va="center", ha="left", fontsize=6.0, color="#1f3a5e")
        else:
            ax.text(d - PAD, y[i] + h/2, f"{d:+.2f}",
                    va="center", ha="right", fontsize=6.0, color="#1f3a5e")
        # Difference annotation on the right (positive side), pushed far
        # enough right that any positive target/distractor value labels
        # cannot overlap it.
        ax.text(3.0, y[i], r"$\Delta_{\!\mathrm{tgt-dis}}=$"
                f"{df_v:+.2f}", va="center", fontsize=5.8, color="#222")

    ax.axvline(0, color="#888", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"logit change at best triplet patch")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(XMIN, XMAX)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02),
              frameon=False, fontsize=6.5,
              handlelength=1.2, handletextpad=0.4)

    save(fig, "fig6_logit_decomp")


# =============================================================================
def main():
    print("Generating figures…")
    # Figure 1 is hand-authored (figures/fig1_paradigm.pdf) and not regenerated.
    fig2_forest()
    fig3_heatmaps()
    fig4_lexical_prior()
    fig5_mechanism_core()
    fig7_evidence()
    print("Done.")


if __name__ == "__main__":
    main()
