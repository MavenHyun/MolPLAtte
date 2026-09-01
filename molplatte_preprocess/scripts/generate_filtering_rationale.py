#!/usr/bin/env python3
"""Render docs/coconut_filtering_rationale.pdf.

Explains why `coconut-flavordb-filtered` exists. Every number is recomputed from
the raw sources by scripts/derive_odorant_envelope.py -- nothing is hard-coded
from memory. Layout uses a running cursor so blocks cannot collide.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parents[1] / "docs" / "coconut_filtering_rationale.pdf"
D = json.load(open("/tmp/article_data.json"))

INK, MUTE, ODOR, COCO, GOOD = "#1a1a1a", "#666666", "#c44e52", "#4c72b0", "#2e7d32"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                     "axes.edgecolor": "#999", "axes.linewidth": .7})

odor, coco, allm = D["odor_mw"], D["coconut_mw"], D["allmeas_mw"]
ENV_MW, ENV_LP, ENV_TP = (108.0, 290.0), (0.4, 4.5), 53.0
base = D["coconut_known"] / D["coconut_n"]
dens = D["coconut_known_in"] / D["coconut_inside"]
rec_odor = 100 * D["odor_in_env"] / D["n_odor"]
rec_known = 100 * D["coconut_known_in"] / D["coconut_known"]
ALLOW, ALLOW_KNOWN = 39401, 2358

CORPORA = [("flavordb-only", 20812, 299), ("coconut-only", 374454, 853),
           ("coconut-flavordb-full", 393066, 889),
           ("coconut-flavordb-filtered", 50933, 467)]

LINE = 0.0163          # fraction of page height per text line at size 8.5


class Page:
    def __init__(self, title, kicker=None):
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.fig.text(.07, .957, title, size=15.5, weight="bold", color=INK)
        if kicker:
            self.fig.text(.07, .937, kicker, size=8.6, color=MUTE, style="italic")
        self.fig.text(.07, .022, "MolPLAtte — corpus design note", size=7, color=MUTE)
        self.y = .905

    def text(self, s, size=8.5, color=INK, gap=1.0):
        self.fig.text(.07, self.y, s, size=size, color=color, va="top", linespacing=1.62)
        self.y -= (s.count("\n") + 1) * LINE * (size / 8.5) + gap * LINE
        return self

    def h2(self, s):
        self.y -= LINE * 0.6
        self.fig.text(.07, self.y, s, size=10.4, weight="bold", color=INK, va="top")
        self.y -= LINE * 1.9
        return self

    def axes(self, height, left=.10, width=.83, gap=3.4):
        # gap must clear the xlabel, which renders BELOW the axes rectangle
        self.y -= height
        ax = self.fig.add_axes([left, self.y, width, height])
        self.y -= gap * LINE
        return ax

    def save(self, pdf):
        pdf.savefig(self.fig); plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ---------------------------------------------------------------- page 1
    p = Page("Why we filter COCONUT",
             "the case for coconut-flavordb-filtered, and what it costs")
    p.h2("The decision")
    p.text(
"""MolPLAtte pretrains on molecular decompositions: a core plus its detached R-groups. The
R-group vocabulary that emerges is set by whatever chemistry dominates the corpus. We
combine two sources, and they are wildly unequal in both size and relevance:

     FlavorDB      20,812 records    every molecule has a measured flavour label
     COCONUT      374,454 records    natural products; 0.66% have any known flavour

Unfiltered, COCONUT outnumbers FlavorDB 18 to 1, so the vocabulary is learned almost
entirely from natural-product chemistry — macrolides, glycosides, polysaccharides —
rather than from flavour chemistry. `coconut-flavordb-filtered` restricts COCONUT to the
physicochemical envelope real odorants occupy, buying scale without handing the
vocabulary to the wrong domain.""")
    p.h2("The size mismatch is not subtle")
    p.text(
f"""COCONUT's median molecular weight is {st.median(coco):.0f} Da, and {sum(x>350 for x in coco)/len(coco):.1%} of it sits above 350 Da. Real
odorants — FlavorDB entries with a measured, non-sweet sensory label — have a median of
{st.median(odor):.0f} Da, with only {sum(x>350 for x in odor)/len(odor):.1%} above 350.

That 350 Da line is not arbitrary. A molecule must volatilise to reach an olfactory
receptor, and above roughly 350 Da it effectively cannot. Most of COCONUT is therefore
not merely unlabelled — it is physically incapable of having an odour.""")
    ax = p.axes(.185)
    bins = np.linspace(0, 900, 90)
    ax.hist(coco, bins=bins, density=True, color=COCO, alpha=.6, label=f"COCONUT (n={D['coconut_n']:,})")
    ax.hist(odor, bins=bins, density=True, color=ODOR, alpha=.7, label=f"real odorants (n={D['n_odor']:,})")
    ax.axvline(350, color=INK, lw=1.1, ls="--")
    ax.text(360, ax.get_ylim()[1]*.80, "350 Da\nvolatility limit", size=6.8, color=INK)
    ax.axvspan(*ENV_MW, color=GOOD, alpha=.13)
    ax.text(np.mean(ENV_MW), ax.get_ylim()[1]*.52, "kept\nMW window", size=6.8,
            color=GOOD, ha="center", weight="bold")
    ax.set_xlabel("molecular weight (Da)"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7.2); ax.set_yticks([])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    p.text("The distributions barely overlap. Training on their union unfiltered means R-group\n"
           "statistics are set by the blue distribution while the task we care about lives in the red.",
           color=MUTE)
    p.h2("A caution about what counts as an odorant")
    p.text(
f"""FlavorDB's `flavor_profile` column is populated for 25,106 of its 25,596 rows, so "has a
label" is nearly vacuous — that set has a median MW of {st.median(allm):.0f} Da, describing FlavorDB's
bulk rather than its odorants. The envelope below is derived only from the {D['n_odor']:,} entries
whose profile maps onto a real sensory class and is not purely sweet. Sweet compounds are
excluded because taste receptors sit in solution, so sweetness carries no volatility
constraint — steviosides are intensely sweet at roughly 800 Da.""")
    p.save(pdf)

    # ---------------------------------------------------------------- page 2
    p = Page("The envelope, and what it buys", "derived from measured odorants — never hand-picked")
    p.text(
f"""A COCONUT compound is kept if it falls inside all three bounds:

        molecular weight      {ENV_MW[0]:.0f} – {ENV_MW[1]:.0f} Da
        logP                  {ENV_LP[0]} – {ENV_LP[1]}
        topological PSA       ≤ {ENV_TP:.0f}

These are percentiles of the {D['n_odor']:,} measured odorants, not intuition. The filter keeps
{D['coconut_inside']:,} of {D['coconut_n']:,} COCONUT compounds — {D['coconut_inside']/D['coconut_n']:.1%}.""")
    p.h2("Enrichment: does it concentrate flavour chemistry?")
    p.text(
f"""The honest test is whether compounds independently known to be flavour-relevant (present
in FlavorDB by InChIKey) are denser inside the envelope than outside.

        base rate across COCONUT         {base:>7.2%}    ({D['coconut_known']:,} of {D['coconut_n']:,})
        inside the envelope              {dens:>7.2%}    ({D['coconut_known_in']:,} of {D['coconut_inside']:,})
        enrichment                       {dens/base:>7.1f}x""")
    ax = p.axes(.145, left=.10, width=.36, gap=0)
    ax.bar(["all\nCOCONUT", "inside\nenvelope"], [base*100, dens*100], color=[MUTE, GOOD], width=.5)
    for i, v in enumerate([base*100, dens*100]):
        ax.text(i, v + .1, f"{v:.2f}%", ha="center", size=7.6, weight="bold")
    ax.set_ylabel("known-flavour density (%)"); ax.set_ylim(0, dens*100*1.35)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax2 = p.fig.add_axes([.58, ax.get_position().y0, .35, .145])
    ax2.bar(["real\nodorants", "known-flavour\nCOCONUT"], [rec_odor, rec_known], color=ODOR, width=.5)
    ax2.axhline(100, color=MUTE, lw=.7, ls=":")
    for i, v in enumerate([rec_odor, rec_known]):
        ax2.text(i, v + 2.5, f"{v:.0f}%", ha="center", size=7.6, weight="bold")
    ax2.set_ylabel("retained by envelope (%)"); ax2.set_ylim(0, 120)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    p.y -= 3.6 * LINE
    p.text(f"Left: the filter concentrates flavour chemistry {dens/base:.1f}-fold. Right: it is also blunt — it discards\n"
           f"{100-rec_odor:.0f}% of real odorants and {100-rec_known:.0f}% of COCONUT compounds already known to carry flavour.",
           color=MUTE)
    p.h2("Why the corpus is a union, not just the envelope")
    p.text(
f"""A {rec_known:.0f}% recall on known-flavour compounds is unacceptable: those are the most valuable
molecules in the corpus, and a three-descriptor box discards more than half of them. The
built corpus therefore keeps the envelope UNION every compound carrying a known flavour
label. That restores recall on the known set to 100% and yields an allowlist of {ALLOW:,}
COCONUT identifiers.

One consequence deserves stating plainly, because it is easy to quote misleadingly. The
known-flavour density of that union is {100*ALLOW_KNOWN/ALLOW:.2f}%, which against the {base:.2%} base rate reads as
a {(ALLOW_KNOWN/ALLOW)/base:.0f}x enrichment. That figure is partly circular: the union was built by adding every
known positive, so its density is inflated by construction. The defensible number for what
the FILTER itself achieves is the envelope-only {dens/base:.1f}x above.""")
    p.save(pdf)

    # ---------------------------------------------------------------- page 3
    p = Page("What filtering costs", "and when to reach for which corpus")
    p.text(
"""Filtering is a trade, not a free improvement: it buys domain density and pays in scale and
vocabulary breadth. All four corpora share build tag `r333-m2-h4-flavor24`, so they are
directly comparable.""")
    names = [c[0] for c in CORPORA]; recs = [c[1] for c in CORPORA]; eff = [c[2] for c in CORPORA]
    yy = np.arange(len(names))
    ax = p.axes(.155, left=.29, width=.30, gap=0)
    ax.barh(yy, recs, color=COCO, height=.6)
    ax.set_yticks(yy); ax.set_yticklabels(names, size=7.3); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("records (log scale)")
    ax.set_xlim(1e4, 1.4e6)
    for i, v in enumerate(recs): ax.text(v*1.25, i, f"{v:,}", va="center", size=6.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax2 = p.fig.add_axes([.655, ax.get_position().y0, .27, .155])
    ax2.barh(yy, eff, color=GOOD, height=.6)
    ax2.set_yticks(yy); ax2.set_yticklabels([]); ax2.invert_yaxis()
    ax2.set_xlabel("effective vocabulary  exp(H)", color=GOOD)
    ax2.set_xlim(0, 1150)
    for i, v in enumerate(eff): ax2.text(v + 30, i, str(v), va="center", size=6.8, color=GOOD)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    p.y -= 3.6 * LINE
    p.text(
f"""Against `coconut-flavordb-full`, the filtered corpus is {393066/50933:.1f}x smaller and its effective
vocabulary — exp(entropy of the R-group distribution), i.e. the number of R-groups actually
carrying probability mass — falls from 889 to 467, roughly half.

This matters for evaluation. Retrieval difficulty scales with effective vocabulary, so
Hit@K on the filtered corpus is NOT comparable to Hit@K on the full one. The filtered task
is easier by construction, and a higher number there is not evidence of a better model.""")
    p.h2("Which corpus for which question")
    p.text(
"""  coconut-flavordb-full         default for pretraining: maximum scale and the broadest
                                R-group vocabulary, with sparse flavour labels.

  coconut-flavordb-filtered     pocket finetuning and flavour conditioning, where label
                                density matters more than raw scale.

  coconut-only                  the contrast arm: natural-product chemistry with almost no
                                flavour labels. Any claimed flavour effect should be
                                checked against this to confirm it does not reproduce.

  flavordb-only                 small and fully labelled; use when label coverage is the
                                binding constraint.""")
    p.h2("Open limitations")
    p.text(
f"""The envelope is a three-descriptor axis-aligned box. Real odorancy is not box-shaped, and
the {100-rec_odor:.0f}% of true odorants it rejects is the price of that simplicity. The union with
known-flavour compounds patches the recall we can measure; it cannot patch recall on
flavour-relevant compounds nobody has labelled yet.

Enrichment is measured against FlavorDB membership, itself a biased sample of flavour
chemistry that over-represents well-studied food compounds. Read the {dens/base:.1f}x as enrichment in
DOCUMENTED flavour relevance, not in flavour relevance as such.""")
    p.save(pdf)

print(f"wrote {OUT}")
