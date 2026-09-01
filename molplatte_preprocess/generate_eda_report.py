#!/usr/bin/env python
"""EDA report for a built MolPallete corpus, as a PDF.

Reports the statistics that actually drive modelling decisions here: what the
decomposition produces, how large the islinked augmentation space is, how skewed
the R-group vocabulary is (which sets the floor any retrieval number must clear),
and whether the attributes the assembly head recovers carry signal or are
structurally constant.

Page layout follows MolDAM_prep/generate_report.py -- matplotlib PdfPages, Agg
backend, one figure per page -- so the two projects' reports read alike.

    python generate_eda_report.py \\
      --corpus /home/mogan/preprocessed/molpallete/coconut-flavordb_full_v2/naveja_recap \\
      --out docs/molpallete_corpus_eda.pdf
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from molpallete_prep.lmdb_store import hydrate
from molpallete_prep.preprocess import path_for
from molpallete_prep.rgroup_library import load_vocabulary

PAGE = (8.5, 11)

#: core / R-group highlight colours (RGB 0-1), colour-blind safe.
_CORE_RGB = (0.30, 0.45, 0.69)
_RG_RGB = (0.87, 0.52, 0.32)


def _draw_decomposition(mol, core_atoms, rgroups, size=(420, 340)):
    """Render a molecule with its core and R-groups highlighted.

    Core atoms in blue, R-group atoms in orange, and the cut bonds drawn wide so
    the decomposition is legible rather than merely colour-coded.
    """
    core = set(int(a) for a in core_atoms)
    rg = set()
    cut_bonds = []
    for r in rgroups:
        rg |= {int(a) for a in r["rgroup_atoms"]}
        b = mol.GetBondBetweenAtoms(int(r["core_linker"]), int(r["rgroup_linker"]))
        if b is not None:
            cut_bonds.append(b.GetIdx())
    hl_atoms = list(core | rg)
    colors = {a: (_CORE_RGB if a in core else _RG_RGB) for a in hl_atoms}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.addStereoAnnotation = True
    o.highlightBondWidthMultiplier = 16
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol,
        highlightAtoms=hl_atoms, highlightAtomColors=colors,
        highlightBonds=cut_bonds,
        highlightBondColors={b: (0.85, 0.15, 0.15) for b in cut_bonds},
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def _page_examples(pdf, examples, title):
    """A grid of rendered decompositions, one row per example."""
    import io
    import matplotlib.image as mpimg
    n = len(examples)
    if not n:
        return
    fig, axs = plt.subplots(3, 2, figsize=PAGE)
    fig.suptitle(title, size=14, weight="bold", x=0.09, ha="left")
    for ax, ex in zip(axs.ravel(), examples):
        ax.imshow(mpimg.imread(io.BytesIO(ex["png"]), format="png"))
        ax.set_title(ex["caption"], size=7.5, loc="left")
        ax.axis("off")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.text(0.09, 0.045,
             "blue = core   orange = R-groups   red = cut bonds",
             size=9, color="0.3")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def _page_title(pdf, title, subtitle=""):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.5, 0.62, title, ha="center", size=24, weight="bold")
    if subtitle:
        for i, line in enumerate(subtitle.split("\n")):
            fig.text(0.5, 0.54 - 0.03 * i, line, ha="center", size=11, color="0.3")
    pdf.savefig(fig); plt.close(fig)


def _page_table(pdf, title, header, rows, note=""):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    ax.set_title(title, size=14, weight="bold", loc="left", pad=18)
    tbl = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.45)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#e8e8e8"); tbl[0, j].set_text_props(weight="bold")
    if note:
        ax.text(0, -0.04 - 0.028 * len(rows) * 0.0, note, transform=ax.transAxes,
                va="top", size=8.5, color="0.25", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _page_text(pdf, title, lines):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.93, title, size=14, weight="bold", va="top")
    fig.text(0.08, 0.88, "\n".join(lines), size=9.5, va="top", family="monospace")
    pdf.savefig(fig); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=6000)
    ap.add_argument("--common-percentile", type=float, default=None,
                    help="also report the effect of MolPLA's instance-level "
                         "common-R-group filter at this percentile")
    ap.add_argument("--vocab", default=None,
                    help="R-group vocabulary (default <corpus>/rgroup_vocab.pkl.gz)")
    a = ap.parse_args()

    root = Path(a.corpus)
    meta = json.loads((root / "__meta__.json").read_text())
    layout = meta.get("layout", "hash3")
    ids = meta["ids"]
    nd = np.asarray(meta["n_decomps"])
    krows = meta["n_rgroups_per_decomp"]
    ka = np.asarray([k for row in krows for k in row])
    variants = np.asarray([sum((1 << k) - 1 for k in row) for row in krows], float)
    src = collections.Counter("flavordb" if i.startswith("FDB") else "coconut" for i in ids)

    # ---- record scan ----
    step = max(1, len(ids) // a.sample)
    core_n, rg_n, mol_n = [], [], []
    ring_cut = joints = 0
    chg = chir = arom = collections.Counter()
    chg, chir, arom = collections.Counter(), collections.Counter(), collections.Counter()
    # Examples span the k range so the page shows what a k=1 decomposition looks
    # like next to a multi-R-group one, rather than six near-identical pictures.
    want_k = [1, 1, 2, 2, 3, 4]
    examples: list[dict] = []
    for mid in ids[::step][: a.sample]:
        try:
            rec = hydrate(torch.load(str(path_for(root, mid, layout)), weights_only=False))
        except Exception:
            continue
        g = rec["original"]
        mol_n.append(int(g.num_nodes))
        chg.update(int(x) for x in g.formal_charge)
        chir.update(int(x) for x in g.chiral_tag)
        m = Chem.MolFromSmiles(rec.get("smiles", "") or "")
        if m is not None and want_k and 8 <= m.GetNumHeavyAtoms() <= 34:
            for d in rec["decompositions"]:
                k = d["n_rgroups"]
                if k in want_k and len(examples) < 6:
                    try:
                        png = _draw_decomposition(m, d["core_atoms"], d["rgroups"])
                    except Exception:
                        break
                    want_k.remove(k)
                    examples.append({
                        "png": png,
                        "caption": (f"{rec['mol_id']}   k={k}   "
                                    f"core {len(d['core_atoms'])} atoms, "
                                    f"R-groups {[len(r['rgroup_atoms']) for r in d['rgroups']]}"),
                    })
                    break
        for d in rec["decompositions"]:
            core_n.append(len(d["core_atoms"]))
            for r in d["rgroups"]:
                rg_n.append(len(r["rgroup_atoms"])); joints += 1
                if m is not None:
                    b = m.GetBondBetweenAtoms(int(r["core_linker"]), int(r["rgroup_linker"]))
                    if b is not None:
                        arom[bool(b.GetIsAromatic())] += 1
                        if b.IsInRing():
                            ring_cut += 1

    vocab = None
    vpath = Path(a.vocab) if a.vocab else root / "rgroup_vocab.pkl.gz"
    if vpath.is_file():
        vocab = load_vocabulary(vpath)

    with PdfPages(a.out) as pdf:
        _page_title(pdf, "MolPallete — Corpus EDA",
                    f"{root.name}\n{root}\n\ngenerated {datetime.now():%Y-%m-%d %H:%M}")

        # A. provenance
        rows = [[k, str(meta[k])[:64]] for k in
                ("sources", "method", "decomposition_params", "core_ratio", "max_cores",
                 "max_rgroups", "keep_stereo", "neutralise", "dedup",
                 "n_duplicates_removed", "size_filter", "graph_hash_version",
                 "rdkit_version", "created_at") if k in meta]
        _page_table(pdf, "A · Provenance", ["field", "value"], rows)

        # B. composition
        rows = [["molecules", f"{len(ids):,}"]]
        for k, v in src.most_common():
            rows.append([f"  from {k}", f"{v:,}  ({v/len(ids):.1%})"])
        rows += [
            ["decompositions", f"{int(nd.sum()):,}"],
            ["  per molecule", f"mean {nd.mean():.2f}, median {int(np.median(nd))}, max {int(nd.max())}"],
            ["R-group slots", f"{int(ka.sum()):,}"],
            ["  per decomposition", f"mean {ka.mean():.2f}"],
            ["k >= 2", f"{float((ka>=2).mean()):.2%} of decompositions"],
        ]
        _page_table(pdf, "B · Composition", ["quantity", "value"], rows,
                    note="k>=2 is what the islinked subset space and the core-decoration "
                         "objective have to work with.")

        # C. augmentation
        rows = [
            ["sampling_unit = molecule", f"{len(ids):,} instances/epoch"],
            ["sampling_unit = decomposition", f"{int(nd.sum()):,} instances/epoch  "
                                              f"({nd.sum()/len(ids):.1f}x)"],
            ["full (mol, core, islinked) space", f"{int(variants.sum()):,}  "
                                                 f"({variants.mean():.1f} per molecule)"],
        ]
        _page_table(pdf, "C · Augmentation space", ["scheme", "size"], rows,
                    note="MolPLA's two stages: one molecule yields many cores; one core "
                         "yields 2^k - 1 islinked subsets.")

        # D. distributions
        fig, axs = plt.subplots(3, 1, figsize=PAGE, gridspec_kw={"hspace": 0.45})
        for ax, (name, xs, col) in zip(axs, (
                ("molecule size (heavy atoms)", mol_n, "#4c72b0"),
                ("core size (atoms)", core_n, "#dd8452"),
                ("R-group size (atoms)", rg_n, "#55a868"))):
            if xs:
                ax.hist(xs, bins=60, color=col, edgecolor="none")
                ax.axvline(st.mean(xs), color="k", ls="--", lw=1,
                           label=f"mean {st.mean(xs):.1f}")
                ax.axvline(np.median(xs), color="k", ls=":", lw=1,
                           label=f"median {np.median(xs):.0f}")
                ax.legend(fontsize=8); ax.set_title(name, size=11)
                ax.set_ylabel("count", size=9)
        fig.suptitle("D · Size distributions", size=14, weight="bold", x=0.09, ha="left")
        pdf.savefig(fig); plt.close(fig)

        _page_examples(pdf, examples,
                       f"E · Example decompositions  (ratio {meta.get('core_ratio')})")

        # F. k distribution + chemistry preserved
        fig, axs = plt.subplots(2, 1, figsize=PAGE, gridspec_kw={"hspace": 0.4})
        kc = collections.Counter(ka.tolist())
        kk = sorted(kc)[:10]
        axs[0].bar([str(k) for k in kk], [kc[k] for k in kk], color="#4c72b0")
        axs[0].set_title("R-groups per decomposition (k)", size=11)
        axs[0].set_xlabel("k"); axs[0].set_ylabel("decompositions")
        for i, k in enumerate(kk):
            axs[0].text(i, kc[k], f"{kc[k]/len(ka):.1%}", ha="center", va="bottom", size=8)
        labels = ["charged atoms\n(--no-neutralise)", "stereocentres\n(keep_stereo)",
                  "ring-bond cuts\n(macrocycles)"]
        n_chg = sum(v for k, v in chg.items() if k != 5)
        n_chir = sum(v for k, v in chir.items() if k != 0)
        vals = [n_chg / max(sum(chg.values()), 1),
                n_chir / max(sum(chir.values()), 1),
                ring_cut / max(joints, 1)]
        axs[1].bar(labels, vals, color=["#c44e52", "#8172b3", "#937860"])
        axs[1].set_title("chemistry the new preprocessing preserves", size=11)
        axs[1].set_ylabel("fraction")
        for i, v in enumerate(vals):
            axs[1].text(i, v, f"{v:.2%}", ha="center", va="bottom", size=9)
        fig.suptitle("F · Decomposition shape & preserved chemistry",
                     size=14, weight="bold", x=0.09, ha="left")
        pdf.savefig(fig); plt.close(fig)

        # F. vocabulary
        if vocab is not None:
            counts = np.asarray([e.count for e in vocab.entries.values()], float)
            p = counts / counts.sum()
            order = np.argsort(-counts)
            cum = np.cumsum(p[order])
            fig, axs = plt.subplots(2, 1, figsize=PAGE, gridspec_kw={"hspace": 0.4})
            axs[0].loglog(np.arange(1, len(counts) + 1), counts[order], color="#4c72b0")
            axs[0].set_title("R-group frequency vs rank (log-log)", size=11)
            axs[0].set_xlabel("rank"); axs[0].set_ylabel("occurrences")
            axs[0].grid(alpha=0.3, which="both")
            axs[1].semilogx(np.arange(1, len(cum) + 1), cum, color="#c44e52")
            axs[1].set_title("cumulative share = the frequency prior a retriever must beat",
                             size=11)
            axs[1].set_xlabel("top-K R-groups"); axs[1].set_ylabel("prior hit@K")
            axs[1].grid(alpha=0.3, which="both")
            for K in (1, 10, 100, 1000):
                if K <= len(cum):
                    axs[1].annotate(f"hit@{K}={cum[K-1]:.3f}", (K, cum[K-1]),
                                    textcoords="offset points", xytext=(6, -10), size=8)
            fig.suptitle("G · R-group library skew", size=14, weight="bold", x=0.09, ha="left")
            pdf.savefig(fig); plt.close(fig)

            rows = [
                ["distinct R-groups", f"{len(vocab):,}"],
                ["occurrences", f"{int(counts.sum()):,}"],
                ["effective size (exp H)", f"{vocab.effective_size():,.0f}"],
                ["top-1 share", f"{p[order[0]]:.2%}"],
                ["top-10 cover", f"{p[order[:10]].sum():.1%}"],
                ["top-100 cover", f"{p[order[:100]].sum():.1%}"],
            ]
            _page_table(pdf, "H · R-group library", ["quantity", "value"], rows,
                        note="Effective size, not the raw count, is the number of classes "
                             "retrieval actually chooses between.")
            keys = vocab.keys_by_frequency()[:20]
            rows = [[f"{vocab.entries[h].count:,}",
                     f"{vocab.entries[h].count/counts.sum():.2%}",
                     (vocab.entries[h].smiles or h[:18])[:44]] for h in keys]
            _page_table(pdf, "I · Most frequent R-groups",
                        ["count", "share", "SMILES"], rows)

            if a.common_percentile is not None:
                import gzip as _gz
                cpath = root / "rgroup_counts.json.gz"
                if cpath.is_file():
                    with _gz.open(cpath, "rt") as fh:
                        cnt = json.load(fh)
                    arr = np.fromiter(cnt.values(), float, len(cnt))
                    thr = float(np.percentile(arr, a.common_percentile))
                    common = {h for h, c in cnt.items() if c >= thr}
                    share = sum(cnt[h] for h in common) / arr.sum()
                    kept_sz, drop_sz, rej = [], [], 0
                    tot_inst = 0
                    for mid in ids[::step][: a.sample]:
                        try:
                            rec = hydrate(torch.load(str(path_for(root, mid, layout)),
                                                     weights_only=False))
                        except Exception:
                            continue
                        for d in rec["decompositions"]:
                            hs = d.get("rgroup_hashes") or []
                            for i_r, r in enumerate(d["rgroups"]):
                                tot_inst += 1
                                n_at = len(r["rgroup_atoms"])
                                if i_r < len(hs) and hs[i_r] in common:
                                    rej += 1; drop_sz.append(n_at)
                                else:
                                    kept_sz.append(n_at)
                    rows = [
                        ["percentile", f"{a.common_percentile}"],
                        ["count threshold", f"{thr:,.0f}"],
                        ["hashes flagged common", f"{len(common):,} of {len(cnt):,}"],
                        ["their share of occurrences", f"{share:.1%}"],
                        ["", ""],
                        ["single-R-group instances rejected",
                         f"{rej:,}/{tot_inst:,}  ({rej/max(tot_inst,1):.1%})"],
                        ["mean R-group size, rejected", f"{np.mean(drop_sz):.2f} atoms"
                            if drop_sz else "-"],
                        ["mean R-group size, kept", f"{np.mean(kept_sz):.2f} atoms"
                            if kept_sz else "-"],
                    ]
                    _page_table(pdf, "J · Instance-level common-R-group filter",
                                ["quantity", "value"], rows,
                                note="Filters INSTANCES, not the library: the retrieval "
                                     "target space and its frequency prior are unchanged. "
                                     "At k=1 the rule reduces to 'drop if the R-group is "
                                     "common', so rejection is far heavier than MolPLA saw.")

    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
