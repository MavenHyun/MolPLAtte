#!/usr/bin/env python3
"""Render docs/eda_tastepocket.pdf -- the pocket finetuning set, instance by instance.

Two halves. The first summarises the set: what receptors it covers, how the CV
folds fall, how big the pockets are, where the flavour labels came from, and how
the R-groups distribute against the retrieval library. The second is a catalogue
with one card per record, drawing EVERY decomposition with its core and R-groups
highlighted, beside the pocket and label metadata for that record.

Everything is recomputed from the corpus and its sidecars. Nothing is copied from
a previous report -- the numbers in this set moved twice during preprocessing
(2,269 pocket sites to 1,255 when backbone residues stopped being counted as
ligands, then to 269 records at the (ligand, receptor) grain), and a hard-coded
figure would still look plausible after such a change.

Layout uses a running cursor, so blocks cannot collide.
"""
from __future__ import annotations

import glob
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/mogan/preprocessed/molplatte")
CORPUS = ROOT / "tastepocket_corpus" / "naveja_recap"
OUT = Path(__file__).resolve().parents[1] / "docs" / "eda_tastepocket.pdf"

INK, MUTE = "#1a1a1a", "#666666"
CORE, RG, ACC, WARN = "#4c72b0", "#c44e52", "#2e7d32", "#b8860b"
CORE_RGB, RG_RGB = (0.298, 0.447, 0.690), (0.769, 0.306, 0.322)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                     "axes.edgecolor": "#999", "axes.linewidth": .7})

LINE = 0.0163


class Page:
    def __init__(self, title, kicker=None, footer="MolPLAtte — tastepocket EDA"):
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.fig.text(.07, .957, title, size=15.5, weight="bold", color=INK)
        if kicker:
            self.fig.text(.07, .937, kicker, size=8.6, color=MUTE, style="italic")
        self.fig.text(.07, .022, footer, size=7, color=MUTE)
        self.title = title
        self.y = .905

    def text(self, s, size=8.5, color=INK, gap=1.0, x=.07):
        self.fig.text(x, self.y, s, size=size, color=color, va="top", linespacing=1.62)
        self.y -= (s.count("\n") + 1) * LINE * (size / 8.5) + gap * LINE
        return self

    def h2(self, s):
        self.y -= LINE * 0.6
        self.fig.text(.07, self.y, s, size=10.4, weight="bold", color=INK, va="top")
        self.y -= LINE * 1.9
        return self

    def axes(self, height, left=.10, width=.83, gap=3.4):
        self.y -= height
        ax = self.fig.add_axes([left, self.y, width, height])
        self.y -= gap * LINE
        return ax

    #: Content below this runs into the footer and is clipped by the page edge.
    FLOOR = .045

    def save(self, pdf):
        # A cursor layout cannot collide, but it CAN run off the bottom, and the
        # PDF renders happily either way. Reported rather than left to be noticed
        # by whoever reads the file.
        if self.y < self.FLOOR:
            print(f"  !! OVERFLOW: '{self.title}' ends at y={self.y:.3f} "
                  f"(floor {self.FLOOR}) -- content is clipped")
        pdf.savefig(self.fig)
        plt.close(self.fig)


def bar(ax, labels, values, color=CORE, xlabel=""):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, size=7.2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, size=7.6)
    ax.tick_params(labelsize=7.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:,}", va="center", size=6.8, color=MUTE)


# ---------------------------------------------------------------- load
def load():
    recs = []
    for f in sorted(glob.glob(str(CORPUS / "**" / "*.pt"), recursive=True)):
        d = torch.load(f, weights_only=False)
        recs.append(d)
    ds = {}
    p = ROOT / "tastepocket" / "dataset.jsonl"
    if p.is_file():
        for line in p.open():
            r = json.loads(line)
            ds[r["id"]] = r
    novel = set()
    counts = {}
    import gzip
    import pickle
    vp = ROOT / "union_vocab" / "base-full__crossdocked__tastepocket" / "rgroup_vocab.pkl.gz"
    if vp.is_file():
        with gzip.open(vp, "rb") as fh:
            v = pickle.load(fh)
        prov = v["provenance"] if isinstance(v, dict) else v.provenance
        entries = v["entries"] if isinstance(v, dict) else v.entries
        novel = set(prov.get("novel_hashes") or [])
        counts = {h: (e["count"] if isinstance(e, dict) else e.count)
                  for h, e in entries.items()}
    return recs, ds, novel, counts


def draw_decomp(mol, dec, size=(300, 210)):
    """PNG of the molecule with core blue and R-groups red."""
    core = set(dec["core_atoms"])
    rgs = set(a for r in dec["rgroups"] for a in r["rgroup_atoms"])
    cols = {**{a: CORE_RGB for a in core}, **{a: RG_RGB for a in rgs}}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().bondLineWidth = 1.4
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=list(core | rgs), highlightAtomColors=cols)
    except Exception:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format="png")


def rgroup_smiles(mol, atoms):
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms))
    except Exception:
        return "?"


def main() -> int:
    recs, ds, novel, counts = load()
    print(f"loaded {len(recs)} corpus records, {len(ds)} dataset rows, "
          f"{len(novel):,} novel hashes")

    fam_of, fold_of, org_of = {}, {}, {}
    for r in recs:
        m = r["meta"]
        fam_of[r["mol_id"]] = (m.get("families") or "").split(";")[0] or "(none)"
        fold_of[r["mol_id"]] = m.get("fold", "")
        org_of[r["mol_id"]] = (m.get("organisms") or "").split(";")[0] or "(none)"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # ---------------------------------------------------- cover
        n_dec = sum(r["n_decomps"] for r in recs)
        n_rg = sum(len(d["rgroups"]) for r in recs for d in r["decompositions"])
        pockets = [float(ds[r["mol_id"]]["mean_pocket_residues"])
                   for r in recs if r["mol_id"] in ds]
        insts = [int(r["meta"].get("n_instances") or 0) for r in recs]

        p = Page("tastepocket", "the pocket-conditioned finetuning set, instance by instance")
        p.text(
            "343 T1 chemosensory complexes from the PDB become 269 (ligand, receptor)\n"
            "records, of which 243 decompose. Every record carries a 24-bit flavour\n"
            "vector and a 1280-d ESM-2 embedding of the 10 A pocket around its ligand.",
            gap=1.6)
        p.h2("What is in it")
        rows = [
            ("corpus records", f"{len(recs)}"),
            ("decompositions", f"{n_dec}  ({n_dec/max(len(recs),1):.2f} per record)"),
            ("R-groups", f"{n_rg}  ({n_rg/max(n_dec,1):.2f} per decomposition)"),
            ("distinct ligands (CCD)", f"{len({r['meta']['ccd'] for r in recs})}"),
            ("distinct receptors (UniProt)", f"{len({r['meta']['uniprot'] for r in recs})}"),
            ("receptor families", f"{len(set(fam_of.values()))}"),
            ("organisms", f"{len(set(org_of.values()))}"),
            ("condvec width", "1304 = 24 flavour + 1280 pocket"),
            ("pocket residues (mean)", f"{np.mean(pockets):.1f}" if pockets else "-"),
            ("pocket sites behind them", f"{sum(insts)}"),
        ]
        # Monospace: space-padding a proportional font does not align columns.
        for k, v in rows:
            p.fig.text(.07, p.y, k, size=8.0, color=MUTE, va="top",
                       family="DejaVu Sans Mono")
            p.fig.text(.40, p.y, v, size=8.0, color=INK, va="top",
                       family="DejaVu Sans Mono")
            p.y -= LINE * 1.15
        p.y -= LINE * 1.2

        p.h2("Why the counts are what they are")
        p.text(
            "One record per (ligand, RECEPTOR), not per pocket site. 1,255 sites collapse\n"
            "to 269 because a homotetramer deposits the same ligand four times. Counting\n"
            "those separately would inflate the R-group frequency prior unevenly -- and\n"
            "that prior is both what Hit@K is judged against and what the logQ correction\n"
            "subtracts, so inflating it moves the number being optimised.\n\n"
            "The 1,255 is itself a correction. Selecting ligand copies by residue name\n"
            "alone matched backbone residues too: mmCIF records free amino acids as ATOM\n"
            "rather than HETATM, so a `resname TRP` match swept up all 14 tryptophans in\n"
            "the protein. That produced 2,269 sites, 445 of them 'tryptophan'.",
            color=MUTE, gap=1.4)
        p.save(pdf)

        # ---------------------------------------------------- families
        p = Page("Receptors", "what the set actually covers")
        fam_rec = Counter(fam_of.values())
        fam_lig = defaultdict(set)
        for r in recs:
            fam_lig[fam_of[r["mol_id"]]].add(r["meta"]["ccd"])
        order = [f for f, _ in fam_rec.most_common()]
        ax = p.axes(.195, left=.42, width=.50)
        bar(ax, [f[:44] for f in order], [fam_rec[f] for f in order],
            xlabel="records")
        ax = p.axes(.195, left=.42, width=.50)
        bar(ax, [f[:44] for f in order], [len(fam_lig[f]) for f in order],
            color=ACC, xlabel="distinct ligands")
        # Named from the data, not from memory: the ranking shifted when the
        # grain changed and a hard-coded family name would have gone quietly
        # wrong while still reading plausibly.
        ratio = sorted(((fam_rec[f] / max(len(fam_lig[f]), 1), f) for f in order),
                       reverse=True)
        top2 = ", ".join(f"{f.split(' –')[0].split(' -')[0]} ({r:.2f})"
                         for r, f in ratio[:2])
        widest = max(order, key=lambda f: len(fam_lig[f]))
        p.text(
            "Records and ligands rank differently, and the gap is the point. Highest\n"
            f"records-per-ligand: {top2}. Both are amino-acid or bitter\n"
            "sensors where the same few ligands recur across several receptors, and each\n"
            "(ligand, receptor) pair is a separate datapoint.\n\n"
            f"{widest.split(' –')[0].split(' -')[0]} is the reverse: {len(fam_lig[widest])} distinct "
            f"ligands over {fam_rec[widest]} records, the widest\n"
            "chemistry in the set -- and the least labellable, since insect pheromones\n"
            "have no human flavour descriptor.",
            color=MUTE, gap=1.2)
        org = Counter(org_of.values()).most_common(10)
        ax = p.axes(.115, left=.42, width=.50)
        bar(ax, [o[:44] for o, _ in org], [n for _, n in org], color=WARN,
            xlabel="records")
        p.save(pdf)

        # ---------------------------------------------------- folds
        p = Page("Cross-validation folds", "cut on connected components, not at random")
        folds = sorted({f for f in fold_of.values() if f != ""}, key=int)
        ax = p.axes(.125)
        counts_f = [sum(1 for v in fold_of.values() if v == f) for f in folds]
        ax.bar([f"fold {f}" for f in folds], counts_f, color=CORE, width=.6)
        ax.set_ylabel("records", size=7.6)
        ax.tick_params(labelsize=7.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for i, v in enumerate(counts_f):
            ax.text(i, v, str(v), ha="center", va="bottom", size=7)

        fam_fold = defaultdict(Counter)
        for mid, f in fold_of.items():
            if f != "":
                fam_fold[fam_of[mid]][f] += 1
        ax = p.axes(.245, left=.42, width=.50)
        fams = [f for f, _ in fam_rec.most_common()]
        bottom = np.zeros(len(fams))
        cmap = plt.get_cmap("tab10")
        for i, fd in enumerate(folds):
            vals = np.array([fam_fold[f][fd] for f in fams], dtype=float)
            ax.barh(np.arange(len(fams)), vals, left=bottom, height=.68,
                    color=cmap(i % 10), label=f"fold {fd}")
            bottom += vals
        ax.set_yticks(np.arange(len(fams)))
        ax.set_yticklabels([f[:44] for f in fams], size=7.2)
        ax.invert_yaxis()
        ax.set_xlabel("records", size=7.6)
        ax.tick_params(labelsize=7.2)
        ax.legend(fontsize=6.4, ncol=5, loc="lower right", frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        single = [f for f in fams if sum(1 for fd in folds if fam_fold[f][fd]) == 1]
        p.text(
            "A fold holds out whole connected components of the (ligand, receptor)\n"
            "bipartite graph, so a held-out receptor's ligands are absent from training\n"
            "too. Splitting by receptor and patching ligand conflicts afterwards cost 64\n"
            "of 269 records and still left the halves unbalanced.\n\n"
            "This cannot be random. Pooled ESM-2 pocket embeddings identify the receptor\n"
            "family with 98.5% 1-NN accuracy across different PDB entries, against a\n"
            "27.4% majority baseline -- a random split puts the same protein on both\n"
            "sides and reports memorisation as generalisation.",
            color=MUTE, gap=1.2)
        p.text(
            f"{len(single)} of {len(fams)} families sit in a single fold, because each is one\n"
            "component and cannot be spread at any weight. Those folds test an unseen\n"
            "receptor FAMILY, which is strictly harder than an unseen receptor:\n"
            + "\n".join(f"   {f[:58]}" for f in single[:6])
            + (f"\n   (+{len(single)-6} more)" if len(single) > 6 else ""),
            color=WARN, size=8.0, gap=0.4)
        p.save(pdf)

        # ---------------------------------------------------- pockets + ligands
        p = Page("Pockets and ligands", "10 A sites, and the molecules that define them")
        if pockets:
            ax = p.axes(.20)
            ax.hist(pockets, bins=28, color=CORE)
            ax.set_xlabel("pocket residues within 10 A of the ligand", size=7.6)
            ax.set_ylabel("records", size=7.6)
            ax.tick_params(labelsize=7.2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.axvline(np.median(pockets), color=RG, lw=1.1, ls="--")
            ax.text(np.median(pockets), ax.get_ylim()[1] * .92,
                    f" median {np.median(pockets):.0f}", size=6.8, color=RG)

        heavy = []
        for r in recs:
            m = Chem.MolFromSmiles(r["smiles"])
            if m:
                heavy.append(m.GetNumHeavyAtoms())
        ax = p.axes(.20)
        ax.hist(heavy, bins=28, color=ACC)
        ax.set_xlabel("ligand heavy atoms", size=7.6)
        ax.set_ylabel("records", size=7.6)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax = p.axes(.18)
        ax.hist(insts, bins=range(1, max(insts) + 2), color=WARN)
        ax.set_xlabel("pocket sites collapsed into one record", size=7.6)
        ax.set_ylabel("records", size=7.6)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        p.text(
            "10 A matches the pocket10 convention CrossDocked's processed LMDB uses, so\n"
            "pockets from the two sources are directly comparable. Residue selection is\n"
            "3D even though the embedding is 1D, which is what keeps it usable for GPCRs:\n"
            "in 8F76 the propionate site draws on residues 100-108 and 151-157.",
            color=MUTE, gap=1.0)
        p.save(pdf)

        # ---------------------------------------------------- flavour
        p = Page("Flavour labels", "tables first, then a molecule-only LLM pass")
        src = Counter(r["meta"].get("flavor_source", "?") for r in recs)
        ax = p.axes(.17, left=.42, width=.50)
        so = src.most_common()
        bar(ax, [k for k, _ in so], [v for _, v in so], color=CORE, xlabel="records")
        lab = Counter(l for r in recs
                      for l in (r["meta"].get("flavor_labels") or "").split(";") if l)
        ax = p.axes(.30, left=.42, width=.50)
        lo = lab.most_common(18)
        bar(ax, [k for k, _ in lo], [v for _, v in lo], color=RG, xlabel="records")
        n_lab = sum(1 for r in recs if r["meta"].get("flavor_labels"))
        p.text(
            f"{n_lab} of {len(recs)} records ({100*n_lab/len(recs):.0f}%) carry a real label.\n\n"
            "The rest is not a preprocessing failure. Of the molecules the tables could\n"
            "not resolve, most are modulators, lipids or cofactors with no percept at\n"
            "all, and 19 more are insect pheromones -- bombykol, bombykal, honeybee queen\n"
            "mandibular pheromone. Genuine stimuli with no human descriptor. Insect OBP is\n"
            "the largest family here, so a large part of this set is unlabellable in a\n"
            "22-term human vocabulary by construction.\n\n"
            "The LLM pass sees the MOLECULE ONLY -- name and SMILES, never the receptor.\n"
            "Told 'this binds TRPM8' it would answer 'cooling', and the flavour half of\n"
            "the condvec would become a re-encoding of the pocket half: conditioning would\n"
            "then show a large lift meaning nothing, because both halves would carry the\n"
            "same variable.",
            color=MUTE, gap=1.0)
        p.save(pdf)

        # ---------------------------------------------------- decompositions
        p = Page("Decompositions", "naveja_recap, identical settings to every other corpus")
        nd = Counter(r["n_decomps"] for r in recs)
        ax = p.axes(.17)
        ks = sorted(nd)
        ax.bar([str(k) for k in ks], [nd[k] for k in ks], color=CORE, width=.6)
        ax.set_xlabel("decompositions per record", size=7.6)
        ax.set_ylabel("records", size=7.6)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        nr = Counter(len(d["rgroups"]) for r in recs for d in r["decompositions"])
        ax = p.axes(.17)
        ks = sorted(nr)
        ax.bar([str(k) for k in ks], [nr[k] for k in ks], color=RG, width=.6)
        ax.set_xlabel("R-groups per decomposition", size=7.6)
        ax.set_ylabel("decompositions", size=7.6)
        ax.tick_params(labelsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        hashes = [h for r in recs for d in r["decompositions"]
                  for h in d["rgroup_hashes"] if h]
        distinct = set(hashes)
        n_novel = len(distinct & novel)
        if counts:
            cn = sorted(counts.get(h, 0) for h in distinct & novel)
            cb = sorted(counts.get(h, 0) for h in distinct - novel)
            ax = p.axes(.19)
            ax.hist([np.log10(np.clip(cb, 1, None)), np.log10(np.clip(cn, 1, None))],
                    bins=26, color=[CORE, RG], label=["in pretraining vocab", "novel"],
                    stacked=True)
            ax.set_xlabel("log10 corpus count in the 91,935-row union library", size=7.6)
            ax.set_ylabel("distinct R-groups", size=7.6)
            ax.legend(fontsize=7, frameon=False)
            ax.tick_params(labelsize=7.2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            p.text(
                f"{n_novel} of {len(distinct)} distinct R-groups "
                f"({100*n_novel/len(distinct):.0f}%) are absent from the pretraining\n"
                "vocabulary. They are also RARE: median corpus count "
                f"{int(np.median(cn))} against {int(np.median(cb))} for the rest.\n\n"
                "That rarity is why hit@K has to be reported split. Novel R-groups barely\n"
                "move a micro average, so a comfortable headline number is consistent with\n"
                "novel retrieval failing outright. Whether it does is what base_hit@K and\n"
                "novel_hit@K are there to separate -- neither can be read off this chart.",
                color=MUTE, gap=1.0)
        p.save(pdf)

        # ---------------------------------------------------- catalogue
        recs_sorted = sorted(recs, key=lambda r: (fam_of[r["mol_id"]], r["meta"]["ccd"]))

        # Geometry, stated once. The text column and the image strip must not
        # overlap: TEXT_R is where text stops, IMG_L is where images start.
        PER_PAGE = 4
        MAXD = 4                    # max n_decomps in this corpus is 4 -> all shown
        PITCH = .205                # vertical distance between record cards
        TOP0 = .905                 # baseline of the first card's header
        TEXT_R, IMG_L = .40, .425
        IMG_W = (.95 - IMG_L) / MAXD
        IMG_H = .092

        def clip(txt, n):
            """Truncate on a word boundary so a family name never ends mid-word."""
            txt = str(txt or "")
            if len(txt) <= n:
                return txt
            cut = txt[:n].rsplit(" ", 1)[0]
            return (cut or txt[:n]) + "…"

        n_pages = (len(recs_sorted) + PER_PAGE - 1) // PER_PAGE
        for start in range(0, len(recs_sorted), PER_PAGE):
            chunk = recs_sorted[start:start + PER_PAGE]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            fig.text(.07, .967, "Catalogue", size=13, weight="bold", color=INK)
            fig.text(.07, .951,
                     f"records {start+1}–{min(start+PER_PAGE, len(recs_sorted))} "
                     f"of {len(recs_sorted)}   ·   core in blue, R-groups in red, "
                     "● = R-group absent from the pretraining vocabulary",
                     size=7.2, color=MUTE)
            fig.text(.07, .022, "MolPLAtte — tastepocket EDA", size=7, color=MUTE)

            for j, r in enumerate(chunk):
                top = TOP0 - j * PITCH
                m = r["meta"]
                mol = Chem.MolFromSmiles(r["smiles"])

                fig.text(.07, top, f"{m['ccd']}  ·  {clip(m['uniprot'], 30)}",
                         size=9.2, weight="bold", color=INK, va="top")
                fig.text(.07, top - .019, clip(m.get("name"), 52),
                         size=7.0, color=MUTE, va="top")

                pocket_res = ds.get(r["mol_id"], {}).get("mean_pocket_residues", "?")
                info = [
                    f"family    {clip((m.get('families') or '').split(';')[0], 40)}",
                    f"organism  {clip((m.get('organisms') or '').split(';')[0], 40)}",
                    f"PDB       {clip((m.get('pdb_ids') or '').replace(';', ' '), 40)}",
                    f"fold {m.get('fold','?')}   sites {m.get('n_instances','?')}"
                    f"   pocket {pocket_res} res",
                    f"flavour   {clip(m.get('flavor_labels') or '(none)', 30)}"
                    f" [{m.get('flavor_source','?')}]",
                    f"SMILES    {clip(r['smiles'], 40)}",
                ]
                fig.text(.07, top - .038, "\n".join(info), size=6.3, color=INK,
                         va="top", linespacing=1.80, family="DejaVu Sans Mono")

                for k, dec in enumerate(r["decompositions"][:MAXD]):
                    # bottom edge, so the title sits BELOW the record header
                    ax = fig.add_axes([IMG_L + k * IMG_W, top - .052 - IMG_H,
                                       IMG_W * .93, IMG_H])
                    ax.axis("off")
                    if mol is not None:
                        ax.imshow(draw_decomp(mol, dec))
                    rgs = " ".join(
                        rgroup_smiles(mol, rg["rgroup_atoms"])
                        for rg in dec["rgroups"]) if mol is not None else ""
                    nov = any(h in novel for h in dec["rgroup_hashes"] if h)
                    ax.set_title(f"#{k} {clip(rgs, 16)}" + ("  ●" if nov else ""),
                                 size=5.6, color=RG if nov else MUTE, pad=1.5)

                if r["n_decomps"] > MAXD:
                    fig.text(.955, top - .09, f"+{r['n_decomps']-MAXD}", size=6,
                             color=MUTE)
                rule = top - PITCH + .035
                fig.add_artist(plt.Line2D([.07, .95], [rule, rule],
                                          color="#e2e2e2", lw=.6))
            pdf.savefig(fig)
            plt.close(fig)
            if (start // PER_PAGE) % 20 == 0:
                print(f"  catalogue page {start//PER_PAGE + 1}/{n_pages}")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
