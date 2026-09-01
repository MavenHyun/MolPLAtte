"""Flavor profiles as sparse condition vectors: what was measured and why it fails."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pathlib import Path as _P
# .../molplatte_preprocess/scripts/this.py -> .../molplatte_preprocess/docs/
OUT=str(_P(__file__).resolve().parents[1]/"docs"/"flavor_conditioning_report.pdf")
A4=(8.27,11.69)
def page(pdf, title, sub=None):
    fig=plt.figure(figsize=A4); fig.subplots_adjust(top=0.93)
    fig.text(0.07,0.95,title,size=15,weight="bold",va="top")
    if sub: fig.text(0.07,0.925,sub,size=9,style="italic",color="#444",va="top")
    return fig
def body(fig,y,txt,size=8.6,mono=True,color="black"):
    fig.text(0.07,y,txt,size=size,va="top",color=color,
             family="monospace" if mono else "sans-serif",linespacing=1.45)

with PdfPages(OUT) as pdf:
    # ---------- cover ----------
    fig=plt.figure(figsize=A4)
    fig.text(0.5,0.72,"Flavor profiles as sparse condition vectors",ha="center",size=19,weight="bold")
    fig.text(0.5,0.675,"MolPLAtte — what was measured, and why the approach was not adopted",
             ha="center",size=10.5,style="italic")
    body(fig,0.58,
"""SUMMARY

MolPLAtte's brief specifies a condition vector that is either protein-pocket
context or neutral. This report records what happened when we tried to build a
flavor-derived condition signal for the 393,066-molecule corpus, and why every
route examined was rejected.

The conclusion is negative and the evidence is quantitative. Flavor labels do
not exist at corpus scale, cannot be derived from chemical class or source
organism at usable precision, and cannot be imputed by a frontier LLM at a rate
that beats a free deterministic baseline. The pretrained model is therefore
UNCONDITIONAL, and that is the recommended state.

One prior failure frames all of it. An earlier 97-bit RDKit fragment condition
vector was found to LEAK: it was computed from the intact molecule G, which
contains the very R-group the retrieval objective must predict. Removing it took
rank1_distinct from 680 to 5,851 and coverage from 11.4% to 100%. Every
candidate below is judged first on whether it repeats that mistake.

  ROUTE                          VERDICT
  measured labels (FlavorDB)     5.29% corpus coverage; usable only for
                                 fine-tuning, not conditioning
  InChIKey join to COCONUT       1.47% ceiling, dominated by an imputed class
  source organism                tested, fails: mango is 4% "fruity"
  chemical class                 tested, fails: alkaloid->bitter is 19% precise
  LLM annotation                 benchmarked: loses to the class prior at n=300
  physical exclusion (MW/TPSA)   VALID, and free: ~209K honest odorless labels

Prepared from measurements run against coconut-flavordb_v5/v6 and FlavorDB
2026-08.""")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 1. label structure ----------
    fig=page(pdf,"1. The label vocabulary is degenerate",
             "FlavorDB 2026-08: 25,106 labelled molecules, 33,785 descriptor assignments")
    ax=fig.add_axes([0.12,0.52,0.80,0.30])
    labs=["sweet-like","sweet","bitter","fruity","green","floral","woody","herbal","waxy","fatty"]
    vals=[13786,8807,738,640,540,309,243,221,211,191]
    ax.barh(range(len(labs))[::-1],vals,color=["#c0392b","#c0392b"]+["#4a7ebb"]*8)
    ax.set_yticks(range(len(labs))[::-1]); ax.set_yticklabels(labs,size=8)
    ax.set_xscale("log"); ax.set_xlabel("molecules (log scale)",size=8)
    ax.tick_params(labelsize=7); ax.set_title("top 10 of 715 descriptors",size=9)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    body(fig,0.46,
"""715 distinct descriptors, but the distribution is pathological:

    sweet-like + sweet     22,593 assignments  =  66.9% of all mass
    appear >= 100x                         24 descriptors
    appear >= 10x                         196
    appear exactly once                   253  (35% of the vocabulary)
    descriptors per molecule             1.35  (22,929 of 25,106 carry ONE)

It is free text, not a taxonomy. sweet/sweet-like, fruity/fruit,
mint/minty/peppermint/spearmint, wood/woody, sulfur/sulfury/sulfurous and
jasmin/jasmine all coexist as separate labels; URL encoding leaked in
(red%20rose, cut%20grass). Axes are mixed: sensory qualities (bitter), source
referents (apple, bacon, taco), intensity (strong, faint) and chemical class
names (aldehydic, phenolic, terpene, pyridine). The last group is circular for
our purposes -- "phenolic" is a statement about structure.

THE sweet-like PROBLEM

sweet-like is 40.8% of all assignments on its own and is almost certainly
IMPUTED rather than measured. Molecules carrying it are dominated by catalogue
identifiers (SCHEMBL, ZINC, CID, AC1...) -- compounds no human has tasted --
while named aroma compounds sit in the small classes. Excluding it is necessary
for any honest evaluation, and doing so removes two thirds of the data.

After synonym folding into 22 sensory classes, the vocabulary covers 94.3% of
genuine (non-sweet-like) assignments. That folded set is what any future
labelling effort should target.""")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 2. coverage ----------
    fig=page(pdf,"2. Coverage is the binding constraint",
             "corpus coconut-flavordb_v6: 393,066 molecules")
    ax=fig.add_axes([0.12,0.60,0.80,0.24])
    seg=[("FlavorDB-sourced\n(labelled)",20812,"#2e8b57"),
         ("InChIKey match",575,"#3cb371"),
         ("MW>350 non-sugar\n(odorless, derivable)",208978,"#4a7ebb"),
         ("MW>350 glycoside\n(needs annotation)",31501,"#e67e22"),
         ("MW<=350\n(needs annotation)",131519,"#c0392b")]
    left=0
    for lab,v,c in seg:
        ax.barh([0],[v],left=left,color=c,edgecolor="white")
        if v>25000: ax.text(left+v/2,0,f"{v:,}",ha="center",va="center",size=7.5,color="white",weight="bold")
        left+=v
    ax.set_xlim(0,left); ax.set_yticks([]); ax.set_xlabel("molecules",size=8); ax.tick_params(labelsize=7)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.legend([plt.Rectangle((0,0),1,1,color=c) for _,_,c in seg],[l.replace("\n"," ") for l,_,_ in seg],
              fontsize=6.2,loc="upper center",bbox_to_anchor=(0.5,-0.42),ncol=2,frameon=False,
              handlelength=1.2,columnspacing=1.4,handletextpad=0.5)
    body(fig,0.455,
"""    measured flavor label, in corpus       20,812      5.29%
    COCONUT matched to FlavorDB by InChIKey    575      0.15%
    -------------------------------------------------------------
    total with a measured label            21,387      5.44%

A condition vector present for 5% of instances does not teach "sweet => this
R-group". It teaches "has a condition => this is a FlavorDB molecule". FlavorDB
and COCONUT differ sharply in chemistry, so the model would learn SOURCE
DISCRIMINATION and it would look like it was working -- the same failure shape
as the 97-bit fragment condvec, in different clothing.

THE JOIN DOES NOT RESCUE IT

Both databases carry InChIKeys, so the join is exact and cheap:

    FlavorDB with InChIKey + label         25,022
    COCONUT with InChIKey                 738,827
      exact InChIKey match                  5,126     0.69%
      connectivity-only (skeleton)          5,729     0.78%
      TOTAL reachable                      10,855     1.47%

A 1.47% ceiling, and the matches are dominated by the imputed sweet-like class
-- flavonoids, saccharides, triterpenoids, i.e. exactly what an automated
"sugar-like => sweet" rule would catch.""")
    pdf.savefig(fig); plt.close(fig)
    print("pages 1-3 done")

    # ---------- 3. organism ----------
    fig=page(pdf,"3. Source organism does not predict flavor",
             "hypothesis: herb-derived compounds taste bitter. Tested, rejected.")
    ax=fig.add_axes([0.12,0.58,0.80,0.26])
    orgs=["Mangifera\nindica","Zingiber\nofficinale","tomato","Capsicum\nannuum"]
    got=[4,5,5,5]; base=[2,4,2,10]; lab=["fruity","woody","fruity","bitter"]
    x=np.arange(len(orgs)); w=0.36
    ax.bar(x-w/2,got,w,label="within organism",color="#4a7ebb")
    ax.bar(x+w/2,base,w,label="corpus baseline",color="#bbbbbb")
    for i,l in enumerate(lab): ax.text(i,max(got[i],base[i])+0.4,l,ha="center",size=7.5,style="italic")
    ax.set_xticks(x); ax.set_xticklabels(orgs,size=7.5); ax.set_ylabel("% of descriptors",size=8)
    ax.legend(fontsize=7,frameon=False); ax.tick_params(labelsize=7)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    body(fig,0.48,
"""The reasoning is attractive: unlike chemical class or an LLM label, source
organism is EXOGENOUS -- not a function of structure -- so conditioning on it
could not be circular. That is a genuine point in its favour.

The data does not support it.

    COCONUT compounds with organism + FlavorDB label     3,089
      ...with a label beyond sweet/sweet-like            1,482
      distinct organisms among them                     21,669

1,482 compounds spread over 21,669 organisms. And where it can be measured, the
enrichment is weak or inverted: mango's compounds are 4% "fruity" against a 2%
base, and Capsicum annuum is DEPLETED in bitter (5% vs 10%).

WHY IT BREAKS

Organism -> compound is one-to-many. A mango contains ~1,500 catalogued
compounds and a handful carry its flavor; the rest are sugars, sterols,
structural terpenoids. "From a bitter herb" does not make a compound bitter, it
makes it co-resident with one.

Two further problems: organisms cover only 37% of COCONUT, and the single
largest "organism" is Homo sapiens (2,786 compounds). The field is a provenance
record full of Streptomyces, E. coli and Penicillium, not a culinary one.

WHAT SURVIVES  Organism is still usable as a FILTER (restrict to culinary
species to raise the prior that a compound is flavor-relevant) or as a
conditioning signal IN ITS OWN RIGHT ("decorate in the manner of Zingiber
metabolites"), which encodes biosynthetic pathway and is leak-free. Neither
requires pretending we know the taste.""")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 4. class ----------
    fig=page(pdf,"4. Chemical class carries signal, but not enough to label",
             "hypothesis: alkaloids taste bitter. True, and still unusable.")
    ax=fig.add_axes([0.12,0.56,0.80,0.28])
    cls=["Carbohydrates","Amino acids\n& Peptides","Alkaloids","Shikimates","Terpenoids","Fatty acids"]
    prec=[65,35,19,46,27,9]; tgt=[1664,5392,41545,24321,26058,9881]
    c=["#2e8b57" if p>=50 else "#e67e22" if p>=30 else "#c0392b" for p in prec]
    ax.bar(range(len(cls)),prec,color=c)
    for i,(p,t) in enumerate(zip(prec,tgt)):
        ax.text(i,p+1.5,f"{p}%",ha="center",size=7.5,weight="bold")
        ax.text(i,2,f"n={t//1000}k",ha="center",size=6.5,color="white")
    ax.axhline(50,ls="--",lw=1,color="#555")
    ax.text(len(cls)-0.4,52,"50% precision",size=6.5,color="#555",ha="right")
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls,size=7); ax.tick_params(labelsize=7)
    ax.set_ylabel("precision of best class rule (%)",size=8); ax.set_ylim(0,75)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    body(fig,0.46,
"""Alkaloid -> bitter is real: 19% against a 6% base rate, a 3.2x enrichment, and
the chemistry is coherent throughout. Bitter peptides enrich 4.3x, fatty acids
DEPLETE to 0.2x while showing fruity/green (they are the ester precursors of
fruit aroma), carbohydrates come out 51% odorless, organosulfur compounds come
out onion 9% / sulfurous 9%. None of this is noise.

It is still not a label.

    targets in a class whose best rule is >=50% precise    1,664   (1.3%)
    targets in a class whose best rule is >=70% precise        0   (0.0%)

"Alkaloid => bitter" is WRONG 81% OF THE TIME, and alkaloids are the single
largest bucket at 41,545 compounds -- a third of the annotation load sits
exactly where class tells you least.

THE DEEPER OBJECTION

np_classifier_pathway is computed deterministically FROM STRUCTURE. Labelling
compounds by class and then training a structure model to predict those labels
teaches it "alkaloid scaffold => bitter", which NPClassifier already states
exactly, for free. It is a lookup table laundered through a neural network, and
the metric would look good precisely because the task is trivial. This is the
same shape as the condvec leak: a structure-derived quantity conditioning a
structure model.

WHAT SURVIVES  Class is a good FEATURE and an excellent STRATIFIER -- report
retrieval per NPClassifier pathway. It is not a flavor label.""")
    pdf.savefig(fig); plt.close(fig)
    print("pages 4-5 done")

    # ---------- 5. LLM ----------
    fig=page(pdf,"5. LLM annotation: promising at n=20, rejected at n=300",
             "deepseek-v4-flash, evidence-tiered prompt, 24-label constrained vocabulary")
    ax=fig.add_axes([0.12,0.60,0.36,0.24])
    ax.bar([0,1],[84.1,70.3],color=["#4a7ebb","#bbbbbb"])
    ax.errorbar([0,1],[84.1,70.3],yerr=[[5.9,5.4],[4.5,4.9]],fmt="none",ecolor="black",capsize=4,lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(["LLM","class prior"],size=7.5)
    ax.set_ylabel("any-hit % (answered subset)",size=7.5); ax.set_ylim(0,100); ax.tick_params(labelsize=7)
    ax.set_title("on the 189 it chose to answer",size=8)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax2=fig.add_axes([0.58,0.60,0.34,0.24])
    ax2.bar([0,1],[159,234],color=["#c0392b","#2e8b57"])
    ax2.set_xticks([0,1]); ax2.set_xticklabels(["LLM","class prior"],size=7.5)
    ax2.set_ylabel("correct labels produced / 300",size=7.5); ax2.tick_params(labelsize=7)
    ax2.set_title("over the WHOLE sample",size=8)
    for s in ("top","right"): ax2.spines[s].set_visible(False)
    body(fig,0.50,
"""Benchmarked against 300 compounds whose flavor is measured, plus 100 unmeasured
ones to test abstention. The prompt required a declared EVIDENCE TIER
(documented / close_analog / structural / none) so that confidence claims are
falsifiable rather than asserted.

WHAT WORKS  Abstention is genuine: 83% [74.5, 89.1] on unmeasured compounds. It
returned zero "bitter" on alkaloid-tagged compounds, and correctly separated
amarogentin (bitter) from glycyrrhizic acid (sweet) -- both large glycosides,
the circularity trap that defeats every structural heuristic. It even flagged
that NPClassifier had mislabelled a synthetic Biginelli scaffold as an alkaloid.

WHAT FAILS  Two things, both invisible at n=20:

  the evidence tiers do not separate
      documented    127/147 = 86.4%   CI [80%, 91%]
      close_analog   21/28  = 75.0%   CI [57%, 87%]
      structural      9/11  = 81.8%   CI [52%, 95%]
  structural is statistically indistinguishable from documented, so the tier
  filter -- the mechanism that was supposed to make these labels safe -- does
  not work. At n=20 it read 100% / 75% / 0%, from three data points.

  it loses to a free baseline
      head-to-head on the 189 it answered   LLM 84.1%  vs  class prior 78.8%
      over the whole 300                    LLM 53.0%  vs  class prior 78.0%
  The LLM is better on what it answers, but abstains on 37% (84 abstained,
  27 no response), so the deterministic lookup produces 47% MORE correct labels
  for nothing. Label precision was 60.6% with recall 45.9% -- it over-predicts.

COST WAS NEVER THE ISSUE  ~$20 for 163K compounds on a cheap non-reasoning
model. Reasoning models were worse AND dearer: qwen3.8-max burned 8,000 output
tokens on 10 compounds and produced no answer; deepseek-v4-pro reasoned itself
INTO the alkaloid class-prior trap that cheaper models avoided.""")
    pdf.savefig(fig); plt.close(fig)

    # ---------- 6. glycoside ----------
    fig=page(pdf,"6. The glycoside trap, and the one route that works",
             "why 'sugar-bearing => sweet' is circular, and what physics gives for free")
    body(fig,0.86,
"""A natural proposal: MW>350 compounds cannot be odorants, but sugar-bearing ones
can still be SWEET, so label them sweet. The measured support looks decisive:

    MW>350, sugar/glycoside, with a FlavorDB label     n=2,018   98.3% sweet
    MW>350, no sugar                                   n=  186   41.4% sweet

It is an artifact. Breaking that 98.3% down by which label:

    ONLY "sweet-like"  (imputed)      1,863    92.3%
    ONLY "sweet"                        118     5.8%
    not sweet at all                     36     1.8%

92.3% carry only the imputed class, almost certainly assigned BECAUSE they are
glycosides. The rule would read FlavorDB's own structural guess back out.

The compounds with real descriptors say the opposite:

    Amarogentin        bitter    <- gentian glycoside, among the most bitter
    Myronate           bitter       substances known
    Neoquassin         bitter
    Swertiamarin       bitter
    Isoquercetin       odorless
    Glycyrrhizic acid  mild;spicy;sweet   <- licorice, genuinely sweet

High-MW glycosides split between intensely sweet and intensely bitter. The sugar
confers solubility and receptor access, not a taste.""")
    body(fig,0.44,
"""THE ONE ROUTE THAT SURVIVES: PHYSICAL EXCLUSION

Volatility is a hard requirement for smell -- a molecule that cannot evaporate
cannot reach an olfactory receptor. That is a statement about physics, not about
missing measurements.

    real odorants (FlavorDB, non-sweet)   median MW 170    8.5% above 350
    COCONUT                               median MW 433   71.2% above 350

This yields ~209,000 honest `odorless` labels for MW>350 non-sugar compounds,
at zero cost and with no model involved.

It does NOT extend to taste. Taste receptors sit in solution; steviosides are
intensely sweet at MW ~800. "Odorless" is derivable; "tasteless" is not.

AND IT MUST NOT BECOME A NEGATIVE-CLASS SHORTCUT  Labelling every unlabelled
COCONUT compound "tasteless" would be the classic positive-unlabeled mistake,
and it is demonstrably false: 5,126 COCONUT compounds already appear in FlavorDB
with real flavors. They were unlabelled in our corpus because nobody had joined
the databases, not because they are tasteless.""",color="black")
    pdf.savefig(fig); plt.close(fig)
    print("pages 6-7 done")

    # ---------- 7. recommendation ----------
    fig=page(pdf,"7. Recommendation",
             "what to do instead, in priority order")
    body(fig,0.86,
"""DO NOT build flavor conditioning into pretraining. There is no data foundation
for it at corpus scale, and every shortcut examined either leaks (structure-
derived), is too sparse (measured labels), or loses to a free baseline (LLM).
The pretrained model is unconditional and that is the correct state.

THE THREE-STATE DESIGN, if flavor work resumes

  positive           measured label (FlavorDB, GoodScents, BitterDB)   ~21K
  provable negative  physicochemically incapable of odor              ~209K
  unknown            small, plausible, never measured                 ~163K

with a `label_source` field per molecule, and UNKNOWN MASKED OUT OF THE LOSS
rather than forced to negative. Training on positives and provable negatives
while abstaining on the rest is the honest version. Two states would silently
convert "not measured" into "has no flavor".

RANKED NEXT STEPS

  1. Use the 21,387 measured labels as an EVALUATION STRATIFIER, not a training
     condition. Report retrieval per flavor class. Costs nothing, needs no
     labels in training, and answers "is this model useful for flavor
     chemistry?" directly.
  2. Apply the ~209K physical odorless labels. Free, honest, model-free.
  3. If conditioning is wanted: pretrain unconditional (done), then FINE-TUNE
     conditionally on the ~21K measured subset. No imputation anywhere.
  4. The one defensible LLM use is narrow: the 31,501 MW>350 glycosides, where
     the class prior is PROVABLY wrong (the sweet/bitter split above) and no
     cheap alternative exists. ~$165 on a strong model. A separate research
     question, not a blocker.

WHAT TO CARRY FORWARD REGARDLESS

  * Any condition vector must pass a SHUFFLE TEST: permute conditions across
    instances and re-score. Unchanged retrieval means the condition is ignored;
    a collapse far beyond what permutation justifies means it is leaking rather
    than conditioning. This single diagnostic would have caught the original
    97-bit condvec immediately instead of after a full seed sweep.
  * `sweet-like` must be excluded from every ground-truth set. It is imputed,
    it is 40.8% of the data, and including it gives credit for reproducing an
    imputation.
  * COCONUT's np_classifier_* and chemical_class ARE available at ~90% coverage
    and are reproducible and auditable. They are the honest conditioning
    candidate if one is wanted -- but they are f(structure), so shuffle-test
    them before trusting any gain.""")
    pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}")
