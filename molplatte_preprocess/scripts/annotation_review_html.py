#!/usr/bin/env python
"""Render LLM flavor annotations as a self-contained HTML page for manual review.

Every annotation is shown with its structure, so a chemist can judge the label
against the molecule rather than against a SMILES string. Filters cover the two
things that actually need auditing: the evidence tier (only `documented` is
usable as a condition bit) and whether the compound has a name -- a documented
claim on an unnamed compound is the fabrication signature.

    python annotation_review_html.py annotations.jsonl -o review.html [--limit N]

Input: one JSON object per line with at least id, smiles, labels, evidence;
optionally name, class, mw, confidence, source, analog, reasoning, truth.
"""
from __future__ import annotations
import argparse, html, json
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

TIER_COLOR={"documented":"#1a7f37","close_analog":"#9a6700","structural":"#bc4c00","none":"#57606a"}

def svg(smiles,size=(210,160)):
    m=Chem.MolFromSmiles(smiles or "")
    if m is None: return "<div class='nostruct'>unparseable</div>"
    d=rdMolDraw2D.MolDraw2DSVG(*size); d.drawOptions().addStereoAnnotation=False
    try: rdMolDraw2D.PrepareAndDrawMolecule(d,m)
    except Exception: return "<div class='nostruct'>render failed</div>"
    d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>","")

def card(r):
    tier=(r.get("evidence") or "none").lower()
    labs=list(r.get("labels") or [])
    unk = labs==["unknown"] or not labs
    name=(r.get("name") or "").strip()
    truth=r.get("truth")
    suspect = tier=="documented" and not name and not unk
    chips="".join(f"<span class='chip {'unk' if l=='unknown' else ''}'>{html.escape(l)}</span>"
                  for l in labs) or "<span class='chip unk'>&mdash;</span>"
    rows=[]
    for k in ("source","analog","reasoning"):
        if r.get(k): rows.append((k,r[k]))
    if truth: rows.append(("measured truth",", ".join(truth) if isinstance(truth,list) else str(truth)))
    meta="".join(f"<div class='kv'><span class='k'>{html.escape(k)}</span>"
                 f"<span class='v'>{html.escape(str(v))}</span></div>" for k,v in rows)
    hit=""
    if truth:
        t=set(truth if isinstance(truth,list) else [truth]); got=set(labs)-{"unknown"}
        hit="<span class='hit yes'>HIT</span>" if got&t else "<span class='hit no'>miss</span>"
    conf=r.get("confidence")
    return f"""
<div class="card" data-tier="{tier}" data-named="{'1' if name else '0'}"
     data-unknown="{'1' if unk else '0'}" data-suspect="{'1' if suspect else '0'}">
  <div class="struct">{svg(r.get('smiles',''))}</div>
  <div class="info">
    <div class="hdr"><code>{html.escape(str(r.get('id','?')))}</code>
      {'<b>'+html.escape(name[:52])+'</b>' if name else "<i class='noname'>no name</i>"}
      {hit}{"<span class='flag'>documented but unnamed</span>" if suspect else ""}</div>
    <div class="sub">{html.escape(str(r.get('class','')))} &middot; MW {html.escape(str(r.get('mw','')))}</div>
    <div class="labels">{chips}
      <span class="tier" style="background:{TIER_COLOR.get(tier,'#57606a')}">{html.escape(tier)}</span>
      {f"<span class='conf'>conf {conf}</span>" if conf is not None else ""}</div>
    {meta}
  </div></div>"""

CSS="""*{box-sizing:border-box}body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d0d7de;padding:12px 20px;z-index:10;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{margin:0 0 4px;font-size:17px}.stats{color:#57606a;font-size:12px}
.controls{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:12px inherit;padding:5px 11px;border:1px solid #d0d7de;background:#fff;border-radius:6px;cursor:pointer}
button.on{background:#0969da;color:#fff;border-color:#0969da}
input[type=search]{padding:5px 9px;border:1px solid #d0d7de;border-radius:6px;font:12px inherit;min-width:220px}
.wrap{padding:16px 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:12px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px;display:flex;gap:10px}
.card[data-suspect="1"]{border-color:#bc4c00;box-shadow:inset 3px 0 0 #bc4c00}
.struct{flex:0 0 210px}.struct svg{width:210px;height:160px}
.nostruct{color:#8c959f;font-size:11px;padding:60px 10px;text-align:center}
.info{flex:1;min-width:0}.hdr{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:2px}
.hdr code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:11px}
.noname{color:#8c959f}.sub{color:#57606a;font-size:11px;margin-bottom:6px}
.labels{margin-bottom:6px;display:flex;gap:4px;flex-wrap:wrap;align-items:center}
.chip{background:#ddf4ff;color:#0969da;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.chip.unk{background:#eff1f3;color:#57606a;font-weight:400}
.tier{color:#fff;padding:2px 7px;border-radius:10px;font-size:10px;text-transform:uppercase;letter-spacing:.3px}
.conf{color:#57606a;font-size:11px}
.flag{background:#fff1e5;color:#bc4c00;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}
.hit{padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700}
.hit.yes{background:#dafbe1;color:#1a7f37}.hit.no{background:#ffebe9;color:#cf222e}
.kv{display:flex;gap:6px;font-size:11px;margin-top:2px}
.k{color:#8c959f;flex:0 0 92px}.v{color:#1f2328;flex:1;min-width:0}"""

JS="""const cards=[...document.querySelectorAll('.card')];let tier='all',extra=null,q='';
function apply(){let n=0;for(const c of cards){
 let ok=(tier==='all'||c.dataset.tier===tier);
 if(ok&&extra==='suspect')ok=c.dataset.suspect==='1';
 if(ok&&extra==='unnamed')ok=c.dataset.named==='0';
 if(ok&&extra==='answered')ok=c.dataset.unknown==='0';
 if(ok&&q)ok=c.textContent.toLowerCase().includes(q);
 c.style.display=ok?'flex':'none';if(ok)n++;}
 document.getElementById('shown').textContent=n;}
document.querySelectorAll('[data-tierbtn]').forEach(b=>b.onclick=()=>{tier=b.dataset.tierbtn;
 document.querySelectorAll('[data-tierbtn]').forEach(x=>x.classList.toggle('on',x===b));apply();});
document.querySelectorAll('[data-extra]').forEach(b=>b.onclick=()=>{const v=b.dataset.extra;
 extra=(extra===v)?null:v;document.querySelectorAll('[data-extra]').forEach(x=>x.classList.toggle('on',x.dataset.extra===extra));apply();});
document.getElementById('q').oninput=e=>{q=e.target.value.toLowerCase();apply();};apply();"""

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("infile")
    ap.add_argument("-o","--out",default="annotation_review.html")
    ap.add_argument("--limit",type=int,default=1500)
    a=ap.parse_args()
    recs=[json.loads(l) for l in open(a.infile) if l.strip()]
    total=len(recs)
    order={"documented":0,"close_analog":1,"structural":2,"none":3}
    recs.sort(key=lambda r:(order.get((r.get("evidence") or "none").lower(),4),
                            0 if (r.get("name") or "").strip() else 1))
    shown=recs[:a.limit]
    tiers=Counter((r.get("evidence") or "none").lower() for r in recs)
    susp=sum(1 for r in recs if (r.get("evidence") or "").lower()=="documented"
             and not (r.get("name") or "").strip() and (r.get("labels") or [])!=["unknown"])
    btns="".join(f'<button data-tierbtn="{t}">{t} ({tiers.get(t,0)})</button>'
                 for t in ("documented","close_analog","structural","none"))
    doc=f"""<!doctype html><meta charset="utf-8"><title>Flavor annotation review</title>
<style>{CSS}</style>
<header><h1>Flavor annotation review</h1>
<div class="stats">{total:,} annotations &middot; showing <span id="shown">0</span> of {len(shown):,} rendered
 &middot; documented {tiers.get('documented',0):,}
 &middot; <b style="color:#bc4c00">{susp:,} documented-but-unnamed</b> (audit these first)</div>
<div class="controls"><button data-tierbtn="all" class="on">all tiers</button>{btns}
<span style="width:12px"></span>
<button data-extra="suspect">documented &amp; unnamed</button>
<button data-extra="unnamed">no name</button>
<button data-extra="answered">answered only</button>
<input id="q" type="search" placeholder="search id, label, source, reasoning..."></div></header>
<div class="wrap">{"".join(card(r) for r in shown)}</div><script>{JS}</script>"""
    Path(a.out).write_text(doc)
    print(f"wrote {a.out}  ({total:,} annotations, {len(shown):,} rendered, "
          f"{Path(a.out).stat().st_size/1048576:.1f} MB)")

if __name__=="__main__": main()
