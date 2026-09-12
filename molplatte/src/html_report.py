"""Self-contained interactive HTML for a LeadOptimizationReport.

One file, no network. Structures are inlined as SVG and docked poses as base64
PNG, so the result can be emailed, archived, or opened years later without the
checkpoint, the pocket, or an internet connection. Nothing is loaded from a CDN
-- a viewer that breaks when a CDN moves is not a record.

Two things it does that the PDF cannot:

* **Sort and filter live.** The columns disagree with each other on purpose --
  a compound can dock better and have no synthetic route -- and the useful
  question is usually "show me the ones that are good on BOTH", which is a
  sort, not a static page.
* **Expand a retrosynthesis route.** The route is a tree of reactions down to
  purchasable starting materials. A table cell cannot hold that; a collapsible
  panel can.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Dict, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

__all__ = ["render_html"]


def _svg(smiles: str, width: int = 210, height: int = 160,
         highlight=None) -> str:
    """Inline SVG for a SMILES, or an empty string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d.drawOptions()
    opts.clearBackground = False
    try:
        if highlight:
            cols = {i: (0.68, 0.85, 0.90) for i in highlight[0]}
            cols.update({i: (1.0, 0.72, 0.60) for i in highlight[1]})
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=list(cols), highlightAtomColors=cols)
        else:
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    except Exception:  # noqa: BLE001 - a missing picture must not lose the row
        return ""
    d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _route_html(node: dict, depth: int = 0) -> str:
    """Render an AIZynthFinder route tree as nested HTML."""
    if not node:
        return ""
    kind = node.get("type")
    kids = node.get("children") or []
    if kind == "mol":
        smi = node.get("smiles", "")
        stock = node.get("in_stock")
        tag = ('<span class="pill buy">in stock</span>' if stock
               else '<span class="pill make">make</span>')
        svg = _svg(smi, 180, 130)
        inner = "".join(_route_html(k, depth + 1) for k in kids)
        return (f'<div class="rnode" style="margin-left:{depth * 14}px">'
                f'<div class="rmol">{svg}'
                f'<div class="rsmi">{html.escape(smi)} {tag}</div></div>'
                f'{inner}</div>')
    # reaction
    rsmi = node.get("smiles", "")
    inner = "".join(_route_html(k, depth + 1) for k in kids)
    return (f'<div class="rrxn" style="margin-left:{depth * 14}px">'
            f'<span class="arrow">&#8593;</span> '
            f'<code>{html.escape(rsmi[:110])}</code></div>{inner}')


def _pose_b64(pose_png: Optional[Path]) -> str:
    if not pose_png or not Path(pose_png).is_file():
        return ""
    data = base64.b64encode(Path(pose_png).read_bytes()).decode()
    return f'<img class="pose" src="data:image/png;base64,{data}">'


_CSS = """
:root{--ink:#1d2027;--mut:#6b7280;--line:#e3e6ea;--good:#0f7b45;--bad:#b3261e;
      --core:#bcd9e6;--rg:#ffb899;--bg:#fff}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     color:var(--ink);background:var(--bg);margin:0;padding:24px 28px}
h1{font-size:19px;margin:0 0 2px} h2{font-size:15px;margin:26px 0 8px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.banner{background:#fdf2f2;border-left:3px solid var(--bad);color:#7f1d1d;
        padding:8px 12px;margin:10px 0 16px;font-size:13px;border-radius:3px}
.meta{display:flex;flex-wrap:wrap;gap:18px;margin:10px 0 18px;font-size:13px}
.meta b{font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;
      vertical-align:middle}
th{cursor:pointer;user-select:none;background:#fafbfc;position:sticky;top:0;
   font-weight:600;white-space:nowrap}
th:hover{background:#f0f2f5} th .ar{color:var(--mut);font-size:10px}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.row:hover{background:#fafbfc}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
      font-weight:600}
.buy{background:#e6f4ec;color:var(--good)} .make{background:#fdf0ef;color:var(--bad)}
.yes{color:var(--good);font-weight:600} .no{color:var(--bad);font-weight:600}
.up{color:var(--good)} .down{color:var(--bad)}
.det{background:#fafbfc;padding:14px 18px;border-bottom:1px solid var(--line)}
.rnode{border-left:2px solid var(--line);padding-left:10px;margin:4px 0}
.rmol{display:flex;align-items:center;gap:10px}
.rsmi{font:11px ui-monospace,monospace;color:var(--mut);word-break:break-all}
.rrxn{margin:2px 0;color:var(--mut)} .arrow{color:var(--ink);font-weight:700}
.pose{max-width:460px;border:1px solid var(--line);border-radius:4px}
.controls{display:flex;gap:12px;align-items:center;margin:0 0 10px;font-size:13px}
input[type=search]{padding:6px 10px;border:1px solid var(--line);border-radius:4px;
                   font-size:13px;width:260px}
label{color:var(--mut);user-select:none}
.foot{color:var(--mut);font-size:12px;margin-top:26px;border-top:1px solid var(--line);
      padding-top:12px}
"""

_JS = """
function sortTable(th){
  const tb=th.closest('table'), idx=[...th.parentNode.children].indexOf(th);
  const asc = th.dataset.dir !== 'asc';
  [...tb.querySelectorAll('th')].forEach(h=>{h.dataset.dir='';
     const a=h.querySelector('.ar'); if(a) a.textContent='';});
  th.dataset.dir = asc?'asc':'desc';
  th.querySelector('.ar').textContent = asc?'\\u25B2':'\\u25BC';
  const body=tb.tBodies[0];
  const groups=[...body.querySelectorAll('tr.row')].map(r=>[r, r.nextElementSibling]);
  groups.sort((x,y)=>{
    const a=x[0].children[idx].dataset.v, b=y[0].children[idx].dataset.v;
    const na=parseFloat(a), nb=parseFloat(b);
    const bothNum = !isNaN(na) && !isNaN(nb);
    if(bothNum) return asc ? na-nb : nb-na;
    return asc ? String(a).localeCompare(b) : String(b).localeCompare(a);
  });
  groups.forEach(g=>{body.appendChild(g[0]); if(g[1]) body.appendChild(g[1]);});
}
function toggle(id){
  const d=document.getElementById(id);
  d.style.display = d.style.display==='table-row' ? 'none' : 'table-row';
}
function applyFilter(){
  const q=(document.getElementById('q').value||'').toLowerCase();
  const onlyMakeable=document.getElementById('mk').checked;
  document.querySelectorAll('tr.row').forEach(r=>{
    const hay=r.dataset.hay, solved=r.dataset.solved;
    let show = !q || hay.includes(q);
    if(onlyMakeable && solved==='False') show=false;
    r.style.display = show?'':'none';
    const d=r.nextElementSibling;
    if(d && d.classList.contains('detrow') && !show) d.style.display='none';
  });
}
"""


def _fmt(v, nd=2):
    if v is None or v == "":
        return "&ndash;"
    if isinstance(v, bool):
        return f'<span class="{"yes" if v else "no"}">{"yes" if v else "NO"}</span>'
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def _delta(v, nd=2, lower_is_better=False):
    if v is None:
        return ""
    cls = "up" if (v < 0) == lower_is_better else "down"
    if v == 0:
        cls = ""
    return f' <span class="{cls}">({v:+.{nd}f})</span>'


def render_html(report, out_path: str | Path, *, title: Optional[str] = None,
                pose_pngs: Optional[Dict[str, Path]] = None) -> Path:
    """Write the report as one self-contained interactive HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = report.table
    ref = report.reference or {}
    mode = "pocket-aware" if report.pocket_used else "pocket-less"
    title = title or f"Lead optimization ({mode})"

    rows = table.to_dict("records") if len(table) else []
    has_vina = any(r.get("vina_score") is not None for r in rows)
    has_retro = any(r.get("retro_solved") is not None for r in rows)

    cols = [("#", "rank"), ("R-group", "rgroup"), ("score", "retrieval_score"),
            ("MW", "MW"), ("logP", "logP"), ("QED", "QED"), ("SA", "SAScore"),
            ("NP", "NPScore")]
    if has_vina:
        cols.append(("vina", "vina_score"))
    if has_retro:
        cols += [("route", "retro_solved"), ("steps", "retro_steps")]
    cols += [("site", "replaced_atoms"), ("novel", "is_novel")]

    head = "".join(f'<th onclick="sortTable(this)">{html.escape(lbl)} '
                   f'<span class="ar"></span></th>' for lbl, _ in cols)

    body = []
    for i, r in enumerate(rows):
        did = f"d{i}"
        cells = []
        for lbl, key in cols:
            v = r.get(key)
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            if key == "rgroup":
                cell = f'<code>{html.escape(str(v))}</code>'
            elif key == "retrieval_score":
                cell = f"{v:+.2f}" if v is not None else "&ndash;"
            elif key == "vina_score":
                cell = (_fmt(v) + _delta(r.get("dvina"), lower_is_better=True))
            elif key in ("MW", "logP", "QED", "SAScore", "NPScore"):
                cell = _fmt(v, 2 if key != "MW" else 1) + _delta(
                    r.get("d" + key), 2 if key != "MW" else 1,
                    lower_is_better=(key == "SAScore"))
            elif key == "retro_solved":
                cell = _fmt(bool(v)) if v is not None else "&ndash;"
            else:
                cell = _fmt(v)
            sort_v = "" if v is None else v
            cells.append(f'<td class="{"num" if num else ""}" '
                         f'data-v="{html.escape(str(sort_v))}">{cell}</td>')

        hay = " ".join(str(r.get(k) or "") for k in
                       ("rgroup", "product", "replaced", "retro_materials")).lower()
        body.append(f'<tr class="row" data-hay="{html.escape(hay)}" '
                    f'data-solved="{r.get("retro_solved")}" '
                    f'onclick="toggle(\'{did}\')">{"".join(cells)}</tr>')

        # ---- detail panel: structure, pose, route
        route = (report.routes or {}).get(r.get("product")) or {}
        det = [f'<div style="display:flex;gap:24px;flex-wrap:wrap">',
               f'<div>{_svg(r.get("product"), 300, 230)}'
               f'<div class="rsmi">{html.escape(str(r.get("product")))}</div></div>']
        pose = (pose_pngs or {}).get(r.get("product"))
        if pose:
            det.append(f"<div>{_pose_b64(pose)}</div>")
        det.append("</div>")
        if route.get("route"):
            mats = route.get("starting_materials") or []
            det.append(f'<h2>Retrosynthesis &mdash; {route.get("n_steps")} steps, '
                       f'{len(mats)} starting material'
                       f'{"s" if len(mats) != 1 else ""}</h2>')
            det.append(_route_html(route["route"]))
        elif route:
            det.append('<h2>Retrosynthesis</h2><p class="no">No route to '
                       'purchasable stock found within the search budget. That '
                       'is not proof none exists.</p>')
        body.append(f'<tr class="detrow" id="{did}" style="display:none">'
                    f'<td class="det" colspan="{len(cols)}">{"".join(det)}</td></tr>')

    banner = ""
    if report.pocket_used and report.pocket_changed_ranking is False:
        banner = ('<div class="banner"><b>Pocket supplied but INERT.</b> The '
                  'ranking is identical without it, so read this as '
                  'flavour-conditioned only.</div>')

    meta = [f"<b>input</b> <code>{html.escape(report.input_smiles)}</code>",
            f"<b>flavour</b> {html.escape(str(report.flavor_condition) or 'none')}",
            f"<b>compounds</b> {len(report.compounds)}"]
    if ref.get("vina_score") is not None:
        meta.append(f"<b>input vina</b> {ref['vina_score']:.2f}")
    if report.redock_rmsd is not None:
        ok = "yes" if report.redock_rmsd < 2 else "no"
        meta.append(f'<b>redock control</b> <span class="{ok}">'
                    f'{report.redock_rmsd:.2f} &#8491;</span>')
    if ref.get("retro_solved") is not None:
        meta.append(f"<b>input route</b> {_fmt(bool(ref['retro_solved']))}")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">{_svg(report.input_smiles, 280, 190)}</div>
{banner}
<div class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>
<div class="controls">
  <input id="q" type="search" placeholder="filter by R-group, product, material"
         oninput="applyFilter()">
  <label><input id="mk" type="checkbox" onchange="applyFilter()"> only with a
    synthetic route</label>
  <span style="color:#6b7280">click a row for structure, pose and route</span>
</div>
<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>
<div class="foot">
Columns are INDEPENDENT and disagree on purpose. <code>score</code> is the
model's logQ-corrected retrieval value; <code>vina</code> is docking, which
knows nothing about the model; <code>route</code> is an AIZynthFinder search,
which knows nothing about either. A compound can dock better and have no route.
Deltas in parentheses are against the input compound.
<code>route = NO</code> means none was found within the search budget, not that
none exists.
</div>
<script>{_JS}</script></body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
