#!/usr/bin/env python3
"""Render a findings markdown document to a print-ready PDF.

Targets paper, not screen: A4, numbered pages, tables that do not split across a
page break, and a type scale that survives being projected in a meeting room.

    python render_report_pdf.py molplatte/docs/FINDINGS.md --out docs/findings.pdf
"""
from __future__ import annotations
import argparse, datetime, re, sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

CSS_TEXT = """
@page {
  size: A4;
  margin: 19mm 17mm 17mm 17mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "DejaVu Sans", sans-serif; font-size: 8.5pt; color: #8A8F98;
  }
  @top-right {
    content: "MolPLAtte — findings";
    font-family: "DejaVu Sans", sans-serif; font-size: 8pt; color: #A5AAB2;
  }
}
@page :first { @top-right { content: ""; } }

body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 9.7pt; line-height: 1.52; color: #17191C; margin: 0;
  hyphens: auto;
}
h1 {
  font-family: "DejaVu Sans", sans-serif; font-size: 20pt; line-height: 1.15;
  margin: 0 0 4mm; color: #0E1114; letter-spacing: -.2pt;
  border-bottom: 1.6pt solid #0F6E6B; padding-bottom: 3mm;
}
h2 {
  font-family: "DejaVu Sans", sans-serif; font-size: 12.5pt; margin: 8mm 0 2.5mm;
  color: #0F6E6B; break-after: avoid; letter-spacing: -.1pt;
}
h3 {
  font-family: "DejaVu Sans", sans-serif; font-size: 10.3pt; margin: 5mm 0 1.8mm;
  color: #21262B; break-after: avoid;
}
p { margin: 0 0 2.4mm; orphans: 2; widows: 2; }
strong { color: #0B0D10; }
hr { border: 0; border-top: .5pt solid #D8DCE0; margin: 6mm 0; }

table {
  border-collapse: collapse; width: 100%; margin: 3mm 0 4mm;
  font-family: "DejaVu Sans", sans-serif; font-size: 8.4pt;
  break-inside: avoid;
}
th, td { padding: 1.5mm 2.4mm; border-bottom: .4pt solid #DEE2E6; text-align: left; }
thead th {
  background: #F2F4F5; color: #40464D; font-size: 7.6pt;
  text-transform: uppercase; letter-spacing: .4pt; border-bottom: .8pt solid #C3C9CF;
}
tbody tr:last-child td { border-bottom: .8pt solid #C3C9CF; }
td { font-variant-numeric: tabular-nums; }

code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8.2pt;
  background: #F2F4F5; padding: .3mm 1mm; border-radius: 1mm; color: #223;
}
pre {
  background: #F7F8F9; border-left: 1.6pt solid #0F6E6B; padding: 2.5mm 3mm;
  font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; line-height: 1.4;
  break-inside: avoid; margin: 3mm 0; overflow-wrap: break-word; white-space: pre-wrap;
}
pre code { background: none; padding: 0; font-size: inherit; }
ul, ol { margin: 0 0 2.8mm; padding-left: 5.5mm; }
li { margin-bottom: 1.1mm; }
blockquote {
  margin: 3mm 0; padding: 2mm 3mm; background: #FDF6E8;
  border-left: 1.6pt solid #B08423; color: #4A3D22; font-size: 9.2pt;
}
.meta {
  font-family: "DejaVu Sans", sans-serif; font-size: 8.4pt; color: #6B7178;
  margin: 0 0 7mm;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subtitle", default=None)
    a = ap.parse_args()

    text = a.source.read_text()
    # strip the relative doc links: they do not resolve on paper
    text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", text)

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

    stamp = datetime.date.today().isoformat()
    sub = a.subtitle or f"generated {stamp} from {a.source.name}"
    html_body = html_body.replace(
        "</h1>", f"</h1><p class='meta'>{sub}</p>", 1)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=f"<article>{html_body}</article>").write_pdf(
        str(a.out), stylesheets=[CSS(string=CSS_TEXT)])
    kb = a.out.stat().st_size / 1024
    print(f"  wrote {a.out}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
