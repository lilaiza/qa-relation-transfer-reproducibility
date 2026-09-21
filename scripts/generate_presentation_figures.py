"""Generate dependency-free SVG figures from the end-to-end held-out run."""
import json
from collections import defaultdict
from pathlib import Path

RUN = Path("results/test_end_to_end_capa1_2026-09-08_v1/test_layer1.json")
OUT = Path("docs/figures")

def svg_bar(labels, series, title, path):
    width, height, left, bottom = 900, 520, 100, 100
    colors = ["#2563eb", "#f97316", "#16a34a"]
    maxv = 1
    barw = 520 / (len(labels) * len(series))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Arial,sans-serif;fill:#172033}}.t{{font-size:24px;font-weight:bold}}.l{{font-size:15px}}.s{{font-size:13px}}</style><rect width="100%" height="100%" fill="white"/><text x="{left}" y="40" class="t">{title}</text><line x1="{left}" y1="100" x2="{left}" y2="{height-bottom}" stroke="#334155"/><line x1="{left}" y1="{height-bottom}" x2="830" y2="{height-bottom}" stroke="#334155"/>']
    for y in range(0, 101, 20):
        yy = height-bottom-(height-bottom-80)*y/100
        parts += [f'<line x1="{left}" y1="{yy}" x2="830" y2="{yy}" stroke="#e2e8f0"/><text x="40" y="{yy+5}" class="s">{y}%</text>']
    for i, label in enumerate(labels):
        x = 180+i*250
        parts.append(f'<text x="{x+35}" y="445" class="l">{label}</text>')
        for j, (_, values) in enumerate(series):
            v=values[i]; h=(height-bottom-80)*v/maxv; bx=x+j*barw
            parts += [f'<rect x="{bx}" y="{height-bottom-h}" width="{barw-5}" height="{h}" fill="{colors[j]}"/><text x="{bx}" y="{height-bottom-h-7}" class="s">{v*100:.1f}%</text>']
    legend_x = 100
    for j, (name, _) in enumerate(series):
        parts.append(f'<rect x="{legend_x}" y="62" width="14" height="14" fill="{colors[j]}"/><text x="{legend_x + 22}" y="74" class="s">{name}</text>')
        legend_x += 40 + len(name) * 8
    path.write_text(''.join(parts)+'</svg>')

rows=json.loads(RUN.read_text())["rows"]; groups=defaultdict(list)
for r in rows: groups[r["condition"]].append(r)
conds=["baseline","same_entity_donor","different_entity_donor"]
top_target=[]; top_donor=[]; reader_target=[]; verifier=[]
for c in conds:
    rs=groups[c]; n=len(rs); top_target.append(sum(r["top_passage_id"]==r["target_passage_id"] for r in rs)/n); top_donor.append(sum(r["top_passage_id"]==r["donor_passage_id"] for r in rs)/n)
    reader_target.append(sum(str(r["reader_answer"]).strip().casefold()==next(p["answer"].strip().casefold() for p in r["retrieval"]["passages"] if p["id"]==r["target_passage_id"]) for r in rs)/n); verifier.append(sum(r["verifier_support_probability"] for r in rs)/n)
svg_bar(["Baseline", "Donor B", "B, other entity"], [("Target evidence A", top_target), ("Donor evidence B", top_donor)], "Localized patch redirects retrieval", OUT / "retrieval_shift.svg")
svg_bar(["Baseline", "Donor B", "B, other entity"], [("Reader answers A", reader_target), ("Verifier passage support", verifier)], "Effect propagates to Reader and Verifier", OUT / "downstream_propagation.svg")
