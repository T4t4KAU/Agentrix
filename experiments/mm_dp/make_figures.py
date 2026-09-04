#!/usr/bin/env python3
"""Generate the three core experiment figures as dependency-free SVG files."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKLOADS = ["small", "medium", "large", "xlarge"]
VISION_TOKENS = {"small": 196, "medium": 1024, "large": 4096, "xlarge": 9216}


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open() as f:
        return list(csv.DictReader(f))


def svg_text(x: float, y: float, value: str, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{value}</text>'


def figure1(rows: list[dict[str, str]], out: Path) -> dict:
    baseline = statistics.median(float(r["ttft_ms"]) for r in rows if r["case"] == "baseline")
    series = {}
    for case in ("same_rank_victim", "different_rank_victim"):
        series[case] = [
            statistics.median(float(r["ttft_ms"]) for r in rows if r["case"] == case and r["workload"] == w) / baseline
            for w in WORKLOADS
        ]
    width, height, left, top, right, bottom = 820, 500, 85, 55, 30, 75
    pw, ph = width - left - right, height - top - bottom
    ymax = 7.0
    xs = [left + i * pw / 3 for i in range(4)]
    y = lambda v: top + ph * (1 - v / ymax)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(0, 8):
        yy = y(tick)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>', svg_text(left - 12, yy + 5, str(tick), anchor="end")]
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#222"/>', f'<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}" stroke="#222"/>']
    for x, w in zip(xs, WORKLOADS):
        parts.append(svg_text(x, top + ph + 25, f'{VISION_TOKENS[w]}'))
    for case, color, label in (("same_rank_victim", "#d62728", "same rank"), ("different_rank_victim", "#1f77b4", "different rank")):
        points = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, series[case]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, v in zip(xs, series[case]):
            parts += [f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>', svg_text(x, y(v) - 10, f'{v:.2f}x', size=12)]
        lx = 590 if case == "same_rank_victim" else 590
        ly = 75 if case == "same_rank_victim" else 100
        parts += [f'<line x1="{lx}" y1="{ly-5}" x2="{lx+32}" y2="{ly-5}" stroke="{color}" stroke-width="3"/>', svg_text(lx + 42, ly, label, anchor="start")]
    parts += [svg_text(width / 2, 30, "Figure 1 — Vision Encode interference on 4K Prefill", 20, weight="bold"), svg_text(width / 2, height - 20, "vision tokens", 16), f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" font-family="sans-serif" font-size="16" text-anchor="middle">victim TTFT slowdown</text>', '</svg>']
    out.write_text("\n".join(parts))
    return {"baseline_ttft_ms": baseline, "slowdown": series}


def figure2(encoder: list[dict[str, str]], p2p: list[dict[str, str]], out: Path) -> dict:
    enc = {
        w: statistics.median(float(r["encoder_forward_ms"]) for r in encoder if r["workload"] == w and int(r["num_encoder_calls"]) == 1)
        for w in WORKLOADS
    }
    copy = {int(r["tokens"]): float(r["median_ms"]) for r in p2p}
    transfer = {w: copy[VISION_TOKENS[w]] for w in WORKLOADS}
    ratio = {w: enc[w] / transfer[w] for w in WORKLOADS}
    width, height, left, top, right, bottom = 820, 500, 85, 55, 30, 75
    pw, ph = width-left-right, height-top-bottom
    lo, hi = math.log10(0.05), math.log10(2000)
    y = lambda v: top + ph * (1 - (math.log10(v)-lo)/(hi-lo))
    xs = [left + (i + 0.5) * pw / 4 for i in range(4)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in (0.05, 0.1, 1, 10, 100, 1000, 2000):
        yy=y(tick); parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>', svg_text(left-12, yy+5, str(tick), anchor="end")]
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#222"/>', f'<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}" stroke="#222"/>']
    for x,w in zip(xs,WORKLOADS):
        for dx,value,color in ((-23,enc[w],"#d62728"),(23,transfer[w],"#2ca02c")):
            yy=y(value); parts.append(f'<rect x="{x+dx-18:.1f}" y="{yy:.1f}" width="36" height="{top+ph-yy:.1f}" fill="{color}"/>')
        parts += [svg_text(x, top+ph+25, str(VISION_TOKENS[w])), svg_text(x, top+12, f'R={ratio[w]:.0f}x', size=12, weight="bold")]
    parts += [svg_text(width/2,30,"Figure 2 — Vision recomputation vs GPU transfer",20,weight="bold"), svg_text(width/2,height-20,"vision tokens",16), f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" font-family="sans-serif" font-size="16" text-anchor="middle">latency (ms, log scale)</text>', '<rect x="590" y="65" width="16" height="16" fill="#d62728"/>', svg_text(615,78,"Encode",anchor="start"), '<rect x="680" y="65" width="16" height="16" fill="#2ca02c"/>', svg_text(705,78,"copy",anchor="start"), '</svg>']
    out.write_text("\n".join(parts))
    return {"encoder_ms": enc, "transfer_ms": transfer, "recompute_transfer_ratio": ratio}


def figure3(rows: list[dict[str, str]], out: Path) -> dict:
    loads = [0,2,4,6]
    deltas = {}
    for w in WORKLOADS:
        deltas[w] = {}
        for load in loads:
            hot = [float(r["ttft_ms"]) for r in rows if r["workload"]==w and int(r["background_load"])==load and r["destination"]=="hot_busy_rank0"]
            cold = [float(r["ttft_ms"]) for r in rows if r["workload"]==w and int(r["background_load"])==load and r["destination"]=="cold_idle_rank1"]
            deltas[w][load] = statistics.median(hot)-statistics.median(cold)
    width,height,left,top,cw,ch=820,500,170,90,145,78
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',svg_text(width/2,30,"Figure 3 — Cache locality / rank load crossover",20,weight="bold"),svg_text(left+2*cw,62,"concurrent 4K Prefill requests on hot rank",15)]
    for j,load in enumerate(loads): parts.append(svg_text(left+(j+0.5)*cw,top-12,str(load),weight="bold"))
    for i,w in enumerate(WORKLOADS):
        parts.append(svg_text(left-15,top+(i+0.5)*ch+5,f'{VISION_TOKENS[w]} tokens',anchor="end"))
        for j,load in enumerate(loads):
            d=deltas[w][load]; strength=min(abs(d)/1800,1); base=(44,160,44) if d<0 else (214,39,40); alpha=0.22+0.65*strength; color=tuple(round(255+(c-255)*alpha) for c in base); fill=f'rgb({color[0]},{color[1]},{color[2]})'; x=left+j*cw;y=top+i*ch
            winner="hot" if d<0 else "cold"
            parts += [f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="white" stroke-width="3"/>',svg_text(x+cw/2,y+ch/2-4,f'{winner} wins',14,weight="bold"),svg_text(x+cw/2,y+ch/2+18,f'Δ={d:+.0f} ms',12)]
    parts += [svg_text(width/2,height-22,"Δ = TTFT(hot busy rank) − TTFT(cold idle rank)",14),'</svg>']
    out.write_text("\n".join(parts))
    return {"hot_minus_cold_ttft_ms": deltas}


def main() -> None:
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    summary = {
        "figure1": figure1(read_csv("interference_calibrated.csv"), out/"figure1_interference.svg"),
        "figure2": figure2(read_csv("encoder_timing_cache_enabled.csv"), read_csv("p2p_gpu0_gpu1.csv"), out/"figure2_recompute_vs_transfer.svg"),
        "figure3": figure3(read_csv("cache_load_conflict.csv"), out/"figure3_cache_load_phase.svg"),
    }
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
