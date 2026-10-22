"""dashboard.html next to qc_summary.csv: one self-contained page (no network) with what needs attention per file,
how each language run went (plan, monitor, re-planning, tool library) and the full summary table, sortable."""
import html
import json
from pathlib import Path

import pandas as pd


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flags(r) -> list:
    """(severity, text) for one summary row: 'bad' needs action, 'warn' is worth a look."""
    out = []
    get = lambda c: r.get(c) if c in r and not (isinstance(r.get(c), float) and pd.isna(r.get(c))) else None
    if get("Corruption") not in (None, "valid"):
        out.append(("bad", f"corrupt: {get('Corruption')}"))
    if str(get("Valid_Extension")) == "False":
        out.append(("warn", "not .wav"))
    if str(get("Upsampled")) == "True":
        orig = _num(get("Estimated_Original_Rate"))
        out.append(("warn", f"upsampled from ~{orig / 1000:g} kHz" if orig else "upsampled"))
    if str(get("Audio_Language_OK")) == "False":
        out.append(("bad", f"sounds like {get('Audio_Language')}"))
    if str(get("Script_OK")) == "False":
        out.append(("warn", "ASR text not all in the expected script"))
    q = get("ASR_Transcript_Quality")
    if q and str(q).startswith("failed"):
        out.append(("bad", f"transcript {q}"))
    if (w := _num(get("WER"))) is not None and w > 0.8:
        out.append(("bad", f"WER {w:.2f}"))
    if (d := _num(get("ASR_Disagreement"))) is not None and d > 1.0:
        out.append(("warn", f"ASR models disagree ({d:.2f})"))
    if (s := _num(get("Low_Score_Chunk_Share"))) is not None and s > 0.3:
        out.append(("warn", f"judge: {s:.0%} of chunks look garbled"))
    if get("Shares_Audio_With"):
        out.append(("bad", f"same audio as {get('Shares_Audio_With')}"))
    if (p := _num(get("Public_Corpus_Matches"))):
        out.append(("bad", f"{p:.0f} matches in a public corpus"))
    if (n := _num(get("Near_Duplicate_Utterances"))):
        out.append(("warn", f"{n:.0f} near-duplicate utterances in other files"))
    if get("Speaker_In_Earlier_Batches"):
        out.append(("warn", "speaker heard in an earlier batch"))
    if (t := _num(get("Segments_With_Timing_Issues"))):
        out.append(("warn", f"{t:.0f} reference time-stamp issues"))
    if get("Other_Scripts_In_Reference"):
        out.append(("warn", f"other scripts in the reference: {get('Other_Scripts_In_Reference')}"))
    return out


def _read(path, kind="csv"):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if kind == "json" else pd.read_csv(path)
    except Exception:
        return None


def _table(df, cls="data"):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            text = "" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))
            cells.append(f"<td title=\"{html.escape(text)}\">{html.escape(text[:160])}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write(out: Path, runs: dict, summary: pd.DataFrame) -> Path:
    info = _read(out / "run_info.json", "json") or {}
    flagged = [(r["Language"], r["Filename"], flags(r)) for _, r in summary.iterrows()]
    hours = pd.to_numeric(summary.get("Duration_s"), errors="coerce").sum() / 3600 if "Duration_s" in summary else 0
    wer = pd.to_numeric(summary.get("WER"), errors="coerce") if "WER" in summary else pd.Series(dtype=float)
    n_bad = sum(any(s == "bad" for s, _ in f) for _, _, f in flagged)
    tiles = [("Files", len(summary)), ("Languages", summary["Language"].nunique()), ("Hours of audio", f"{hours:.1f}"),
             ("Files needing action", n_bad), ("Mean WER (files with a reference)", f"{wer.mean():.2f}" if wer.notna().any() else "-")]

    runs_html = []
    for lang, run in runs.items():
        res = run / "results"
        plan = _read(res / "plan.json", "json") or {}
        mon = _read(res / "monitor_log.csv")
        rep = _read(res / "replan_log.csv")
        lib = _read(res / "tool_library_log.csv")
        parts = [f"<h3>{html.escape(lang.title())}</h3><ul>"]
        if plan:
            parts.append(f"<li>Tasks run: {', '.join(map(str, plan.get('tasks', [])))} (structure from {html.escape(str(plan.get('structure_from')))})</li>")
            for key, what in (("added_by_checker", "added by the prompt checker"), ("added_by_keywords", "added because the prompt names them"),
                              ("added_as_dependencies", "added as dependencies"), ("skipped_no_reference", "skipped (no reference transcripts)")):
                if plan.get(key):
                    parts.append(f"<li>{what}: {', '.join(map(str, plan[key]))}</li>")
        if mon is not None and len(mon):
            last = mon.groupby("Task").tail(1)
            counts = last.Verdict.value_counts().to_dict()
            parts.append(f"<li>Monitor (last verdict per step): {', '.join(f'{k} {v}' for k, v in counts.items())}</li>")
            for r in last[last.Verdict != "OK"].itertuples():
                parts.append(f"<li class='warn'>Step {r.Task} ({html.escape(str(r.Step))}): {html.escape(str(r.Verdict))}: {html.escape(str(r.Reason))}</li>")
        if rep is not None and len(rep):
            for r in rep.itertuples():
                parts.append(f"<li>Re-planning, step {r.Task}: {'re-run' if r.Rerun else 'not re-run'}"
                             f"{' -> ' + html.escape(str(r.After_Rerun)) if r.Rerun else ''} ({html.escape(str(r.Planner_Reason))})</li>")
        if lib is not None and len(lib):
            for r in lib.itertuples():
                parts.append(f"<li>Tool library: saved script for {html.escape(str(r.Output))} {'passed' if r.Passed else 'failed'}</li>")
        parts.append("</ul>")
        runs_html.append("".join(parts))

    attention = [f"<tr><td>{html.escape(l)}</td><td>{html.escape(f)}</td><td>"
                 + " ".join(f"<span class='chip {s}'>{html.escape(t)}</span>" for s, t in fl) + "</td></tr>"
                 for l, f, fl in flagged if fl]
    models = ", ".join(f"{k.replace('_model', '')}: {v.get('name') if isinstance(v, dict) else v}"
                       for k, v in info.items() if k.endswith("_model"))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpeechQM Report</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--bad:#b42318;--badbg:#fde8e6;--warn:#8a5a00;--warnbg:#fdf3dc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--card:#1f1f1d;--line:#34332f;--bad:#ff9b8f;--badbg:#3d1b17;--warn:#f0c46a;--warnbg:#382c12}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1400px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} h3{{font-size:15px;margin:12px 0 4px}}
.muted{{color:var(--muted)}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}} .tile b{{display:block;font-size:22px}}
.chip{{display:inline-block;border-radius:10px;padding:1px 8px;margin:2px;font-size:12px}} .chip.bad{{background:var(--badbg);color:var(--bad)}}
.chip.warn{{background:var(--warnbg);color:var(--warn)}} li.warn{{color:var(--warn)}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top;white-space:nowrap}}
table.data td{{max-width:260px;overflow:hidden;text-overflow:ellipsis}} th{{position:sticky;top:0;background:var(--card);cursor:pointer}}
input{{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);width:min(360px,100%);margin:6px 0}}
.runs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}} .runs>div{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px 12px}}
</style></head><body>
<h1>SpeechQM quality report</h1>
<div class="muted">{html.escape(str(info.get('started', '')))} to {html.escape(str(info.get('finished', '')))} &middot; {html.escape(models)}</div>
<div class="tiles">{''.join(f"<div class='tile'><span class='muted'>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>" for k, v in tiles)}</div>
<h2>Files that need attention ({len(attention)})</h2>
<div class="wrap"><table><thead><tr><th>Language</th><th>File</th><th>Findings</th></tr></thead><tbody>{''.join(attention) or '<tr><td colspan=3>None</td></tr>'}</tbody></table></div>
<h2>Runs</h2><div class="runs">{''.join(f'<div>{r}</div>' for r in runs_html)}</div>
<h2>All results</h2><input id="q" placeholder="Filter rows" oninput="filt()">
<div class="wrap" id="all">{_table(summary)}</div>
<script>
function filt(){{const q=document.getElementById('q').value.toLowerCase();document.querySelectorAll('#all tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')}}
document.querySelectorAll('#all th').forEach((th,i)=>th.onclick=()=>{{const tb=th.closest('table').tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort((a,b)=>{{const x=a.cells[i].textContent,y=b.cells[i].textContent,nx=parseFloat(x),ny=parseFloat(y);const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c}});rows.forEach(r=>tb.appendChild(r))}});
</script></body></html>"""
    path = Path(out) / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    return path
