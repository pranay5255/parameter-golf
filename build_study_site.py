#!/usr/bin/env python3
"""
Build a static HTML study site for every NanoGPT speedrun record.
Parses research.md, PR data, and log files to create a comprehensive guide.
"""

import re
import json
import os
import html
import glob
from pathlib import Path

BASE = Path("/home/pranay5255/parameter-golf")
LOGS = BASE / "logs" / "records"
PR_DATA = BASE / "pr_data"
OUT = BASE / "study_site"

# ── Parse research.md ───────────────────────────────────────────────────────

def parse_records(md_path):
    """Parse the markdown table rows into structured records, line by line."""
    lines = md_path.read_text().split('\n')
    records = []
    current_track = None
    row_idx = 0  # unique index for dedup

    for line in lines:
        # Detect track sections
        if 'Record time | Description' in line:
            # Look back to determine which track
            continue
        if '## Speedrun track 2' in line:
            current_track = "track_2_medium"
            continue
        if current_track is None and '| # | Record time' in line:
            current_track = "track_1_short"
            continue
        if line.startswith('| -'):
            continue

        # Match table rows: "NUM | TIME | DESC | DATE | LOG | CONTRIBUTORS"
        # Lines start with a number, not with |
        m = re.match(r'^(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*$', line)
        if not m:
            continue

        if current_track is None:
            current_track = "track_1_short"

        num, time_str, desc, date, log_col, contributors = m.groups()

        # Extract time value
        time_match = re.search(r'([\d.]+)\s*(minutes|hours|min)', time_str)
        time_val = time_match.group(1) if time_match else time_str.strip()
        time_unit = "hours" if time_match and "hour" in time_match.group(2) else "minutes"

        # Extract description text and link
        desc_link = re.search(r'\[([^\]]+)\]\(([^)]+)\)', desc)
        desc_text = desc_link.group(1) if desc_link else desc.strip()
        desc_url = desc_link.group(2) if desc_link else None

        # Extract ALL log paths from the log column AND description column
        log_paths = re.findall(r'\[log\]\(([^)]+)\)', log_col)

        # Extract PR links from log column
        pr_links = re.findall(r'\[PR\]\(([^)]+)\)', log_col)
        pr_numbers = []
        for p in pr_links:
            pm = re.search(r'/pull/(\d+)', p)
            if pm:
                pr_numbers.append(pm.group(1).rstrip('/'))

        row_idx += 1
        records.append({
            "track": current_track,
            "number": int(num),
            "row_idx": row_idx,  # unique per line
            "time": time_val,
            "time_unit": time_unit,
            "description": desc_text,
            "description_url": desc_url,
            "date": date.strip(),
            "log_paths": log_paths,
            "pr_numbers": pr_numbers,
            "contributors": contributors.strip(),
            "raw_time_str": time_str.strip(),
        })
    return records


# ── Load PR data ────────────────────────────────────────────────────────────

def load_pr_data(pr_num):
    meta_path = PR_DATA / f"pr_{pr_num}_meta.json"
    files_path = PR_DATA / f"pr_{pr_num}_files.json"
    meta = {}
    files = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except:
            pass
    if files_path.exists():
        for line in files_path.read_text().strip().split('\n'):
            if line.strip():
                try:
                    files.append(json.loads(line))
                except:
                    pass
    return meta, files


# ── Parse log files for metrics ─────────────────────────────────────────────

def parse_log_metrics(log_path):
    """Extract key metrics from a training log file."""
    if not log_path.exists():
        return None

    text = log_path.read_text(errors='replace')
    metrics = {
        "has_code": text.strip().startswith("import"),
        "val_losses": [],
        "total_steps": None,
        "train_time_ms": None,
        "step_avg_ms": None,
        "peak_memory": None,
    }

    # Extract validation losses
    for m in re.finditer(r'val_loss:([\d.]+)', text):
        metrics["val_losses"].append(float(m.group(1)))

    # Extract final training time
    time_matches = list(re.finditer(r'train_time:(\d+)ms', text))
    if time_matches:
        metrics["train_time_ms"] = int(time_matches[-1].group(1))

    # Step avg
    avg_matches = list(re.finditer(r'step_avg:([\d.]+)ms', text))
    if avg_matches:
        metrics["step_avg_ms"] = float(avg_matches[-1].group(1))

    # Total steps
    step_matches = list(re.finditer(r'step:(\d+)/(\d+)', text))
    if step_matches:
        metrics["total_steps"] = int(step_matches[-1].group(2))

    # Peak memory
    mem_match = re.search(r'peak memory allocated:\s*(\d+)\s*MiB', text)
    if mem_match:
        metrics["peak_memory"] = int(mem_match.group(1))

    return metrics


# ── Extract code sections from log files ────────────────────────────────────

def extract_code_from_log(log_path):
    """Some log files embed the full training code before the training output."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors='replace')
    if not text.strip().startswith("import"):
        return None
    # Find where the training output starts (step: lines or similar)
    code_end = re.search(r'^(step:\d|s:\d|warmup|kernel|starting|Rank \d)', text, re.MULTILINE)
    if code_end:
        return text[:code_end.start()].rstrip()
    # If no training output found, might be all code
    if len(text) > 50000:
        return text[:50000]  # safety cap
    return None


# ── Categorize each record's change type ────────────────────────────────────

CATEGORIES = {
    "architecture": ["embedding", "rotary", "relu", "qk-norm", "skip connection", "u-net", "attention",
                     "head", "mlp", "softcap", "value embed", "smear", "bigram", "hyperconnection",
                     "paired head", "partitioned", "backout", "flatten"],
    "optimizer": ["muon", "adam", "learning rate", "lr", "momentum", "weight decay", "newton-schulz",
                  "polar express", "normuon", "cautious", "snoo", "ema"],
    "systems": ["pytorch", "compile", "fp8", "bfloat16", "bf16", "triton", "kernel", "fused",
                "distributed", "all-reduce", "reduce_scatter", "async", "overlap", "transpose"],
    "data": ["batch size", "sequence length", "window", "document", "eos", "bos", "token",
             "yarn", "max_seq_len", "max_doc_len"],
    "training": ["schedule", "cooldown", "warmup", "step", "accumulate", "gradient",
                 "multi-token", "prediction", "untie", "retie", "freeze"],
}

def categorize_record(desc):
    desc_lower = desc.lower()
    cats = []
    for cat, keywords in CATEGORIES.items():
        if any(kw in desc_lower for kw in keywords):
            cats.append(cat)
    return cats if cats else ["other"]


# ── HTML generation ─────────────────────────────────────────────────────────

def esc(s):
    return html.escape(str(s)) if s else ""

def render_diff_html(patch_text):
    """Render a unified diff patch as colored HTML."""
    if not patch_text:
        return '<p class="muted">Diff too large or binary file</p>'
    lines = patch_text.split('\n')
    out = []
    for line in lines:
        escaped = esc(line)
        if line.startswith('@@'):
            out.append(f'<span class="diff-hunk">{escaped}</span>')
        elif line.startswith('+'):
            out.append(f'<span class="diff-add">{escaped}</span>')
        elif line.startswith('-'):
            out.append(f'<span class="diff-del">{escaped}</span>')
        else:
            out.append(f'<span class="diff-ctx">{escaped}</span>')
    return '\n'.join(out)


def time_display(rec):
    t = rec["time"]
    u = rec["time_unit"]
    if u == "hours":
        return f"{t} hours"
    return f"{t} min"


def build_site():
    records = parse_records(BASE / "research.md")
    print(f"Parsed {len(records)} records")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records").mkdir(exist_ok=True)

    # ── Collect all data ────────────────────────────────────────────────
    all_data = []
    for rec in records:
        cats = categorize_record(rec["description"])

        # Load PR data
        pr_meta_list = []
        pr_files_list = []
        for pr_num in rec["pr_numbers"]:
            meta, files = load_pr_data(pr_num)
            pr_meta_list.append((pr_num, meta, files))
            pr_files_list.extend(files)

        # Find and parse log files
        log_metrics_list = []
        code_snapshots = []
        for lp in rec["log_paths"]:
            full_path = LOGS.parent / lp  # logs/../records/...  -> need LOGS base
            # Actually log_paths are like records/track_1_short/...
            full_path = BASE / "logs" / lp
            if full_path.is_file():
                m = parse_log_metrics(full_path)
                if m:
                    log_metrics_list.append((lp, m))
                code = extract_code_from_log(full_path)
                if code:
                    code_snapshots.append((lp, code))
            elif full_path.is_dir():
                # Directory of logs
                for f in sorted(full_path.glob("*.txt")) + sorted(full_path.glob("*.log")):
                    m = parse_log_metrics(f)
                    if m:
                        log_metrics_list.append((str(f.relative_to(BASE / "logs")), m))
                    code = extract_code_from_log(f)
                    if code:
                        code_snapshots.append((str(f.relative_to(BASE / "logs")), code))

        # Identify code-changing files in PRs (train_gpt.py, kernels, etc.)
        code_files = []
        other_files = []
        for pf in pr_files_list:
            fn = pf.get("filename", "")
            if fn.endswith(".py") and "record" not in fn and "data/" not in fn:
                code_files.append(pf)
            elif fn.endswith(".py"):
                other_files.append(pf)

        all_data.append({
            "rec": rec,
            "categories": cats,
            "pr_meta_list": pr_meta_list,
            "code_files": code_files,
            "other_files": other_files,
            "log_metrics": log_metrics_list,
            "code_snapshots": code_snapshots,
        })

    # ── Build index.html ────────────────────────────────────────────────
    track1 = [d for d in all_data if d["rec"]["track"] == "track_1_short"]
    track2 = [d for d in all_data if d["rec"]["track"] == "track_2_medium"]

    # Compute category stats
    cat_counts = {}
    for d in all_data:
        for c in d["categories"]:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    index_html = generate_index(track1, track2, cat_counts, all_data)
    (OUT / "index.html").write_text(index_html)
    print(f"Wrote index.html")

    # ── Build individual record pages ───────────────────────────────────
    for d in all_data:
        rec = d["rec"]
        slug = make_slug(rec)
        d["slug"] = slug
        page = generate_record_page(d, slug, all_data)
        (OUT / "records" / f"{slug}.html").write_text(page)

    print(f"Wrote {len(all_data)} record pages")
    print(f"Site built at: {OUT}")


# ── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --purple: #bc8cff; --orange: #f0883e;
    --diff-add-bg: #12261e; --diff-del-bg: #2d1215;
    --diff-add-text: #56d364; --diff-del-text: #f85149;
    --diff-hunk-bg: #1c2233;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 0; max-width: 100%;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }

/* Header */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2233 100%);
    border-bottom: 1px solid var(--border);
    padding: 3rem 2rem;
    text-align: center;
}
.hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
.hero h1 span { color: var(--accent); }
.hero .subtitle { color: var(--muted); font-size: 1rem; max-width: 700px; margin: 0 auto; }

/* Navigation tabs */
.tabs {
    display: flex; gap: 0; border-bottom: 1px solid var(--border);
    background: var(--surface); padding: 0 2rem;
    position: sticky; top: 0; z-index: 100;
}
.tab {
    padding: 0.8rem 1.5rem; cursor: pointer;
    border-bottom: 2px solid transparent;
    color: var(--muted); transition: all 0.2s;
    font-family: inherit; font-size: 0.85rem;
    background: none; border-top: none; border-left: none; border-right: none;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Stats bar */
.stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem; padding: 1.5rem 0;
}
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.2rem; text-align: center;
}
.stat-card .number { font-size: 2rem; font-weight: bold; color: var(--accent); }
.stat-card .label { color: var(--muted); font-size: 0.8rem; margin-top: 0.3rem; }

/* Category badges */
.badge {
    display: inline-block; padding: 0.15rem 0.6rem;
    border-radius: 12px; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-architecture { background: #1f2d5a; color: #79b8ff; }
.badge-optimizer { background: #2d1f3d; color: var(--purple); }
.badge-systems { background: #2d2a1f; color: var(--yellow); }
.badge-data { background: #1f2d2a; color: var(--green); }
.badge-training { background: #2d1f1f; color: var(--orange); }
.badge-other { background: #1f2222; color: var(--muted); }

/* Filter bar */
.filters {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    padding: 1rem 0; align-items: center;
}
.filter-btn {
    padding: 0.4rem 1rem; border-radius: 20px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--muted); cursor: pointer; font-family: inherit;
    font-size: 0.8rem; transition: all 0.2s;
}
.filter-btn:hover, .filter-btn.active {
    border-color: var(--accent); color: var(--accent);
    background: rgba(88, 166, 255, 0.1);
}

/* Timeline */
.timeline { position: relative; padding: 1rem 0; }
.timeline::before {
    content: ''; position: absolute; left: 24px; top: 0; bottom: 0;
    width: 2px; background: var(--border);
}

/* Record cards */
.record-card {
    position: relative; margin-left: 50px; margin-bottom: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.2rem 1.5rem;
    transition: all 0.2s; cursor: pointer;
}
.record-card:hover {
    border-color: var(--accent);
    transform: translateX(4px);
}
.record-card::before {
    content: ''; position: absolute; left: -34px; top: 1.5rem;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--accent); border: 2px solid var(--bg);
}
.record-header {
    display: flex; justify-content: space-between;
    align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;
}
.record-num {
    font-size: 0.75rem; color: var(--muted);
    background: rgba(255,255,255,0.05); padding: 0.1rem 0.5rem;
    border-radius: 4px;
}
.record-title { font-size: 1rem; font-weight: 600; flex: 1; min-width: 200px; }
.record-time {
    font-size: 1.2rem; font-weight: bold; color: var(--green);
    white-space: nowrap;
}
.record-meta {
    display: flex; gap: 1.5rem; margin-top: 0.5rem;
    font-size: 0.8rem; color: var(--muted); flex-wrap: wrap;
}
.record-badges { display: flex; gap: 0.4rem; margin-top: 0.5rem; flex-wrap: wrap; }

/* Tab content */
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Chart area */
.chart-container {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 2rem; margin: 1rem 0;
    position: relative; height: 400px;
}
.chart-svg { width: 100%; height: 100%; }

/* ── Record detail page ─────────────────────── */
.detail-hero {
    background: linear-gradient(135deg, #0d1117, #161b22);
    padding: 2rem; border-bottom: 1px solid var(--border);
}
.breadcrumb { color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }
.detail-title { font-size: 2rem; margin-bottom: 0.5rem; }
.detail-time { font-size: 1.5rem; color: var(--green); margin-bottom: 1rem; }

.section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin: 1rem 0; overflow: hidden;
}
.section-header {
    padding: 1rem 1.5rem; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: 1rem;
    display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; user-select: none;
}
.section-header:hover { background: rgba(255,255,255,0.02); }
.section-body { padding: 1.5rem; }

/* Diff viewer */
.diff-viewer {
    background: #0d1117; border-radius: 6px; overflow-x: auto;
    font-size: 0.8rem; line-height: 1.5;
}
.diff-file-header {
    padding: 0.8rem 1rem; background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
}
.diff-file-name { font-weight: 600; }
.diff-stats {
    font-size: 0.75rem;
}
.diff-stats .add { color: var(--green); }
.diff-stats .del { color: var(--red); }
.diff-content {
    padding: 0.5rem 1rem; white-space: pre; overflow-x: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.diff-add { color: var(--diff-add-text); background: var(--diff-add-bg); display: block; }
.diff-del { color: var(--diff-del-text); background: var(--diff-del-bg); display: block; }
.diff-hunk { color: var(--purple); background: var(--diff-hunk-bg); display: block; padding: 0.2rem 0; }
.diff-ctx { display: block; }

/* Metrics table */
.metrics-table {
    width: 100%; border-collapse: collapse;
}
.metrics-table th, .metrics-table td {
    padding: 0.5rem 1rem; text-align: left;
    border-bottom: 1px solid var(--border);
}
.metrics-table th { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }

/* Code viewer */
.code-viewer {
    background: #0d1117; border-radius: 6px;
    font-size: 0.75rem; line-height: 1.5;
    max-height: 600px; overflow: auto;
}
.code-viewer pre {
    padding: 1rem; white-space: pre; font-family: 'SF Mono', monospace;
    counter-reset: line;
}
.code-viewer pre .line {
    display: block; counter-increment: line;
}
.code-viewer pre .line::before {
    content: counter(line); display: inline-block; width: 4em;
    text-align: right; margin-right: 1em; color: var(--muted);
    opacity: 0.4; font-size: 0.7rem;
}

/* PR body */
.pr-body {
    padding: 1.5rem; font-size: 0.9rem; line-height: 1.8;
    color: var(--text);
}
.pr-body h1, .pr-body h2, .pr-body h3 { margin: 1rem 0 0.5rem; color: var(--accent); }
.pr-body p { margin: 0.5rem 0; }
.pr-body code {
    background: rgba(255,255,255,0.07); padding: 0.1rem 0.4rem;
    border-radius: 3px; font-size: 0.85em;
}
.pr-body pre { background: #0d1117; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 0.5rem 0; }
.pr-body pre code { background: none; padding: 0; }
.pr-body img { max-width: 100%; border-radius: 6px; margin: 0.5rem 0; }
.pr-body ul, .pr-body ol { padding-left: 1.5rem; margin: 0.5rem 0; }

/* Nav buttons */
.nav-buttons {
    display: flex; justify-content: space-between; padding: 2rem 0;
}
.nav-btn {
    padding: 0.6rem 1.5rem; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); cursor: pointer; font-family: inherit;
    font-size: 0.85rem; transition: all 0.2s;
}
.nav-btn:hover { border-color: var(--accent); color: var(--accent); }

.muted { color: var(--muted); font-style: italic; }

/* Collapsible */
.collapsible-content { display: none; }
.collapsible-content.open { display: block; }
.toggle-icon { transition: transform 0.2s; display: inline-block; }
.toggle-icon.open { transform: rotate(90deg); }

/* Study notes */
.study-note {
    background: rgba(88, 166, 255, 0.05);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem; margin: 1rem 0;
    border-radius: 0 6px 6px 0;
}
.study-note h4 { color: var(--accent); margin-bottom: 0.5rem; font-size: 0.9rem; }
.study-note p, .study-note ul { font-size: 0.85rem; color: var(--muted); }
.study-note ul { padding-left: 1.2rem; }

/* Responsive */
@media (max-width: 768px) {
    .container { padding: 1rem; }
    .hero h1 { font-size: 1.5rem; }
    .stats { grid-template-columns: repeat(2, 1fr); }
    .record-header { flex-direction: column; }
    .timeline::before { left: 14px; }
    .record-card { margin-left: 35px; }
    .record-card::before { left: -27px; }
}
"""


# ── Study analysis for each record ─────────────────────────────────────────

RECORD_ANALYSIS = {
    # Track 1
    (1, "track_1_short"): {
        "what": "llm.c baseline - standard GPT-2 small training",
        "why": "Establishes the 45-minute baseline on 8xH100. Uses standard AdamW optimizer with vanilla transformer architecture.",
        "key_concepts": ["GPT-2 architecture", "AdamW optimizer", "Cross-entropy loss", "FineWeb dataset"],
        "change_type": "baseline",
    },
    (2, "track_1_short"): {
        "what": "Tuned learning rate & added rotary positional embeddings",
        "why": "RoPE provides better length generalization than learned positional embeddings. LR tuning is low-hanging fruit.",
        "key_concepts": ["Rotary Position Embeddings (RoPE)", "Learning rate tuning", "Positional encoding"],
        "change_type": "architecture + hyperparameter",
    },
    (3, "track_1_short"): {
        "what": "Introduced the Muon optimizer",
        "why": "Muon uses spectral steepest descent (Newton-Schulz orthogonalization) instead of Adam's element-wise scaling. ~1.5x better sample efficiency with lower memory.",
        "key_concepts": ["Muon optimizer", "Newton-Schulz iteration", "Spectral norm", "Orthogonalization"],
        "change_type": "optimizer",
    },
    (4, "track_1_short"): {
        "what": "Muon improvements",
        "why": "Refined Newton-Schulz coefficients and Muon hyperparameters for faster convergence.",
        "key_concepts": ["Quintic polynomial coefficients", "Non-convergent coefficients for speed"],
        "change_type": "optimizer",
    },
    (5, "track_1_short"): {
        "what": "Modernized architecture: Pad embeddings, ReLU², zero-init projections, QK-Norm",
        "why": "ReLU² provides sharper activation than GELU. QK-Norm stabilizes attention. Zero-init projections (muP-like) improve training dynamics at init.",
        "key_concepts": ["ReLU² activation", "QK-Norm", "Zero initialization", "muP (maximal update parameterization)"],
        "change_type": "architecture",
    },
    (6, "track_1_short"): {
        "what": "Distributed Muon overhead across GPUs",
        "why": "Newton-Schulz iterations are compute-heavy. Distributing across GPUs amortizes the cost.",
        "key_concepts": ["Distributed optimizer", "Communication-computation overlap"],
        "change_type": "systems",
    },
    (7, "track_1_short"): {
        "what": "Upgraded to PyTorch 2.5.0",
        "why": "Newer PyTorch brings improved torch.compile, better CUDA kernels, and general optimizations.",
        "key_concepts": ["torch.compile improvements", "Framework upgrades"],
        "change_type": "systems",
    },
    (8, "track_1_short"): {
        "what": "Untied embedding and LM head weights",
        "why": "Separate embedding and head matrices allow each to specialize. Costs more parameters but improves loss.",
        "key_concepts": ["Weight tying vs untying", "Embedding/LM head independence"],
        "change_type": "architecture",
    },
    (9, "track_1_short"): {
        "what": "Value embeddings + skip connections + momentum warmup + logit softcap",
        "why": "Value embeddings (Zhou et al. 2024) mix extra learned vectors into attention values. Skip connections from embedding improve gradient flow. Logit softcapping prevents extreme logit values.",
        "key_concepts": ["Value Residual Learning", "Embedding skip connections", "Logit softcapping", "Momentum warmup"],
        "change_type": "architecture + training",
    },
    (10, "track_1_short"): {
        "what": "BFloat16 activations",
        "why": "Using bf16 for intermediate activations saves memory bandwidth without significant accuracy loss on H100s.",
        "key_concepts": ["Mixed precision training", "BFloat16", "Memory bandwidth optimization"],
        "change_type": "systems",
    },
    (11, "track_1_short"): {
        "what": "U-net pattern skip connections & doubled learning rate",
        "why": "U-net style connections (early layers skip to late layers) create richer gradient paths. Higher LR is possible with these stabilizing connections.",
        "key_concepts": ["U-Net skip connections", "Layer-to-layer shortcuts", "Learning rate scaling"],
        "change_type": "architecture + training",
    },
    (12, "track_1_short"): {
        "what": "1024-ctx dense causal attention → 64K-ctx FlexAttention",
        "why": "FlexAttention (PyTorch's flexible attention API) enables efficient long-context with sliding windows. Massive context expansion improves language modeling.",
        "key_concepts": ["FlexAttention", "Sliding window attention", "Long-context training", "Block masking"],
        "change_type": "architecture + systems",
    },
    (13, "track_1_short"): {
        "what": "Attention window warmup",
        "why": "Start training with small attention windows and gradually increase. Saves compute in early training when long-range dependencies aren't yet useful.",
        "key_concepts": ["Curriculum learning", "Window size scheduling", "Compute efficiency"],
        "change_type": "training",
    },
    (14, "track_1_short"): {
        "what": "Value Embeddings - learned vectors mixed into attention values",
        "why": "Extra embedding tables that are added to value projections, allowing the model to inject position/content-dependent bias.",
        "key_concepts": ["Value embeddings", "Attention augmentation", "Learned biases"],
        "change_type": "architecture",
    },
    (15, "track_1_short"): {
        "what": "U-net pattern value embeddings + code optimizations",
        "why": "Applying the U-net skip pattern to value embeddings. Systems optimizations reduce overhead.",
        "key_concepts": ["U-net value embeddings", "Code optimization", "MFU improvements"],
        "change_type": "architecture + systems",
    },
    (16, "track_1_short"): {
        "what": "Split value embeddings, block sliding window, separate block mask",
        "why": "Splitting value embeddings allows different patterns per head group. Block-level sliding windows are more GPU-friendly.",
        "key_concepts": ["Split embeddings", "Block-level masks", "GPU-friendly attention patterns"],
        "change_type": "architecture + systems",
    },
    (17, "track_1_short"): {
        "what": "Sparsify value embeddings, improve rotary embeddings, drop attention layer",
        "why": "Sparse value embeddings reduce compute. Improved RoPE frequencies. Removing a redundant attention layer saves time without hurting loss.",
        "key_concepts": ["Sparse embeddings", "Layer pruning", "RoPE frequency tuning"],
        "change_type": "architecture",
    },
    (18, "track_1_short"): {
        "what": "Lower logit softcap from 30 to 15",
        "why": "A tighter softcap (15 vs 30) more aggressively regularizes extreme predictions, improving generalization.",
        "key_concepts": ["Logit softcapping", "Regularization", "Hyperparameter sensitivity"],
        "change_type": "training",
    },
    (19, "track_1_short"): {
        "what": "FP8 head, offset logits, lr decay to 0.1 instead of 0.0",
        "why": "FP8 matmul for the LM head saves significant compute on H100. Non-zero LR floor prevents the optimizer from stalling at end of training.",
        "key_concepts": ["FP8 computation", "H100 FP8 tensor cores", "LR schedule design", "Asymmetric rescaling"],
        "change_type": "systems + training",
    },
    (20, "track_1_short"): {
        "what": "Merged QKV weights, long-short attention, attention scale, lower Adam epsilon, batched Muon",
        "why": "QKV weight fusion reduces kernel launch overhead. Long-short attention (Gemma 2) alternates window sizes. Batching Muon updates improves throughput.",
        "key_concepts": ["QKV fusion", "Long-short sliding window (Gemma 2)", "Batched optimizer updates"],
        "change_type": "architecture + optimizer + systems",
    },
    (21, "track_1_short"): {
        "what": "Reduced batch size",
        "why": "Smaller batches mean more parameter updates per token. At this stage, update frequency matters more than gradient quality.",
        "key_concepts": ["Batch size vs update frequency tradeoff", "Critical batch size"],
        "change_type": "training",
    },
    (22, "track_1_short"): {
        "what": "Faster gradient all-reduce",
        "why": "Optimized the distributed communication pattern for gradient synchronization, reducing the wall-clock overhead of multi-GPU training.",
        "key_concepts": ["Gradient all-reduce", "NCCL communication", "Ring all-reduce optimization"],
        "change_type": "systems",
    },
    (23, "track_1_short"): {
        "what": "Overlap computation and gradient communication",
        "why": "By overlapping backward pass computation with gradient communication, we hide the latency of multi-GPU synchronization.",
        "key_concepts": ["Computation-communication overlap", "Async gradient sync", "Pipeline parallelism"],
        "change_type": "systems",
    },
    (24, "track_1_short"): {
        "what": "Replace gradient all_reduce with reduce_scatter",
        "why": "reduce_scatter only gives each GPU its own shard of the gradient, avoiding the cost of broadcasting the full result back. Combined with optimizer sharding.",
        "key_concepts": ["reduce_scatter vs all_reduce", "Zero-style optimizer sharding", "Communication bandwidth"],
        "change_type": "systems",
    },
    (25, "track_1_short"): {
        "what": "Upgrade to PyTorch 2.9.0.dev",
        "why": "Nightly PyTorch builds often contain significant performance improvements in torch.compile and CUDA kernels.",
        "key_concepts": ["Framework version impact", "torch.compile improvements"],
        "change_type": "systems",
    },
    (26, "track_1_short"): {
        "what": "Align training batch starts with EoS, increase cooldown fraction",
        "why": "Starting sequences at sentence boundaries gives cleaner training signal. Longer LR cooldown fraction (0.45) allows more fine-grained optimization at end of training.",
        "key_concepts": ["Document boundary alignment", "BOS/EOS tokens", "LR cooldown schedule"],
        "change_type": "data + training",
    },
    (27, "track_1_short"): {
        "what": "Transpose MLP matrix + Triton kernel for symmetric matmul",
        "why": "Transposing one MLP matrix enables symmetric matmul in Muon's Newton-Schulz step. Custom Triton kernel exploits this symmetry for ~2x speedup in orthogonalization.",
        "key_concepts": ["Symmetric matrix multiplication", "Triton kernels", "Matrix layout optimization", "Newton-Schulz optimization"],
        "change_type": "optimizer + systems",
    },
    (28, "track_1_short"): {
        "what": "Sparse attention gate",
        "why": "A learned gate that can zero out attention contributions, effectively making attention sparse. Reduces compute for uninformative attention patterns.",
        "key_concepts": ["Sparse attention", "Gating mechanisms", "Attention sparsity"],
        "change_type": "architecture",
    },
    (29, "track_1_short"): {
        "what": "Flash Attention 3 with variable-length sequences",
        "why": "FA3 has higher SM utilization on Hopper GPUs than FlexAttention. Variable-length packing (flash_attn_varlen_func) enables efficient document masking with max_doc_len=2048.",
        "key_concepts": ["Flash Attention 3", "Hopper GPU SM utilization", "Variable-length attention", "Document packing"],
        "change_type": "systems + architecture",
    },
    (30, "track_1_short"): {
        "what": "Drop first MLP layer",
        "why": "The first MLP layer contributes less than later layers. Removing it saves compute with minimal loss impact, as early layers primarily do token mixing via attention.",
        "key_concepts": ["Layer pruning", "Compute allocation across layers", "Early vs late layer roles"],
        "change_type": "architecture",
    },
    (31, "track_1_short"): {
        "what": "Dynamic YaRN (Yet another RoPE extensioN)",
        "why": "YaRN adjusts RoPE frequency scaling dynamically as window sizes change during training, preventing the attention pattern degradation that occurs when extending context.",
        "key_concepts": ["YaRN", "RoPE frequency scaling", "Context extension", "Dynamic frequency adjustment"],
        "change_type": "architecture + training",
    },
    (32, "track_1_short"): {
        "what": "Optimize distributed training + improve skip gating + bf16 usage",
        "why": "Multiple small improvements: vectorized sigmoid for gates, improved bfloat16 usage reducing memory bandwidth, and distributed training optimizations.",
        "key_concepts": ["Vectorized operations", "Memory bandwidth optimization", "Skip connection gating"],
        "change_type": "systems + architecture",
    },
    (33, "track_1_short"): {
        "what": "Async data loading + extended final layer attention window for validation",
        "why": "Asynchronous data batching hides I/O latency. Extending the attention window in the final layer during validation improves eval loss without training cost.",
        "key_concepts": ["Async data pipeline", "Validation-time tricks", "Attention window at eval time"],
        "change_type": "systems + training",
    },
    (34, "track_1_short"): {
        "what": "Smear token embeddings 1 position forward",
        "why": "A 'smear' module mixes each token's embedding with the next position, giving the model a slight lookahead. Simple linear interpolation.",
        "key_concepts": ["Token smearing", "1-token lookahead", "Embedding mixing"],
        "change_type": "architecture",
    },
    (35, "track_1_short"): {
        "what": "Drop first attention layer + extend long windows for validation",
        "why": "First attention layer is least impactful. Extending all long-window layers to full context during validation gives free eval improvement.",
        "key_concepts": ["Layer pruning", "Validation-time window extension", "Compute reallocation"],
        "change_type": "architecture + training",
    },
    (36, "track_1_short"): {
        "what": "Custom Muon sizing + shared reduce_scatter for MLP and attention",
        "why": "Custom parameter group sizing for Muon optimizer. Combining MLP and attention gradient communications into a single reduce_scatter call reduces communication overhead.",
        "key_concepts": ["Optimizer parameter grouping", "Communication fusion", "reduce_scatter batching"],
        "change_type": "optimizer + systems",
    },
    (37, "track_1_short"): {
        "what": "Compute cross entropy in BF16 during training",
        "why": "Cross-entropy in BF16 is faster and the precision loss is acceptable during training (FP32 still used for validation).",
        "key_concepts": ["Mixed precision loss computation", "BF16 cross entropy", "Training vs validation precision"],
        "change_type": "systems",
    },
    (38, "track_1_short"): {
        "what": "Polar Express - replacement for Newton-Schulz in Muon",
        "why": "Polar Express is a more efficient method to compute the polar decomposition (orthogonalization) used by Muon. Fewer iterations needed than Newton-Schulz.",
        "key_concepts": ["Polar decomposition", "Polar Express algorithm", "Newton-Schulz alternative", "Matrix orthogonalization"],
        "change_type": "optimizer",
    },
    (39, "track_1_short"): {
        "what": "Update Adam params every other step + reduce batch size",
        "why": "Adam parameters (embeddings, heads) need fewer updates than Muon parameters. Skipping every other Adam step saves compute. Smaller batch size increases update frequency.",
        "key_concepts": ["Heterogeneous update frequencies", "Adam vs Muon update cadence", "Batch size reduction"],
        "change_type": "optimizer + training",
    },
    (40, "track_1_short"): {
        "what": "Backout architecture + hyperparameter tuning + lambda padding optimization",
        "why": "'Backout' enables model to back out contributions from early layers before prediction. Skip connection from 2/3 point. Misc hyperparameter tuning and padding optimizations.",
        "key_concepts": ["Backout connections", "Layer contribution gating", "Hyperparameter tuning"],
        "change_type": "architecture + training",
    },
    (41, "track_1_short"): {
        "what": "NorMuon optimizer",
        "why": "NorMuon normalizes Muon updates by their spectral norm, providing more stable training dynamics. Based on the NorMuon paper (arXiv:2510.05491).",
        "key_concepts": ["NorMuon", "Spectral normalization of updates", "Optimizer stability"],
        "change_type": "optimizer",
    },
    (42, "track_1_short"): {
        "what": "Update NorMuon learning rate and step logic",
        "why": "Tuning NorMuon's LR and fixing step logic to properly account for the normalization. Better calibrated updates.",
        "key_concepts": ["NorMuon LR tuning", "Step logic correction"],
        "change_type": "optimizer",
    },
    (43, "track_1_short"): {
        "what": "Cautious Weight Decay with schedule tied to LR",
        "why": "Cautious WD only decays weights when the decay direction agrees with the gradient direction. Schedule ties decay strength to LR to prevent over-regularization late in training.",
        "key_concepts": ["Cautious weight decay", "Gradient-aware regularization", "WD scheduling"],
        "change_type": "optimizer + training",
    },
    (44, "track_1_short"): {
        "what": "Backward hooks on Adam for gradient synchronization",
        "why": "Using backward hooks to trigger Adam gradient sync immediately when gradients are ready, rather than waiting for the full backward pass to complete.",
        "key_concepts": ["Backward hooks", "Eager gradient synchronization", "Profiling-guided optimization"],
        "change_type": "systems",
    },
    (45, "track_1_short"): {
        "what": "Refine skip architecture + exponential decay initialization",
        "why": "Improved skip connection pattern and updated exponential decay initialization for better training dynamics at startup.",
        "key_concepts": ["Skip connection refinement", "Exponential decay of residual stream", "Initialization strategies"],
        "change_type": "architecture",
    },
    (46, "track_1_short"): {
        "what": "Batch size schedule",
        "why": "Start training with smaller batches (more updates) then increase batch size as training progresses. Follows critical batch size theory.",
        "key_concepts": ["Batch size scheduling", "Critical batch size", "Curriculum training"],
        "change_type": "training",
    },
    (47, "track_1_short"): {
        "what": "Multiply attention lambda with weight instead of data, fix warmup",
        "why": "Moving the lambda multiplication from data to weights is mathematically equivalent but more compute-efficient (done once per weight, not per token).",
        "key_concepts": ["Attention lambda optimization", "Weight vs data multiplication", "Warmup fix"],
        "change_type": "architecture + systems",
    },
    (48, "track_1_short"): {
        "what": "Speed up Muon + additional pre-multiply lambda + reshape matrices + NorMuon axis update",
        "why": "Multiple optimizer optimizations: faster Muon via matrix reshaping, pre-multiplying lambda constants, and updating NorMuon normalization axis.",
        "key_concepts": ["Muon speedup", "Matrix reshape for efficiency", "Pre-multiplication optimization"],
        "change_type": "optimizer + systems",
    },
    (49, "track_1_short"): {
        "what": "Partial Key Offset",
        "why": "Offset only a portion of the key vectors, reducing the computational overhead while preserving most of the positional information benefit.",
        "key_concepts": ["Partial key offset", "Attention key manipulation", "Compute-accuracy tradeoff"],
        "change_type": "architecture",
    },
    (50, "track_1_short"): {
        "what": "Extend Cautious Weight Decay to Adam parameters",
        "why": "Previously only applied to Muon parameters. Extending CWD to Adam-managed parameters (embeddings, heads) provides consistent regularization.",
        "key_concepts": ["Cautious weight decay for Adam", "Consistent regularization across parameter groups"],
        "change_type": "optimizer",
    },
    (51, "track_1_short"): {
        "what": "Retie embedding to LM head + retune FP8 scales",
        "why": "Re-tying weights saves parameters and compute. FP8 scale retuning compensates for the changed weight sharing dynamics.",
        "key_concepts": ["Weight tying/untying strategy", "FP8 scale calibration", "Parameter efficiency"],
        "change_type": "architecture + systems",
    },
    (52, "track_1_short"): {
        "what": "Smooth scalars via beta increase + smear gate LR + freeze scalars during transitions",
        "why": "Increasing EMA beta for scalar parameters smooths their updates. Freezing scalars during architectural transitions (like untying) prevents instability.",
        "key_concepts": ["Scalar smoothing", "EMA beta scheduling", "Transition stability", "Gate learning rate"],
        "change_type": "training",
    },
    (53, "track_1_short"): {
        "what": "Multi-token prediction + untie embed/lm_head at 2/3 training",
        "why": "MTP predicts multiple future tokens simultaneously, providing richer gradient signal. Untying weights partway through training allows specialization after initial shared learning.",
        "key_concepts": ["Multi-token prediction", "Dynamic weight tying", "Training phase transitions", "Auxiliary prediction heads"],
        "change_type": "architecture + training",
    },
    (54, "track_1_short"): {
        "what": "Asymmetric logit rescale",
        "why": "Different scaling for positive and negative logits before softmax, allowing the model to be more confident in predictions while controlling extreme values asymmetrically.",
        "key_concepts": ["Asymmetric rescaling", "Logit manipulation", "Softcap alternatives"],
        "change_type": "architecture",
    },
    (55, "track_1_short"): {
        "what": "Gates on value embeddings and skip connections",
        "why": "Learned gates on value embeddings and skip connections let the model dynamically control information flow, enabling or suppressing different pathways per layer.",
        "key_concepts": ["Gated connections", "Learned information flow", "Dynamic skip connections"],
        "change_type": "architecture",
    },
    (56, "track_1_short"): {
        "what": "Optimize and compile Adam + increase buffer precision + move gates to Adam",
        "why": "torch.compile on Adam optimizer reduces Python overhead. Higher precision Adam buffers prevent accumulation errors. Moving gate parameters to Adam gives them independent optimization.",
        "key_concepts": ["Compiled optimizer", "Adam buffer precision", "Parameter bank assignment"],
        "change_type": "optimizer + systems",
    },
    (57, "track_1_short"): {
        "what": "BFloat16 attn/mlp weights + mixed precision Muon + interweaved Adam/Muon",
        "why": "Storing weights in BF16 saves memory bandwidth. Mixed precision Muon keeps high precision only where needed. Interweaving Adam and Muon steps avoids stale gradients.",
        "key_concepts": ["Mixed precision weights", "Interleaved optimizer steps", "Memory bandwidth optimization"],
        "change_type": "optimizer + systems",
    },
    (58, "track_1_short"): {
        "what": "Paired Head Attention",
        "why": "Pairs of attention heads share computation (e.g., shared QK but different V), effectively doubling head count at reduced cost.",
        "key_concepts": ["Paired head attention", "Head sharing", "Attention efficiency"],
        "change_type": "architecture",
    },
    (59, "track_1_short"): {
        "what": "Fused Triton kernel for linear relu² MLP step",
        "why": "Fusing the linear projection + ReLU² activation into a single Triton kernel eliminates intermediate memory reads/writes.",
        "key_concepts": ["Kernel fusion", "Triton kernels", "Memory bandwidth elimination", "ReLU² fusion"],
        "change_type": "systems",
    },
    (60, "track_1_short"): {
        "what": "Fused Triton kernel for softcapped multi-token prediction cross entropy",
        "why": "Fusing the softcap + MTP cross-entropy into one kernel avoids materializing large intermediate tensors and reduces memory traffic.",
        "key_concepts": ["Cross-entropy kernel fusion", "MTP loss fusion", "Softcap integration"],
        "change_type": "systems",
    },
    (61, "track_1_short"): {
        "what": "Unified optimizers and transposed LM head",
        "why": "Unifying Adam and Muon into a single optimizer loop reduces scheduling overhead. Transposing LM head enables more efficient matmul layout.",
        "key_concepts": ["Optimizer unification", "LM head transpose", "Memory layout optimization"],
        "change_type": "optimizer + systems",
    },
    (62, "track_1_short"): {
        "what": "Bigram Hash Embedding",
        "why": "A hash-based bigram embedding captures 2-gram statistics cheaply. The hash function maps character pairs to embedding indices, providing subword information without a full bigram vocabulary.",
        "key_concepts": ["Bigram hash embedding", "Feature hashing", "N-gram features", "Cheap embedding augmentation"],
        "change_type": "architecture",
    },
    (63, "track_1_short"): {
        "what": "Untie value embeddings",
        "why": "Separate value embeddings per layer group rather than shared, allowing each layer to learn specialized value representations.",
        "key_concepts": ["Value embedding independence", "Per-layer specialization"],
        "change_type": "architecture",
    },
    (64, "track_1_short"): {
        "what": "Tuned nonzero attention V and O initialization",
        "why": "Instead of zero-init for V and O projections, use carefully tuned small nonzero init. This provides better gradient flow at start of training (mimetic initialization).",
        "key_concepts": ["Mimetic initialization", "V/O projection init", "Non-zero initialization"],
        "change_type": "architecture",
    },
    (65, "track_1_short"): {
        "what": "Group value embeddings into single parameter",
        "why": "Consolidating multiple value embedding parameters into one fused parameter reduces optimizer overhead and enables more efficient memory access patterns.",
        "key_concepts": ["Parameter fusion", "Value embedding consolidation", "Memory access patterns"],
        "change_type": "systems",
    },
    (66, "track_1_short"): {
        "what": "Upgrade to PyTorch 2.10",
        "why": "Framework upgrade providing compiler improvements and better kernel generation.",
        "key_concepts": ["Framework upgrade", "torch.compile improvements"],
        "change_type": "systems",
    },
    (67, "track_1_short"): {
        "what": "Tune fused softcap kernels + fuse FP8 quantization in LM head",
        "why": "Tuning Triton kernel parameters (block sizes, num_warps) for the softcap kernel. Fusing FP8 quantization into the LM head forward pass eliminates a separate quantization step.",
        "key_concepts": ["Kernel tuning", "FP8 quantization fusion", "Triton autotuning"],
        "change_type": "systems",
    },
    (68, "track_1_short"): {
        "what": "Move bigram hash computation to GPU",
        "why": "Previously computed on CPU. Moving to GPU eliminates the CPU→GPU transfer bottleneck for bigram hash embeddings.",
        "key_concepts": ["CPU to GPU migration", "Host-to-device transfer elimination"],
        "change_type": "systems",
    },
    (69, "track_1_short"): {
        "what": "Kernel optimizations",
        "why": "Fine-tuning various Triton kernel parameters including block sizes, warp counts, and pipeline stages for better hardware utilization.",
        "key_concepts": ["Triton kernel optimization", "Block size tuning", "Hardware utilization"],
        "change_type": "systems",
    },
    (70, "track_1_short"): {
        "what": "Tune value embedding layout and ve_gates",
        "why": "Optimizing the memory layout of value embeddings and their gating parameters for better cache locality and fewer memory transactions.",
        "key_concepts": ["Memory layout optimization", "Cache locality", "Value embedding tuning"],
        "change_type": "architecture + systems",
    },
    (71, "track_1_short"): {
        "what": "Sparse bigram gradient communication + optimized CPU loading",
        "why": "Bigram embedding gradients are sparse (most entries zero). Using sparse communication avoids sending zeros. CPU data loading optimization reduces host overhead.",
        "key_concepts": ["Sparse gradient communication", "Sparse all-reduce", "CPU loading optimization"],
        "change_type": "systems",
    },
    (72, "track_1_short"): {
        "what": "Increase minimum LR + max_seq_len schedule",
        "why": "Higher minimum LR prevents learning from stalling at end of training. max_seq_len schedule starts with shorter sequences and increases, similar to batch size scheduling.",
        "key_concepts": ["Minimum LR floor", "Sequence length scheduling", "Training curriculum"],
        "change_type": "training",
    },
    (73, "track_1_short"): {
        "what": "Partitioned Hyperconnections",
        "why": "A generalization of skip connections where the residual stream is partitioned into segments with learned mixing coefficients. More expressive than simple skip connections.",
        "key_concepts": ["Partitioned hyperconnections", "Residual stream partitioning", "Learned mixing coefficients"],
        "change_type": "architecture",
    },
    (74, "track_1_short"): {
        "what": "Flattened GPT forward + remove post-attention lambdas + transpose kernels",
        "why": "Flattening the forward pass removes nested function calls, enabling better torch.compile optimization. Removing post-attention lambdas simplifies computation. Custom transpose kernels.",
        "key_concepts": ["Forward pass flattening", "Compiler-friendly code", "Custom transpose kernels"],
        "change_type": "systems + architecture",
    },
    (75, "track_1_short"): {
        "what": "Cross entropy kernel optimizations",
        "why": "Optimized Triton kernel for cross-entropy loss computation with better memory access patterns and reduced register pressure.",
        "key_concepts": ["Cross-entropy kernel optimization", "Register pressure", "Memory coalescing"],
        "change_type": "systems",
    },
    (76, "track_1_short"): {
        "what": "Reuse and tune backward transpose kernel",
        "why": "Reusing the forward transpose kernel in the backward pass and tuning its parameters eliminates a redundant kernel.",
        "key_concepts": ["Kernel reuse", "Transpose optimization", "Backward pass efficiency"],
        "change_type": "systems",
    },
    (77, "track_1_short"): {
        "what": "Replace partitioned hyperconnections with single saved activation",
        "why": "Simplification: instead of full partitioned hyperconnections, save a single activation and use it as a skip. Fewer parameters, similar performance.",
        "key_concepts": ["Architecture simplification", "Saved activation skip", "Occam's razor in architecture"],
        "change_type": "architecture",
    },
    # Track 2 records
    (1, "track_2_medium"): {
        "what": "llm.c baseline for GPT-2 Medium (350M parameters)",
        "why": "Establishes the 5.8 hour baseline for the larger GPT-2 Medium model targeting 2.92 validation loss.",
        "key_concepts": ["GPT-2 Medium architecture", "350M parameters", "Scaling from Small to Medium"],
        "change_type": "baseline",
    },
    (2, "track_2_medium"): {
        "what": "Scale up GPT-2 Small track speedrun techniques",
        "why": "Direct transfer of optimizations from the Small track (Muon, modern arch, etc.) to the Medium model.",
        "key_concepts": ["Technique transfer across scales", "Scaling speedrun methods"],
        "change_type": "architecture + optimizer",
    },
    (3, "track_2_medium"): {
        "what": "Standard weight decay",
        "why": "Adding conventional weight decay regularization to prevent overfitting in the larger model.",
        "key_concepts": ["Weight decay", "Regularization at scale"],
        "change_type": "optimizer",
    },
    (4, "track_2_medium"): {
        "what": "Tuned Muon Newton-Schulz coefficients",
        "why": "The optimal Newton-Schulz polynomial coefficients may differ at larger scale. Retuning for the Medium model.",
        "key_concepts": ["Scale-dependent optimizer tuning", "Newton-Schulz coefficient optimization"],
        "change_type": "optimizer",
    },
    (9, "track_2_medium"): {
        "what": "Add two value embeddings",
        "why": "Adding value embeddings to the Medium track, similar to Small track records 14-17.",
        "key_concepts": ["Value embeddings at scale", "Cross-track technique transfer"],
        "change_type": "architecture",
    },
    (12, "track_2_medium"): {
        "what": "Snoo Optimizer - outer optimizer around Adam and Muon",
        "why": "Snoo wraps both Adam and Muon with an outer optimization loop, providing meta-learning-like adaptation of their updates.",
        "key_concepts": ["Snoo optimizer", "Outer optimizer", "Meta-learning-inspired optimization"],
        "change_type": "optimizer",
    },
    (13, "track_2_medium"): {
        "what": "EMA Wrapper on Muon",
        "why": "Exponential Moving Average of Muon-optimized weights provides smoother model for evaluation, similar to Polyak averaging.",
        "key_concepts": ["EMA", "Polyak averaging", "Weight averaging for generalization"],
        "change_type": "optimizer",
    },
    (16, "track_2_medium"): {
        "what": "Smear + Multi-Token Prediction for Medium track",
        "why": "Bringing the smear module and MTP from the Small track to Medium, with scale-appropriate tuning.",
        "key_concepts": ["Smear module at scale", "MTP at scale", "Technique transfer"],
        "change_type": "architecture",
    },
    (18, "track_2_medium"): {
        "what": "Bulk transfer of Short track features",
        "why": "Wholesale import of many optimizations from the Small track that had accumulated since the Medium track last synced.",
        "key_concepts": ["Bulk technique transfer", "Cross-track synchronization"],
        "change_type": "architecture + optimizer + systems",
    },
}

def get_analysis(rec_num, track):
    key = (rec_num, track)
    return RECORD_ANALYSIS.get(key, None)


def make_slug(rec):
    """Create a unique, filesystem-safe slug for a record."""
    date_safe = rec['date'].replace('/', '-').replace(' ', '')
    return f"{rec['track']}_r{rec['row_idx']:02d}_{rec['number']}_{date_safe}"


# ── Generate index page ─────────────────────────────────────────────────────

def generate_index(track1, track2, cat_counts, all_data):
    # Build timeline chart data
    t1_points = []
    for d in track1:
        try:
            t = float(d["rec"]["time"])
            if d["rec"]["time_unit"] == "hours":
                t *= 60
            t1_points.append((d["rec"]["number"], t, d["rec"]["description"][:40]))
        except:
            pass

    t2_points = []
    for d in track2:
        try:
            t = float(d["rec"]["time"])
            if d["rec"]["time_unit"] == "hours":
                t *= 60
            t2_points.append((d["rec"]["number"], t, d["rec"]["description"][:40]))
        except:
            pass

    # SVG chart for track 1
    def make_chart_svg(points, label):
        if not points:
            return ""
        max_t = max(p[1] for p in points)
        min_t = min(p[1] for p in points)
        max_n = max(p[0] for p in points)
        min_n = min(p[0] for p in points)
        w, h = 900, 340
        pad_l, pad_r, pad_t, pad_b = 70, 30, 20, 50

        def sx(n):
            if max_n == min_n: return pad_l + (w - pad_l - pad_r) / 2
            return pad_l + (n - min_n) / (max_n - min_n) * (w - pad_l - pad_r)
        def sy(t):
            if max_t == min_t: return pad_t + (h - pad_t - pad_b) / 2
            return pad_t + (t - min_t) / (max_t - min_t) * (h - pad_t - pad_b)

        svg = f'<svg viewBox="0 0 {w} {h}" class="chart-svg">'
        # Grid lines
        for i in range(5):
            t_val = min_t + (max_t - min_t) * i / 4
            y = sy(t_val)
            svg += f'<line x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}" stroke="#30363d" stroke-width="0.5"/>'
            svg += f'<text x="{pad_l-10}" y="{y+4}" fill="#8b949e" font-size="11" text-anchor="end">{t_val:.1f}m</text>'

        # Line
        path_d = " ".join(f"{'M' if i==0 else 'L'}{sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(points))
        svg += f'<path d="{path_d}" fill="none" stroke="#58a6ff" stroke-width="2"/>'

        # Points
        for p in points:
            x, y = sx(p[0]), sy(p[1])
            svg += f'<circle cx="{x}" cy="{y}" r="4" fill="#58a6ff" stroke="#0d1117" stroke-width="2">'
            svg += f'<title>Record #{p[0]}: {p[1]:.2f}min - {esc(p[2])}</title></circle>'

        # X axis label
        svg += f'<text x="{w/2}" y="{h-5}" fill="#8b949e" font-size="12" text-anchor="middle">Record Number</text>'
        svg += f'<text x="15" y="{h/2}" fill="#8b949e" font-size="12" text-anchor="middle" transform="rotate(-90,15,{h/2})">Time (minutes)</text>'
        svg += '</svg>'
        return svg

    t1_svg = make_chart_svg(t1_points, "Track 1: GPT-2 Small")
    t2_svg = make_chart_svg(t2_points, "Track 2: GPT-2 Medium")

    # Count records with PRs, code, etc.
    n_with_pr = sum(1 for d in all_data if d["pr_meta_list"])
    n_with_code = sum(1 for d in all_data if d["code_snapshots"])
    n_with_diff = sum(1 for d in all_data if d["code_files"])
    total_log_files = sum(len(d["log_metrics"]) for d in all_data)

    def render_card(d):
        rec = d["rec"]
        slug = make_slug(rec)
        badges = "".join(f'<span class="badge badge-{c}">{c}</span>' for c in d["categories"])
        pr_info = f'PR #{d["pr_meta_list"][0][0]}' if d["pr_meta_list"] else ""
        has_code = "Code" if d["code_snapshots"] else ""
        has_diff = "Diff" if d["code_files"] else ""
        meta_parts = [f'<span>{esc(rec["date"])}</span>']
        if rec["contributors"]:
            meta_parts.append(f'<span>{esc(rec["contributors"][:60])}</span>')
        if pr_info:
            meta_parts.append(f'<span>{pr_info}</span>')
        if has_code:
            meta_parts.append(f'<span style="color:var(--green)">Has Code Snapshot</span>')
        if has_diff:
            meta_parts.append(f'<span style="color:var(--yellow)">Has Code Diff</span>')

        return f'''
        <a href="records/{slug}.html" style="text-decoration:none;color:inherit">
        <div class="record-card" data-categories="{' '.join(d['categories'])}" data-track="{rec['track']}">
            <div class="record-header">
                <span class="record-num">#{rec['number']}</span>
                <span class="record-title">{esc(rec['description'])}</span>
                <span class="record-time">{time_display(rec)}</span>
            </div>
            <div class="record-badges">{badges}</div>
            <div class="record-meta">{"".join(meta_parts)}</div>
        </div>
        </a>'''

    t1_cards = "\n".join(render_card(d) for d in track1)
    t2_cards = "\n".join(render_card(d) for d in track2)

    cat_filters = "".join(
        f'<button class="filter-btn" data-cat="{cat}" onclick="toggleFilter(this)">{cat} ({cnt})</button>'
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoGPT Speedrun Study Guide - 77 Records, Every Code Change</title>
<style>{CSS}</style>
</head>
<body>

<div class="hero">
    <h1>NanoGPT Speedrun <span>Study Guide</span></h1>
    <p class="subtitle">
        From 45 minutes to 1.4 minutes: every optimization, every code diff, every technique.
        A comprehensive study of {len(all_data)} world records across 2 tracks.
    </p>
</div>

<div class="tabs">
    <button class="tab active" onclick="showTab('overview')">Overview</button>
    <button class="tab" onclick="showTab('track1')">Track 1: GPT-2 Small ({len(track1)})</button>
    <button class="tab" onclick="showTab('track2')">Track 2: GPT-2 Medium ({len(track2)})</button>
    <button class="tab" onclick="showTab('chart')">Speed Chart</button>
</div>

<div class="container">

<!-- Overview tab -->
<div id="tab-overview" class="tab-content active">
    <div class="stats">
        <div class="stat-card">
            <div class="number">{len(all_data)}</div>
            <div class="label">Total Records</div>
        </div>
        <div class="stat-card">
            <div class="number">{n_with_pr}</div>
            <div class="label">Records with PRs</div>
        </div>
        <div class="stat-card">
            <div class="number">{n_with_diff}</div>
            <div class="label">Records with Code Diffs</div>
        </div>
        <div class="stat-card">
            <div class="number">{n_with_code}</div>
            <div class="label">Full Code Snapshots</div>
        </div>
        <div class="stat-card">
            <div class="number">{total_log_files}</div>
            <div class="label">Training Logs</div>
        </div>
        <div class="stat-card">
            <div class="number">31x</div>
            <div class="label">Total Speedup (Track 1)</div>
        </div>
    </div>

    <h2 style="margin: 1.5rem 0 0.5rem">Techniques by Category</h2>
    <div class="filters">
        <button class="filter-btn active" data-cat="all" onclick="toggleFilter(this)">all ({len(all_data)})</button>
        {cat_filters}
    </div>

    <div class="study-note">
        <h4>How to Use This Guide</h4>
        <ul>
            <li>Each record card links to a <strong>detailed study page</strong> with code diffs, PR descriptions, training metrics, and analysis.</li>
            <li>Use the <strong>category filters</strong> to focus on specific types of changes (architecture, optimizer, systems, etc.).</li>
            <li>The <strong>Speed Chart</strong> tab shows the progression of training speed over time.</li>
            <li>Code diffs show exactly what changed in <code>train_gpt.py</code> for each record.</li>
            <li>Full code snapshots (embedded in log files) let you see the complete training script at that point in time.</li>
        </ul>
    </div>

    <h2 style="margin: 1.5rem 0 0.5rem">Track 1: GPT-2 Small (3.28 loss target)</h2>
    <div class="timeline" id="timeline-all-t1">
        {t1_cards}
    </div>

    <h2 style="margin: 1.5rem 0 0.5rem">Track 2: GPT-2 Medium (2.92 loss target)</h2>
    <div class="timeline" id="timeline-all-t2">
        {t2_cards}
    </div>
</div>

<!-- Track 1 tab -->
<div id="tab-track1" class="tab-content">
    <h2 style="margin: 1rem 0">Track 1: GPT-2 Small — 45 min → 1.435 min</h2>
    <div class="filters">
        <button class="filter-btn active" data-cat="all" onclick="toggleFilter(this)">all</button>
        {cat_filters}
    </div>
    <div class="timeline">
        {t1_cards}
    </div>
</div>

<!-- Track 2 tab -->
<div id="tab-track2" class="tab-content">
    <h2 style="margin: 1rem 0">Track 2: GPT-2 Medium — 5.8 hours → 17.35 min</h2>
    <div class="timeline">
        {t2_cards}
    </div>
</div>

<!-- Chart tab -->
<div id="tab-chart" class="tab-content">
    <h2 style="margin: 1rem 0">Speed Progression</h2>
    <h3 style="color: var(--muted); margin: 0.5rem 0">Track 1: GPT-2 Small</h3>
    <div class="chart-container">{t1_svg}</div>
    <h3 style="color: var(--muted); margin: 1.5rem 0 0.5rem">Track 2: GPT-2 Medium</h3>
    <div class="chart-container">{t2_svg}</div>
</div>

</div>

<script>
function showTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
}}

function toggleFilter(btn) {{
    const cat = btn.dataset.cat;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.record-card').forEach(card => {{
        if (cat === 'all') {{
            card.parentElement.style.display = '';
        }} else {{
            const cats = card.dataset.categories.split(' ');
            card.parentElement.style.display = cats.includes(cat) ? '' : 'none';
        }}
    }});
}}
</script>
</body>
</html>"""


# ── Generate individual record page ─────────────────────────────────────────

def simple_markdown_to_html(md):
    """Very basic markdown to HTML conversion."""
    if not md:
        return ""
    # Escape HTML first
    text = esc(md)
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Code blocks
    text = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', text, flags=re.DOTALL)
    # Links - unescape the already-escaped chars for href
    def fix_link(m):
        link_text = m.group(1)
        url = m.group(2).replace('&amp;', '&').replace('&#x27;', "'")
        return f'<a href="{url}" target="_blank">{link_text}</a>'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', fix_link, text)
    # Images
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1"/>', text)
    # Line breaks to paragraphs (simple)
    text = re.sub(r'\n\n+', '</p><p>', text)
    # Lists
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>(\s*<li>.*?</li>)*)', r'<ul>\1</ul>', text, flags=re.DOTALL)
    return f'<p>{text}</p>'


def generate_record_page(data, slug, all_data):
    rec = data["rec"]
    cats = data["categories"]
    badges = "".join(f'<span class="badge badge-{c}">{c}</span>' for c in cats)

    analysis = get_analysis(rec["number"], rec["track"])
    # Auto-generate analysis from categories if no manual entry
    if not analysis:
        cats = data["categories"]
        analysis = {
            "what": rec["description"],
            "why": f"This record contributes to the speedrun through changes in: {', '.join(cats)}.",
            "key_concepts": [rec["description"]],
            "change_type": " + ".join(cats),
        }

    # Navigation
    same_track = [d for d in all_data if d["rec"]["track"] == rec["track"]]
    idx = next((i for i, d in enumerate(same_track) if d["rec"]["row_idx"] == rec["row_idx"]), -1)
    prev_slug = None
    next_slug = None
    if idx > 0:
        prev_slug = make_slug(same_track[idx-1]["rec"])
    if idx < len(same_track) - 1:
        next_slug = make_slug(same_track[idx+1]["rec"])

    # ── Sections ────────────────────────────────────────────────────────

    # 1. Study Notes / Analysis
    analysis_html = ""
    if analysis:
        concepts = "".join(f"<li>{esc(c)}</li>" for c in analysis.get("key_concepts", []))
        analysis_html = f'''
        <div class="section">
            <div class="section-header">Study Notes</div>
            <div class="section-body">
                <div class="study-note">
                    <h4>What Changed</h4>
                    <p>{esc(analysis["what"])}</p>
                </div>
                <div class="study-note">
                    <h4>Why It Works</h4>
                    <p>{esc(analysis["why"])}</p>
                </div>
                <div class="study-note">
                    <h4>Key Concepts to Understand</h4>
                    <ul>{concepts}</ul>
                </div>
                <div class="study-note">
                    <h4>Change Type</h4>
                    <p>{esc(analysis["change_type"])}</p>
                </div>
            </div>
        </div>'''

    # 2. PR Description & Discussion
    pr_sections = ""
    for pr_num, meta, files in data["pr_meta_list"]:
        title = meta.get("title", f"PR #{pr_num}")
        body = meta.get("body", "")
        user = meta.get("user", "")
        adds = meta.get("additions", 0)
        dels = meta.get("deletions", 0)
        changed = meta.get("changed_files", 0)

        body_html = simple_markdown_to_html(body) if body else '<p class="muted">No description provided</p>'

        pr_sections += f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="toggle-icon open">▶</span> PR #{pr_num}: {esc(title)}</span>
                <span class="diff-stats">
                    <span class="add">+{adds}</span> / <span class="del">-{dels}</span> in {changed} files
                    — by @{esc(user)}
                </span>
            </div>
            <div class="collapsible-content open">
                <div class="pr-body">{body_html}</div>
            </div>
        </div>'''

    # 3. Code Diffs
    diff_sections = ""
    for pr_num, meta, files in data["pr_meta_list"]:
        # Separate code files from record log files
        code_diffs = []
        record_files = []
        for f in files:
            fn = f.get("filename", "")
            patch = f.get("patch")
            if fn.endswith(".py") and "record" not in fn.lower() and patch:
                code_diffs.append(f)
            elif fn in ["requirements.txt", ".gitignore", "run.sh", "Dockerfile"] and patch:
                code_diffs.append(f)
            elif patch and not fn.startswith("records/"):
                code_diffs.append(f)
            else:
                record_files.append(f)

        if code_diffs:
            diff_html = ""
            for cf in code_diffs:
                fn = cf["filename"]
                adds = cf.get("additions", 0)
                dels = cf.get("deletions", 0)
                patch = cf.get("patch", "")
                status = cf.get("status", "modified")
                status_label = {"modified": "M", "added": "A", "removed": "D"}.get(status, status[0].upper())

                rendered_diff = render_diff_html(patch) if patch else '<p class="muted">Binary or too large</p>'

                diff_html += f'''
                <div class="diff-viewer">
                    <div class="diff-file-header">
                        <span class="diff-file-name">[{status_label}] {esc(fn)}</span>
                        <span class="diff-stats"><span class="add">+{adds}</span> <span class="del">-{dels}</span></span>
                    </div>
                    <div class="diff-content">{rendered_diff}</div>
                </div>'''

            diff_sections += f'''
            <div class="section">
                <div class="section-header" onclick="toggleSection(this)">
                    <span><span class="toggle-icon open">▶</span> Code Changes (PR #{pr_num})</span>
                    <span>{len(code_diffs)} file(s) modified</span>
                </div>
                <div class="collapsible-content open">
                    <div class="section-body">{diff_html}</div>
                </div>
            </div>'''

        if record_files:
            rec_html = "<ul>"
            for rf in record_files:
                fn = rf["filename"]
                adds = rf.get("additions", 0)
                rec_html += f'<li style="color:var(--muted);font-size:0.85rem">{esc(fn)} <span class="add" style="color:var(--green)">+{adds}</span></li>'
            rec_html += "</ul>"
            diff_sections += f'''
            <div class="section">
                <div class="section-header" onclick="toggleSection(this)">
                    <span><span class="toggle-icon">▶</span> Record/Log Files Added (PR #{pr_num})</span>
                    <span>{len(record_files)} file(s)</span>
                </div>
                <div class="collapsible-content">
                    <div class="section-body">{rec_html}</div>
                </div>
            </div>'''

    # 4. Training Metrics
    metrics_html = ""
    if data["log_metrics"]:
        rows = ""
        for lp, m in data["log_metrics"][:10]:  # limit to 10 logs
            final_loss = f'{m["val_losses"][-1]:.4f}' if m["val_losses"] else "—"
            train_time = f'{m["train_time_ms"]/1000:.1f}s' if m["train_time_ms"] else "—"
            steps = str(m["total_steps"]) if m["total_steps"] else "—"
            step_avg = f'{m["step_avg_ms"]:.2f}ms' if m["step_avg_ms"] else "—"
            mem = f'{m["peak_memory"]} MiB' if m["peak_memory"] else "—"
            has_code_tag = ' <span style="color:var(--green)">[has code]</span>' if m["has_code"] else ""
            short_name = Path(lp).name[:40]

            rows += f'''<tr>
                <td>{esc(short_name)}{has_code_tag}</td>
                <td>{final_loss}</td>
                <td>{train_time}</td>
                <td>{steps}</td>
                <td>{step_avg}</td>
                <td>{mem}</td>
            </tr>'''

        extra_note = ""
        total_logs = len(data["log_metrics"])
        if total_logs > 10:
            extra_note = f'<p class="muted" style="padding:1rem">Showing 10 of {total_logs} log files</p>'

        metrics_html = f'''
        <div class="section">
            <div class="section-header" onclick="toggleSection(this)">
                <span><span class="toggle-icon open">▶</span> Training Metrics ({total_logs} runs)</span>
            </div>
            <div class="collapsible-content open">
                <div class="section-body">
                    <table class="metrics-table">
                        <tr><th>Log File</th><th>Val Loss</th><th>Train Time</th><th>Steps</th><th>Step Avg</th><th>Peak Memory</th></tr>
                        {rows}
                    </table>
                    {extra_note}
                </div>
            </div>
        </div>'''

    # 5. Code Snapshots
    code_html = ""
    if data["code_snapshots"]:
        for lp, code in data["code_snapshots"][:2]:  # limit to 2 snapshots
            lines = code.split('\n')
            # Render with line numbers
            numbered = "".join(f'<span class="line">{esc(l)}</span>' for l in lines[:500])
            truncated_note = f'<p class="muted" style="padding:0.5rem 1rem">Showing first 500 of {len(lines)} lines</p>' if len(lines) > 500 else ""

            code_html += f'''
            <div class="section">
                <div class="section-header" onclick="toggleSection(this)">
                    <span><span class="toggle-icon">▶</span> Full Code Snapshot: {esc(Path(lp).name[:50])}</span>
                    <span>{len(lines)} lines</span>
                </div>
                <div class="collapsible-content">
                    <div class="section-body">
                        {truncated_note}
                        <div class="code-viewer"><pre>{numbered}</pre></div>
                    </div>
                </div>
            </div>'''

    # 6. No PR data fallback
    no_pr_note = ""
    if not data["pr_meta_list"] and not data["code_files"]:
        desc_link = ""
        if rec["description_url"]:
            desc_link = f'<p>External writeup: <a href="{esc(rec["description_url"])}" target="_blank">{esc(rec["description_url"])}</a></p>'
        no_pr_note = f'''
        <div class="section">
            <div class="section-header">Record Details</div>
            <div class="section-body">
                <div class="study-note">
                    <h4>Note</h4>
                    <p>This record predates the PR-based workflow. Code changes were committed directly
                    to the repository. The description and external links below provide context for what changed.</p>
                    {desc_link}
                </div>
            </div>
        </div>'''

    track_label = "Track 1: GPT-2 Small" if rec["track"] == "track_1_short" else "Track 2: GPT-2 Medium"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Record #{rec['number']} — {esc(rec['description'])} | NanoGPT Study Guide</title>
<style>{CSS}</style>
</head>
<body>

<div class="detail-hero">
    <div class="container">
        <div class="breadcrumb">
            <a href="../index.html">Study Guide</a> → {track_label} → Record #{rec['number']}
        </div>
        <div class="detail-title">#{rec['number']}: {esc(rec['description'])}</div>
        <div class="detail-time">{time_display(rec)}</div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center">
            {badges}
            <span style="color:var(--muted);margin-left:1rem">{esc(rec['date'])}</span>
            <span style="color:var(--muted)">by {esc(rec['contributors'])}</span>
        </div>
    </div>
</div>

<div class="container">
    {analysis_html}
    {no_pr_note}
    {pr_sections}
    {diff_sections}
    {metrics_html}
    {code_html}

    <div class="nav-buttons">
        {'<a href="'+prev_slug+'.html" class="nav-btn">← Previous Record</a>' if prev_slug else '<span></span>'}
        <a href="../index.html" class="nav-btn">Back to Index</a>
        {'<a href="'+next_slug+'.html" class="nav-btn">Next Record →</a>' if next_slug else '<span></span>'}
    </div>
</div>

<script>
function toggleSection(header) {{
    const content = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');
    content.classList.toggle('open');
    if (icon) icon.classList.toggle('open');
}}
</script>
</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_site()
