"""
Nexus Pipeline — local web UI.

A single Flask page that runs the whole Nexus regulatory-document pipeline
end-to-end and shows every step as it happens: pick (or upload) a regulator
PDF, watch Steps 1-7 stream their live output and per-step artifact summaries,
then ask a natural-language question (Step 6) grounded in the indexed corpus.

It shells out to the existing CLI scripts (extract_document.py, ... ,
answer_query.py) via subprocess — the same commands you'd run by hand — so the
UI is a thin, honest wrapper: nothing is computed here that the scripts don't
already produce, and the artifacts written to Nexus/ are exactly the ones a
manual run produces.

Usage:
    python pipeline_web.py
    (then open http://127.0.0.1:5001)
"""

import json
import subprocess
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
NEXUS = ROOT / "Nexus"
INPUT_DIR = NEXUS / "input"

# The pipeline, in order. Steps 1-7 are per-document (keyed by `stem`); Step 6
# (answer_query) is corpus-wide and handled separately by /api/query.
#   - `arg`: "pdf" -> pass `--pdf <path>`, "stem" -> pass `--stem <stem>`
#   - `skip_llm`: whether the script accepts --skip-llm (honored when the
#     user ticks "skip LLM" for a fast, offline dry run)
#   - `artifact`: (relative path template, summary-fn key) read after the step
STEPS = [
    {"key": "extract",   "num": "1",  "title": "Extract",   "script": "extract_document.py",
     "arg": "pdf",  "skip_llm": True,  "blurb": "PDF → Markdown + metadata"},
    {"key": "structure", "num": "2",  "title": "Structure",  "script": "structure_document.py",
     "arg": "stem", "skip_llm": True,  "blurb": "Recover the Part → Section → Clause tree"},
    {"key": "interpret", "num": "3",  "title": "Interpret",  "script": "interpret_document.py",
     "arg": "stem", "skip_llm": True,  "blurb": "Tag each clause with regulatory meaning"},
    {"key": "clauses",   "num": "3.5", "title": "Clauses",   "script": "export_clauses.py",
     "arg": "stem", "skip_llm": False, "blurb": "Flatten the tree into a clause register"},
    {"key": "ontology",  "num": "4",  "title": "Ontology",   "script": "build_ontology.py",
     "arg": "stem", "skip_llm": False, "blurb": "Map clauses to canonical concepts"},
    {"key": "graph",     "num": "5A", "title": "Graph",      "script": "build_graph.py",
     "arg": "stem", "skip_llm": False, "blurb": "Assemble the knowledge graph"},
    {"key": "index",     "num": "5B", "title": "Index",      "script": "build_index.py",
     "arg": "stem", "skip_llm": False, "blurb": "Build the semantic-retrieval index"},
]


# --- Artifact summaries -----------------------------------------------------
# Each returns a small dict rendered as key/value chips under the step, plus a
# path to the full JSON artifact the user can expand. Failures are swallowed:
# a missing/oddly-shaped artifact just yields no summary rather than a 500.

def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_nodes(nodes):
    total = 0
    by_type: dict[str, int] = {}
    stack = list(nodes or [])
    while stack:
        n = stack.pop()
        total += 1
        t = n.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
        stack.extend(n.get("children") or [])
    return total, by_type


def summarize(step_key: str, stem: str) -> dict | None:
    if step_key == "extract":
        m = _load(NEXUS / "metadata" / f"{stem}_metadata.json")
        if not m:
            return None
        return {"artifact": f"metadata/{stem}_metadata.json", "fields": {
            "Title": m.get("title") or "—",
            "Regulator": m.get("regulator") or "—",
            "Pages": m.get("total_pages"),
            "Sections": m.get("total_sections"),
            "Words": m.get("total_words"),
            "Status": m.get("extraction_status"),
        }}
    if step_key == "structure":
        s = _load(NEXUS / "structure" / f"{stem}_structure.json")
        if not s:
            return None
        total, by_type = _count_nodes(s.get("structure"))
        return {"artifact": f"structure/{stem}_structure.json", "fields": {
            "Total nodes": total,
            **{t.capitalize(): c for t, c in sorted(by_type.items())},
        }}
    if step_key == "interpret":
        s = _load(NEXUS / "semantics" / f"{stem}_semantics.json")
        if not s:
            return None
        total, _ = _count_nodes(s.get("structure"))
        return {"artifact": f"semantics/{stem}_semantics.json", "fields": {
            "Nodes tagged": total,
        }}
    if step_key == "clauses":
        c = _load(NEXUS / "clauses" / f"{stem}_clauses.json")
        if not c:
            return None
        return {"artifact": f"clauses/{stem}_clauses.json", "fields": {
            "Clauses": c.get("clause_count", len(c.get("clauses", []))),
        }}
    if step_key == "ontology":
        o = _load(NEXUS / "ontology" / f"{stem}_ontology.json")
        if not o:
            return None
        cov = o.get("coverage", {})
        return {"artifact": f"ontology/{stem}_ontology.json", "fields": {
            "Clauses": cov.get("total"),
            "Mapped": cov.get("mapped"),
            "Unmapped": cov.get("unmapped"),
            "Coverage": f"{cov.get('coverage_pct', 0)}%",
        }}
    if step_key == "graph":
        g = _load(NEXUS / "graph" / f"{stem}_graph.json")
        if not g:
            return None
        st = g.get("stats", {})
        return {"artifact": f"graph/{stem}_graph.json", "fields": {
            "Nodes": st.get("nodes"),
            "Edges": st.get("edges"),
            **{k: v for k, v in (st.get("nodes_by_type") or {}).items()},
        }}
    if step_key == "index":
        ix = _load(NEXUS / "index" / f"{stem}_index.json")
        if not ix:
            return None
        dense = ix.get("vector_type") == "dense"
        fields = {
            "Backend": ix.get("backend"),
            "Retrieval": "semantic (embeddings)" if dense else "lexical (TF-IDF)",
            "Clauses indexed": ix.get("clause_count"),
        }
        if dense:
            fields["Model"] = ix.get("embedding_model")
            fields["Dimensions"] = ix.get("dim")
        else:
            fields["Vocabulary"] = ix.get("vocabulary_size")
        return {"artifact": f"index/{stem}_index.json", "fields": fields}
    return None


# --- Run orchestration (Server-Sent Events) ---------------------------------

def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def run_pipeline(pdf_path: Path, stem: str, skip_llm: bool, index_backend: str):
    """Generator yielding SSE frames as each step runs."""
    yield _sse("start", {"stem": stem, "steps": [
        {"key": s["key"], "num": s["num"], "title": s["title"], "blurb": s["blurb"]} for s in STEPS
    ]})

    for step in STEPS:
        yield _sse("step_start", {"key": step["key"]})

        cmd = [sys.executable, str(ROOT / step["script"])]
        if step["arg"] == "pdf":
            cmd += ["--pdf", str(pdf_path)]
        else:
            cmd += ["--stem", stem]
        if skip_llm and step["skip_llm"]:
            cmd += ["--skip-llm"]
        # Step 5B: choose lexical (tfidf) vs neural (embedding) retrieval.
        if step["key"] == "index" and index_backend and index_backend != "tfidf":
            cmd += ["--backend", index_backend]

        yield _sse("log", {"key": step["key"], "line": "$ " + " ".join(
            (c if " " not in c else f'"{c}"') for c in cmd[1:])})

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    yield _sse("log", {"key": step["key"], "line": line})
            code = proc.wait()
        except Exception as e:
            yield _sse("step_done", {"key": step["key"], "ok": False, "error": str(e)})
            yield _sse("fatal", {"key": step["key"], "error": str(e)})
            return

        if code != 0:
            yield _sse("step_done", {"key": step["key"], "ok": False,
                                     "error": f"exited with code {code}"})
            yield _sse("fatal", {"key": step["key"], "error": f"Step failed (exit {code})"})
            return

        yield _sse("step_done", {"key": step["key"], "ok": True,
                                 "summary": summarize(step["key"], stem)})

    yield _sse("done", {"stem": stem})


@app.route("/api/run")
def api_run():
    pdf = (request.args.get("pdf") or "").strip()
    skip_llm = request.args.get("skip_llm") == "1"
    # embed=1 -> neural embeddings (semantic search); else tfidf lexical.
    index_backend = "embedding" if request.args.get("embed") == "1" else "tfidf"
    pdf_path = (INPUT_DIR / pdf) if not Path(pdf).is_absolute() else Path(pdf)
    if not pdf_path.exists():
        return jsonify({"error": f"PDF not found: {pdf}"}), 404
    stem = pdf_path.stem
    return Response(run_pipeline(pdf_path, stem, skip_llm, index_backend), mimetype="text/event-stream",
                   headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/documents")
def api_documents():
    docs = []
    if INPUT_DIR.exists():
        for p in sorted(INPUT_DIR.glob("*")):
            if p.suffix.lower() == ".pdf":
                stem = p.stem
                processed = (NEXUS / "index" / f"{stem}_index.json").exists()
                docs.append({"file": p.name, "stem": stem, "processed": processed})
    return jsonify({"documents": docs})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify({"error": "no file"}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "only PDF files are supported"}), 400
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(f.filename).name
    dest = INPUT_DIR / safe
    f.save(str(dest))
    return jsonify({"file": dest.name, "stem": dest.stem})


@app.route("/api/artifact")
def api_artifact():
    rel = (request.args.get("path") or "").strip()
    target = (NEXUS / rel).resolve()
    if NEXUS.resolve() not in target.parents and target != NEXUS.resolve():
        return jsonify({"error": "path outside Nexus"}), 400
    if not target.exists():
        return jsonify({"error": "not found"}), 404
    return Response(target.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    top_k = int(data.get("top_k") or 6)
    no_llm = bool(data.get("no_llm"))

    cmd = [sys.executable, str(ROOT / "answer_query.py"), "--query", query, "--top-k", str(top_k)]
    if data.get("stem"):
        cmd += ["--stem", str(data["stem"])]
    if no_llm:
        cmd += ["--no-llm"]

    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if proc.returncode != 0:
        return jsonify({"error": (proc.stdout or "") + (proc.stderr or "")}), 500

    # answer_query writes the run to Nexus/answers/<ts>.json; grab the newest.
    answers = sorted((NEXUS / "answers").glob("answer_*.json"))
    result = _load(answers[-1]) if answers else None
    if result is None:
        return jsonify({"error": "no answer produced", "log": proc.stdout}), 500
    return jsonify({"result": result, "log": proc.stdout})


PAGE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nexus Pipeline</title>
<style>
  :root {
    --bg:#0f1116; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
    --text:#e6e8ee; --muted:#9aa2b1; --accent:#5b9dff; --ok:#3ecf8e;
    --warn:#f0b429; --err:#ff6b6b; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,sans-serif;
    background:var(--bg); color:var(--text); }
  header { padding:18px 24px; border-bottom:1px solid var(--border);
    display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:19px; margin:0; font-weight:650; letter-spacing:.2px; }
  header .sub { color:var(--muted); font-size:13px; }
  .wrap { max-width:1080px; margin:0 auto; padding:22px 24px 60px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:18px; margin-bottom:20px; }
  .card h2 { font-size:14px; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:0 0 14px; font-weight:600; }
  .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  select, input[type=text] { background:var(--panel2); color:var(--text);
    border:1px solid var(--border); border-radius:8px; padding:9px 11px; font-size:14px; }
  select { min-width:280px; }
  input[type=text] { flex:1; min-width:240px; }
  button { background:var(--accent); color:#08122a; border:none; border-radius:8px;
    padding:9px 18px; font-size:14px; font-weight:600; cursor:pointer; }
  button.secondary { background:var(--panel2); color:var(--text); border:1px solid var(--border); }
  button:disabled { opacity:.45; cursor:default; }
  label.chk { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; cursor:pointer; }
  .hint { color:var(--muted); font-size:12.5px; margin-top:10px; }
  .steps { display:flex; flex-direction:column; gap:12px; }
  .step { border:1px solid var(--border); border-radius:10px; overflow:hidden;
    background:var(--panel2); opacity:.55; transition:opacity .2s; }
  .step.active, .step.done, .step.err { opacity:1; }
  .step-head { display:flex; align-items:center; gap:12px; padding:12px 14px; cursor:pointer; }
  .badge { width:34px; height:34px; border-radius:8px; background:var(--panel);
    border:1px solid var(--border); display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-size:12px; font-weight:600; flex-shrink:0; }
  .step.done .badge { border-color:var(--ok); color:var(--ok); }
  .step.err .badge { border-color:var(--err); color:var(--err); }
  .step.active .badge { border-color:var(--accent); color:var(--accent); }
  .step-title { font-weight:600; font-size:14.5px; }
  .step-blurb { color:var(--muted); font-size:12.5px; }
  .spacer { flex:1; }
  .status { font-size:12px; font-family:var(--mono); color:var(--muted); }
  .step.active .status { color:var(--accent); }
  .step.done .status { color:var(--ok); }
  .step.err .status { color:var(--err); }
  .spin { display:inline-block; width:12px; height:12px; border:2px solid var(--border);
    border-top-color:var(--accent); border-radius:50%; animation:spin .7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .step-body { display:none; border-top:1px solid var(--border); padding:12px 14px; }
  .step.open .step-body { display:block; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }
  .chip { background:var(--panel); border:1px solid var(--border); border-radius:20px;
    padding:4px 11px; font-size:12.5px; }
  .chip b { color:var(--accent); font-weight:600; }
  pre.log { background:#0b0d12; border:1px solid var(--border); border-radius:8px;
    padding:10px 12px; margin:0; font-family:var(--mono); font-size:12px; line-height:1.55;
    max-height:220px; overflow:auto; white-space:pre-wrap; word-break:break-word; color:#c7ccd6; }
  .artifact-link { font-size:12px; color:var(--accent); cursor:pointer; margin-top:8px;
    display:inline-block; text-decoration:underline; }
  pre.artifact { background:#0b0d12; border:1px solid var(--border); border-radius:8px;
    padding:10px 12px; margin-top:8px; font-family:var(--mono); font-size:11.5px;
    max-height:320px; overflow:auto; display:none; color:#a9d5c0; }
  .answer { line-height:1.6; }
  .answer .grade { font-family:var(--mono); font-size:12px; color:var(--muted); margin-bottom:10px; }
  .ev { border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin-bottom:8px;
    background:var(--panel2); font-size:13px; }
  .ev .meta { color:var(--muted); font-size:11.5px; font-family:var(--mono); margin-bottom:4px; }
  .ev .rev { color:var(--warn); }
  .err-box { color:var(--err); font-size:13px; white-space:pre-wrap; font-family:var(--mono); }
  .q-row { display:flex; gap:10px; flex-wrap:wrap; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Nexus Pipeline</h1>
  <span class="sub">Regulator PDF → structure → meaning → ontology → graph → index → answers</span>
</header>
<div class="wrap">

  <div class="card">
    <h2>1 &middot; Pick a document</h2>
    <div class="controls">
      <select id="docSelect"></select>
      <button class="secondary" id="uploadBtn">Upload PDF…</button>
      <input type="file" id="fileInput" accept="application/pdf" style="display:none">
      <label class="chk"><input type="checkbox" id="useEmbed" checked> neural embeddings (semantic search)</label>
      <label class="chk"><input type="checkbox" id="skipLlm"> skip LLM (fast dry run)</label>
      <div class="spacer"></div>
      <button id="runBtn">Run pipeline ▶</button>
    </div>
    <div class="hint" id="pickHint">Documents already processed are marked ✓. Re-running overwrites their artifacts.</div>
  </div>

  <div class="card" id="pipeCard" style="display:none">
    <h2>2 &middot; Pipeline</h2>
    <div class="steps" id="steps"></div>
  </div>

  <div class="card" id="queryCard">
    <h2>3 &middot; Ask the corpus (Step 6 &middot; answer_query)</h2>
    <div class="q-row">
      <input type="text" id="queryInput" placeholder="e.g. What must LFIs do when credit risk increases significantly?">
      <label class="chk"><input type="checkbox" id="noLlm"> evidence only</label>
      <button id="askBtn">Ask</button>
    </div>
    <div class="q-row" style="margin-top:8px; align-items:center">
      <span style="color:var(--muted); font-size:13px">Search in:</span>
      <select id="queryScope" style="min-width:320px"></select>
    </div>
    <div class="hint">Choose one document to keep the answer grounded in it, or “All indexed documents” to search the whole corpus.</div>
    <div id="answer" style="margin-top:16px"></div>
  </div>

</div>
<script>
const $ = s => document.querySelector(s);
const docSelect = $('#docSelect');

async function loadDocs() {
  const r = await fetch('/api/documents');
  const d = await r.json();
  // Remember the current selections so rebuilding the lists doesn't snap them
  // back to the first option (the reason the dropdown reverted to 155MD).
  const prevDoc = docSelect.value;
  docSelect.innerHTML = '';
  if (!d.documents.length) {
    docSelect.innerHTML = '<option value="">(no PDFs in Nexus/input — upload one)</option>';
    return;
  }
  const scope = $('#queryScope');
  const prevScope = scope.value;
  scope.innerHTML = '<option value="">All indexed documents</option>';
  for (const doc of d.documents) {
    const o = document.createElement('option');
    o.value = doc.file;
    o.dataset.stem = doc.stem;
    o.dataset.processed = doc.processed ? '1' : '';
    o.textContent = (doc.processed ? '✓ ' : '· ') + doc.file;
    docSelect.appendChild(o);
    // Only indexed (processed) documents are searchable in the query section.
    if (doc.processed) {
      const so = document.createElement('option');
      so.value = doc.stem;
      so.textContent = doc.stem;
      scope.appendChild(so);
    }
  }
  if (prevDoc) docSelect.value = prevDoc;    // restore prior document selection
  if (prevScope) scope.value = prevScope;
}
loadDocs();

$('#uploadBtn').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData(); fd.append('pdf', file);
  $('#pickHint').textContent = 'Uploading ' + file.name + '…';
  const r = await fetch('/api/upload', { method:'POST', body:fd });
  const d = await r.json();
  if (d.error) { $('#pickHint').textContent = 'Upload failed: ' + d.error; return; }
  await loadDocs();
  docSelect.value = d.file;
  $('#pickHint').textContent = 'Uploaded ' + d.file + '. Click Run pipeline.';
};

const stepsEl = $('#steps');
let stepIndex = {};

function renderSteps(steps) {
  stepsEl.innerHTML = '';
  stepIndex = {};
  for (const s of steps) {
    const el = document.createElement('div');
    el.className = 'step';
    el.innerHTML = `
      <div class="step-head">
        <div class="badge">${s.num}</div>
        <div>
          <div class="step-title">${s.title}</div>
          <div class="step-blurb">${s.blurb}</div>
        </div>
        <div class="spacer"></div>
        <div class="status" data-role="status">queued</div>
      </div>
      <div class="step-body">
        <div class="chips" data-role="chips"></div>
        <pre class="log" data-role="log"></pre>
        <span class="artifact-link" data-role="alink" style="display:none">view full artifact</span>
        <pre class="artifact" data-role="artifact"></pre>
      </div>`;
    el.querySelector('.step-head').onclick = () => el.classList.toggle('open');
    stepsEl.appendChild(el);
    stepIndex[s.key] = el;
  }
}

function setStatus(el, txt, spin) {
  const s = el.querySelector('[data-role=status]');
  s.innerHTML = (spin ? '<span class="spin"></span> ' : '') + txt;
}

$('#runBtn').onclick = () => {
  const pdf = docSelect.value;
  if (!pdf) { $('#pickHint').textContent = 'Pick or upload a PDF first.'; return; }
  const skip = $('#skipLlm').checked ? '1' : '0';
  const embed = $('#useEmbed').checked ? '1' : '0';
  $('#runBtn').disabled = true;
  $('#pipeCard').style.display = 'block';
  $('#pipeCard').scrollIntoView({ behavior:'smooth' });

  const es = new EventSource('/api/run?pdf=' + encodeURIComponent(pdf) + '&skip_llm=' + skip + '&embed=' + embed);

  es.addEventListener('start', e => {
    const d = JSON.parse(e.data);
    renderSteps(d.steps);
  });
  es.addEventListener('step_start', e => {
    const el = stepIndex[JSON.parse(e.data).key];
    el.className = 'step active open';
    setStatus(el, 'running', true);
  });
  es.addEventListener('log', e => {
    const d = JSON.parse(e.data);
    const log = stepIndex[d.key].querySelector('[data-role=log]');
    log.textContent += d.line + '\n';
    log.scrollTop = log.scrollHeight;
  });
  es.addEventListener('step_done', e => {
    const d = JSON.parse(e.data);
    const el = stepIndex[d.key];
    if (!d.ok) { el.className = 'step err open'; setStatus(el, d.error || 'failed', false); return; }
    el.className = 'step done';
    setStatus(el, 'done', false);
    if (d.summary) {
      const chips = el.querySelector('[data-role=chips]');
      chips.innerHTML = '';
      for (const [k, v] of Object.entries(d.summary.fields || {})) {
        if (v === null || v === undefined) continue;
        const c = document.createElement('span');
        c.className = 'chip';
        c.innerHTML = k + ': <b>' + v + '</b>';
        chips.appendChild(c);
      }
      if (d.summary.artifact) {
        const alink = el.querySelector('[data-role=alink]');
        const art = el.querySelector('[data-role=artifact]');
        alink.style.display = 'inline-block';
        alink.onclick = async () => {
          if (art.style.display === 'block') { art.style.display = 'none'; return; }
          art.textContent = 'loading…'; art.style.display = 'block';
          const r = await fetch('/api/artifact?path=' + encodeURIComponent(d.summary.artifact));
          const txt = await r.text();
          try { art.textContent = JSON.stringify(JSON.parse(txt), null, 2); }
          catch { art.textContent = txt; }
        };
      }
    }
  });
  es.addEventListener('fatal', e => { es.close(); $('#runBtn').disabled = false; });
  es.addEventListener('done', e => {
    es.close();
    $('#runBtn').disabled = false;
    loadDocs();
  });
  es.onerror = () => { es.close(); $('#runBtn').disabled = false; };
};

$('#askBtn').onclick = async () => {
  const query = $('#queryInput').value.trim();
  if (!query) return;
  const out = $('#answer');
  const stem = $('#queryScope').value;   // "" = all indexed documents
  out.innerHTML = '<span class="spin"></span> reasoning over ' + (stem ? escapeHtml(stem) : 'the whole corpus') + '…';
  $('#askBtn').disabled = true;
  try {
    const body = { query, no_llm: $('#noLlm').checked };
    if (stem) body.stem = stem;
    const r = await fetch('/api/query', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) { out.innerHTML = '<div class="err-box">' + escapeHtml(d.error) + '</div>'; return; }
    renderAnswer(d.result);
  } catch (err) {
    out.innerHTML = '<div class="err-box">' + escapeHtml(String(err)) + '</div>';
  } finally {
    $('#askBtn').disabled = false;
  }
};

function escapeHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderAnswer(res) {
  const out = $('#answer');
  let html = '';
  const conf = res.confidence || {};
  html += '<div class="answer">';
  html += '<div class="grade">confidence: ' + escapeHtml(conf.grade || conf.band || JSON.stringify(conf)) +
          ' &middot; corpus: ' + escapeHtml((res.corpus && (res.corpus.documents || res.corpus.document_count)) || '—') + '</div>';
  if (res.answer) html += '<p>' + escapeHtml(res.answer).replace(/\n/g,'<br>') + '</p>';
  else html += '<p><i>No LLM synthesis — structured evidence only.</i></p>';
  html += '<h3 style="font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px">Evidence</h3>';
  for (const ev of (res.evidence || [])) {
    html += '<div class="ev">';
    html += '<div class="meta">' + escapeHtml(ev.document_id || '') + ' &middot; clause ' + escapeHtml(ev.clause_no || ev.clause_id || '') +
            (ev.page ? ' &middot; p.' + ev.page : '') + ' &middot; score ' + (ev.score != null ? ev.score.toFixed(3) : '—') + '</div>';
    if (ev.title) html += '<b>' + escapeHtml(ev.title) + '</b><br>';
    html += escapeHtml(ev.concept || ev.canonical_label || '');
    if (ev.review_required) html += ' <span class="rev">⚠ review required</span>';
    html += '</div>';
  }
  if ((res.ungrounded_citations || []).length)
    html += '<div class="err-box">⚠ ungrounded citations: ' + escapeHtml(JSON.stringify(res.ungrounded_citations)) + '</div>';
  html += '</div>';
  out.innerHTML = html;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
