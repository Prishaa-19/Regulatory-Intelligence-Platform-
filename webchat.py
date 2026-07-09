"""Temporary local web UI for the Azure Grok chatbot.

Runs a small Flask server: serves a single chat page and proxies messages
to Azure server-side, so the API key never reaches the browser.

Usage:
    python webchat.py
    (then open http://127.0.0.1:5000)
"""

from flask import Flask, jsonify, render_template_string, request

from chatbot import DEFAULT_MODEL_KEY, DEFAULT_SYSTEM_PROMPT, MODELS, build_caller

app = Flask(__name__)

_config = MODELS[DEFAULT_MODEL_KEY]
_call = build_caller(_config["provider"])
_model_name = _config["model"]

conversation = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{{ model_name }} chat</title>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0; font-family: system-ui, sans-serif;
    background: #ffffff; color: #1a1a1a;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 12px 16px; border-bottom: 1px solid #e0e0e0;
    font-weight: 600;
  }
  #log {
    flex: 1; overflow-y: auto; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .msg { max-width: 70%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; }
  .msg p { margin: 0 0 8px; }
  .msg p:last-child { margin-bottom: 0; }
  .msg pre { background: #f0f0f0; border-radius: 8px; padding: 10px; overflow-x: auto; }
  .msg code { background: #f0f0f0; border-radius: 4px; padding: 1px 5px; font-size: 0.9em; }
  .msg pre code { background: none; padding: 0; }
  .msg ul, .msg ol { margin: 4px 0; padding-left: 22px; }
  .msg table { border-collapse: collapse; margin: 4px 0; max-width: 100%; display: block; overflow-x: auto; }
  .msg th, .msg td { border: 1px solid #d8d8d8; padding: 6px 10px; text-align: left; }
  .msg th { background: #eceef2; font-weight: 600; }
  .user table, .user th, .user td { border-color: rgba(255,255,255,0.35); }
  .user th { background: rgba(255,255,255,0.15); }
  .user { align-self: flex-end; background: #2f6fed; color: white; }
  .user code, .user pre { background: rgba(255,255,255,0.2); }
  .assistant { align-self: flex-start; background: #f5f5f7; border: 1px solid #e0e0e0; }
  .error { align-self: center; color: #d3232f; font-size: 0.9em; }
  form {
    display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e0e0e0;
  }
  input {
    flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid #d0d0d0;
    background: #ffffff; color: #1a1a1a; font-size: 1em;
  }
  button {
    padding: 10px 18px; border-radius: 8px; border: none;
    background: #2f6fed; color: white; font-size: 1em; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<header>{{ model_name }}</header>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Type a message..." autofocus>
  <button id="send">Send</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Minimal markdown -> HTML (headers, bold/italic, inline+block code, lists, links).
function renderMarkdown(src) {
  const blocks = [];
  let text = escapeHtml(src);

  text = text.replace(/```([\\s\\S]*?)```/g, (_, code) => {
    blocks.push('<pre><code>' + code.replace(/^\\n/, '') + '</code></pre>');
    return '\\u0000' + (blocks.length - 1) + '\\u0000';
  });

  const lines = text.split('\\n');
  let html = '';
  let listType = null;
  let paragraph = [];

  function flushParagraph() {
    if (paragraph.length) {
      html += '<p>' + paragraph.join('<br>') + '</p>';
      paragraph = [];
    }
  }
  function closeList() {
    if (listType) { html += '</' + listType + '>'; listType = null; }
  }
  function inline(s) {
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
    s = s.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
    s = s.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }
  function splitRow(line) {
    let cells = line.split('|');
    if (cells.length && cells[0].trim() === '') cells.shift();
    if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();
    return cells.map(c => c.trim());
  }
  function isTableSeparator(line) {
    return /^\\|?\\s*:?-{2,}:?\\s*(\\|\\s*:?-{2,}:?\\s*)*\\|?$/.test(line);
  }

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    const heading = line.match(/^(#{1,6})\\s+(.*)/);
    const ul = line.match(/^[-*]\\s+(.*)/);
    const ol = line.match(/^\\d+\\.\\s+(.*)/);
    const nextLine = lines[i + 1] !== undefined ? lines[i + 1].trim() : undefined;
    const isTableStart = line.includes('|') && nextLine !== undefined && isTableSeparator(nextLine);

    if (isTableStart) {
      flushParagraph(); closeList();
      const headerCells = splitRow(line);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().includes('|') && lines[i].trim() !== '') {
        bodyRows.push(splitRow(lines[i].trim()));
        i++;
      }
      html += '<table><thead><tr>' + headerCells.map(c => '<th>' + inline(c) + '</th>').join('') + '</tr></thead>';
      if (bodyRows.length) {
        html += '<tbody>' + bodyRows.map(r => '<tr>' + r.map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>').join('') + '</tbody>';
      }
      html += '</table>';
      continue;
    } else if (heading) {
      flushParagraph(); closeList();
      const level = heading[1].length;
      html += '<h' + level + '>' + inline(heading[2]) + '</h' + level + '>';
    } else if (ul) {
      flushParagraph();
      if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
      html += '<li>' + inline(ul[1]) + '</li>';
    } else if (ol) {
      flushParagraph();
      if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; }
      html += '<li>' + inline(ol[1]) + '</li>';
    } else if (line === '') {
      flushParagraph(); closeList();
    } else {
      closeList();
      paragraph.push(inline(line));
    }
    i++;
  }
  flushParagraph();
  closeList();

  html = html.replace(/\\u0000(\\d+)\\u0000/g, (_, i) => blocks[Number(i)]);
  return html;
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'assistant') {
    div.innerHTML = renderMarkdown(text);
  } else {
    div.textContent = text;
  }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMessage('user', text);
  sendBtn.disabled = true;
  const thinking = addMessage('assistant', 'Thinking...');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'request failed');
    thinking.innerHTML = renderMarkdown(data.reply);
  } catch (err) {
    thinking.remove();
    addMessage('error', 'Error: ' + err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, model_name=_model_name)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    conversation.append({"role": "user", "content": user_message})
    try:
        reply = _call(_model_name, conversation)
    except Exception as e:
        conversation.pop()
        return jsonify({"error": str(e)}), 500

    conversation.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
