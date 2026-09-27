"""Databricks App: a small RAG chat UI over the varnam-agent Gold vector index.

Self-contained (doesn't import rag_common) since Databricks Apps deploys from
a single source folder. Auth is via WorkspaceClient()'s default resolution,
which inside a deployed Databricks App picks up the app's own service
principal credentials automatically -- no personal token involved.
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request, Response
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

CATALOG = "workspace"
SCHEMA = "rag_demo"
VS_INDEX = f"{CATALOG}.{SCHEMA}.gold_chunks_index"
EMBED_ENDPOINT = "databricks-gte-large-en"
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TOP_K = 5
MAX_HISTORY_MESSAGES = 6  # 3 turns; bounds prompt growth in a long session

w = WorkspaceClient()


app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>varnam-agent RAG chat</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --border: #262b36; --text: #e6e8ec;
    --muted: #8b93a5; --accent: #6ea8fe; --user-bubble: #2b3446; --bot-bubble: #1c2029;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 2px;
  }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header p { margin: 0; font-size: 12px; color: var(--muted); }
  #messages {
    flex: 1; overflow-y: auto; padding: 20px; display: flex;
    flex-direction: column; gap: 16px; max-width: 760px; width: 100%;
    margin: 0 auto;
  }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: var(--user-bubble); }
  .msg.bot { align-self: flex-start; background: var(--bot-bubble); border: 1px solid var(--border); }
  .msg.bot.pending { color: var(--muted); }
  .sources { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px; }
  .sources summary { cursor: pointer; font-size: 12px; color: var(--muted); }
  .source-item { margin-top: 8px; padding: 8px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; font-size: 12px; }
  .source-item .meta { color: var(--accent); font-weight: 600; margin-bottom: 4px; }
  .source-item .snippet { color: var(--muted); }
  form {
    display: flex; gap: 8px; padding: 16px 20px; border-top: 1px solid var(--border);
    max-width: 760px; width: 100%; margin: 0 auto; box-sizing: border-box;
  }
  input {
    flex: 1; background: var(--panel); border: 1px solid var(--border); color: var(--text);
    padding: 10px 14px; border-radius: 8px; font-size: 14px; outline: none;
  }
  input:focus { border-color: var(--accent); }
  button {
    background: var(--accent); color: #0b0d11; border: none; padding: 10px 18px;
    border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
  <header>
    <h1>varnam-agent</h1>
    <p>RAG chat over the arXiv sample &mdash; retrieval + generation on Databricks</p>
  </header>
  <div id="messages"></div>
  <form id="form">
    <input id="input" type="text" placeholder="Ask about the ingested papers..." autocomplete="off" />
    <button type="submit" id="send">Ask</button>
  </form>
<script>
const messages = document.getElementById('messages');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
let history = [];

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function renderSources(container, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement('details');
  details.className = 'sources';
  const summary = document.createElement('summary');
  summary.textContent = `${sources.length} source${sources.length > 1 ? 's' : ''}`;
  details.appendChild(summary);
  sources.forEach((s, i) => {
    const item = document.createElement('div');
    item.className = 'source-item';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `[${i + 1}] ${s.source || 'unknown'} · score ${s.score?.toFixed(3) ?? 'n/a'}`;
    const snippet = document.createElement('div');
    snippet.className = 'snippet';
    snippet.textContent = (s.chunk_text || '').slice(0, 240) + ((s.chunk_text || '').length > 240 ? '…' : '');
    item.appendChild(meta);
    item.appendChild(snippet);
    details.appendChild(item);
  });
  container.appendChild(details);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  send.disabled = true;
  addMessage('user', query);
  const pending = addMessage('bot pending', 'Thinking…');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history }),
    });
    const data = await res.json();
    pending.classList.remove('pending');
    if (data.error) {
      pending.textContent = 'Error: ' + data.error;
    } else {
      pending.textContent = data.answer;
      renderSources(pending, data.sources);
      history.push({ role: 'user', content: query }, { role: 'assistant', content: data.answer });
      history = history.slice(-12); // keep last 6 turns client-side too
    }
  } catch (err) {
    pending.classList.remove('pending');
    pending.textContent = 'Request failed: ' + err;
  } finally {
    send.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    history = data.get("history") or []
    if not query:
        return jsonify({"error": "empty query"}), 400

    try:
        return _answer(query, history)
    except Exception as e:
        # Any backend failure (index mid-rebuild, endpoint timeout, ...) must
        # still come back as JSON -- Flask's default error page is HTML, and
        # the client's `await res.json()` would otherwise throw a confusing
        # "not valid JSON" error instead of surfacing anything useful.
        app.logger.exception("chat request failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502


def _answer(query: str, history: list) -> Response:
    embed_resp = w.serving_endpoints.query(name=EMBED_ENDPOINT, input=[query])
    query_vector = embed_resp.data[0].embedding

    results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        columns=["chunk_id", "doc_id", "chunk_text", "source_dataset"],
        query_vector=query_vector,
        num_results=TOP_K,
    )
    columns = [c.name for c in results.manifest.columns]
    hits = []
    for row in results.result.data_array or []:
        record = dict(zip(columns, row))
        hits.append({
            "chunk_text": record.get("chunk_text"),
            "source": record.get("source_dataset"),
            "doc_id": record.get("doc_id"),
            "score": row[-1],
        })

    if not hits:
        return jsonify({"answer": "No matching content found in the index.", "sources": []})

    context = "\n\n---\n\n".join(f"[{i + 1}] {h['chunk_text']}" for i, h in enumerate(hits))
    system_prompt = (
        "You are a research assistant. Answer the user's question using ONLY the "
        "numbered context snippets below, citing sources inline like [1] or [2]. "
        "If the context doesn't contain the answer, say so honestly instead of guessing. "
        "Earlier turns of this conversation may be included for follow-up context.\n\n"
        f"Context:\n{context}"
    )

    history_messages = []
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            history_messages.append(ChatMessage(
                role=ChatMessageRole.USER if role == "user" else ChatMessageRole.ASSISTANT,
                content=content,
            ))

    chat_resp = w.serving_endpoints.query(
        name=CHAT_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt)]
        + history_messages
        + [ChatMessage(role=ChatMessageRole.USER, content=query)],
        max_tokens=700,
    )
    answer = chat_resp.choices[0].message.content

    return jsonify({"answer": answer, "sources": hits})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    app.run(host="0.0.0.0", port=port)
