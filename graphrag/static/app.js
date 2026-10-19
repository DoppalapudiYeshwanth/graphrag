"use strict";
/* Graph RAG web app. All server text is inserted with textContent, never innerHTML. */

const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "How is homomorphic encryption used to protect biometric data?",
  "What approaches reduce energy use in wireless networks?",
  "How do biometric authentication schemes use smart cards?",
];

let busyUploading = false;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "Please check your input.";
    } catch (_) { /* keep the default message */ }
    throw new Error(detail);
  }
  return response;
}

function showError(message) {
  const box = $("error");
  box.textContent = message || "";
  box.hidden = !message;
}

/* ── workspace ─────────────────────────────────────────────────────────── */
async function loadWorkspace() {
  try {
    const data = await (await api("/api/workspace")).json();
    renderDocuments(data.documents);
    $("lim-chars").textContent = data.limits.max_chars.toLocaleString();
    $("ttl").textContent = data.limits.ttl_days;
    $("capacity").textContent =
      `Shared capacity left today: ${data.capacity.questions_left} questions, ${data.capacity.chunks_left} text chunks`;
    $("dropzone").classList.toggle("busy", data.documents.length >= data.limits.max_documents);
  } catch (err) {
    showError(err.message);
  }
}

function renderDocuments(documents) {
  const list = $("doc-list");
  list.replaceChildren();
  for (const doc of documents) {
    const remove = el("button", { class: "icon-btn", type: "button", "aria-label": `Delete ${doc.title}`, text: "Delete" });
    remove.addEventListener("click", () => deleteDocument(doc.doc_id));
    list.append(el("li", {},
      el("span", { class: "name", text: doc.title, title: doc.title }),
      el("span", { class: "meta", text: `${doc.chunks} part${doc.chunks === 1 ? "" : "s"}` }),
      remove));
  }
  $("docs-empty").hidden = documents.length > 0;
  $("delete-all").hidden = documents.length === 0;
}

async function deleteDocument(docId) {
  try {
    await api(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    await loadWorkspace();
  } catch (err) { showError(err.message); }
}

$("delete-all").addEventListener("click", async () => {
  if (!confirm("Delete all of your uploaded documents?")) return;
  try {
    await api("/api/workspace", { method: "DELETE" });
    await loadWorkspace();
  } catch (err) { showError(err.message); }
});

/* ── upload with streamed progress ─────────────────────────────────────── */
function setStatus(text, fraction, failed = false) {
  const box = $("upload-status");
  box.hidden = false;
  box.classList.toggle("failed", failed);
  $("status-text").textContent = text;
  $("bar-fill").style.width = `${Math.round((fraction || 0) * 100)}%`;
}

async function uploadFile(file) {
  if (busyUploading || !file) return;
  busyUploading = true;
  showError("");
  $("dropzone").classList.add("busy");
  setStatus(`Reading ${file.name}…`, 0.02);

  const form = new FormData();
  form.append("file", file);
  try {
    const response = await api("/api/upload", { method: "POST", body: form });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) handleUploadEvent(JSON.parse(line));
      }
    }
  } catch (err) {
    setStatus(err.message, 0, true);
  } finally {
    busyUploading = false;
    $("file").value = "";
    await loadWorkspace();
  }
}

function handleUploadEvent(event) {
  if (event.event === "start") {
    setStatus(`Analysing ${event.title} (${event.chunks} parts)…`, 0.05);
  } else if (event.event === "progress") {
    setStatus(`Analysing part ${event.done} of ${event.total}…`, 0.05 + 0.9 * (event.done / event.total));
  } else if (event.event === "done") {
    const note = event.failed_chunks ? ` (${event.failed_chunks} parts could not be analysed)` : "";
    setStatus(`Ready: ${event.entities} concepts and ${event.relationships} relationships found${note}.`, 1);
  } else if (event.event === "error") {
    setStatus(event.detail, 0, true);
  }
}

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("file").click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
$("file").addEventListener("change", (e) => uploadFile(e.target.files[0]));
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("over"); });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove("over"); });
}
dropzone.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));

/* ── answers ───────────────────────────────────────────────────────────── */
/* Minimal, safe formatting of the model's text: paragraphs, lists, **bold**, [chunk-id] citations. */
function appendInline(parent, text, sourceIds, onCite) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[[A-Za-z0-9-]+::\d+\])/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      parent.append(el("strong", { text: part.slice(2, -2) }));
    } else if (/^\[[A-Za-z0-9-]+::\d+\]$/.test(part)) {
      const id = part.slice(1, -1);
      const index = sourceIds.indexOf(id);
      if (index >= 0) {
        const chip = el("button", { class: "cite", type: "button", title: `Show source ${id}`, text: String(index + 1) });
        chip.addEventListener("click", () => onCite(id));
        parent.append(chip);
      }
    } else {
      parent.append(document.createTextNode(part));
    }
  }
}

function renderAnswer(text, sourceIds, onCite) {
  const box = el("div", { class: "answer" });
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const item = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      const ordered = /^\d/.test(line);
      if (!list || list.tagName !== (ordered ? "OL" : "UL")) {
        list = el(ordered ? "ol" : "ul");
        box.append(list);
      }
      const li = el("li");
      appendInline(li, item[1], sourceIds, onCite);
      list.append(li);
    } else if (line) {
      list = null;
      const p = el("p");
      appendInline(p, line.replace(/^#{1,6}\s+/, ""), sourceIds, onCite);
      box.append(p);
    }
  }
  return box;
}

function renderResult(question, data) {
  const sourceIds = data.sources.map((s) => s.chunk_id);
  const sourceNodes = new Map();

  const sources = el("details", {}, el("summary", { text: `Sources (${data.sources.length})` }));
  data.sources.forEach((s, i) => {
    const node = el("div", { class: "source" },
      el("div", { class: "id" },
        `${i + 1}. ${s.chunk_id}`,
        el("span", { class: `badge${s.via === "graph" ? " graph" : ""}`, text: s.via === "graph" ? "found via graph" : "similar" })),
      el("p", { text: s.text }));
    sourceNodes.set(s.chunk_id, node);
    sources.append(node);
  });

  const onCite = (id) => {
    sources.open = true;
    const node = sourceNodes.get(id);
    sourceNodes.forEach((n) => n.classList.remove("hl"));
    node.classList.add("hl");
    node.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const card = el("div", { class: "card" },
    el("h3", { text: question }),
    renderAnswer(data.answer, sourceIds, onCite),
    sources);

  if (data.facts.length) {
    const facts = el("ul", { class: "facts" });
    for (const f of data.facts) {
      facts.append(el("li", {},
        `${f.source} `, el("span", { class: "rel", text: f.relation }), ` ${f.target}`));
    }
    card.append(el("details", {}, el("summary", { text: `Knowledge-graph facts (${data.facts.length})` }), facts));
  }
  return card;
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) return;
  showError("");

  const button = $("ask-btn");
  button.disabled = true;
  button.textContent = "Thinking…";
  const pending = el("div", { class: "card thinking", text: "Searching and reading the sources…" });
  $("results").prepend(pending);

  try {
    const scope = document.querySelector('input[name="scope"]:checked').value;
    const response = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, scope }),
    });
    pending.replaceWith(renderResult(question, await response.json()));
  } catch (err) {
    pending.remove();
    showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
    loadWorkspace();
  }
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("ask-form").requestSubmit();
});

for (const text of EXAMPLES) {
  const chip = el("button", { class: "chip", type: "button", text });
  chip.addEventListener("click", () => { $("question").value = text; $("question").focus(); });
  $("examples").append(chip);
}

loadWorkspace();
