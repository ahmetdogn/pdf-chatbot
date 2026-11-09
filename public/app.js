const $ = (id) => document.getElementById(id);
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const MAX_REQUEST_BYTES = 4 * 1024 * 1024; // vercel limit ist 4.5 mb
const chatHistory = [];

// ---- embeddings im browser ----

let extractorPromise = null;

function getExtractor() {
  if (!extractorPromise) {
    // dynamisch laden, falls das cdn nicht geht läuft die app trotzdem (nur mit bm25)
    extractorPromise = import(TRANSFORMERS_URL)
      .then(({ pipeline }) => pipeline("feature-extraction", MODEL, { dtype: "q8" }))
      .catch((err) => {
        console.warn("embedding modell konnte nicht geladen werden", err);
        return null;
      });
  }
  return extractorPromise;
}

async function embed(texts, onProgress) {
  const extractor = await getExtractor();
  if (!extractor) return null;
  const result = [];
  for (let i = 0; i < texts.length; i += 16) {
    const out = await extractor(texts.slice(i, i + 16), { pooling: "mean", normalize: true });
    // auf 4 nachkommastellen runden, sonst wird der request zu groß
    for (const vec of out.tolist()) result.push(vec.map((x) => Math.round(x * 1e4) / 1e4));
    onProgress?.(Math.min(i + 16, texts.length), texts.length);
  }
  return result;
}

// ---- speicher (indexeddb) ----

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("pdf-chat", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("docs", { keyPath: "id" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function db(mode, fn) {
  const conn = await openDb();
  return new Promise((resolve, reject) => {
    const tx = conn.transaction("docs", mode);
    const req = fn(tx.objectStore("docs"));
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = () => reject(tx.error);
  });
}

const getDocs = () => db("readonly", (s) => s.getAll());
const saveDoc = (doc) => db("readwrite", (s) => s.put(doc));
const deleteDoc = (id) => db("readwrite", (s) => s.delete(id));

// ---- dokumente ----

async function renderDocuments() {
  const docs = await getDocs();
  const list = $("doc-list");
  list.innerHTML = "";
  if (!docs.length) {
    list.innerHTML = '<li class="none">Noch keine Dokumente</li>';
    return;
  }
  for (const doc of docs) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="meta">${doc.pages} S.</span>
      <button title="Entfernen" aria-label="Entfernen">✕</button>`;
    li.querySelector(".name").textContent = doc.filename;
    li.querySelector(".name").title = doc.filename;
    li.querySelector("button").onclick = async () => {
      if (!confirm(`„${doc.filename}“ entfernen?`)) return;
      await deleteDoc(doc.id);
      renderDocuments();
    };
    list.appendChild(li);
  }
}

function setStatus(text, isError = false) {
  $("upload-status").className = isError ? "status error" : "status";
  $("upload-status").textContent = text;
}

async function upload(files) {
  const existing = new Set((await getDocs()).map((d) => d.id));
  for (const file of files) {
    setStatus(`Lade ${file.name} hoch...`);
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/parse", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(`${file.name}: ${body.detail || "Upload fehlgeschlagen"}`, true);
      return;
    }
    if (existing.has(body.id)) {
      setStatus(`${file.name} ist schon vorhanden.`);
      continue;
    }

    setStatus("Lade Embedding-Modell (beim ersten Mal ca. 120 MB)...");
    const vectors = await embed(
      body.chunks.map((c) => c.text),
      (done, total) => setStatus(`${file.name}: Embeddings ${done}/${total}`)
    );
    body.chunks.forEach((c, i) => (c.embedding = vectors ? vectors[i] : null));

    await saveDoc({ id: body.id, filename: body.filename, pages: body.pages, chunks: body.chunks });
    setStatus(
      vectors
        ? `${file.name}: ${body.pages} Seiten, ${body.chunks.length} Abschnitte`
        : `${file.name} geladen, aber Modell nicht verfügbar - nur Stichwortsuche`
    );
  }
  renderDocuments();
}

const dropzone = $("dropzone");
$("file-input").onchange = (e) => upload(e.target.files);
["dragenter", "dragover"].forEach((t) =>
  dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((t) =>
  dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files));

// ---- antworten anzeigen ----

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// bisschen markdown + [n] als klickbare quelle
function renderAnswer(text, sources) {
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[(\d+)\]/g, (m, n) =>
        sources[n - 1] ? `<button class="cite" data-n="${n}">${n}</button>` : m
      );
  let html = "";
  let inList = false;
  for (const line of text.split("\n")) {
    const item = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    if (item) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(item[1])}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  if (inList) html += "</ul>";
  return html;
}

function showSource(src) {
  $("source-title").textContent = `[${src.n}] ${src.filename} · Seite ${src.page}`;
  $("source-text").textContent = src.text;
  $("source-dialog").showModal();
}

function addMessage(role, text = "") {
  $("empty")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "user") div.textContent = text;
  else div.innerHTML = '<div class="body typing"></div><div class="sources"></div>';
  $("messages").appendChild(div);
  div.scrollIntoView({ block: "end" });
  return div;
}

// ---- chat ----

async function sendChat(payload) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Demo-Password": localStorage.getItem("demo-password") || "",
      },
      body: payload,
    });
    if (res.status !== 401) return res;
    const pw = prompt(attempt ? "Falsches Passwort, nochmal:" : "Demo-Passwort:");
    if (pw === null) return res;
    localStorage.setItem("demo-password", pw);
  }
}

async function ask(question) {
  addMessage("user", question);
  const msg = addMessage("assistant");
  const body = msg.querySelector(".body");
  let sources = [];
  let answer = "";

  const fail = (text) => {
    body.classList.remove("typing");
    body.textContent = text;
    msg.classList.add("error");
  };

  msg.addEventListener("click", (e) => {
    const n = e.target.closest("[data-n]")?.dataset.n;
    if (n && sources[n - 1]) showSource(sources[n - 1]);
  });

  const docs = await getDocs();
  const chunks = docs.flatMap((d) => d.chunks);
  const hasVectors = chunks.length && chunks.every((c) => c.embedding);
  const queryEmbedding = hasVectors ? (await embed([question]))?.[0] : null;

  const payload = JSON.stringify({ question, query_embedding: queryEmbedding, chunks, history: chatHistory });
  if (payload.length > MAX_REQUEST_BYTES) {
    fail("Zu viele Dokumente auf einmal, bitte ein paar entfernen.");
    return;
  }

  const res = await sendChat(payload);
  if (!res.ok) {
    fail(res.status === 401 ? "Ohne Passwort geht's leider nicht." : "Anfrage fehlgeschlagen.");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();
    for (const raw of events) {
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] || "{}");
      if (event === "sources") {
        sources = data.sources;
        msg.querySelector(".sources").innerHTML = sources
          .map((s) => `<button data-n="${s.n}"><b>${s.n}</b>${escapeHtml(s.filename)}, S. ${s.page}</button>`)
          .join("");
      } else if (event === "delta") {
        answer += data.text;
        body.innerHTML = renderAnswer(answer, sources);
        msg.scrollIntoView({ block: "end" });
      } else if (event === "error") {
        msg.classList.add("error");
        answer += (answer ? "\n\n" : "") + data.message;
        body.innerHTML = renderAnswer(answer, sources);
      }
    }
  }
  body.classList.remove("typing");

  // nur quellen anzeigen die auch zitiert wurden
  const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => m[1]));
  if (cited.size) {
    msg.querySelectorAll(".sources button").forEach((b) => {
      if (!cited.has(b.dataset.n)) b.remove();
    });
  }
  chatHistory.push({ role: "user", content: question }, { role: "assistant", content: answer });
}

const form = $("chat-form");
const textarea = $("question");
form.onsubmit = async (e) => {
  e.preventDefault();
  const q = textarea.value.trim();
  if (!q) return;
  textarea.value = "";
  $("send").disabled = true;
  try { await ask(q); } finally { $("send").disabled = false; textarea.focus(); }
};
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
});

renderDocuments();
