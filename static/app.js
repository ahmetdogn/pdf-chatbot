const $ = (id) => document.getElementById(id);
const chatHistory = [];

async function loadDocuments() {
  const res = await fetch("/api/documents");
  const docs = await res.json();
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
      await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
      loadDocuments();
    };
    list.appendChild(li);
  }
}

async function upload(files) {
  const status = $("upload-status");
  for (const file of files) {
    status.className = "status";
    status.textContent = `Lade ${file.name} hoch...`;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/documents", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      status.className = "status error";
      status.textContent = `${file.name}: ${body.detail || "Upload fehlgeschlagen"}`;
      return;
    }
    status.textContent = body.duplicate
      ? `${file.name} ist schon vorhanden.`
      : `${file.name}: ${body.pages} Seiten, ${body.chunks} Abschnitte`;
  }
  loadDocuments();
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

async function ask(question) {
  addMessage("user", question);
  const msg = addMessage("assistant");
  const body = msg.querySelector(".body");
  let sources = [];
  let answer = "";

  msg.addEventListener("click", (e) => {
    const n = e.target.closest("[data-n]")?.dataset.n;
    if (n && sources[n - 1]) showSource(sources[n - 1]);
  });

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: chatHistory }),
  });
  if (!res.ok) {
    body.classList.remove("typing");
    body.textContent = "Anfrage fehlgeschlagen.";
    msg.classList.add("error");
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

loadDocuments();
