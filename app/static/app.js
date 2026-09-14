document.addEventListener("DOMContentLoaded", () => {
  // Dynamic API Base URL (Relative if served together on Render, or remote URL if frontend is deployed on Vercel/Netlify)
  const API_BASE = window.API_BASE || "";

  // Elements
  const messagesArea = document.getElementById("messagesArea");
  const welcomeCard = document.getElementById("welcomeCard");
  const chatForm = document.getElementById("chatForm");
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");

  // Admin Modal Elements
  const adminModal = document.getElementById("adminModal");
  const openAdminBtn = document.getElementById("openAdminBtn");
  const closeAdminBtn = document.getElementById("closeAdminBtn");
  const adminAuthSection = document.getElementById("adminAuthSection");
  const adminUploadSection = document.getElementById("adminUploadSection");
  const adminLoginForm = document.getElementById("adminLoginForm");
  const adminUsernameInput = document.getElementById("adminUsername");
  const adminPasswordInput = document.getElementById("adminPassword");
  const loginError = document.getElementById("loginError");
  const adminLogoutBtn = document.getElementById("adminLogoutBtn");

  // File Upload Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const selectedFileName = document.getElementById("selectedFileName");
  const uploadBtn = document.getElementById("uploadBtn");
  const pipelineProgress = document.getElementById("pipelineProgress");
  const uploadResult = document.getElementById("uploadResult");
  const adminDocsList = document.getElementById("adminDocsList");
  const refreshDocsBtn = document.getElementById("refreshDocsBtn");

  // Logs Modal Elements
  const logsModal = document.getElementById("logsModal");
  const openLogsBtn = document.getElementById("openLogsBtn");
  const closeLogsBtn = document.getElementById("closeLogsBtn");
  const logsContainer = document.getElementById("logsContainer");

  let selectedFile = null;

  // ==========================================================================
  // Quick Prompt Chips
  // ==========================================================================
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query) {
        userInput.value = query;
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // ==========================================================================
  // Chat Form Submission
  // ==========================================================================
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Hide welcome card once first message is sent
    if (welcomeCard) {
      welcomeCard.style.display = "none";
    }

    // Append User Message
    appendMessage(query, "user");
    userInput.value = "";
    sendBtn.disabled = true;

    // Append Loading State
    const loadingId = appendLoadingMessage();

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query })
      });

      if (!response.ok) {
        const errText = await response.text();
        let errMsg = `Server error (${response.status})`;
        try {
          const errData = JSON.parse(errText);
          if (typeof errData.detail === "string") {
            errMsg = errData.detail;
          } else if (errData.detail) {
            errMsg = JSON.stringify(errData.detail);
          } else if (errData.message) {
            errMsg = errData.message;
          }
        } catch (e) {
          if (errText && errText.trim().length > 0 && errText.length < 250) {
            errMsg = `${errText} (${response.status})`;
          }
        }
        throw new Error(errMsg);
      }

      const data = await response.json();
      removeLoadingMessage(loadingId);
      appendBotResponse(data);

    } catch (err) {
      removeLoadingMessage(loadingId);
      appendErrorMessage(`Failed to get response: ${err.message}`);
    } finally {
      sendBtn.disabled = false;
      userInput.focus();
    }
  });

  // ==========================================================================
  // Message Rendering
  // ==========================================================================
  function appendMessage(text, sender) {
    const row = document.createElement("div");
    row.className = `message-row ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    if (sender === "user") {
      row.appendChild(bubble);
    } else {
      const avatar = document.createElement("div");
      avatar.className = "avatar bot-avatar";
      avatar.textContent = "🤖";
      row.appendChild(avatar);
      row.appendChild(bubble);
    }

    messagesArea.appendChild(row);
    scrollToBottom();
  }

  function appendLoadingMessage() {
    const id = "loading-" + Date.now();
    const row = document.createElement("div");
    row.className = "message-row bot";
    row.id = id;

    const avatar = document.createElement("div");
    avatar.className = "avatar bot-avatar";
    avatar.textContent = "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;">
        <div class="spinner"></div>
        <span style="font-size:13px;color:#64748b;">Consulting HR Knowledge Base & LangGraph Agent...</span>
      </div>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesArea.appendChild(row);
    scrollToBottom();
    return id;
  }

  function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendErrorMessage(errorText) {
    const row = document.createElement("div");
    row.className = "message-row bot";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.style.borderColor = "#fecaca";
    bubble.style.backgroundColor = "#fef2f2";
    bubble.innerHTML = `<span style="color:#b91c1c;font-weight:600;">⚠️ ${escapeHtml(errorText)}</span>`;

    row.appendChild(bubble);
    messagesArea.appendChild(row);
    scrollToBottom();
  }

  function appendBotResponse(data) {
    const row = document.createElement("div");
    row.className = "message-row bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar bot-avatar";
    avatar.textContent = "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    // 1. Source Badge
    const badge = document.createElement("div");
    const srcType = data.source_type || "kb";
    badge.className = `source-badge ${srcType}`;
    if (srcType === "kb") {
      badge.innerHTML = `🏢 Company HR Policy Document`;
    } else if (srcType === "web") {
      badge.innerHTML = `🌐 External Web Search (Tavily)`;
    } else {
      badge.innerHTML = `ℹ️ General Guidance`;
    }
    bubble.appendChild(badge);

    // 2. Answer Body
    const contentDiv = document.createElement("div");
    contentDiv.className = "bot-content";
    contentDiv.innerHTML = formatMarkdown(data.answer);
    bubble.appendChild(contentDiv);

    // 3. Citations Section
    if (data.citations && data.citations.length > 0) {
      const citBox = document.createElement("div");
      citBox.className = "citations-box";

      const citTitle = document.createElement("div");
      citTitle.className = "citations-title";
      citTitle.textContent = "Verified Sources & Citations:";
      citBox.appendChild(citTitle);

      const citList = document.createElement("div");
      citList.className = "citation-list";

      const seen = new Set();
      data.citations.forEach((c) => {
        const key = c.source + (c.page_number ? `_p${c.page_number}` : "");
        if (!seen.has(key)) {
          seen.add(key);
          const item = document.createElement("div");
          item.className = "citation-item";

          if (c.source.startsWith("http://") || c.source.startsWith("https://") || c.source.includes("http")) {
            // Web citation with link
            const urlMatch = c.source.match(/https?:\/\/[^\s)]+/);
            const url = urlMatch ? urlMatch[0] : c.source;
            item.innerHTML = `🔗 <a href="${url}" target="_blank" rel="noopener noreferrer">${escapeHtml(c.source)}</a>`;
          } else {
            // Document citation with page
            const pageStr = c.page_number ? ` (Page ${c.page_number})` : "";
            item.innerHTML = `📄 <strong>${escapeHtml(c.source)}</strong>${pageStr}`;
          }
          citList.appendChild(item);
        }
      });

      citBox.appendChild(citList);
      bubble.appendChild(citBox);
    }

    // 4. Collapsible Decision Trace
    if (data.decision_trace && data.decision_trace.length > 0) {
      const traceCard = document.createElement("div");
      traceCard.className = "trace-card";

      const toggleBtn = document.createElement("button");
      toggleBtn.className = "trace-toggle-btn";
      toggleBtn.innerHTML = `
        <span>🔍 View Agent Decision Trace (${data.decision_trace.length} steps)</span>
        <span class="arrow">▼</span>
      `;

      const traceContent = document.createElement("div");
      traceContent.className = "trace-content";
      traceContent.innerHTML = data.decision_trace.map((step) => escapeHtml(step)).join("<br>");

      toggleBtn.addEventListener("click", () => {
        const isOpen = traceContent.classList.toggle("open");
        toggleBtn.querySelector(".arrow").textContent = isOpen ? "▲" : "▼";
      });

      traceCard.appendChild(toggleBtn);
      traceCard.appendChild(traceContent);
      bubble.appendChild(traceCard);
    }

    // 5. Feedback Buttons
    const feedbackRow = document.createElement("div");
    feedbackRow.className = "feedback-row";

    const upBtn = document.createElement("button");
    upBtn.className = "fb-btn";
    upBtn.innerHTML = "👍 Helpful";

    const downBtn = document.createElement("button");
    downBtn.className = "fb-btn";
    downBtn.innerHTML = "👎 Not accurate";

    const fbStatus = document.createElement("span");
    fbStatus.className = "fb-status";

    const sendFeedback = async (rating) => {
      upBtn.disabled = true;
      downBtn.disabled = true;
      try {
        await fetch(`${API_BASE}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: data.question,
            answer: data.answer,
            rating: rating
          })
        });
        fbStatus.textContent = "Thank you for the feedback!";
      } catch (e) {
        fbStatus.textContent = "Feedback recorded.";
      }
    };

    upBtn.addEventListener("click", () => {
      upBtn.classList.add("active");
      sendFeedback("up");
    });
    downBtn.addEventListener("click", () => {
      downBtn.classList.add("active");
      sendFeedback("down");
    });

    feedbackRow.appendChild(upBtn);
    feedbackRow.appendChild(downBtn);
    feedbackRow.appendChild(fbStatus);
    bubble.appendChild(feedbackRow);

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesArea.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesArea.scrollTop = messagesArea.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Markdown headers
    html = html.replace(/^### (.*$)/gim, "<h4 style='margin:10px 0 4px;font-size:14px;color:#1e293b;'>$1</h4>");
    html = html.replace(/^## (.*$)/gim, "<h3 style='margin:12px 0 6px;font-size:15px;color:#1e293b;'>$1</h3>");
    // Bullet items
    html = html.replace(/^\* (.*$)/gim, "<li style='margin-left:18px;'>$1</li>");
    html = html.replace(/^- (.*$)/gim, "<li style='margin-left:18px;'>$1</li>");
    // Line breaks
    html = html.replace(/\n\n/g, "<p style='margin-bottom:8px;'></p>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  // ==========================================================================
  // Admin Portal & Authentication
  // ==========================================================================
  function updateAdminUI() {
    const auth = sessionStorage.getItem("admin_auth");
    if (auth) {
      adminAuthSection.style.display = "none";
      adminUploadSection.style.display = "block";
      fetchAdminDocs();
    } else {
      adminAuthSection.style.display = "block";
      adminUploadSection.style.display = "none";
    }
  }

  openAdminBtn.addEventListener("click", () => {
    adminModal.classList.add("active");
    if (!sessionStorage.getItem("admin_auth")) {
      adminUsernameInput.value = "";
      adminPasswordInput.value = "";
    }
    updateAdminUI();
  });

  closeAdminBtn.addEventListener("click", () => {
    adminModal.classList.remove("active");
  });

  adminModal.addEventListener("click", (e) => {
    if (e.target === adminModal) {
      adminModal.classList.remove("active");
    }
  });

  adminLoginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const user = adminUsernameInput.value.trim();
    const pass = adminPasswordInput.value.trim();

    // Store credentials in session
    sessionStorage.setItem("admin_auth", JSON.stringify({ user, pass }));
    loginError.style.display = "none";
    adminUsernameInput.value = "";
    adminPasswordInput.value = "";
    updateAdminUI();
  });

  adminLogoutBtn.addEventListener("click", () => {
    sessionStorage.removeItem("admin_auth");
    adminUsernameInput.value = "";
    adminPasswordInput.value = "";
    updateAdminUI();
  });

  // ==========================================================================
  // File Upload & Automated Ingestion Pipeline
  // ==========================================================================
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    uploadBtn.disabled = false;
    uploadResult.style.display = "none";
  }

  uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    const authData = JSON.parse(sessionStorage.getItem("admin_auth") || "{}");
    if (!authData.user || !authData.pass) {
      alert("Please log in again.");
      sessionStorage.removeItem("admin_auth");
      updateAdminUI();
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("username", authData.user);
    formData.append("password", authData.pass);

    uploadBtn.disabled = true;
    pipelineProgress.style.display = "flex";
    uploadResult.style.display = "none";

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Upload failed (${response.status})`);
      }

      uploadResult.className = "upload-result-card";
      uploadResult.innerHTML = `
        <strong>✅ Ingestion Complete!</strong><br>
        Document: <em>${escapeHtml(data.filename)}</em><br>
        Chunks Extracted & Upserted: <strong>${data.chunks_indexed}</strong><br>
        Status: ${escapeHtml(data.message)}
      `;
      uploadResult.style.display = "block";

      // Reset selection
      selectedFile = null;
      fileInput.value = "";
      selectedFileName.textContent = "";
      fetchAdminDocs();

    } catch (err) {
      uploadResult.className = "upload-result-card";
      uploadResult.style.backgroundColor = "#fef2f2";
      uploadResult.style.borderColor = "#fecaca";
      uploadResult.style.color = "#991b1b";
      uploadResult.innerHTML = `<strong>❌ Error:</strong> ${escapeHtml(err.message)}`;
      uploadResult.style.display = "block";
    } finally {
      pipelineProgress.style.display = "none";
      uploadBtn.disabled = selectedFile === null;
    }
  });

  // ==========================================================================
  // Fetch Admin Docs
  // ==========================================================================
  async function fetchAdminDocs() {
    adminDocsList.innerHTML = "<span style='font-size:12px;color:#64748b;'>Loading indexed documents...</span>";
    try {
      const res = await fetch(`${API_BASE}/admin/docs`);
      const data = await res.json();

      if (!data.documents || data.documents.length === 0) {
        adminDocsList.innerHTML = "<span style='font-size:12px;color:#64748b;'>No documents uploaded yet.</span>";
        return;
      }

      adminDocsList.innerHTML = data.documents.map((d) => `
        <div class="doc-item">
          <div class="doc-name">📄 ${escapeHtml(d.filename)}</div>
          <div class="doc-meta">${(d.size_bytes / 1024).toFixed(1)} KB</div>
        </div>
      `).join("");

    } catch (e) {
      adminDocsList.innerHTML = `<span style='color:#dc2626;font-size:12px;'>Failed to load documents: ${e.message}</span>`;
    }
  }

  refreshDocsBtn.addEventListener("click", fetchAdminDocs);

  // ==========================================================================
  // Logs Modal
  // ==========================================================================
  openLogsBtn.addEventListener("click", async () => {
    logsModal.classList.add("active");
    logsContainer.innerHTML = "<span>Fetching latest traces...</span>";
    try {
      const res = await fetch(`${API_BASE}/logs?limit=10`);
      const data = await res.json();
      if (!data.logs || data.logs.length === 0) {
        logsContainer.innerHTML = "<span style='color:#64748b;'>No query logs recorded yet. Ask a question first!</span>";
        return;
      }

      logsContainer.innerHTML = data.logs.slice().reverse().map((log) => `
        <div class="log-entry">
          <div class="log-entry-header">
            <span>Query: "${escapeHtml(log.question)}"</span>
            <span>Source: ${escapeHtml(log.source_type.toUpperCase())}</span>
          </div>
          <ul class="log-steps">
            ${log.decision_trace.map((step) => `<li>→ ${escapeHtml(step)}</li>`).join("")}
          </ul>
        </div>
      `).join("");
    } catch (e) {
      logsContainer.innerHTML = `<span style='color:#ef4444;'>Failed to load logs: ${e.message}</span>`;
    }
  });

  closeLogsBtn.addEventListener("click", () => {
    logsModal.classList.remove("active");
  });

  logsModal.addEventListener("click", (e) => {
    if (e.target === logsModal) {
      logsModal.classList.remove("active");
    }
  });
});
