// Frontend Logic for Medhas Memory Engine Web Dashboard

document.addEventListener("DOMContentLoaded", () => {
    let currentUserId = document.getElementById("global-user-id").value;
    let currentSessionId = "";

    // 1. Tab Navigation
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");

            if (targetId === "tab-letta") loadWorkingMemory();
            if (targetId === "tab-graph") loadKnowledgeGraph();
            if (targetId === "tab-procedural") loadProceduralPlaybooks();
        });
    });

    // 2. Initialize Session
    async function initSession() {
        currentUserId = document.getElementById("global-user-id").value;
        try {
            const res = await fetch(`/api/session/create?user_id=${encodeURIComponent(currentUserId)}`, { method: "POST" });
            const data = await res.json();
            currentSessionId = data.session_id;
            document.getElementById("global-session-id").innerText = currentSessionId.slice(0, 8) + "...";
        } catch (e) {
            console.error("Session creation error:", e);
        }
    }
    initSession();

    document.getElementById("btn-new-session").addEventListener("click", () => {
        initSession();
        document.getElementById("chat-messages-list").innerHTML = `
            <div class="chat-system-banner">
                ✨ New session initialized: ${currentSessionId}. Ready for fresh turns.
            </div>
        `;
    });

    // 3. Chat Turn Execution
    const chatInput = document.getElementById("chat-input-text");
    const sendBtn = document.getElementById("btn-send-message");
    const messagesList = document.getElementById("chat-messages-list");

    async function sendTurn() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Append user bubble
        const userBubble = document.createElement("div");
        userBubble.className = "chat-bubble user";
        userBubble.innerText = text;
        messagesList.appendChild(userBubble);

        chatInput.value = "";
        messagesList.scrollTop = messagesList.scrollHeight;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: currentUserId,
                    session_id: currentSessionId,
                    message: text
                })
            });
            const data = await res.json();

            // Append assistant bubble
            const botBubble = document.createElement("div");
            botBubble.className = "chat-bubble assistant";
            botBubble.innerText = data.response;
            messagesList.appendChild(botBubble);

            // Update inspection metrics
            document.getElementById("turn-latency-badge").innerText = `${data.latency_ms} ms`;
            document.getElementById("raw-context-inspector").innerText = JSON.stringify(data, null, 2);

            messagesList.scrollTop = messagesList.scrollHeight;
        } catch (e) {
            const errBubble = document.createElement("div");
            errBubble.className = "chat-bubble assistant";
            errBubble.innerText = "Error executing turn: " + e.message;
            messagesList.appendChild(errBubble);
        }
    }

    sendBtn.addEventListener("click", sendTurn);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendTurn();
        }
    });

    // 4. Letta Core RAM Blocks Manager
    async function loadWorkingMemory() {
        currentUserId = document.getElementById("global-user-id").value;
        const container = document.getElementById("ram-blocks-list");
        container.innerHTML = "<div class='empty-state'>Fetching RAM blocks...</div>";

        try {
            const res = await fetch(`/api/working-memory?user_id=${encodeURIComponent(currentUserId)}`);
            const data = await res.json();
            const blocks = data.blocks || {};

            container.innerHTML = "";
            const keys = Object.keys(blocks);

            if (keys.length === 0) {
                container.innerHTML = "<div class='empty-state'>No core RAM blocks found.</div>";
                return;
            }

            keys.forEach(k => {
                const b = blocks[k];
                const card = document.createElement("div");
                card.className = "ram-block-card";
                card.innerHTML = `
                    <header>
                        <span class="ram-block-title">${k}</span>
                        <button class="btn btn-secondary btn-xs btn-del" data-label="${k}">Delete</button>
                    </header>
                    <p style="font-size: 12px; color: #475569;">${typeof b === 'object' ? b.description || '' : ''}</p>
                    <pre class="code-block" style="margin-top: 6px;">${typeof b === 'object' ? b.value || '' : b}</pre>
                `;
                container.appendChild(card);
            });

            // Bind delete buttons
            document.querySelectorAll(".btn-del").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const label = btn.getAttribute("data-label");
                    await fetch(`/api/working-memory/block?user_id=${encodeURIComponent(currentUserId)}&label=${encodeURIComponent(label)}`, { method: "DELETE" });
                    loadWorkingMemory();
                });
            });

        } catch (e) {
            container.innerHTML = `<div class='empty-state'>Error loading RAM blocks: ${e.message}</div>`;
        }
    }

    document.getElementById("btn-refresh-blocks").addEventListener("click", loadWorkingMemory);

    document.getElementById("form-create-block").addEventListener("submit", async (e) => {
        e.preventDefault();
        const label = document.getElementById("new-block-label").value;
        const desc = document.getElementById("new-block-desc").value;
        const val = document.getElementById("new-block-val").value;

        await fetch("/api/working-memory/block", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: currentUserId, label, description: desc, value: val })
        });

        document.getElementById("form-create-block").reset();
        loadWorkingMemory();
    });

    document.getElementById("btn-run-doctor").addEventListener("click", async () => {
        const docDiv = document.getElementById("doctor-audit-results");
        docDiv.innerHTML = "<p>Running Letta Memory Doctor audit...</p>";

        const res = await fetch(`/api/working-memory/doctor?user_id=${encodeURIComponent(currentUserId)}`);
        const data = await res.json();
        docDiv.innerHTML = `<pre class="code-block">${JSON.stringify(data, null, 2)}</pre>`;
    });

    // 5. Mem0 & LightRAG Search Playground
    document.getElementById("btn-run-hybrid-search").addEventListener("click", async () => {
        const q = document.getElementById("search-query-input").value.trim();
        if (!q) return;

        const res = await fetch("/api/atomic-memory/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: currentUserId, query: q, limit: 5, session_id: currentSessionId })
        });
        const data = await res.json();
        const container = document.getElementById("search-results-list");
        container.innerHTML = "";

        if (data.results.length === 0) {
            container.innerHTML = "<div class='empty-state'>No facts found.</div>";
            return;
        }

        data.results.forEach(r => {
            const div = document.createElement("div");
            div.className = "ram-block-card mb-2";
            div.innerHTML = `
                <div style="font-weight: 500;">${r.fact_text}</div>
                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">
                    RRF Score: ${r.rrf_score.toFixed(4)} | Similarity: ${r.similarity.toFixed(2)} | Importance: ${r.importance_score}
                </div>
            `;
            container.appendChild(div);
        });
    });

    document.getElementById("btn-run-dual-search").addEventListener("click", async () => {
        const q = document.getElementById("search-query-input").value.trim();
        if (!q) return;

        const res = await fetch("/api/atomic-memory/dual-search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: currentUserId, query: q, limit: 5, session_id: currentSessionId })
        });
        const data = await res.json();
        document.getElementById("dual-search-results-json").innerText = JSON.stringify(data, null, 2);
    });

    document.getElementById("btn-load-all-facts").addEventListener("click", async () => {
        const container = document.getElementById("all-facts-table-container");
        const res = await fetch(`/api/atomic-memory/facts?user_id=${encodeURIComponent(currentUserId)}`);
        const data = await res.json();

        if (data.facts.length === 0) {
            container.innerHTML = "<div class='empty-state'>No active facts in memory.</div>";
            return;
        }

        let html = `<table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left;">
            <thead><tr style="border-bottom:1px solid #E2E8F0;"><th style="padding:6px;">Fact Text</th><th style="padding:6px;">Importance</th><th style="padding:6px;">Created At</th></tr></thead>
            <tbody>`;
        data.facts.forEach(f => {
            html += `<tr style="border-bottom:1px solid #F1F5F9;">
                <td style="padding:6px;">${f.fact_text}</td>
                <td style="padding:6px;">${f.importance_score}</td>
                <td style="padding:6px;">${new Date(f.created_at).toLocaleString()}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        container.innerHTML = html;
    });

    // 6. Graphiti & Cognee Knowledge Graph
    async function loadKnowledgeGraph() {
        const svg = document.getElementById("knowledge-graph-svg");
        svg.innerHTML = "";

        const res = await fetch(`/api/graph?user_id=${encodeURIComponent(currentUserId)}`);
        const data = await res.json();
        const nodes = data.nodes || [];
        const links = data.links || [];

        if (nodes.length === 0) {
            svg.innerHTML = "<text x='50%' y='50%' text-anchor='middle'>No graph nodes created yet. Execute a turn to extract entities.</text>";
            return;
        }

        // Render simple SVG Node-Link graph
        const width = svg.clientWidth || 600;
        const height = 440;
        const radius = 24;

        nodes.forEach((n, idx) => {
            const angle = (idx / nodes.length) * 2 * Math.PI;
            const x = width / 2 + Math.cos(angle) * (width / 3.5);
            const y = height / 2 + Math.sin(angle) * (height / 3.5);

            n.x = x;
            n.y = y;
        });

        // Draw Links
        links.forEach(l => {
            const src = nodes.find(n => n.id === l.source);
            const tgt = nodes.find(n => n.id === l.target);
            if (src && tgt) {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", src.x);
                line.setAttribute("y1", src.y);
                line.setAttribute("x2", tgt.x);
                line.setAttribute("y2", tgt.y);
                line.setAttribute("stroke", "#CBD5E1");
                line.setAttribute("stroke-width", "2");
                svg.appendChild(line);
            }
        });

        // Draw Nodes
        nodes.forEach(n => {
            const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", n.x);
            circle.setAttribute("cy", n.y);
            circle.setAttribute("r", radius);
            circle.setAttribute("fill", "#EFF6FF");
            circle.setAttribute("stroke", "#2563EB");
            circle.setAttribute("stroke-width", "2");
            g.appendChild(circle);

            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", n.x);
            text.setAttribute("y", n.y + 4);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("font-weight", "600");
            text.setAttribute("font-size", "11px");
            text.textContent = n.name;
            g.appendChild(text);

            svg.appendChild(g);
        });
    }

    document.getElementById("btn-load-graph").addEventListener("click", loadKnowledgeGraph);

    document.getElementById("btn-export-graph-json").addEventListener("click", async () => {
        const res = await fetch(`/api/graph?user_id=${encodeURIComponent(currentUserId)}`);
        const data = await res.json();
        alert(JSON.stringify(data, null, 2));
    });

    document.getElementById("btn-run-ppr").addEventListener("click", async () => {
        const input = document.getElementById("ppr-seed-input").value.trim();
        const seeds = input ? input.split(",").map(s => s.trim()) : ["Postgres DB"];

        const res = await fetch("/api/graph/ppr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: currentUserId, seed_nodes: seeds })
        });
        const data = await res.json();
        document.getElementById("ppr-results-list").innerHTML = `<pre class="code-block">${JSON.stringify(data, null, 2)}</pre>`;
    });

    // 7. Procedural Playbooks
    async function loadProceduralPlaybooks() {
        const container = document.getElementById("procedural-playbooks-container");
        const res = await fetch(`/api/procedural?user_id=${encodeURIComponent(currentUserId)}&task=deploy rust microservice`);
        const data = await res.json();
        container.innerHTML = `<pre class="code-block">${JSON.stringify(data, null, 2)}</pre>`;
    }
    document.getElementById("btn-load-playbooks").addEventListener("click", loadProceduralPlaybooks);

    document.getElementById("btn-run-dream-cycle").addEventListener("click", async () => {
        const res = await fetch(`/api/atomic-memory/dream-cycle?user_id=${encodeURIComponent(currentUserId)}`, { method: "POST" });
        const data = await res.json();
        document.getElementById("dream-cycle-result").innerText = JSON.stringify(data, null, 2);
    });
});
