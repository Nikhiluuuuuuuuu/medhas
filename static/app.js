/* ==========================================================================
   MEDHAS AI - FRONTEND APPLICATION SCRIPT
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const userIdInput = document.getElementById('user-id-input');
  const btnSwitchUser = document.getElementById('btn-switch-user');
  const btnNewChat = document.getElementById('btn-new-chat');
  const sessionsList = document.getElementById('sessions-list');
  const sessionCountBadge = document.getElementById('session-count');
  
  const currentSessionTitle = document.getElementById('current-session-title');
  const currentSessionSubtitle = document.getElementById('current-session-subtitle');
  const welcomeScreen = document.getElementById('welcome-screen');
  const messageList = document.getElementById('message-list');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');

  const btnToggleInspector = document.getElementById('btn-toggle-inspector');
  const btnCloseInspector = document.getElementById('btn-close-inspector');
  const inspectorDrawer = document.getElementById('inspector-drawer');
  const btnQuickDream = document.getElementById('btn-quick-dream');
  const btnTriggerDreamDrawer = document.getElementById('btn-trigger-dream-drawer');

  // Tab Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const workingBlocksContainer = document.getElementById('working-blocks-container');
  const factsContainer = document.getElementById('facts-container');
  const factSearchInput = document.getElementById('fact-search-input');
  const btnRefreshRam = document.getElementById('btn-refresh-ram');
  const btnRefreshFacts = document.getElementById('btn-refresh-facts');
  const btnRefreshGraph = document.getElementById('btn-refresh-graph');

  // App State
  let currentUserId = userIdInput.value.trim() || 'test_user_verified_10_10';
  let currentSessionId = localStorage.getItem(`medhas_session_${currentUserId}`) || null;
  let sessions = JSON.parse(localStorage.getItem(`medhas_sessions_${currentUserId}`) || '[]');
  let isProcessing = false;

  // Configure Marked Markdown Renderer
  marked.setOptions({
    highlight: function(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
    breaks: true
  });

  // Initialize Feather Icons
  feather.replace();

  // Initialize Application
  init();

  async function init() {
    setupEventListeners();

    if (sessions.length > 0 && !currentSessionId) {
      currentSessionId = sessions[0].id;
    }

    if (!currentSessionId) {
      await createNewSession();
    } else {
      renderSessionsList();
      await loadTranscript(currentSessionId);
    }

    refreshAllMemoryInspectors();
  }

  function setupEventListeners() {
    // Switch User
    btnSwitchUser.addEventListener('click', async () => {
      const newUserId = userIdInput.value.trim();
      if (!newUserId) return;
      currentUserId = newUserId;
      currentSessionId = localStorage.getItem(`medhas_session_${currentUserId}`) || null;
      sessions = JSON.parse(localStorage.getItem(`medhas_sessions_${currentUserId}`) || '[]');

      if (!currentSessionId) {
        await createNewSession();
      } else {
        renderSessionsList();
        await loadTranscript(currentSessionId);
      }
      refreshAllMemoryInspectors();
    });

    // New Chat
    btnNewChat.addEventListener('click', async () => {
      await createNewSession();
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
    });

    // Send Form
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleSendMessage();
    });

    // Enter Key Handler (Shift+Enter for newline)
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    // Inspector Drawer Toggles
    btnToggleInspector.addEventListener('click', () => {
      inspectorDrawer.classList.toggle('closed');
    });
    btnCloseInspector.addEventListener('click', () => {
      inspectorDrawer.classList.add('closed');
    });

    // Dream Cycle Triggers
    btnQuickDream.addEventListener('click', triggerDreamCycle);
    btnTriggerDreamDrawer.addEventListener('click', triggerDreamCycle);

    // Inspector Tabs
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${targetTab}`).classList.add('active');
      });
    });

    // Memory Refresh Buttons
    btnRefreshRam.addEventListener('click', fetchWorkingRAM);
    btnRefreshFacts.addEventListener('click', fetchAtomicFacts);
    btnRefreshGraph.addEventListener('click', fetchKnowledgeGraph);

    // Fact Search Filter
    factSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      document.querySelectorAll('.fact-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(q) ? 'flex' : 'none';
      });
    });
  }

  // Session Management
  async function createNewSession() {
    try {
      const res = await fetch(`/session/create?user_id=${encodeURIComponent(currentUserId)}`, { method: 'POST' });
      const data = await res.json();

      if (data.status === 'success') {
        currentSessionId = data.session_id;
        const newSessionObj = { id: currentSessionId, createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
        
        sessions.unshift(newSessionObj);
        localStorage.setItem(`medhas_sessions_${currentUserId}`, JSON.stringify(sessions));
        localStorage.setItem(`medhas_session_${currentUserId}`, currentSessionId);

        renderSessionsList();
        clearMessageArea();
        updateHeaderInfo();
      }
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  }

  function renderSessionsList() {
    sessionsList.innerHTML = '';
    sessionCountBadge.textContent = sessions.length;

    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = `session-item ${s.id === currentSessionId ? 'active' : ''}`;
      item.innerHTML = `
        <i data-feather="message-square"></i>
        <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Session ${s.id.substring(0, 8)}...</span>
        <span class="subtext">${s.createdAt || ''}</span>
      `;
      item.addEventListener('click', () => switchSession(s.id));
      sessionsList.appendChild(item);
    });
    feather.replace();
  }

  async function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    currentSessionId = sessionId;
    localStorage.setItem(`medhas_session_${currentUserId}`, currentSessionId);
    renderSessionsList();
    await loadTranscript(sessionId);
    updateHeaderInfo();
  }

  function updateHeaderInfo() {
    currentSessionTitle.textContent = `Session: ${currentSessionId ? currentSessionId.substring(0, 12) + '...' : 'None'}`;
    currentSessionSubtitle.textContent = `User: ${currentUserId}`;
  }

  // Transcript & Messages
  async function loadTranscript(sessionId) {
    clearMessageArea();
    try {
      const res = await fetch(`/session/transcript/${sessionId}`);
      const data = await res.json();

      if (data.status === 'success' && data.messages && data.messages.length > 0) {
        welcomeScreen.style.display = 'none';
        data.messages.forEach(msg => {
          appendMessageRow(msg.role, msg.content);
        });
        scrollToBottom();
      } else {
        welcomeScreen.style.display = 'flex';
      }
    } catch (err) {
      console.error('Failed to load transcript:', err);
      welcomeScreen.style.display = 'flex';
    }
  }

  function clearMessageArea() {
    messageList.innerHTML = '';
    welcomeScreen.style.display = 'flex';
  }

  async function handleSendMessage() {
    const text = userInput.value.trim();
    if (!text || isProcessing) return;

    // Hide welcome screen
    welcomeScreen.style.display = 'none';

    // Clear & reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    isProcessing = true;
    btnSend.disabled = true;

    // Render User Message
    appendMessageRow('user', text);
    scrollToBottom();

    // Render Typing Indicator
    const typingRow = appendTypingIndicator();
    scrollToBottom();

    try {
      const res = await fetch('/turn/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          session_id: currentSessionId,
          message: text
        })
      });

      const data = await res.json();
      typingRow.remove();

      if (data.status === 'success') {
        appendMessageRow('assistant', data.response);
        scrollToBottom();
        refreshAllMemoryInspectors();
      } else {
        appendMessageRow('assistant', `⚠️ Error: ${data.detail || 'Failed to process response'}`);
      }
    } catch (err) {
      typingRow.remove();
      appendMessageRow('assistant', `⚠️ Network Error: Unable to reach backend engine.`);
    } finally {
      isProcessing = false;
      btnSend.disabled = false;
    }
  }

  window.sendSuggestedMessage = function(text) {
    userInput.value = text;
    handleSendMessage();
  };

  function appendMessageRow(role, content) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? 'U' : '<i data-feather="cpu"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'user') {
      bubble.textContent = content;
    } else {
      bubble.innerHTML = marked.parse(content);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    messageList.appendChild(row);

    feather.replace();
    return row;
  }

  function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i data-feather="cpu"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble typing-bubble';
    bubble.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messageList.appendChild(row);
    feather.replace();
    return row;
  }

  function scrollToBottom() {
    const container = document.getElementById('messages-container');
    container.scrollTop = container.scrollHeight;
  }

  // Memory Inspector Fetching
  function refreshAllMemoryInspectors() {
    fetchWorkingRAM();
    fetchAtomicFacts();
    fetchKnowledgeGraph();
  }

  async function fetchWorkingRAM() {
    try {
      const res = await fetch(`/working/blocks?user_id=${encodeURIComponent(currentUserId)}`);
      const data = await res.json();

      if (data.blocks) {
        workingBlocksContainer.innerHTML = '';
        Object.entries(data.blocks).forEach(([name, block]) => {
          const card = document.createElement('div');
          card.className = 'ram-block-card';
          card.innerHTML = `
            <h5><i data-feather="layers"></i> ${name}</h5>
            <pre>${typeof block.value === 'object' ? JSON.stringify(block.value, null, 2) : block.value || '(Empty)'}</pre>
          `;
          workingBlocksContainer.appendChild(card);
        });
        feather.replace();
      }
    } catch (err) {
      console.error('Failed to fetch RAM blocks:', err);
    }
  }

  async function fetchAtomicFacts() {
    try {
      const res = await fetch(`/memory/facts?user_id=${encodeURIComponent(currentUserId)}`);
      const data = await res.json();

      if (data.facts) {
        factsContainer.innerHTML = '';
        if (data.facts.length === 0) {
          factsContainer.innerHTML = '<div class="loading-spinner">No atomic facts stored yet.</div>';
          return;
        }

        data.facts.forEach(fact => {
          const item = document.createElement('div');
          item.className = 'fact-item active-fact';
          item.innerHTML = `
            <div class="fact-text">${fact.fact_text || fact}</div>
            <div class="fact-meta">
              <span>Confidence: ${fact.confidence_score || '0.90'}</span>
              <span>Layer 3 Mem0</span>
            </div>
          `;
          factsContainer.appendChild(item);
        });
      }
    } catch (err) {
      console.error('Failed to fetch facts:', err);
    }
  }

  let cachedGraphData = null;

  const btnExpandGraph = document.getElementById('btn-expand-graph');
  const btnCloseGraphModal = document.getElementById('btn-close-graph-modal');
  const graphModal = document.getElementById('graph-modal');

  if (btnExpandGraph) {
    btnExpandGraph.addEventListener('click', () => {
      if (graphModal) {
        graphModal.classList.add('open');
        if (cachedGraphData) {
          setTimeout(() => renderD3Graph('modal-graph-svg', cachedGraphData), 50);
        }
      }
    });
  }

  if (btnCloseGraphModal) {
    btnCloseGraphModal.addEventListener('click', () => {
      if (graphModal) graphModal.classList.remove('open');
    });
  }

  async function fetchKnowledgeGraph() {
    try {
      const res = await fetch(`/memory/graph?user_id=${encodeURIComponent(currentUserId)}`);
      const data = await res.json();

      if (data.status === 'success' && data.graph) {
        cachedGraphData = data.graph;
        const { nodes, links, stats } = data.graph;
        document.getElementById('stat-nodes').textContent = stats ? stats.total_nodes : nodes.length;
        document.getElementById('stat-edges').textContent = stats ? stats.total_active_edges : links.length;

        // Render D3 SVG graph in drawer
        renderD3Graph('graph-svg', cachedGraphData);

        const nodesList = document.getElementById('graph-nodes-list');
        nodesList.innerHTML = '';

        if (nodes.length === 0) {
          nodesList.innerHTML = '<div class="loading-spinner">No entity nodes created yet.</div>';
          return;
        }

        nodes.forEach(n => {
          const item = document.createElement('div');
          item.className = 'node-item';
          item.innerHTML = `
            <span class="node-name">${n.name}</span>
            <span class="node-type">${n.entity_type || 'Entity'}</span>
          `;
          nodesList.appendChild(item);
        });
      }
    } catch (err) {
      console.error('Failed to fetch graph:', err);
    }
  }

  function renderD3Graph(svgId, graphData) {
    if (!graphData || !graphData.nodes || typeof d3 === 'undefined') return;
    const svg = d3.select(`#${svgId}`);
    svg.selectAll('*').remove();

    const svgElement = document.getElementById(svgId);
    const width = svgElement ? (svgElement.clientWidth || (svgId === 'graph-svg' ? 300 : 900)) : 300;
    const height = svgElement ? (svgElement.clientHeight || (svgId === 'graph-svg' ? 260 : 600)) : 260;

    const nodes = graphData.nodes.map(d => ({ ...d }));
    const links = graphData.links.map(d => ({ ...d }));

    if (nodes.length === 0) return;

    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', (event) => {
      g.attr('transform', event.transform);
    }));

    const colorScale = d3.scaleOrdinal()
      .domain(['Person', 'Company', 'Skill', 'Entity', 'Role'])
      .range(['#4f46e5', '#059669', '#d97706', '#7c3aed', '#0284c7']);

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('class', 'd3-link');

    const linkLabel = g.append('g')
      .selectAll('text')
      .data(links)
      .enter().append('text')
      .attr('class', 'd3-link-label')
      .text(d => d.relationship || 'LINK');

    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', 14)
      .attr('fill', d => colorScale(d.entity_type || 'Entity'))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('class', 'd3-node');

    node.append('text')
      .attr('dy', 26)
      .attr('class', 'd3-node-label')
      .text(d => d.name);

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      linkLabel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2 - 3);

      node
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }

  async function triggerDreamCycle() {
    try {
      const res = await fetch(`/memory/dream?user_id=${encodeURIComponent(currentUserId)}`, { method: 'POST' });
      const data = await res.json();

      if (data.status === 'success') {
        refreshAllMemoryInspectors();
        alert('🌙 Dream Cycle Reflection Completed! Long-term memory consolidated.');
      }
    } catch (err) {
      alert('Failed to trigger dream cycle.');
    }
  }
});

