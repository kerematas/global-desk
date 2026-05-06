/*
  frontend/app.js — Client-side chat logic for Global Desk.

  Handles the single-conversation UI: sending messages to the /api/chat
  endpoint, rendering assistant replies with formatted text and source
  links, persisting chat history in localStorage, and managing the
  typing indicator and starter prompt chips.
*/

// localStorage key for saving the conversation between page refreshes
const STORAGE_KEY = "global-desk-single-chat-v2";

// default greeting shown on first load and after clearing the chat
const STARTER_MESSAGES = [
  {
    role: "assistant",
    author: "Global Desk",
    skipForApi: true,  // don't send the greeting to the API as part of history
    text:
      "Hi! Ask a question about CPT, OPT, travel, taxes, work authorization, or maintaining status. I’ll answer using the existing Global Desk knowledge base."
  }
];

// Grab all of the elements we need once at startup.
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const chatMessages = document.getElementById("chatMessages");
const clearChatButton = document.getElementById("clearChatButton");
const sendButton = document.getElementById("sendButton");
const promptChips = document.querySelectorAll(".prompt-chip");

// Keep the current conversation in memory.
let messages = loadMessages();

// Track whether the interface is waiting on a response.
let isWaitingForReply = false;

/*
  Read saved messages from localStorage.
  If nothing is saved yet, fall back to the starter conversation.
*/
function loadMessages() {
  const saved = window.localStorage.getItem(STORAGE_KEY);

  if (!saved) {
    return [...STARTER_MESSAGES];
  }

  try {
    const parsed = JSON.parse(saved);
    return Array.isArray(parsed) && parsed.length > 0
      ? parsed
      : [...STARTER_MESSAGES];
  } catch (error) {
    console.error("Could not parse saved chat history:", error);
    return [...STARTER_MESSAGES];
  }
}

/*
  Save the current conversation so a refresh keeps the same single chat.
*/
function saveMessages() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

/*
  Build the chat history array that gets sent to the backend.
  We exclude the latest message (sent separately) and the starter greeting.
*/
function buildApiHistory() {
  return messages
    .slice(0, -1)                                    // everything except the newest message
    .filter((message) => !message.skipForApi)         // skip the starter greeting
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      role: message.role,
      content: message.text
    }));
}

/*
  Render the full message list from scratch.
  This is simple and perfectly fine for a small single-chat prototype.
*/
function renderMessages() {
  chatMessages.innerHTML = "";

  messages.forEach((message) => {
    chatMessages.appendChild(createMessageElement(message));
  });

  scrollToBottom();
}

/*
  Build one message row.
  The markup matches the CSS classes in styles.css.
*/
function createMessageElement(message) {
  const row = document.createElement("article");
  row.className = `message-row ${message.role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = message.role === "assistant" ? "GD" : "You";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = message.author;

  const body = document.createElement("div");
  body.className = "message-text";

  appendFormattedMessage(body, message.text);

  bubble.append(role, body);

  if (message.role === "assistant" && Array.isArray(message.sources) && message.sources.length > 0) {
    bubble.appendChild(createSourcesElement(message.sources));
  }

  row.append(avatar, bubble);
  return row;
}

/*
  Render a simple formatted message body.
  We support the most useful structures for this project:
  short paragraphs, section labels, bullet lists, and numbered lists.
*/
function appendFormattedMessage(container, text) {
  const blocks = String(text)
    .trim()
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  if (blocks.length === 0) {
    const paragraph = document.createElement("p");
    paragraph.textContent = "";
    container.appendChild(paragraph);
    return;
  }

  blocks.forEach((block) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) {
      return;
    }

    // check if the first line looks like a section title (ends with ":")
    const normalizedTitle = lines[0]
      .replace(/^[-*]\s+/, "")
      .replace(/^\d+\.\s+/, "");

    const hasSectionTitle = normalizedTitle.endsWith(":") && lines.length > 1;
    const contentLines = hasSectionTitle ? lines.slice(1) : lines;

    if (hasSectionTitle) {
      const title = document.createElement("p");
      title.className = "message-section-title";
      title.textContent = normalizedTitle;
      container.appendChild(title);
    }

    // if all lines start with - or *, render as a bullet list
    if (contentLines.every((line) => /^[-*]\s+/.test(line))) {
      const list = document.createElement("ul");
      list.className = "message-list";

      contentLines.forEach((line) => {
        const item = document.createElement("li");
        item.textContent = line.replace(/^[-*]\s+/, "");
        list.appendChild(item);
      });

      container.appendChild(list);
      return;
    }

    // if all lines start with a number, render as a numbered list
    if (contentLines.every((line) => /^\d+\.\s+/.test(line))) {
      const list = document.createElement("ol");
      list.className = "message-list";

      contentLines.forEach((line) => {
        const item = document.createElement("li");
        item.textContent = line.replace(/^\d+\.\s+/, "");
        list.appendChild(item);
      });

      container.appendChild(list);
      return;
    }

    // otherwise just render as a plain paragraph
    const paragraph = document.createElement("p");
    paragraph.textContent = contentLines.join("\n");
    container.appendChild(paragraph);
  });
}

/*
  Show source URLs in a lightweight footer under assistant messages.
*/
function createSourcesElement(sources) {
  const wrapper = document.createElement("div");
  wrapper.className = "message-sources";

  const label = document.createElement("p");
  label.className = "message-sources-label";
  label.textContent = "Sources";
  wrapper.appendChild(label);

  const list = document.createElement("div");
  list.className = "message-sources-list";

  sources.forEach((sourceItem) => {
    const link = document.createElement("a");
    link.className = "message-source-link";
    link.href = sourceItem.source;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = sourceItem.source;
    list.appendChild(link);
  });

  wrapper.appendChild(list);
  return wrapper;
}

/*
  Add the temporary typing indicator while the assistant is "thinking."
*/
function showTypingIndicator() {
  const row = document.createElement("article");
  row.className = "message-row assistant";
  row.id = "typingIndicator";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "GD";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const role = document.createElement("p");
  role.className = "message-role";
  role.textContent = "Global Desk";

  const typing = document.createElement("div");
  typing.className = "typing-bubble";

  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.className = "typing-dot";
    typing.appendChild(dot);
  }

  bubble.append(role, typing);
  row.append(avatar, bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
}

/*
  Remove the typing indicator once we have a response.
*/
function hideTypingIndicator() {
  const typingIndicator = document.getElementById("typingIndicator");

  if (typingIndicator) {
    typingIndicator.remove();
  }
}

/*
  Small helper so the newest message stays visible.
*/
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/*
  Keep the textarea height comfortable as the user types.
*/
function autoResizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
}

/*
  Send the latest message plus prior chat turns to the backend API.
  The backend stays stateless, so the browser sends the current single-chat
  history with each request.
*/
async function getAssistantReply(userMessage) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      message: userMessage,
      history: buildApiHistory()
    })
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Something went wrong while contacting the server.");
  }

  return data;
}

/*
  Main submit handler — runs when the user sends a message.
*/
async function handleSubmit(event) {
  event.preventDefault();

  const text = messageInput.value.trim();

  // don't send empty messages or double-send while waiting
  if (!text || isWaitingForReply) {
    return;
  }

  isWaitingForReply = true;
  sendButton.disabled = true;

  // add the user's message to the chat right away
  messages.push({
    role: "user",
    author: "You",
    text
  });

  saveMessages();
  renderMessages();

  // clear the input and show the typing dots
  messageInput.value = "";
  autoResizeInput();
  showTypingIndicator();

  try {
    // send to the backend and wait for the RAG answer
    const result = await getAssistantReply(text);

    hideTypingIndicator();

    // add the assistant's response to the chat
    messages.push({
      role: "assistant",
      author: "Global Desk",
      text: result.answer,
      sources: result.sources || []
    });

    saveMessages();
    renderMessages();
  } catch (error) {
    // if something goes wrong, show the error as a chat message
    console.error("Could not create assistant reply:", error);
    hideTypingIndicator();

    messages.push({
      role: "assistant",
      author: "Global Desk",
      text: error.message || "Something went wrong while contacting the server."
    });

    saveMessages();
    renderMessages();
  } finally {
    isWaitingForReply = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
}

/*
  Clear the current single conversation and restore the starter message.
*/
function handleClearChat() {
  messages = [...STARTER_MESSAGES];
  saveMessages();
  renderMessages();
  messageInput.value = "";
  autoResizeInput();
  messageInput.focus();
}

/*
  Clicking a starter prompt sends that question immediately.
  This makes the layout feel more interactive without adding extra UI.
*/
function handlePromptChipClick(event) {
  const prompt = event.currentTarget.dataset.prompt;

  if (!prompt) {
    return;
  }

  messageInput.value = prompt;
  autoResizeInput();
  chatForm.requestSubmit();
}

/*
  Let Enter send the message, while Shift + Enter still adds a new line.
*/
function handleComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
}

// Initial render on page load.
renderMessages();
autoResizeInput();

// Wire up the main interactions.
chatForm.addEventListener("submit", handleSubmit);
messageInput.addEventListener("input", autoResizeInput);
messageInput.addEventListener("keydown", handleComposerKeydown);
clearChatButton.addEventListener("click", handleClearChat);
promptChips.forEach((chip) => chip.addEventListener("click", handlePromptChipClick));
